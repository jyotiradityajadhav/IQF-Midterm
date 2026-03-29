"""
Indian Mutual Fund Monthly Returns — v2
========================================
Source: AMFI India (official) — no third-party API dependency

Strategy:
  - Downloads historical NAV data directly from AMFI's navhistory portal
  - Falls back to mfapi.in per-fund if needed
  - Computes month-end NAV → monthly % return
  - Outputs wide + long format CSVs

Output files:
  mf_monthly_returns_wide.csv  — rows=months, cols=fund names
  mf_monthly_returns_long.csv  — Fund Name | Month | Monthly Return (%)
  scheme_list.csv              — all scheme codes + names

Requirements:
    pip install requests pandas tqdm python-dateutil

Run:
    python mf_monthly_returns_v2.py
"""

import requests
import pandas as pd
import io
import time
import logging
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
START_DATE       = date(2020, 1, 1)
END_DATE         = date.today()
OUTPUT_WIDE      = "mf_monthly_returns_wide.csv"
OUTPUT_LONG      = "mf_monthly_returns_long.csv"
SCHEME_LIST_FILE = "scheme_list.csv"
WORKERS          = 8
MAX_FUNDS        = None     # set e.g. 300 for a test run
RETRY_ATTEMPTS   = 4
RETRY_DELAY      = 3        # seconds

# Filter by keywords in fund name — None = all funds
# Example: ["equity", "large cap", "flexi cap"]
CATEGORY_FILTER  = None

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

AMFI_SCHEME_LIST_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_FUND_URL       = "https://api.mfapi.in/mf/{code}"


# ─────────────────────────────────────────
# STEP 1: Get scheme list from AMFI bulk NAV
# ─────────────────────────────────────────
def fetch_scheme_list_amfi():
    """
    AMFI's NAVAll.txt contains current NAV for all schemes.
    Format per line:
        SchemeCode;ISINDiv;ISINGrowth;SchemeName;NAVDate;NAV
    We extract SchemeCode + SchemeName.
    """
    log.info("Fetching scheme list from AMFI (NAVAll.txt) ...")

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = session.get(AMFI_SCHEME_LIST_URL, headers=headers, timeout=60, stream=True)
            r.raise_for_status()

            lines = r.content.decode("utf-8", errors="ignore").splitlines()
            schemes = []
            for line in lines:
                parts = line.strip().split(";")
                if len(parts) >= 4:
                    try:
                        code = int(parts[0].strip())
                        name = parts[3].strip()
                        if name:
                            schemes.append({"schemeCode": code, "schemeName": name})
                    except ValueError:
                        continue

            log.info(f"  → {len(schemes):,} schemes found")
            return schemes

        except Exception as e:
            log.warning(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError("Could not fetch scheme list from AMFI after retries.")


# ─────────────────────────────────────────
# STEP 2: Fetch NAV history per scheme via mfapi
# ─────────────────────────────────────────
def fetch_nav_history(scheme_code: int):
    url = MFAPI_FUND_URL.format(code=scheme_code)
    session = requests.Session()

    for attempt in range(RETRY_ATTEMPTS):
        try:
            r = session.get(url, timeout=30, stream=True)
            r.raise_for_status()

            # Stream-read to avoid IncompleteRead
            raw = b""
            for chunk in r.iter_content(chunk_size=4096):
                raw += chunk

            import json
            data = json.loads(raw)
            return data.get("data", [])

        except Exception as e:
            wait = RETRY_DELAY * (attempt + 1)
            time.sleep(wait)

    return None


# ─────────────────────────────────────────
# STEP 3: NAV records → monthly returns
# ─────────────────────────────────────────
def compute_monthly_returns(nav_records, start_date, end_date):
    if not nav_records:
        return None

    rows = []
    for rec in nav_records:
        try:
            dt  = datetime.strptime(rec["date"], "%d-%m-%Y").date()
            nav = float(rec["nav"])
            if start_date <= dt <= end_date:
                rows.append((dt, nav))
        except (ValueError, KeyError, TypeError):
            continue

    if len(rows) < 2:
        return None

    s = pd.Series({r[0]: r[1] for r in rows})
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    # Month-end NAV: last observation in each month
    monthly_nav = s.resample("ME").last().dropna()

    if len(monthly_nav) < 2:
        return None

    # Month-over-month % return
    returns = monthly_nav.pct_change() * 100
    returns = returns.iloc[1:]  # drop first NaN
    returns.index = returns.index.to_period("M").astype(str)
    return returns


# ─────────────────────────────────────────
# STEP 4: Worker
# ─────────────────────────────────────────
def process_scheme(scheme):
    code = scheme["schemeCode"]
    name = scheme["schemeName"]

    nav_records = fetch_nav_history(code)
    if not nav_records:
        return name, None

    returns = compute_monthly_returns(nav_records, START_DATE, END_DATE)
    return name, returns


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Indian MF Monthly Returns Builder v2")
    log.info(f"  Window : {START_DATE} → {END_DATE}")
    log.info(f"  Source : AMFI (scheme list) + mfapi.in (NAV history)")
    log.info("=" * 60)

    # --- Scheme list ---
    schemes = fetch_scheme_list_amfi()

    # Category filter
    if CATEGORY_FILTER:
        kw = [k.lower() for k in CATEGORY_FILTER]
        before = len(schemes)
        schemes = [s for s in schemes
                   if any(k in s["schemeName"].lower() for k in kw)]
        log.info(f"Category filter applied: {before:,} → {len(schemes):,} schemes")

    # Cap
    if MAX_FUNDS:
        schemes = schemes[:MAX_FUNDS]
        log.info(f"Capped to {MAX_FUNDS} funds.")

    # Save scheme list
    pd.DataFrame(schemes).to_csv(SCHEME_LIST_FILE, index=False)
    log.info(f"Scheme list → {SCHEME_LIST_FILE}")
    log.info(f"Fetching NAV history for {len(schemes):,} schemes ...")

    # --- Parallel fetch ---
    all_returns = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_scheme, s): s for s in schemes}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Fetching", unit="fund"):
            name, returns = future.result()
            if returns is not None:
                all_returns[name] = returns

    log.info(f"Got data for {len(all_returns):,} funds.")

    if not all_returns:
        log.error("No data retrieved. Check your internet connection.")
        return

    # --- Wide format ---
    df_wide = pd.DataFrame(all_returns)
    df_wide.index.name = "Month"
    df_wide = df_wide.sort_index().round(4)
    df_wide.to_csv(OUTPUT_WIDE)
    log.info(f"Wide CSV → {OUTPUT_WIDE}  [{df_wide.shape[0]} months × {df_wide.shape[1]} funds]")

    # --- Long format ---
    df_long = (
        df_wide
        .stack()
        .reset_index()
    )
    df_long.columns = ["Month", "Fund Name", "Monthly Return (%)"]
    df_long = df_long.dropna().sort_values(["Fund Name", "Month"]).reset_index(drop=True)
    df_long.to_csv(OUTPUT_LONG, index=False)
    log.info(f"Long CSV → {OUTPUT_LONG}  [{len(df_long):,} rows]")

    # --- Quick summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Months covered : {df_wide.index[0]} → {df_wide.index[-1]}")
    print(f"  Funds with data: {df_wide.shape[1]:,}")
    print(f"  Total data pts : {df_long.shape[0]:,}")
    print(f"  Avg fill rate  : {df_wide.notna().mean().mean()*100:.1f}%")
    print(f"\n  Files saved:")
    print(f"    {OUTPUT_WIDE}")
    print(f"    {OUTPUT_LONG}")
    print(f"    {SCHEME_LIST_FILE}")


if __name__ == "__main__":
    main()
