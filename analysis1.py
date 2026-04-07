Code for Analysis

# -*- coding: utf-8 -*-

"""
Mutual Fund Persistence Analysis (Enhanced Version)
==================================================

Includes:
1. First-order transitions (basic persistence)
2. Second-order transitions (conditional persistence)
3. Path analysis (multi-step movement)
4. Momentum vs Reversal classification
5. Duration analysis (stickiness)

Regression REMOVED.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
INPUT_WIDE       = "mf_filtered_wide_2021_2026.csv"
INPUT_NIFTY      = "nifty50_monthly_2021_2026.csv"
INPUT_VIX        = "india_vix_monthly_2021_2026.csv"
INPUT_FACTORS    = "factors_monthly_2021_2026.csv"

RESULTS_DIR      = "results_2021_2026"
RANKING_WINDOW   = 12
HOLDING_WINDOW   = 12
N_QUINTILES      = 5
VIX_HIGH_THRESH  = 20.0

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
def load_data():
    df_wide = pd.read_csv(INPUT_WIDE, index_col=0)
    df_wide.index = pd.PeriodIndex(df_wide.index, freq="M")
    df_wide = df_wide.sort_index()

    nifty, vix = None, None

    if os.path.exists(INPUT_NIFTY):
        nifty = pd.read_csv(INPUT_NIFTY, index_col=0)
        nifty.index = pd.PeriodIndex(nifty.index, freq="M")
        nifty = nifty.squeeze()

    if os.path.exists(INPUT_VIX):
        vix = pd.read_csv(INPUT_VIX, index_col=0)
        vix.index = pd.PeriodIndex(vix.index, freq="M")
        vix = vix.squeeze()

    return df_wide, nifty, vix

# ─────────────────────────────────────────
# REGIMES
# ─────────────────────────────────────────
def define_regimes(df_wide, nifty, vix):
    months = df_wide.index

    if nifty is not None:
        nifty = nifty.reindex(months)
        cum = (1 + nifty/100).rolling(12).apply(np.prod) - 1
        bull = cum > 0
    else:
        proxy = df_wide.median(axis=1)
        cum = (1 + proxy/100).rolling(12).apply(np.prod) - 1
        bull = cum > 0

    if vix is not None:
        highvol = vix.reindex(months) > VIX_HIGH_THRESH
    else:
        highvol = df_wide.std(axis=1) > df_wide.std(axis=1).quantile(0.75)

    regime = pd.DataFrame(index=months)
    regime["Regime"] = np.where(highvol, "HighVol",
                        np.where(bull, "Bull", "Bear"))
    return regime

# ─────────────────────────────────────────
# ROLLING QUINTILES
# ─────────────────────────────────────────
def compute_rolling_quintiles(df):
    months = df.index
    rows = []
    step = 6

    for i in range(RANKING_WINDOW, len(months) - HOLDING_WINDOW, step):
        rank_period = months[i-RANKING_WINDOW:i]
        hold_period = months[i:i+HOLDING_WINDOW]

        rank_df = df.loc[rank_period].dropna(axis=1)
        if rank_df.shape[1] < 10:
            continue

        cum = (1 + rank_df/100).prod() - 1
        quint = pd.qcut(cum, N_QUINTILES,
                        labels=[f"Q{i}" for i in range(1,6)])

        fwd = df.loc[hold_period, rank_df.columns].mean()

        for fund in rank_df.columns:
            rows.append([
                str(months[i]), fund,
                quint[fund], fwd[fund]
            ])

    return pd.DataFrame(rows, columns=[
        "Date","Fund","Quintile","FwdReturn"
    ])

# ─────────────────────────────────────────
# FIRST ORDER
# ─────────────────────────────────────────
def first_order_transition(quint_df):
    pivot = quint_df.pivot(index="Date", columns="Fund", values="Quintile")
    dates = pivot.index

    counts = pd.DataFrame(0, index=[f"Q{i}" for i in range(1,6)],
                             columns=[f"Q{i}" for i in range(1,6)])

    for i in range(len(dates)-1):
        curr = pivot.loc[dates[i]]
        nxt  = pivot.loc[dates[i+1]]

        common = curr.dropna().index.intersection(nxt.dropna().index)

        for f in common:
            counts.loc[curr[f], nxt[f]] += 1

    return counts.div(counts.sum(axis=1), axis=0)*100

# ─────────────────────────────────────────
# SECOND ORDER
# ─────────────────────────────────────────
def second_order_transition(quint_df):
    pivot = quint_df.pivot(index="Date", columns="Fund", values="Quintile")
    dates = pivot.index

    data = {}

    for i in range(2, len(dates)):
        a = pivot.loc[dates[i-2]]
        b = pivot.loc[dates[i-1]]
        c = pivot.loc[dates[i]]

        common = a.dropna().index.intersection(
            b.dropna().index
         ).intersection(
            c.dropna().index
         )

        for f in common:
            key = (a[f], b[f])
            nxt = c[f]

            if key not in data:
                data[key] = {f"Q{i}":0 for i in range(1,6)}

            data[key][nxt] += 1

    rows = []
    for k,v in data.items():
        total = sum(v.values())
        probs = {kk: vv/total*100 for kk,vv in v.items()}
        rows.append({"From":f"{k[0]}→{k[1]}", **probs})

    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# PATH ANALYSIS
# ─────────────────────────────────────────
def path_analysis(quint_df):
    pivot = quint_df.pivot(index="Date", columns="Fund", values="Quintile")
    dates = pivot.index

    paths = {}

    for i in range(3, len(dates)):
        window = dates[i-3:i+1]
        sub = pivot.loc[window]

        for f in sub.columns:
            seq = sub[f].dropna()
            if len(seq)==4:
                p = tuple(seq.values)
                paths[p] = paths.get(p,0)+1

    df = pd.DataFrame([
        {"Path":"→".join(p),"Count":c}
        for p,c in paths.items()
    ]).sort_values("Count",ascending=False)

    return df

# ─────────────────────────────────────────
# PATH CLASSIFICATION
# ─────────────────────────────────────────
def classify_paths(df):
    def classify(p):
        nums = [int(x[1]) for x in p.split("→")]
        if nums == sorted(nums): return "Momentum"
        if nums == sorted(nums, reverse=True): return "Reversal"
        return "Mixed"

    df["Type"] = df["Path"].apply(classify)
    summary = df.groupby("Type")["Count"].sum().reset_index()
    return df, summary

# ─────────────────────────────────────────
# DURATION
# ─────────────────────────────────────────
def duration_analysis(quint_df):
    pivot = quint_df.pivot(index="Date", columns="Fund", values="Quintile")

    rows = []

    for f in pivot.columns:
        s = pivot[f].dropna()

        prev = None
        length = 0

        for q in s:
            if q == prev:
                length += 1
            else:
                if prev:
                    rows.append([f, prev, length])
                prev = q
                length = 1

        if prev:
            rows.append([f, prev, length])

    return pd.DataFrame(rows, columns=["Fund","Quintile","Duration"])

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    df, nifty, vix = load_data()

    quint_df = compute_rolling_quintiles(df)

    # 1st order
    t1 = first_order_transition(quint_df)
    t1.to_csv(f"{RESULTS_DIR}/first_order.csv")

    # 2nd order
    t2 = second_order_transition(quint_df)
    t2.to_csv(f"{RESULTS_DIR}/second_order.csv", index=False)

    # paths
    paths = path_analysis(quint_df)
    paths.to_csv(f"{RESULTS_DIR}/paths.csv", index=False)

    # classification
    paths, summary = classify_paths(paths)
    summary.to_csv(f"{RESULTS_DIR}/path_types.csv", index=False)

    # duration
    dur = duration_analysis(quint_df)
    dur.to_csv(f"{RESULTS_DIR}/durations.csv", index=False)

    print("DONE. Check results folder.")

if __name__ == "__main__":
    main()
