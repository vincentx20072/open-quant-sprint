"""
Track 10 Part 2: does the rank IC of simple signals differ between high and
low absorption-ratio hours?

Run from the repo root:
    python research/vincent/part2_ic_by_absorption.py            # full panel
    python research/vincent/part2_ic_by_absorption.py --smoke    # first 2,000 hours, shapes only

Choices (pre-registered in the PR before the first full-panel run):
    window = 168, k = 1 (primary) and k = 4
    signals: MOM (template.py raw 24h signal), REV (minus the last hour's log return),
             FLOW (baseline_flow.py weights)
    forward return: next hour's log return on the full hourly grid,
                    computed on the full panel before any split
    cutoffs: AR terciles on the first 40 % of hours; compare on the last 60 %
    test: mean IC(high) - mean IC(low), 95 % CI from a 1-day block bootstrap
          (10,000 draws, seed 0); 7-day blocks reported as a sensitivity
    Holm over the 6 tests (3 signals x 2 values of k)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.contrib.vincent import (  # noqa: E402
    absorption_ratio,
    day_block_bootstrap,
    forward_returns,
    holm,
    hourly_log_returns,
    label_regimes,
    rank_ic_by_hour,
    regime_cutoffs,
    to_wide,
)
from src.features import parkinson_volatility  # noqa: E402
from strategies.baseline_flow import compute_weights as baseline_flow_weights  # noqa: E402

WINDOW = 168
KS = (1, 4)
TRAIN_FRAC = 0.4
N_BOOT = 10_000
SEED = 0
PRIMARY = ("MOM", 1)


# the three signals as hours x symbols frames on the same hourly grid as the forward returns
def signals(df):
    p_vol = parkinson_volatility(df, window=24).replace(0.0, np.nan)
    mom = df.groupby("symbol")["close"].pct_change(24) / p_vol
    return {
        # MOM: the raw 24h signal from strategies/template.py (before z-score, EWM, deadband)
        "MOM": to_wide(df.assign(mom=mom), "mom"),
        # REV: minus the last hour's return
        "REV": -hourly_log_returns(df),
        # FLOW: the weights of strategies/baseline_flow.py
        "FLOW": to_wide(df.assign(flow=baseline_flow_weights(df)), "flow"),
    }


# absorption ratio, one value per hour (it is the same for every symbol)
def ar_by_hour(df, k):
    return to_wide(df.assign(ar=absorption_ratio(df, window=WINDOW, k=k)), "ar").mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train_1h.parquet"))
    ap.add_argument("--smoke", action="store_true", help="first 2,000 hours, print shapes and timings only")
    ap.add_argument("--out", default=str(ROOT / "research" / "vincent" / "results"))
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if args.smoke:
        keep = df["timestamp"].drop_duplicates().iloc[:2000]
        df = df[df["timestamp"].isin(keep)].reset_index(drop=True)

    fwd = forward_returns(df)  # full panel, before any split
    grid = fwd.index
    sigs = signals(df)
    ics = {name: rank_ic_by_hour(s, fwd) for name, s in sigs.items()}
    ars = {k: ar_by_hour(df, k) for k in KS}

    n_train = int(len(grid) * TRAIN_FRAC)
    test = grid[n_train:]

    if args.smoke:
        print(f"rows {len(df)}, hours {len(grid)}, train hours {n_train}, test hours {len(test)}")
        for name, s in sigs.items():
            print(f"{name}: shape {s.shape}, same grid as fwd {s.index.equals(grid)}, non-NaN share {s.notna().to_numpy().mean():.3f}")
        for k, a in ars.items():
            print(f"AR k={k}: same grid {a.index.equals(grid)}, non-NaN hours {a.notna().sum()}")
        print(f"elapsed {time.time() - t0:.1f}s (no IC values printed in smoke mode)")
        return

    rows = []
    for k in KS:
        lo, hi = regime_cutoffs(ars[k], TRAIN_FRAC)
        reg = label_regimes(ars[k], lo, hi).loc[test]
        for name in sigs:
            ic = ics[name].loc[test]
            day = day_block_bootstrap(ic, reg, n_boot=N_BOOT, block_days=1, seed=SEED)
            week = day_block_bootstrap(ic, reg, n_boot=N_BOOT, block_days=7, seed=SEED)
            rows.append(
                {
                    "signal": name,
                    "k": k,
                    "cut_lo": lo,
                    "cut_hi": hi,
                    "ic_all_test": ic.mean(),
                    "ic_mid": ic[reg == "mid"].mean(),
                    **day,
                    "ci7_lo": week["ci_lo"],
                    "ci7_hi": week["ci_hi"],
                    "p7": week["p"],
                }
            )
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"])
    res["primary"] = [(s, k) == PRIMARY for s, k in zip(res["signal"], res["k"])]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "part2_results.csv", index=False)

    print(f"hours {len(grid)}, cutoff period {grid[0]} .. {grid[n_train - 1]}, test {test[0]} .. {test[-1]}")
    for k in KS:
        r = res[res["k"] == k].iloc[0]
        print(f"k={k}: AR cutoffs low <= {r.cut_lo:.4f}, high > {r.cut_hi:.4f}")
    print()
    print("| signal | k | mean IC high AR | mean IC low AR | diff (high - low) | 95% CI (1-day blocks) | p | p Holm | 95% CI (7-day blocks) | hours high / low |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in res.to_dict("records"):
        star = " **(primary)**" if r["primary"] else ""
        print(
            f"| {r['signal']}{star} | {r['k']} | {r['ic_high']:+.4f} | {r['ic_low']:+.4f} | {r['diff']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p']:.4f} | {r['p_holm']:.4f} | "
            f"[{r['ci7_lo']:+.4f}, {r['ci7_hi']:+.4f}] | {r['n_high']} / {r['n_low']} |"
        )
    print()
    print("Mean IC on the whole last 60 % (all regimes), and in the middle third (k=1):")
    for r in res[res["k"] == 1].to_dict("records"):
        print(f"  {r['signal']}: all {r['ic_all_test']:+.4f}, mid {r['ic_mid']:+.4f}")
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
