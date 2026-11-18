"""Checks on the threshold-recalibration simulation.

1. Theory. For a continuous TB score distribution F, a threshold at the m-th lowest of k local TB
   scores leaves population sensitivity 1 - F(X_(m)), and F(X_(m)) ~ Beta(m, k - m + 1) whatever
   F is, so P(sensitivity >= 0.9) = I_0.1(m, k - m + 1) (regularised incomplete beta). The 5th
   percentile with linear interpolation lies between the order statistics m = floor((k-1) 0.05) + 1
   and m + 1, so its probability is bracketed by the two values.
2. Misspecification. The simulation is repeated with the site's TB score distribution modelled
   (a) by the Gaussian kernel density with half and with double the Silverman bandwidth, and
   (b) by the empirical distribution itself (calibration cases drawn with replacement from the
   site's TB scores, sensitivity judged on all of them: the plug-in estimate), and compared with
   (c) the held-out check (calibrate on k TB images, judge on the remaining ones).

Writes OUT_DIR/theory.csv and OUT_DIR/misspecification.csv.
Usage: python scripts/analysis/recalibration_checks.py OUT_DIR [B]
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc

sys.path.insert(0, str(Path(__file__).parent))
from recalibration import ARMS, KS, R, TBDensity, probs, threshold  # noqa: E402

TARGET_Q = 0.05


def theory() -> pd.DataFrame:
    rows = []
    for k in KS:
        h = (k - 1) * TARGET_Q
        m = math.floor(h) + 1
        lo, hi = betainc(m + 1, k - m, 0.1) if m + 1 <= k else np.nan, betainc(m, k - m + 1, 0.1)
        rows.append(
            {
                "k_cases": k,
                "order_stat_m": m,
                "p_sens90_lower": float(lo),
                "p_sens90_upper": float(hi),
            }
        )
    return pd.DataFrame(rows)


def simulate(pos: np.ndarray, neg: np.ndarray, k: int, b: int, mode: str, rng) -> dict:
    se, sp = np.empty(b), np.empty(b)
    dens = TBDensity(pos)
    if mode.startswith("kde_x"):
        dens.h *= float(mode.removeprefix("kde_x"))
    for i in range(b):
        if mode == "empirical":
            cal = rng.choice(pos, k, replace=True)
            thr, _ = threshold(cal, "q05")
            se[i] = (pos >= thr).mean()
        elif mode == "held-out":
            idx = rng.permutation(len(pos))
            thr, _ = threshold(pos[idx[:k]], "q05")
            se[i] = (pos[idx[k:]] >= thr).mean()
        else:
            thr, _ = threshold(dens.sample(k, rng), "q05")
            se[i] = dens.sensitivity(thr)
        sp[i] = (neg < thr).mean()
    return {
        "p_sens90": float((se >= 0.9).mean()),
        "p_who": float(((se >= 0.9) & (sp >= 0.7)).mean()),
    }


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    out.mkdir(parents=True, exist_ok=True)
    th = theory()
    th.to_csv(out / "theory.csv", index=False)
    print(th.round(3).to_string(index=False))
    rng = np.random.default_rng(0)
    rows = []
    for label, (_family, sites) in ARMS.items():
        pattern = sites.get("tbx11k")
        if pattern is None:
            continue
        for run in sorted(glob.glob(str(R / pattern))):
            run = Path(run)
            if not (run / "predictions_test.csv").exists():
                continue
            y, p = probs(run, "test", "tbx11k")
            pos, neg = p[y == 1], p[y == 0]
            for k in (20, 40, 80):
                for mode in ("kde_x1", "kde_x0.5", "kde_x2", "empirical", "held-out"):
                    rows.append({"model": label, "run": run.name, "k_cases": k, "mode": mode,
                                 **simulate(pos, neg, k, b, mode, rng)})  # fmt: skip
        print(f"done {label}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "per_run.csv", index=False)
    m = d.groupby(["model", "k_cases", "mode"])[["p_sens90", "p_who"]].mean().reset_index()
    m.to_csv(out / "misspecification.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(
            m.pivot_table(index=["model", "k_cases"], columns="mode", values="p_who")
            .round(2)
            .to_string()
        )
        print(
            m.pivot_table(index="k_cases", columns="mode", values="p_sens90").round(2).to_string()
        )


if __name__ == "__main__":
    main()
