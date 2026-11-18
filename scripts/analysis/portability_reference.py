"""Reference points for reading the deployed operating points of Paper A (saved results only).

Writes to OUT_DIR:
  no_shift.csv        what a model with NO site shift would deliver: the threshold is set on the source
                      validation TB cases, so population sensitivity follows the Beta law of its order
                      statistic; test counts are binomial. Range of sensitivity for one run and for the mean
                      of an arm's runs, and the chance of meeting the WHO minimum (90/70) on the point
                      estimate and on both exact lower bounds, for a true specificity of 90 %.
  arm_no_shift.csv    each arm's mean delivered sensitivity against that range.
  shared_shift.csv    shared-split ensembles: external minus internal sensitivity and specificity at one
                      threshold, with Wald 95 % CIs (threshold noise cancels in the difference).
  sick_negatives.csv  TBX11K per run: active TB against sick-but-non-TB images, with exact lower bounds.
  symptomatic.csv     TBX11K programme scenario with symptomatic negatives (sick-but-non-TB and inactive TB).
  rank_rules.csv      smallest k with P(sensitivity >= 90 %) >= 0.90 for the m-th lowest of k TB scores.

Usage: python scripts/analysis/portability_reference.py OUT_DIR
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

R = Path("~/tbmxai_runs").expanduser()
A = R / "analysis"
N = {"active": 660, "inactive": 139, "sick": 3662, "healthy": 3800}
SITES = {"montgomery": (58, 80), "tbx11k": (800, 7462), "shenzhen": (336, 326)}
CARTRIDGE = 7.97


def lower(k: np.ndarray, n: int) -> np.ndarray:
    k = np.asarray(k)
    out = np.zeros(k.shape, float)
    pos = k > 0
    out[pos] = beta.ppf(0.025, k[pos], n - k[pos] + 1)
    return out


def rank(k: int) -> int:
    """Rank from the bottom of the validation TB score that sets the 90 % threshold."""
    return k - math.ceil(0.9 * k - 1e-9) + 1


def simulate(k: int, n_tb: int, n_non: int, rng: np.random.Generator, draws: int = 200_000) -> dict:
    sens = 1 - rng.beta(rank(k), k - rank(k) + 1, draws)
    tp, tn = rng.binomial(n_tb, sens), rng.binomial(n_non, 0.90, draws)
    point = (tp / n_tb >= 0.9) & (tn / n_non >= 0.7)
    bounds = (lower(tp, n_tb) >= 0.9) & (lower(tn, n_non) >= 0.7)
    lo, hi = np.percentile(tp / n_tb, [2.5, 97.5])
    return {"sens_lo": lo, "sens_hi": hi, "p_point": point.mean(), "p_bounds": bounds.mean()}


def val_cases(pattern: str) -> int:
    f = sorted(glob.glob(str(R / pattern / "predictions_val.csv")))[0]
    return int(pd.read_csv(f, usecols=["y_true"])["y_true"].sum())


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    k_sz = val_cases("transfer/multisite_raddino_from_shenzhen/seed1")
    k_tb = val_cases("multisite_retuned/imagenet/seed1")
    rows = []
    for site, (n_tb, n_non) in SITES.items():
        k = k_tb if site == "shenzhen" else k_sz
        rows.append({"site": site, "val_tb": k, "n_tb": n_tb, "n_non_tb": n_non,
                     **simulate(k, n_tb, n_non, rng)})  # fmt: skip
    pd.DataFrame(rows).to_csv(out / "no_shift.csv", index=False)

    op = pd.read_csv(A / "operating_points" / "operating_points.csv")
    op = op[(op["population"] == "all") & (op["unit"] == "run")]
    arms = []
    for (arm, site), g in op.groupby(["arm", "site"], sort=False):
        k, r, n = (k_tb if site == "shenzhen" else k_sz), len(g), SITES[site][0]
        s = 1 - rng.beta(rank(k), k - rank(k) + 1, (100_000, r))
        m = (rng.binomial(n, s) / n).mean(axis=1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        arms.append({"arm": arm, "site": site, "runs": r, "mean_sens": g["sens"].mean(), "lo": lo, "hi": hi,
                     "outside": not lo <= g["sens"].mean() <= hi})  # fmt: skip
    pd.DataFrame(arms).to_csv(out / "arm_no_shift.csv", index=False)

    s = pd.read_csv(A / "shared_split" / "summary.csv")
    n = {"montgomery": (58, 80), "tbx11k": (800, 7462), "shenzhen (internal)": (68, 65)}
    shift = []
    for arm, g in s.groupby("arm", sort=False):
        g = g.assign(site=g["site"].str.lower()).set_index("site")
        i = g.loc["shenzhen (internal)"]
        for site in ("montgomery", "tbx11k"):
            e = g.loc[site]
            for col, pos in (("sens", 0), ("spec", 1)):
                a, b = n[site][pos], n["shenzhen (internal)"][pos]
                d = e[col] - i[col]
                se = math.sqrt(e[col] * (1 - e[col]) / a + i[col] * (1 - i[col]) / b)
                shift.append({"arm": arm, "site": site, "measure": col, "diff": d, "lower": d - 1.96 * se,
                              "upper": d + 1.96 * se, "excludes_zero": abs(d) > 1.96 * se})  # fmt: skip
    pd.DataFrame(shift).to_csv(out / "shared_shift.csv", index=False)

    t = pd.read_csv(A / "operating_points" / "tbx11k_classes.csv")
    t["arm"] = t["arm"].str.replace("(registered)", "(pre-specified)", regex=False)
    sick, prog = [], []
    for _, r in t.iterrows():
        tp = round(r["flagged_tb_active"] * N["active"])
        tn = round((1 - r["flagged_sick_non_tb"]) * N["sick"])
        se_lo, sp_lo = lower(np.array([tp]), N["active"])[0], lower(np.array([tn]), N["sick"])[0]
        sick.append({"arm": r["arm"], "run": r["run"], "sens": tp / N["active"], "sens_lo": se_lo,
                     "spec": tn / N["sick"], "spec_lo": sp_lo, "meets_point": tp / N["active"] >= 0.9 and tn / N["sick"] >= 0.7,
                     "meets_bounds": se_lo >= 0.9 and sp_lo >= 0.7})  # fmt: skip
        se = r["flagged_tb_active"]
        neg = N["sick"] + N["inactive"]
        sp = (
            1
            - (N["sick"] * r["flagged_sick_non_tb"] + N["inactive"] * r["flagged_tb_obsolete"])
            / neg
        )
        for prev in (0.05, 0.10, 0.20):
            ref = 1000 * (prev * se + (1 - prev) * (1 - sp))
            det = 1000 * prev * se
            prog.append({"arm": r["arm"], "run": r["run"], "prevalence": prev, "referrals": ref, "tests_per_case": ref / det,
                         "missed": 1000 * prev * (1 - se), "saved": 1 - ref / 1000})  # fmt: skip
    pd.DataFrame(sick).to_csv(out / "sick_negatives.csv", index=False)
    pd.DataFrame(prog).to_csv(out / "symptomatic.csv", index=False)

    rules = []
    for m in range(1, 6):
        k = next(k for k in range(m, 400) if beta.cdf(0.1, m, k - m + 1) >= 0.90)
        rules.append({"rank": m, "k": k, "assurance": beta.cdf(0.1, m, k - m + 1),
                      **{f"tested_at_{p}pct": math.ceil(k / (p / 100)) for p in (5, 10, 20)}})  # fmt: skip
    pd.DataFrame(rules).to_csv(out / "rank_rules.csv", index=False)


if __name__ == "__main__":
    main()
