"""Local threshold recalibration.

A site recalibrates a deployed model (a single run, the deployed artefact; see operating_points.py) by
collecting k locally confirmed TB cases and setting the threshold from their scores.

Design choices, and why:
  * No finite-population artefact. v2 drew the k cases from the site's own TB images and judged
    the threshold on all of them, including the k drawn ones (on Montgomery, k = 40 is 69 % of its
    58 TB images). Here, the site's TB score distribution is modelled with a Gaussian kernel density
    on the logit scale (Silverman bandwidth). Calibration cases are drawn from it, and population
    sensitivity is that density's survival function at the threshold. Specificity is judged on all
    the site's non-TB images, which calibration never uses. A held-out check on TBX11K (calibrate on
    k TB images, judge on the remaining TB images) is reported alongside.
  * Rules with a fixed effective target. ``q10`` / ``q05``: the threshold is the 10th / 5th
    percentile of the k local scores (linear interpolation), so the intended sensitivity is 90 % /
    95 % at every k. ``cp90``: the highest threshold whose one-sided 95 % Clopper-Pearson lower
    bound on sample sensitivity is at least 90 %. It needs k >= 29, and below that the site refers
    everyone (recorded as not achievable).
  * One reference for all sites. With a continuous score distribution, the population sensitivity
    reached by a percentile rule does not depend on the distribution, so the chance of reaching
    90 % sensitivity depends only on the rule and k, not on the model or the site. A perfectly
    discriminating model (specificity 100 %) is therefore one reference for every site.

Outputs (OUT_DIR): per_run.csv; summary.csv (mean over runs, with the range over runs of p_who).

Usage: python scripts/analysis/recalibration.py OUT_DIR [B]
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import beta, norm

R = Path("~/tbmxai_runs").expanduser()

# label -> (init family, {site: run glob})
ARMS_BASE = {
    "DenseNet-121, ImageNet": ("ImageNet", {"montgomery": "nlm_tb_tuned/cxr_densenet_seeds/seed*",
                                            "tbx11k": "multisite/cxr_from_shenzhen/seed*"}),
    "DenseNet-121, MIMIC": ("MIMIC", {"montgomery": "transfer/nlm_densenet_mimicpt/seed*"}),
    "DenseNet-121, MIMIC, frozen": ("MIMIC", {"montgomery": "transfer/nlm_densenet_mimicpt_frozen/seed*"}),
    "ConvNeXt-T, ImageNet": ("ImageNet", {"montgomery": "nlm_tb_tuned/cxr_convnext_seeds/seed*",
                                          "tbx11k": "multisite/cxr_convnext_from_shenzhen/seed*"}),
    "ConvNeXt-T, MIMIC": ("MIMIC", {"montgomery": "transfer/nlm_convnext_mimicpt/seed*",
                                    "tbx11k": "transfer/multisite_convnext_mimicpt_from_shenzhen/seed*"}),
    "ConvNeXt-T, MIMIC, frozen": ("MIMIC", {"montgomery": "transfer/nlm_convnext_mimicpt_frozen/seed*"}),
    "TorchXRayVision": ("Public FM", {"montgomery": "transfer/nlm_xrv/seed*",
                                      "tbx11k": "transfer/multisite_xrv_from_shenzhen/seed*"}),
    "TorchXRayVision, frozen": ("Public FM", {"montgomery": "transfer/nlm_xrv_frozen/seed*",
                                              "tbx11k": "transfer/multisite_xrv_frozen_from_shenzhen/seed*"}),
    "RAD-DINO": ("Public FM", {"montgomery": "transfer/nlm_raddino/seed*",
                               "tbx11k": "transfer/multisite_raddino_from_shenzhen/seed*"}),
    "RAD-DINO, frozen": ("Public FM", {"montgomery": "transfer/nlm_raddino_frozen/seed*",
                                       "tbx11k": "transfer/multisite_raddino_frozen_from_shenzhen/seed*"}),
}


def probs(run: Path, split: str, site: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(run / f"predictions_{split}.csv")
    if site is not None and "site" in d:
        d = d[d["site"] == site]
    p = d[[c for c in d if c.startswith("prob_")][-1]].to_numpy(float)
    return d["y_true"].to_numpy(int), p


ARMS = {
    **ARMS_BASE,
    "RAD-DINO, retuned": (
        "Public FM",
        {
            "montgomery": "transfer/raddino_lr/nlm_*/seed*",
            "tbx11k": "transfer/raddino_lr/multisite_*/seed*",
        },
    ),
}
KS = (10, 20, 30, 40, 60, 80, 100)
RULES = ("q10", "q05", "cp90")
PREVALENCES = (0.01, 0.05, 0.10)


class TBDensity:
    """Kernel density of a site's TB scores on the logit scale."""

    def __init__(self, scores: np.ndarray) -> None:
        self.x = logit(np.clip(scores, 1e-6, 1 - 1e-6))
        n = len(self.x)
        iqr = np.subtract(*np.quantile(self.x, [0.75, 0.25]))
        spread = min(self.x.std(ddof=1), iqr / 1.34) if iqr > 0 else self.x.std(ddof=1)
        self.h = 0.9 * spread * n ** (-0.2)

    def sample(self, k: int, rng: np.random.Generator) -> np.ndarray:
        return expit(rng.choice(self.x, k) + self.h * rng.standard_normal(k))

    def sensitivity(self, thr: float) -> float:
        if not np.isfinite(thr):
            return 1.0
        t = logit(np.clip(thr, 1e-6, 1 - 1e-6))
        return float(norm.sf((t - self.x) / self.h).mean())


def threshold(scores: np.ndarray, rule: str) -> tuple[float, bool]:
    """Threshold from the k local TB scores. Returns (threshold, achievable)."""
    if rule == "q10":
        return float(np.quantile(scores, 0.10)), True
    if rule == "q05":
        return float(np.quantile(scores, 0.05)), True
    s = np.sort(scores)[::-1]  # threshold s[j] catches j + 1 of the k cases
    k = len(s)
    caught = np.arange(1, k + 1)
    lower = np.where(caught == 0, 0.0, beta.ppf(0.05, caught, k - caught + 1))
    ok = lower >= 0.90
    if not ok.any():
        return -np.inf, False  # cannot certify: refer everyone
    return float(s[int(np.argmax(ok))]), True


def summarise(se: np.ndarray, sp: np.ndarray, ach: np.ndarray) -> dict[str, float]:
    who = (se >= 0.9) & (sp >= 0.7)
    return {
        "sens_median": float(np.median(se)),
        "sens_p10": float(np.quantile(se, 0.1)),
        "spec_median": float(np.median(sp)),
        "spec_p10": float(np.quantile(sp, 0.1)),
        "p_sens90": float((se >= 0.9).mean()),
        "p_who": float(who.mean()),
        "p_achievable": float(ach.mean()),
    }


def simulate(
    y: np.ndarray, p: np.ndarray, k: int, b: int, rng: np.random.Generator, heldout: bool
) -> list[dict]:
    pos, neg = p[y == 1], p[y == 0]
    dens = TBDensity(pos)
    out = []
    for rule in RULES:
        se, sp, ach = np.empty(b), np.empty(b), np.empty(b, bool)
        for i in range(b):
            if heldout:
                idx = rng.permutation(len(pos))
                cal, rest = pos[idx[:k]], pos[idx[k:]]
                thr, ach[i] = threshold(cal, rule)
                se[i] = (rest >= thr).mean()
            else:
                thr, ach[i] = threshold(dens.sample(k, rng), rule)
                se[i] = dens.sensitivity(thr)
            sp[i] = (neg < thr).mean()
        out.append(
            {
                "rule": rule,
                "k_cases": k,
                "evaluation": "held-out" if heldout else "density",
                **summarise(se, sp, ach),
                **{f"patients_at_{int(q * 100)}pct": int(np.ceil(k / q)) for q in PREVALENCES},
            }
        )
    return out


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    # reference: perfect discrimination (any continuous TB distribution; specificity 100 %)
    y_ref = np.r_[np.ones(2000, int), np.zeros(2000, int)]
    p_ref = np.r_[rng.uniform(0.5, 1.0, 2000), rng.uniform(0.0, 0.5, 2000)]
    for k in KS:
        for r in simulate(y_ref, p_ref, k, b, rng, heldout=False):
            rows.append(
                {"model": "Perfect discrimination (reference)", "family": "reference",
                 "site": "any", "run": "-", **r}
            )  # fmt: skip
    for label, (family, sites) in ARMS.items():
        for site, pattern in sites.items():
            for run in sorted(glob.glob(str(R / pattern))):
                run = Path(run)
                if not (run / "predictions_test.csv").exists():
                    continue
                y, p = probs(run, "test", site)
                n_pos = int(y.sum())
                for k in KS:
                    if k > n_pos:
                        continue
                    modes = [False] + ([True] if site == "tbx11k" else [])
                    for heldout in modes:
                        for r in simulate(y, p, k, b, rng, heldout):
                            rows.append(
                                {"model": label, "family": family, "site": site,
                                 "run": run.name, **r}
                            )  # fmt: skip
            print(f"done {label}", flush=True)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    keys = ["site", "model", "family", "evaluation", "rule", "k_cases"]
    num = [c for c in per_run.columns if c not in (*keys, "run")]
    g = per_run.groupby(keys, sort=False)
    summ = g[num].mean().reset_index()
    summ["runs"] = g.size().to_numpy()
    summ["p_who_min"] = g["p_who"].min().to_numpy()
    summ["p_who_max"] = g["p_who"].max().to_numpy()
    summ.to_csv(out / "summary.csv", index=False)
    show = summ[(summ["rule"] == "q05") & summ["k_cases"].isin([20, 40, 100])][
        ["site", "model", "evaluation", "k_cases", "sens_median", "spec_median", "p_sens90",
         "p_who", "p_who_min", "p_who_max"]
    ]  # fmt: skip
    with pd.option_context("display.width", 220, "display.max_rows", 400):
        print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
