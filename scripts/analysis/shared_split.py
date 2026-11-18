"""Shared-split analysis.

Six Shenzhen-trained arms, five seeds each, all on ONE Shenzhen train / validation / internal-test
split (configs/transfer/internal_shared, split_seed: 0): seeds vary only initialisation and data
order. The seed ensemble therefore has genuine validation scores, and its threshold (highest score
reaching 90 % sensitivity on the averaged validation scores) is a valid deployed artefact.

Per arm and test set (shared internal Shenzhen test, Montgomery, TBX11K), for the ensemble and for
single runs: AUROC, sensitivity and specificity at the source threshold with exact 95 % CIs, the
WHO minimum (90/70) on the point estimate and on both lower bounds, oracle specificity at 90 %
sensitivity, and the sensitivity error (delivered minus 90 %).

Usage: python scripts/analysis/shared_split.py OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

from tbmxai.evaluation.threshold import select_threshold

R = Path("~/tbmxai_runs").expanduser()
RUNS = R / "transfer" / "internal_shared"
TBX = R / "analysis" / "internal_shared_tbx11k"
ARMS = {
    "convnext_imagenet": "ConvNeXt-T, ImageNet",
    "convnext_mimic": "ConvNeXt-T, MIMIC",
    "densenet_imagenet": "DenseNet-121, ImageNet",
    "xrv": "TorchXRayVision",
    "raddino_frozen": "RAD-DINO, frozen",
    "raddino_retuned": "RAD-DINO, re-tuned",
}
SEEDS = range(1, 6)


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def prob(d: pd.DataFrame) -> pd.Series:
    return d["p"] if "p" in d else d[[c for c in d if c.startswith("prob_")][-1]]


def evaluate(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pos = p >= thr
    tp, fn = int((pos & (y == 1)).sum()), int((~pos & (y == 1)).sum())
    tn, fp = int((~pos & (y == 0)).sum()), int((pos & (y == 0)).sum())
    se, sp = tp / (tp + fn), tn / (tn + fp)
    (se_lo, _), (sp_lo, _) = cp(tp, tp + fn), cp(tn, tn + fp)
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return {
        "auroc": float(roc_auc_score(y, p)), "sens": se, "spec": sp, "sens_lo": se_lo,
        "spec_lo": sp_lo, "sens_error": se - 0.9,
        "oracle_spec90": float((1 - fpr[tpr >= 0.9 - 1e-12]).max()),
        "who_point": bool(se >= 0.9 and sp >= 0.7), "who_bounds": bool(se_lo >= 0.9 and sp_lo >= 0.7),
        "n_tb": tp + fn, "n_non_tb": tn + fp,
    }  # fmt: skip


def threshold(y: np.ndarray, p: np.ndarray) -> float:
    return float(select_threshold(y, p, method="sensitivity", target_sensitivity=0.9))


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, label in ARMS.items():
        vals, tests, tbxs = [], [], []
        for s in SEEDS:
            run = RUNS / arm / f"seed{s}"
            v = pd.read_csv(run / "predictions_val.csv").set_index("sample_id")
            t = pd.read_csv(run / "predictions_test.csv").set_index("sample_id")
            x = pd.read_csv(TBX / f"{arm}__seed{s}.csv").set_index("sample_id")
            v["p"], t["p"], x["p"] = prob(v), prob(t), prob(x)
            vals.append(v), tests.append(t), tbxs.append(x)
        # the split is shared: every seed must hold the same validation and test samples
        assert all(set(v.index) == set(vals[0].index) for v in vals), arm
        assert all(set(t.index) == set(tests[0].index) for t in tests), arm
        units = [("ensemble", "5 seeds", vals, tests, tbxs)] + [
            ("run", f"seed{s}", [vals[i]], [tests[i]], [tbxs[i]]) for i, s in enumerate(SEEDS)
        ]
        for unit, name, vv, tt, xx in units:
            yv = vals[0]["y_true"].to_numpy(int)
            pv = pd.concat([v["p"] for v in vv], axis=1).loc[vals[0].index].mean(axis=1).to_numpy()
            thr = threshold(yv, pv)
            pt = pd.concat([t["p"] for t in tt], axis=1).loc[tests[0].index].mean(axis=1)
            px = pd.concat([x["p"] for x in xx], axis=1).loc[tbxs[0].index].mean(axis=1)
            sets = {
                "shenzhen (internal)": (tests[0]["site"] == "shenzhen", tests[0], pt),
                "montgomery": (tests[0]["site"] == "montgomery", tests[0], pt),
                "tbx11k": (pd.Series(True, index=tbxs[0].index), tbxs[0], px),
            }
            for site, (mask, frame, p) in sets.items():
                y = frame.loc[mask, "y_true"].to_numpy(int)
                rows.append({"arm": label, "unit": unit, "run": name, "site": site, "threshold": thr,
                             **evaluate(y, p[mask].to_numpy(float), thr)})  # fmt: skip
        print(f"done {label}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "operating_points.csv", index=False)
    runs = df[df["unit"] == "run"].groupby(["arm", "site"], sort=False)
    summ = df[df["unit"] == "ensemble"].set_index(["arm", "site"])[
        ["auroc", "sens", "spec", "sens_lo", "spec_lo", "oracle_spec90", "who_point", "who_bounds"]
    ].join(runs.agg(runs_sens_min=("sens", "min"), runs_sens_max=("sens", "max"),
                    runs_spec_min=("spec", "min"), runs_spec_max=("spec", "max"),
                    runs_who_point=("who_point", "sum")))  # fmt: skip
    summ.reset_index().to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
