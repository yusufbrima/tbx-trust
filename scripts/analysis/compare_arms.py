"""Pre-specified comparisons C1–C5 (and exploratory C7) of the analysis plan, with its decision rules.

Each arm = a set of seed runs (or one run). Per external site, an arm's test probabilities are
averaged over seeds (aligned by sample_id); the two arms are then compared with a paired
patient-level bootstrap (2,000 resamples) and DeLong. Decision rule (analysis plan,
"Statistical analysis"): a difference is meaningful only if |diff| >= 0.02 AUROC, its bootstrap
p-value survives Holm correction within the confirmatory family (C1, C2, C5), and the
two arms' per-seed AUROC ranges overlap by at most half of the narrower range.

Usage: python scripts/analysis/compare_arms.py OUT_DIR
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tbmxai.core.types import Predictions
from tbmxai.evaluation.compare import ModelComparator

R = Path("~/tbmxai_runs").expanduser()
MARGIN = 0.02

# (id, family, question, arm A, arm B, site)
# C5 uses the TBX11K-only re-tune (scripts/experiments/tbx11k_retune.sh: settings selected on TBX11K
# validation), so its Shenzhen result is independent of Shenzhen; the first C5 runs reused settings
# tuned on Shenzhen validation.
COMPARISONS = [
    ("C1", "confirmatory", "MIMIC vs ImageNet init, ConvNeXt-T (Shenzhen-trained)",
     "nlm_tb_tuned/cxr_convnext_seeds/seed*", "transfer/nlm_convnext_mimicpt/seed*", "montgomery"),
    ("C1", "confirmatory", "MIMIC vs ImageNet init, ConvNeXt-T (Shenzhen-trained)",
     "multisite/cxr_convnext_from_shenzhen/seed*", "transfer/multisite_convnext_mimicpt_from_shenzhen/seed*", "tbx11k"),
    ("C2", "confirmatory", "Frozen vs fine-tuned MIMIC ConvNeXt-T",
     "transfer/nlm_convnext_mimicpt/seed*", "transfer/nlm_convnext_mimicpt_frozen/seed*", "montgomery"),
    ("C3", "supporting", "ConvNeXt-T vs DenseNet-121 (ImageNet, tuned)",
     "nlm_tb_tuned/cxr_densenet_seeds/seed*", "nlm_tb_tuned/cxr_convnext_seeds/seed*", "montgomery"),
    ("C4", "supporting", "MIMIC vs ImageNet init, DenseNet-121",
     "nlm_tb_tuned/cxr_densenet_seeds/seed*", "transfer/nlm_densenet_mimicpt/seed*", "montgomery"),
    ("C4", "supporting", "Frozen MIMIC vs ImageNet fine-tuned, DenseNet-121",
     "nlm_tb_tuned/cxr_densenet_seeds/seed*", "transfer/nlm_densenet_mimicpt_frozen/seed*", "montgomery"),
    ("C5", "confirmatory", "MIMIC vs ImageNet init, ConvNeXt-T (TBX11K-trained)",
     "multisite_retuned/imagenet/seed*", "multisite_retuned/mimic/seed*", "shenzhen"),
    ("C5", "confirmatory", "MIMIC vs ImageNet init, ConvNeXt-T (TBX11K-trained)",
     "multisite_retuned/imagenet/seed*", "multisite_retuned/mimic/seed*", "montgomery"),
    # C7 (exploratory): public CXR foundation models as an external reference for C1. The plan
    # named this reference but no comparison, so these rows are outside the Holm family.
    ("C7", "exploratory", "TorchXRayVision vs MIMIC-pretrained ConvNeXt-T",
     "transfer/nlm_convnext_mimicpt/seed*", "transfer/nlm_xrv/seed*", "montgomery"),
    ("C7", "exploratory", "TorchXRayVision vs MIMIC-pretrained ConvNeXt-T",
     "transfer/multisite_convnext_mimicpt_from_shenzhen/seed*", "transfer/multisite_xrv_from_shenzhen/seed*", "tbx11k"),
    ("C7", "exploratory", "RAD-DINO (fine-tuned) vs MIMIC-pretrained ConvNeXt-T",
     "transfer/nlm_convnext_mimicpt/seed*", "transfer/nlm_raddino/seed*", "montgomery"),
    ("C7", "exploratory", "RAD-DINO (fine-tuned) vs MIMIC-pretrained ConvNeXt-T",
     "transfer/multisite_convnext_mimicpt_from_shenzhen/seed*", "transfer/multisite_raddino_from_shenzhen/seed*", "tbx11k"),
    ("C7", "exploratory", "RAD-DINO (frozen) vs MIMIC-pretrained ConvNeXt-T",
     "transfer/nlm_convnext_mimicpt/seed*", "transfer/nlm_raddino_frozen/seed*", "montgomery"),
    ("C7", "exploratory", "RAD-DINO (frozen) vs MIMIC-pretrained ConvNeXt-T",
     "transfer/multisite_convnext_mimicpt_from_shenzhen/seed*", "transfer/multisite_raddino_frozen_from_shenzhen/seed*", "tbx11k"),
]


def arm(pattern: str, site: str | None) -> tuple[Predictions, list[float], int]:
    runs = sorted(glob.glob(str(R / pattern)))
    if not runs:
        raise FileNotFoundError(pattern)
    frames, seed_auc = [], []
    for run in runs:
        d = pd.read_csv(Path(run) / "predictions_test.csv")
        if site is not None and "site" in d:
            d = d[d["site"] == site]
        d["p"] = d[[c for c in d if c.startswith("prob_")][-1]]
        seed_auc.append(float(roc_auc_score(d["y_true"], d["p"])))
        frames.append(d[["sample_id", "patient_id", "y_true", "p"]])
    m = pd.concat(frames).groupby("sample_id").agg(
        patient_id=("patient_id", "first"), y_true=("y_true", "first"), p=("p", "mean"), k=("p", "size"))
    if m["k"].nunique() != 1:
        raise ValueError(f"{pattern}: seeds do not cover the same samples")
    m = m.sort_index()
    pred = Predictions(m["y_true"].to_numpy(), m["p"].to_numpy(), sample_ids=m.index.to_numpy(),
                       group_ids=m["patient_id"].astype(str).to_numpy())
    return pred, seed_auc, len(runs)


def overlap_fraction(a: list[float], b: list[float]) -> float:
    lo, hi = max(min(a), min(b)), min(max(a), max(b))
    narrow = min(max(a) - min(a), max(b) - min(b))
    if hi <= lo:
        return 0.0
    return 1.0 if narrow == 0 else (hi - lo) / narrow


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, family, question, pa, pb, site in COMPARISONS:
        (a, sa, na), (b, sb, nb) = arm(pa, site), arm(pb, site)
        res = ModelComparator(["auroc", "spec_at_sens@0.9"], n_bootstrap=2000, correction="none",
                              seed=0).compare({"A": a, "B": b})
        pw = res.pairwise.set_index("metric")
        auc = pw.loc["auroc"]
        rows.append({
            "id": cid, "family": family, "question": question, "site": site or "mimic test",
            "arm_a": pa.split("/")[-2] if "*" in pa.split("/")[-1] else pa, "arm_b": pb.split("/")[-2] if "*" in pb.split("/")[-1] else pb,
            "seeds_a": na, "seeds_b": nb, "n_test": len(a.y_true),
            "auroc_a": float(auc["value_a"]), "auroc_b": float(auc["value_b"]),
            "diff_b_minus_a": -float(auc["diff"]), "diff_lower": -float(auc["diff_upper"]),
            "diff_upper": -float(auc["diff_lower"]), "p_bootstrap": float(auc["p_bootstrap"]),
            "p_delong": float(auc["p_delong"]),
            "spec90_a": float(pw.loc["spec_at_sens90", "value_a"]),
            "spec90_b": float(pw.loc["spec_at_sens90", "value_b"]),
            "seed_auc_a": f"{np.mean(sa):.3f} ({min(sa):.3f}–{max(sa):.3f})",
            "seed_auc_b": f"{np.mean(sb):.3f} ({min(sb):.3f}–{max(sb):.3f})",
            "seed_overlap": overlap_fraction(sa, sb) if na > 1 and nb > 1 else np.nan,
        })
    df = pd.DataFrame(rows)
    conf = df["family"] == "confirmatory"
    p = df.loc[conf, "p_bootstrap"].to_numpy()
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    df["p_holm"] = np.nan
    df.loc[conf, "p_holm"] = adj
    big = df["diff_b_minus_a"].abs() >= MARGIN
    ci_ok = (df["diff_lower"] > 0) | (df["diff_upper"] < 0)
    p_ok = df["p_holm"].fillna(df["p_bootstrap"]) < 0.05
    seeds_ok = df["seed_overlap"].isna() | (df["seed_overlap"] <= 0.5)
    df["decision"] = np.where(big & ci_ok & p_ok & seeds_ok,
                              np.where(df["diff_b_minus_a"] > 0, "B better", "A better"),
                              "no clear difference")
    df.to_csv(out / "decision_table.csv", index=False)
    show = ["id", "site", "arm_a", "arm_b", "auroc_a", "auroc_b", "diff_b_minus_a", "diff_lower",
            "diff_upper", "p_holm", "p_delong", "seed_overlap", "decision"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print(df[show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
