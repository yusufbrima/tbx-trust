"""MIMIC TB site, patient-level re-analysis from saved per-image predictions (no GPU).

Patient-level re-analysis of the image-level MIMIC TB site:

  * one film per patient. A TB patient's film is the first TB-episode study taken no later than
    7 days after treatment start (first pyrazinamide or ethambutol order, drugs specific to
    active TB). A negative patient's film is their first study. Within a study, PA is preferred
    over AP.
  * pulmonary positives only. The episode study is coded pulmonary/respiratory/miliary
    (``label_pulmonary_tb``), or the patient had M. tuberculosis in a respiratory specimen.
  * TB rule-out negatives. Negative patients with an AFB smear or culture ordered and no
    M. tuberculosis ever isolated; the film is their first study within 30 days of the first AFB
    order.
  * the plan threshold. Each run's threshold reaches 90 % sensitivity on its own
    source-validation predictions. Each run is a deployable artefact (seeds use different
    validation splits, so a seed ensemble has no source threshold; the ensemble gives AUROC
    only).
  * flag rates by CheXpert finding among the negatives' films, and AUROC by view (AP / PA).

Writes OUT_DIR/{per_run.csv, summary.csv, findings.csv, cohort.json}. Aggregates only. The
per-image predictions it reads stay on the analysis machine (PhysioNet DUA).

Usage: python scripts/analysis/mimic_tb_patient.py OUT_DIR
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tbmxai.evaluation.threshold import select_threshold

R = Path("~/tbmxai_runs").expanduser()
SEL = Path("~/data/tbmxai/mimic_subset/selection").expanduser()
SITE = Path("~/data/tbmxai/mimic_tb_site").expanduser()
PRED = R / "analysis/mimic_tb_site_v2/predictions"
PHYS = Path("~/data/physionet").expanduser()
MICRO = PHYS / "mimiciv/3.1/hosp/microbiologyevents.csv.gz"
RX = PHYS / "mimiciv/3.1/hosp/prescriptions.csv.gz"
CHEXPERT = PHYS / "mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"

RESPIRATORY = ("SPUTUM", "BRONCHOALVEOLAR LAVAGE", "BRONCHIAL WASHINGS", "BRONCHIAL BRUSH")
AFB_TESTS = ("ACID FAST CULTURE", "ACID FAST SMEAR", "BLOOD/AFB CULTURE")
TB_DRUGS = ("pyrazinamide", "ethambutol")
FINDINGS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# arm label (as in mimic_tb_site_v2) -> run glob; FM arms saw MIMIC-CXR in pretraining
ARMS = {
    "DenseNet-121, ImageNet (tuned)": "nlm_tb_tuned/cxr_densenet_seeds/seed*",
    "ConvNeXt-T, ImageNet": "nlm_tb_tuned/cxr_convnext_seeds/seed*",
    "DenseNet-121, MIMIC-pretrained": "transfer/nlm_densenet_mimicpt/seed*",
    "DenseNet-121, MIMIC-pretrained, frozen": "transfer/nlm_densenet_mimicpt_frozen/seed*",
    "ConvNeXt-T, MIMIC-pretrained": "transfer/nlm_convnext_mimicpt/seed*",
    "ConvNeXt-T, MIMIC-pretrained, frozen": "transfer/nlm_convnext_mimicpt_frozen/seed*",
    "TorchXRayVision DenseNet-121": "transfer/nlm_xrv/seed*",
    "TorchXRayVision DenseNet-121, frozen": "transfer/nlm_xrv_frozen/seed*",
    "RAD-DINO": "transfer/nlm_raddino/seed*",
    "RAD-DINO, frozen": "transfer/nlm_raddino_frozen/seed*",
    "RAD-DINO, retuned": "transfer/raddino_lr/nlm_{best}/seed*",
}
LEAKY = ("TorchXRayVision", "RAD-DINO")


def slug(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label)


def microbiology(subjects: set[int]) -> pd.DataFrame:
    cols = ["subject_id", "chartdate", "spec_type_desc", "test_name", "org_name"]
    parts = []
    for c in pd.read_csv(MICRO, usecols=cols, chunksize=1_000_000):
        parts.append(c[c["subject_id"].isin(subjects)])
    m = pd.concat(parts)
    m["chartdate"] = pd.to_datetime(m["chartdate"])
    m["mtb"] = m["org_name"].fillna("").str.upper().str.contains("TUBERCULOSIS")
    m["afb"] = m["test_name"].isin(AFB_TESTS)
    m["resp"] = m["spec_type_desc"].isin(RESPIRATORY)
    return m


def treatment_start(subjects: set[int]) -> pd.Series:
    parts = []
    for c in pd.read_csv(RX, usecols=["subject_id", "starttime", "drug"], chunksize=2_000_000):
        c = c[c["subject_id"].isin(subjects)]
        parts.append(c[c["drug"].fillna("").str.lower().str.contains("|".join(TB_DRUGS))])
    rx = pd.concat(parts)
    return pd.to_datetime(rx["starttime"]).groupby(rx["subject_id"]).min()


def cohort() -> tuple[pd.DataFrame, dict]:
    """One row per selected film: dicom, patient, y, set flags, view, CheXpert findings."""
    studies = pd.read_parquet(SEL / "studies.parquet")
    images = pd.read_parquet(SEL / "images.parquet")
    site = pd.read_csv(SITE / "labels.csv")  # studies in the image-level site, with tb label
    img = images.merge(site[["study_id", "tb"]], on="study_id")
    img = img.merge(
        studies[["study_id", "study_time", "label_pulmonary_tb"]], on="study_id", how="left"
    )
    img["study_time"] = pd.to_datetime(img["study_time"])
    img["view_rank"] = img["view_position"].map({"PA": 0, "AP": 1}).fillna(2)
    img = img.sort_values(["subject_id", "study_time", "view_rank", "dicom_id"])

    subjects = set(img["subject_id"].unique())
    micro = microbiology(subjects)
    mtb_any = set(micro.loc[micro["mtb"], "subject_id"])
    mtb_resp = set(micro.loc[micro["mtb"] & micro["resp"], "subject_id"])
    first_afb = micro[micro["afb"]].groupby("subject_id")["chartdate"].min()
    pos_img = img[img["tb"] == "tb"]
    tx = treatment_start(set(pos_img["subject_id"]))

    # positives: first eligible TB-episode film per patient
    pos_img = pos_img.assign(tx=pos_img["subject_id"].map(tx))
    eligible = pos_img["tx"].isna() | (
        pos_img["study_time"] <= pos_img["tx"] + pd.Timedelta(7, "D")
    )
    pos = pos_img[eligible].groupby("subject_id").head(1).copy()
    pulm_study = pos_img.groupby("subject_id")["label_pulmonary_tb"].max().fillna(0) == 1
    pos["pulmonary"] = pos["subject_id"].map(pulm_study) | pos["subject_id"].isin(mtb_resp)
    pos["y"] = 1

    neg_img = img[(img["tb"] == "non_tb") & ~img["subject_id"].isin(mtb_any)]
    neg = neg_img.groupby("subject_id").head(1).copy()
    neg["y"] = 0
    # rule-out negatives: first film within 30 days of the first AFB order
    ro = neg_img.assign(afb=neg_img["subject_id"].map(first_afb)).dropna(subset=["afb"])
    ro = ro[(ro["study_time"] - ro["afb"]).abs() <= pd.Timedelta(30, "D")]
    ro = ro.groupby("subject_id").head(1).copy()
    ro["y"] = 0

    chex = pd.read_csv(CHEXPERT)[["study_id", *FINDINGS]]
    frames = []
    for name, d in (("pos", pos), ("neg", neg), ("ruleout", ro)):
        d = d.merge(chex, on="study_id", how="left")
        d["set"] = name
        frames.append(d)
    films = pd.concat(frames, ignore_index=True)
    films["pulmonary"] = films["pulmonary"].fillna(False).astype(bool)
    info = {
        "tb_patients_image_level": int(pos_img["subject_id"].nunique()),
        "tb_patients_with_eligible_film": len(pos),
        "tb_patients_pulmonary": int(pos["pulmonary"].sum()),
        "tb_patients_with_treatment_start": int(pos["tx"].notna().sum()),
        "negative_patients": len(neg),
        "ruleout_patients": len(ro),
        "negatives_dropped_for_mtb": int(
            img.loc[img["tb"] == "non_tb", "subject_id"].isin(mtb_any).sum()
        ),
        "views_pos": pos["view_position"].value_counts().to_dict(),
        "views_neg": neg["view_position"].value_counts().to_dict(),
        "patient_prevalence_pulmonary": float(
            pos["pulmonary"].sum() / (pos["pulmonary"].sum() + len(neg))
        ),
    }
    return films, info


def plan_threshold(run: Path) -> float:
    v = pd.read_csv(run / "predictions_val.csv")
    p = v[[c for c in v if c.startswith("prob_")][-1]].to_numpy(float)
    return float(
        select_threshold(v["y_true"].to_numpy(int), p, method="sensitivity", target_sensitivity=0.9)
    )


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def evaluate(films: pd.DataFrame, best: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, finding_rows, ens = [], [], []
    for arm, pattern in ARMS.items():
        runs = sorted(glob.glob(str(R / pattern.format(best=best))))
        probs = []
        for run in map(Path, runs):
            f = PRED / f"{slug(arm)}__{run.name}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["dicom_id"] = d["sample_id"].str.split("|").str[-1].str.removesuffix(".jpg")
            p = films["dicom_id"].map(d.set_index("dicom_id")["p"])
            if p.isna().any():
                raise ValueError(f"{arm} {run.name}: {int(p.isna().sum())} films without a score")
            probs.append(p.to_numpy(float))
            thr = plan_threshold(run)
            flag = probs[-1] >= thr
            row = {"arm": arm, "run": run.name, "leaky": arm.startswith(LEAKY), "threshold": thr}
            for tag, pos_mask in (("pulm", films["pulmonary"]), ("alltb", films["y"] == 1)):
                pm = (films["set"] == "pos") & pos_mask
                for neg_name in ("neg", "ruleout"):
                    m = pm | (films["set"] == neg_name)
                    row[f"auroc_{tag}_{neg_name}"] = auc(films.loc[m, "y"].to_numpy(), probs[-1][m])
                row[f"sens_{tag}"] = float(flag[pm].mean())
            for view in ("PA", "AP"):
                m = ((films["set"] == "pos") & films["pulmonary"]) | (films["set"] == "neg")
                m &= films["view_position"] == view
                row[f"auroc_pulm_neg_{view}"] = auc(films.loc[m, "y"].to_numpy(), probs[-1][m])
            for neg_name in ("neg", "ruleout"):
                row[f"flag_{neg_name}"] = float(flag[films["set"] == neg_name].mean())
            rows.append(row)
            negs = films["set"] == "neg"
            for fnd in FINDINGS:
                m = negs & (films[fnd] == 1)
                finding_rows.append(
                    {
                        "arm": arm,
                        "run": run.name,
                        "finding": fnd,
                        "n": int(m.sum()),
                        "flag_rate": float(flag[m].mean()) if m.any() else np.nan,
                    }
                )
        if probs:
            pe = np.mean(probs, axis=0)
            m = ((films["set"] == "pos") & films["pulmonary"]) | (films["set"] == "neg")
            y = films.loc[m, "y"].to_numpy()
            rng = np.random.default_rng(0)
            idx_pos, idx_neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            boots = [
                auc(y[t], pe[m][t])
                for t in (
                    np.concatenate(
                        [rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, len(idx_neg))]
                    )
                    for _ in range(2000)
                )
            ]
            ens.append(
                {
                    "arm": arm,
                    "seeds": len(probs),
                    "auroc_ens_pulm_neg": auc(y, pe[m]),
                    "lo": float(np.quantile(boots, 0.025)),
                    "hi": float(np.quantile(boots, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(finding_rows), pd.DataFrame(ens)


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((R / "transfer/raddino_lr/selection.json").read_text())
    best = sel["selected"].removeprefix("nlm_")
    films, info = cohort()
    (out / "cohort.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str), flush=True)
    per_run, findings, ens = evaluate(films, best)
    per_run.to_csv(out / "per_run.csv", index=False)
    findings.to_csv(out / "findings.csv", index=False)
    num = [c for c in per_run if c not in ("arm", "run", "leaky")]
    summ = per_run.groupby(["arm", "leaky"])[num].agg(["mean", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index().merge(ens, on="arm")
    summ.to_csv(out / "summary.csv", index=False)
    cols = [
        "arm",
        "leaky",
        "auroc_ens_pulm_neg",
        "lo",
        "hi",
        "auroc_pulm_neg_mean",
        "auroc_pulm_ruleout_mean",
        "sens_pulm_mean",
        "flag_neg_mean",
        "flag_ruleout_mean",
        "auroc_pulm_neg_PA_mean",
        "auroc_pulm_neg_AP_mean",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ[cols].round(3).to_string(index=False))
        fs = findings.groupby(["finding"]).agg(n=("n", "first"), flag=("flag_rate", "mean"))
        print(fs.round(3).to_string())


if __name__ == "__main__":
    main()
