"""MIMIC-CXR as a fourth, real-patient external TB test site.

Builds (once) a standard dataset of frontal MIMIC images:
  * positives: every image of a study positive for ``tb_episode`` (65 patients; all were
    excluded from MIMIC pretraining),
  * negatives: every image of the dev_split *test* patients (never seen by pretraining or any
    model) whose study is ``tb_episode`` negative; studies with an unknown label are dropped.
Then scores saved runs on it: AUROC, AUPRC, specificity at 90 % sensitivity, and sensitivity /
specificity at each run's own validation-chosen threshold (per run, then summarised over
seeds), with a patient-level bootstrap CI on the seed-averaged probabilities.

TorchXRayVision and RAD-DINO were trained on MIMIC-CXR (possibly these patients), so their
rows are flagged ``leakage_free = False``.

Usage:
  python scripts/analysis/mimic_tb_site.py build
  python scripts/analysis/mimic_tb_site.py score OUT_DIR LABEL=RUN_GLOB [LABEL=RUN_GLOB ...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEL = Path("~/data/tbmxai/mimic_subset/selection").expanduser()
JPG = Path("~/data/physionet/mimic-cxr-jpg/2.1.0").expanduser()
OUT_DS = Path("~/data/tbmxai/mimic_tb_site").expanduser()


def build() -> None:
    from tbmxai.data.dataset_format import DatasetSpec, LabelSpec

    studies = pd.read_parquet(SEL / "studies.parquet")
    images = pd.read_parquet(SEL / "images.parquet")
    dev = pd.read_csv(SEL / "dev_split.csv").set_index("subject_id")["dev_split"]
    studies["dev_split"] = studies["subject_id"].map(dev)
    lab = studies["label_tb_episode"]
    pos = studies[lab.fillna(0) == 1]
    neg = studies[(studies["dev_split"] == "test") & (lab == 0)]
    keep = pd.concat([pos.assign(tb="tb"), neg.assign(tb="non_tb")])[
        ["subject_id", "study_id", "tb"]
    ]
    img = images.merge(keep, on=["subject_id", "study_id"])
    img = img[[(JPG / p).exists() for p in img["path"]]]
    OUT_DS.mkdir(parents=True, exist_ok=True)
    labels = (
        img[["subject_id", "study_id", "tb"]]
        .drop_duplicates("study_id")
        .rename(columns={"subject_id": "patient_id"})
    )
    labels["site"] = "mimic"
    labels.to_csv(OUT_DS / "labels.csv", index=False)
    images_out = img.rename(columns={"subject_id": "patient_id"})[["patient_id", "study_id"]].copy()
    images_out["image_path"] = [str(JPG / p) for p in img["path"]]
    images_out.to_csv(OUT_DS / "images.csv", index=False)
    DatasetSpec(
        name="mimic_tb_site",
        description="MIMIC-CXR external TB test site: tb_episode-positive studies (65 patients) + "
        "tb_episode-negative studies of the dev_split test patients. Evaluation only.",
        patient_ids="provided",
        labels={"tb": LabelSpec(classes=["non_tb", "tb"], description="Active TB episode (±30 d)")},
    ).save(OUT_DS)
    info = {
        "images": len(images_out),
        "studies": len(labels),
        "patients": int(labels["patient_id"].nunique()),
        "tb_patients": int(labels.loc[labels["tb"] == "tb", "patient_id"].nunique()),
        "tb_images": int(img["tb"].eq("tb").sum()),
    }
    (OUT_DS / "build_info.json").write_text(json.dumps(info, indent=2))
    print(info)


def predict(run: Path) -> pd.DataFrame:
    from tbmxai import ExperimentConfig, Pipeline
    from tbmxai.models.base import Classifier

    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    cfg.data.dataset = OUT_DS
    cfg.data.label_column = "tb"
    cfg.data.use = ["images"]
    cfg.data.class_names = None
    cfg.view.modalities = ["imaging"]
    cfg.view.metadata_columns = ["site"]
    cfg.view.missing_modalities = "drop"
    cfg.protocol.path = None
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    model = Classifier.load(run / "model.pkl")
    pr = model.predict(view)
    return pd.DataFrame(
        {
            "sample_id": pr.sample_ids,
            "patient_id": view.patient_ids,
            "y_true": pr.y_true,
            "p": pr.y_prob[:, 1],
        }
    )


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    pred = p >= thr
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "spec_at_sens90": float((1 - fpr[tpr >= 0.9 - 1e-12]).max()),
        "sensitivity": float(pred[y == 1].mean()),
        "specificity": float((~pred)[y == 0].mean()),
    }


def score(out: Path, specs: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    per_run, pooled = [], []
    for spec in specs:
        label, pattern = spec.split("=", 1)
        runs = sorted(glob.glob(str(Path(pattern).expanduser())))
        frames = []
        for run in runs:
            run = Path(run)
            print(f"== {label}: {run}", flush=True)
            d = predict(run)
            tf = run / "threshold.json"
            thr = json.loads(tf.read_text())["threshold"] if tf.exists() else 0.5
            per_run.append(
                {
                    "model": label,
                    "run": run.name,
                    "threshold": thr,
                    **metrics(d["y_true"].to_numpy(), d["p"].to_numpy(), thr),
                }
            )
            frames.append(d)
            pred_dir = out / "predictions"
            pred_dir.mkdir(exist_ok=True)
            slug = "".join(ch if ch.isalnum() else "_" for ch in label)
            d.to_csv(pred_dir / f"{slug}__{run.name}.csv", index=False)  # local only (PhysioNet DUA)
            pd.DataFrame(per_run).to_csv(out / "per_run.csv", index=False)
        if not frames:
            continue
        m = (
            pd.concat(frames)
            .groupby("sample_id")
            .agg(patient_id=("patient_id", "first"), y_true=("y_true", "first"), p=("p", "mean"))
        )
        y, p, g = m["y_true"].to_numpy(), m["p"].to_numpy(), m["patient_id"].astype(str).to_numpy()
        rng = np.random.default_rng(0)
        pats = np.unique(g)
        idx_by = {pt: np.flatnonzero(g == pt) for pt in pats}
        boots = []
        for _ in range(1000):
            take = np.concatenate([idx_by[pt] for pt in rng.choice(pats, len(pats))])
            if len(np.unique(y[take])) == 2:
                boots.append(roc_auc_score(y[take], p[take]))
        pooled.append(
            {
                "model": label,
                "seeds": len(frames),
                "n_images": len(y),
                "n_patients": len(pats),
                "tb_images": int(y.sum()),
                "auroc_ensemble": float(roc_auc_score(y, p)),
                "auroc_lower": float(np.quantile(boots, 0.025)),
                "auroc_upper": float(np.quantile(boots, 0.975)),
                "auprc_ensemble": float(average_precision_score(y, p)),
                "leakage_free": not any(
                    k in label.lower() for k in ("xrv", "torchxrayvision", "rad-dino", "raddino")
                ),
            }
        )
    pr = pd.DataFrame(per_run)
    summ = (
        pr.groupby("model")
        .agg(
            auroc=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc=("auprc", "mean"),
            spec_at_sens90=("spec_at_sens90", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
        )
        .reset_index()
    )
    summ = summ.merge(pd.DataFrame(pooled), on="model")
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        score(Path(sys.argv[2]).expanduser(), sys.argv[3:])
