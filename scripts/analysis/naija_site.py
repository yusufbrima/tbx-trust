"""Naija-CXR (Rasheed Shekoni Federal University Teaching Hospital, Dutse, Nigeria) as an
exploratory African external site for the Shenzhen-trained TB models.

Naija-CXR (Zenodo 10.5281/zenodo.21781842, v1.1, CC BY 4.0, non-commercial research use) has 200
patients, one frontal image each, and a radiologist report + impression per patient. TB labels
are radiological impressions (8 patients), not microbiological, so the site cannot support an
AUROC claim; what it can measure is how often a model, at its deployed threshold, flags
radiologically normal Nigerian patients and patients with other disease (mostly cardiac) as TB.
Added after the analysis plan was written: report as exploratory.

Classes (from the impression, rules below, every assignment written to labels.csv):
  tb:         impression mentions PTB / tuberculosis
  normal:     impression is "normal study" and nothing else
  other:      any other chest impression (cardiomegaly, heart failure, pneumonia, ...)
  excluded:   non-chest radiographs (abdominal films), dropped

Usage:
  python scripts/analysis/naija_site.py build
  python scripts/analysis/naija_site.py dedup              # perceptual-hash check vs our sites
  python scripts/analysis/naija_site.py score OUT_DIR LABEL=RUN_GLOB [LABEL=RUN_GLOB ...]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = Path("~/data/naija_cxr/raw/RSFUTH_CXR").expanduser()
OUT_DS = Path("~/data/tbmxai/naija_cxr").expanduser()
# Our training / test sites, for the duplicate check (dataset folder, raw image root).
SITES = {
    "nlm_tb": (Path("~/data/tbmxai/nlm_tb"), Path("~/data/nlm_tb")),
    "tbx11k": (Path("~/data/tbmxai/tbx11k"), Path("~/data/tbx11k")),
}
TB = re.compile(r"\bPTB\b|tubercul|koch", re.I)
NORMAL = re.compile(r"^\s*normal (study|chest)\.?\s*$", re.I)
NON_CHEST = re.compile(
    r"\babdomen\b|bowel|valvulae|hemidiaphragms? .*radiolucent|perforation", re.I
)


def classify(report: str, impression: str) -> str:
    if NON_CHEST.search(report) or NON_CHEST.search(impression):
        return "excluded"
    if TB.search(impression):
        return "tb"
    if NORMAL.match(impression):
        return "normal"
    return "other"


def build() -> None:
    from tbmxai.data.dataset_format import DatasetSpec, LabelSpec

    d = pd.read_csv(RAW / "CXR_Labels.csv")
    d["Report"] = d["Report"].fillna("").astype(str)
    d["Impression"] = d["Impression"].fillna("").astype(str).str.strip()
    d["naija_class"] = [classify(r, i) for r, i in zip(d["Report"], d["Impression"], strict=True)]
    # one JPEG per folder; three are misnamed (P000031.jpg, P000044.jpg, a camera-style name)
    d["image_path"] = [
        str(next(iter(sorted((RAW / pid).glob("*.jpg"))), RAW / pid / "missing"))
        for pid in d["PID"]
    ]
    missing = [p for p in d["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    OUT_DS.mkdir(parents=True, exist_ok=True)
    d.rename(columns={"PID": "patient_id"}).assign(
        impression=lambda x: x["Impression"].str.replace(r"\s+", " ", regex=True)
    )[["patient_id", "naija_class", "impression"]].to_csv(
        OUT_DS / "classification.csv", index=False
    )
    d = d[d["naija_class"] != "excluded"]
    labels = pd.DataFrame(
        {
            "patient_id": d["PID"],
            "tb": np.where(d["naija_class"] == "tb", "tb", "non_tb"),
            "naija_class": d["naija_class"],
            "site": "naija",
        }
    )
    labels.to_csv(OUT_DS / "labels.csv", index=False)
    d.rename(columns={"PID": "patient_id"})[["patient_id", "image_path"]].to_csv(
        OUT_DS / "images.csv", index=False
    )
    d.rename(columns={"PID": "patient_id"}).assign(
        report_text=lambda x: x["Report"].str.strip() + "\nIMPRESSION: " + x["Impression"]
    )[["patient_id", "report_text"]].to_csv(OUT_DS / "reports.csv", index=False)
    DatasetSpec(
        name="naija_cxr",
        description="Naija-CXR (RSFUTH Dutse, Nigeria; Zenodo 10.5281/zenodo.21781842): frontal "
        "CXRs with radiologist reports. TB = radiological impression. Evaluation only.",
        patient_ids="provided",
        # naija_class stays a plain column (metadata): declared labels other than the target are
        # dropped from the cohort to prevent leakage.
        labels={
            "tb": LabelSpec(
                classes=["non_tb", "tb"], description="PTB in the radiologist impression"
            )
        },
    ).save(OUT_DS)
    info = {
        "patients": len(labels),
        "excluded_non_chest": int(
            (pd.read_csv(OUT_DS / "classification.csv")["naija_class"] == "excluded").sum()
        ),
        **{f"n_{k}": int(v) for k, v in labels["naija_class"].value_counts().items()},
    }
    (OUT_DS / "build_info.json").write_text(json.dumps(info, indent=2))
    print(info)


def dhash(path: str, size: int = 16) -> np.ndarray:
    """Difference hash on a grayscale thumbnail: robust to resizing and re-compression."""
    from PIL import Image

    with Image.open(path) as im:
        im.draft("L", (size * 8, size * 8))  # fast JPEG downscale
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


def dedup() -> None:
    """Nearest-neighbour Hamming distance of every Naija image to every image of our sites."""
    naija = pd.read_csv(OUT_DS / "images.csv")
    hn = np.stack([dhash(p) for p in naija["image_path"]])
    rows = []
    for site, (ds, raw) in SITES.items():
        imgs = pd.read_csv(ds.expanduser() / "images.csv")
        paths = [
            p if Path(p).is_absolute() else str(raw.expanduser() / p) for p in imgs["image_path"]
        ]
        hs = np.stack([dhash(p) for p in paths])
        dist = np.concatenate(  # (n_naija, n_site) bits differing; chunked to bound memory
            [(hn[:, None, :] != hs[None, k : k + 500, :]).sum(-1) for k in range(0, len(hs), 500)],
            axis=1,
        )
        j = dist.argmin(1)
        for i in range(len(naija)):
            rows.append(
                {
                    "patient_id": naija["patient_id"][i],
                    "site": site,
                    "nearest": imgs["patient_id"].iloc[j[i]],
                    "hamming": int(dist[i, j[i]]),
                }
            )
        print(
            f"{site}: {len(paths)} images; min distance {dist.min()} bits of {hn.shape[1]}",
            flush=True,
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DS / "dedup.csv", index=False)
    # Unrelated CXRs typically differ in ~30-45 % of bits; re-encoded copies in < 5 %.
    sus = out[out["hamming"] <= 0.1 * hn.shape[1]]
    print(f"suspected duplicates (<=10 % bits differ): {len(sus)}")
    if len(sus):
        print(sus.to_string(index=False))


def predict(run: Path) -> pd.DataFrame:
    from tbmxai import ExperimentConfig, Pipeline
    from tbmxai.models.base import Classifier

    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    cfg.data.dataset = OUT_DS
    cfg.data.label_column = "tb"
    cfg.data.use = ["images"]
    cfg.data.class_names = None
    cfg.view.modalities = ["imaging"]
    cfg.view.metadata_columns = ["naija_class"]
    cfg.view.missing_modalities = "drop"
    cfg.protocol.path = None
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    pr = Classifier.load(run / "model.pkl").predict(view)
    return pd.DataFrame(
        {
            "patient_id": view.patient_ids,
            "y_true": pr.y_true,
            "p": pr.y_prob[:, 1],
            "naija_class": view.metadata["naija_class"].astype(str).to_numpy(),
        }
    )


def val_threshold(run: Path) -> float:
    """Plan rule: highest threshold reaching 90 % sensitivity on source validation."""
    from tbmxai.evaluation.threshold import select_threshold

    v = pd.read_csv(run / "predictions_val.csv")
    p = v[[c for c in v if c.startswith("prob_")][-1]]
    return select_threshold(
        v["y_true"].to_numpy(), p.to_numpy(), method="sensitivity", target_sensitivity=0.9
    )


def score(out: Path, specs: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        label, pattern = spec.split("=", 1)
        for run in sorted(glob.glob(str(Path(pattern).expanduser()))):
            run = Path(run)
            print(f"== {label}: {run}", flush=True)
            # Saved scores are reused, so a change of threshold rule never re-reads the images.
            saved = out / f"pred_{re.sub(r'[^A-Za-z0-9]+', '_', label)}_{run.name}.csv"
            cols = ["patient_id", "y_true", "p", "naija_class"]
            d = pd.read_csv(saved)[cols] if saved.exists() else predict(run)
            thr = val_threshold(run)
            flag = d["p"] >= thr
            row = {
                "model": label,
                "run": run.name,
                "threshold": thr,
                "auroc_tb_vs_rest": roc_auc_score(d["y_true"], d["p"]),
            }
            for cls in ("normal", "other", "tb"):
                m = d["naija_class"] == cls
                row[f"flagged_{cls}"] = float(flag[m].mean())
                row[f"n_{cls}"] = int(m.sum())
            rows.append(row)
            d.assign(model=label, run=run.name, flagged=flag).to_csv(
                out / f"pred_{re.sub(r'[^A-Za-z0-9]+', '_', label)}_{run.name}.csv", index=False
            )
            pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)
    pr = pd.DataFrame(rows)
    summ = (
        pr.groupby("model", sort=False)
        .agg(
            seeds=("run", "size"),
            auroc_tb_vs_rest=("auroc_tb_vs_rest", "mean"),
            flagged_normal=("flagged_normal", "mean"),
            flagged_other=("flagged_other", "mean"),
            flagged_tb=("flagged_tb", "mean"),
            n_normal=("n_normal", "first"),
            n_other=("n_other", "first"),
            n_tb=("n_tb", "first"),
        )
        .reset_index()
    )
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "dedup":
        dedup()
    else:
        score(Path(sys.argv[2]).expanduser(), sys.argv[3:])
