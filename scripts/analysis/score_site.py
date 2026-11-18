"""Score saved runs on another site's standard dataset (images only), without retraining.

Used for the shared-split runs (configs/transfer/internal_shared/): models trained on one
fixed Shenzhen split are tested on Montgomery during training and scored here on TBX11K, so one set
of runs covers both external sites. Writes one CSV per run with sample_id, patient_id, y_true and
p (positive-class probability), plus every label column of the dataset for later stratification.

Usage: python scripts/analysis/score_site.py DATASET_DIR LABEL_COLUMN OUT_DIR RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def predict(run: Path, dataset: Path, label: str) -> pd.DataFrame:
    from tbmxai import ExperimentConfig, Pipeline
    from tbmxai.models.base import Classifier

    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    cfg.data.dataset = dataset
    cfg.data.label_column = label
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
    out = pd.DataFrame(
        {
            "sample_id": pr.sample_ids,
            "patient_id": view.patient_ids,
            "y_true": pr.y_true,
            "p": pr.y_prob[:, 1],
        }
    )
    labels = pd.read_csv(dataset / "labels.csv").drop_duplicates("patient_id")
    return out.merge(labels, on="patient_id", how="left", suffixes=("", "_label"))


def main() -> None:
    dataset, label, out = Path(sys.argv[1]).expanduser(), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for run in map(Path, sys.argv[4:]):
        dest = out / f"{run.parent.name}__{run.name}.csv"
        if dest.exists():
            print(f"skip {dest.name} (exists)", flush=True)
            continue
        predict(run, dataset, label).to_csv(dest, index=False)
        print(f"wrote {dest.name}", flush=True)


if __name__ == "__main__":
    main()
