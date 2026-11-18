"""Lesion-level saliency on TBX11K.

For Shenzhen-trained models tested on TBX11K, Grad-CAM maps of the TB-positive TBX11K images are
compared with the TB bounding boxes drawn by TBX11K's radiologists (train + val annotations,
512 x 512 images). Per image:
  box_energy   share of positive Grad-CAM energy inside the union of TB boxes
  box_area     share of the image covered by the boxes
  box_ratio    box_energy / box_area (1 = no preference for the lesion)
  hit          whether the saliency peak falls inside a box enlarged by 16 px (pointing game)
Lesions are split by TBX11K category (active / obsolete / uncertain), and by whether the model
flagged the image at its plan threshold (90 % sensitivity on Shenzhen validation).

Usage: python scripts/analysis/saliency_boxes.py OUT_DIR LABEL=RUN_GLOB [LABEL=RUN_GLOB ...]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.evaluation.threshold import select_threshold
from tbmxai.explain.saliency import GradCAMExplainer
from tbmxai.models.base import Classifier

ANN = Path("~/data/tbx11k/TBX11K/TBX11K/annotations/json").expanduser()
CAT = {1: "active", 2: "obsolete", 3: "uncertain"}
TOL = 16  # pointing-game tolerance in pixels of the 512 x 512 image


def boxes() -> dict[str, list[tuple[list[float], str]]]:
    out: dict[str, list[tuple[list[float], str]]] = {}
    for f in ("TBX11K_train.json", "TBX11K_val.json"):
        d = json.loads((ANN / f).read_text())
        name = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
        for a in d["annotations"]:
            out.setdefault(name[a["image_id"]], []).append((a["bbox"], CAT[a["category_id"]]))
    return out


def mask(bxs: list[tuple[list[float], str]], size: int, pad: int = 0) -> np.ndarray:
    m = np.zeros((size, size), bool)
    s = size / 512
    for (x, y, w, h), _ in bxs:
        x0, y0 = max(0, int((x - pad) * s)), max(0, int((y - pad) * s))
        x1, y1 = (
            min(size, int(np.ceil((x + w + pad) * s))),
            min(size, int(np.ceil((y + h + pad) * s))),
        )
        m[y0:y1, x0:x1] = True
    return m


def analyse(label: str, run: Path, bx: dict) -> pd.DataFrame:
    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    stem = pd.Series(view.patient_ids).astype(str).str.removeprefix("tbx11k_")
    site = view.metadata["site"].astype(str).to_numpy()
    keep = np.flatnonzero((site == "tbx11k") & (view.y == 1) & stem.isin(bx.keys()).to_numpy())
    sub = view.subset(keep)
    model = Classifier.load(run / "model.pkl")
    thr = plan_threshold(run)
    exp = GradCAMExplainer(model, target_classes=[1], batch_size=8).explain(sub)
    maps = np.clip(exp.attributions[:, 0], 0, None)
    rows = []
    for i, pid in enumerate(sub.patient_ids):
        sal = maps[i]
        size = sal.shape[0]
        b = bx[str(pid).removeprefix("tbx11k_")]
        m, mp = mask(b, size), mask(b, size, TOL)
        total = float(sal.sum()) or 1e-12
        peak = np.unravel_index(int(sal.argmax()), sal.shape)
        kinds = {k for _, k in b}
        rows.append(
            {
                "model": label,
                "run": run.name,
                "image": str(pid),
                "kind": "active"
                if "active" in kinds
                else ("obsolete" if kinds == {"obsolete"} else "uncertain"),
                "box_energy": float(sal[m].sum()) / total,
                "box_area": float(m.mean()),
                "box_ratio": float(sal[m].sum()) / total / max(float(m.mean()), 1e-6),
                "hit": bool(mp[peak]),
                "flagged": bool(exp.extra["prob"][i] >= thr),
            }
        )
    return pd.DataFrame(rows)


def plan_threshold(run: Path) -> float:
    v = pd.read_csv(run / "predictions_val.csv")
    return select_threshold(
        v["y_true"].to_numpy(),
        v[[c for c in v if c.startswith("prob_")][-1]].to_numpy(),
        method="sensitivity",
        target_sensitivity=0.9,
    )


def reflag(out: Path, specs: list[str]) -> pd.DataFrame:
    """Recompute only the ``flagged`` column from saved predictions (no Grad-CAM, no GPU)."""
    per = pd.read_csv(out / "per_image.csv")
    dirs = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]).expanduser().parent for s in specs}
    for (label, name), idx in per.groupby(["model", "run"]).groups.items():
        run = dirs[label] / name
        t = pd.read_csv(run / "predictions_test.csv")
        t = t[t["site"] == "tbx11k"].set_index("patient_id")
        prob = t[[c for c in t if c.startswith("prob_")][-1]]
        per.loc[idx, "flagged"] = per.loc[idx, "image"].map(prob).to_numpy() >= plan_threshold(run)
    per["flagged"] = per["flagged"].astype(bool)
    per.to_csv(out / "per_image.csv", index=False)
    return per


def main() -> None:
    redo = sys.argv[1] == "reflag"
    args = sys.argv[2:] if redo else sys.argv[1:]
    out = Path(args[0]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if redo:
        per = reflag(out, args[1:])
    else:
        bx = boxes()
        frames = []
        for spec in args[1:]:
            label, pattern = spec.split("=", 1)
            for run in sorted(glob.glob(str(Path(pattern).expanduser()))):
                print(f"== {label}: {run}", flush=True)
                frames.append(analyse(label, Path(run), bx))
                pd.concat(frames).to_csv(out / "per_image.csv", index=False)
        per = pd.concat(frames)
    summ = (
        per.groupby(["model", "kind"])
        .agg(
            images=("image", "nunique"),
            runs=("run", "nunique"),
            box_energy=("box_energy", "mean"),
            box_area=("box_area", "mean"),
            box_ratio=("box_ratio", "mean"),
            pointing=("hit", "mean"),
            flagged=("flagged", "mean"),
        )
        .reset_index()
    )
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
