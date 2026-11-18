"""Quantitative saliency on Montgomery: lung focus, pointing game and a randomisation check.

For every trained CXR (or CXR + tabular) run given, Grad-CAM maps for the TB output are
computed on all Montgomery images and compared with the expert lung masks shipped with the
Montgomery set (ManualMask/leftMask + rightMask):

* lung_energy   share of the (non-negative) Grad-CAM mass inside the lungs
* lung_area     share of the image covered by the lungs (what a uniform map would score)
* lung_ratio    lung_energy / lung_area (> 1: the map concentrates on the lungs)
* peak_in_lung  whether the map's maximum lies inside the lungs (pointing game)

Model randomisation check (Adebayo et al., 2018): the same maps from a copy of the network
with every layer re-initialised; a faithful map should change, so the Spearman correlation
between trained and randomised maps (64 x 64) should be low.

Images and masks go through the same square resize as in training, so they align.

Usage: python scripts/analysis/saliency_lungs.py OUT_DIR LABEL=RUN_GLOB [LABEL=RUN_GLOB ...]
"""

from __future__ import annotations

import copy
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.explain.saliency import GradCAMExplainer
from tbmxai.models.base import Classifier

MASKS = Path("~/data/nlm_tb/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask").expanduser()
N_SANITY = 24


def lung_mask(sample_id: str, size: tuple[int, int]) -> np.ndarray:
    name = sample_id.split("|")[-1]
    parts = [np.asarray(Image.open(MASKS / side / name).convert("L").resize(size, Image.NEAREST)) > 127
             for side in ("leftMask", "rightMask")]
    return parts[0] | parts[1]


def randomised(model: Classifier) -> Classifier:
    rnd = copy.deepcopy(model)
    torch.manual_seed(0)
    for module in rnd.network_.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return rnd


def small(m: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(m)[None, None].float()
    return torch.nn.functional.interpolate(t, size=(64, 64), mode="area")[0, 0].numpy().ravel()


def analyse(label: str, run: Path) -> pd.DataFrame:
    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    site = view.metadata["site"].astype(str).to_numpy()
    mont = view.subset(np.flatnonzero(site == "montgomery"))
    model = Classifier.load(run / "model.pkl")
    exp = GradCAMExplainer(model, target_classes=[1], batch_size=8).explain(mont)
    maps = np.clip(exp.attributions[:, 0], 0, None)  # (n, H, W)
    rows = []
    for i, sid in enumerate(exp.sample_ids):
        sal = maps[i]
        mask = lung_mask(str(sid), (sal.shape[1], sal.shape[0]))
        total = float(sal.sum()) or 1e-12
        energy = float(sal[mask].sum()) / total
        area = float(mask.mean())
        peak = np.unravel_index(int(sal.argmax()), sal.shape)
        rows.append({
            "model": label, "run": run.name, "sample_id": str(sid), "y_true": int(exp.extra["y_true"][i]),
            "prob_tb": float(exp.extra["prob"][i]), "outcome": str(exp.extra["category"][i]),
            "lung_energy": energy, "lung_area": area, "lung_ratio": energy / area if area else np.nan,
            "peak_in_lung": bool(mask[peak]),
        })
    out = pd.DataFrame(rows)
    # randomisation check on a fixed subset
    idx = np.arange(min(N_SANITY, len(mont)))
    sub = mont.subset(idx)
    rnd = GradCAMExplainer(randomised(model), target_classes=[1], batch_size=8).explain(sub)
    rmaps = np.clip(rnd.attributions[:, 0], 0, None)
    rho = [spearmanr(small(maps[j]), small(rmaps[j])).statistic for j in idx]
    out["rand_spearman"] = np.nan
    out.loc[idx, "rand_spearman"] = rho
    return out


def main() -> None:
    out_dir = Path(sys.argv[1]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for spec in sys.argv[2:]:
        label, pattern = spec.split("=", 1)
        for run in sorted(glob.glob(str(Path(pattern).expanduser()))):
            print(f"== {label}: {run}", flush=True)
            frames.append(analyse(label, Path(run)))
            pd.concat(frames).to_csv(out_dir / "per_image.csv", index=False)
    per = pd.concat(frames)
    per_run = per.groupby(["model", "run"]).agg(
        lung_energy=("lung_energy", "mean"), lung_ratio=("lung_ratio", "mean"),
        pointing=("peak_in_lung", "mean"), rand_spearman=("rand_spearman", "mean"),
    ).reset_index()
    summary = per_run.groupby("model").agg(
        runs=("run", "size"),
        lung_energy=("lung_energy", "mean"), lung_energy_sd=("lung_energy", "std"),
        lung_ratio=("lung_ratio", "mean"), lung_ratio_sd=("lung_ratio", "std"),
        pointing=("pointing", "mean"), pointing_sd=("pointing", "std"),
        rand_spearman=("rand_spearman", "mean"),
    ).reset_index()
    summary["lung_area"] = float(per.drop_duplicates("sample_id")["lung_area"].mean())
    per_run.to_csv(out_dir / "per_run.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    (out_dir / "info.json").write_text(json.dumps({"runs": sys.argv[2:], "n_sanity": N_SANITY}, indent=2))


if __name__ == "__main__":
    main()
