"""Figure: Grad-CAM against TBX11K's radiologist-drawn TB boxes, two models on the same images.

Images are drawn at random (fixed seed) from the TBX11K TB images with boxes: four with an active
lesion and two with inactive (obsolete) lesions only, so the figure is not a selection of good cases.
Each panel prints that image's box ratio (share of saliency inside the boxes / boxes' share of the
image) and the model's TB probability. Grad-CAM exactly as in scripts/analysis/saliency_boxes.py.
TBX11K is public (CC BY 4.0), so its images may be shown.

Usage: python scripts/reporting/saliency_overlays.py OUT_STEM LABEL=RUN_DIR [LABEL=RUN_DIR ...]
       --compact as the first argument: the first two active and the first inactive image of the same
       draw, models as rows, boxes drawn on the maps (main-text version).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from saliency_boxes import boxes, mask  # noqa: E402

from tbmxai import ExperimentConfig, Pipeline  # noqa: E402
from tbmxai.explain.saliency import GradCAMExplainer  # noqa: E402
from tbmxai.models.base import Classifier  # noqa: E402

IMG = Path("~/data/tbx11k/TBX11K/TBX11K/imgs/tb").expanduser()
COLOR = {"active": "#00e5ff", "obsolete": "#ffd400", "uncertain": "#ffffff"}


def pick(bx: dict, available: set[str], seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    kind = {s: ("active" if any(k == "active" for _, k in b) else
                "obsolete" if all(k == "obsolete" for _, k in b) else "uncertain")
            for s, b in bx.items() if s in available}  # fmt: skip
    act = sorted(s for s, k in kind.items() if k == "active")
    old = sorted(s for s, k in kind.items() if k == "obsolete")
    return list(rng.choice(act, 4, replace=False)) + list(rng.choice(old, 2, replace=False))


def maps_for(run: Path, stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cfg = ExperimentConfig.from_yaml(run / "config.yaml")
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    ids = pd.Series(view.patient_ids).astype(str).str.removeprefix("tbx11k_")
    site = view.metadata["site"].astype(str).to_numpy()
    idx = [int(np.flatnonzero((ids == s).to_numpy() & (site == "tbx11k"))[0]) for s in stems]
    sub = view.subset(np.array(idx))
    model = Classifier.load(run / "model.pkl")
    exp = GradCAMExplainer(model, target_classes=[1], batch_size=8).explain(sub)
    return np.clip(exp.attributions[:, 0], 0, None), np.asarray(exp.extra["prob"])


def compact(out: Path, stems: list[str], results: list, bx: dict) -> None:
    cols = [stems[0], stems[1], stems[4]]  # first two active, first inactive (same random draw)
    pos = [0, 1, 4]
    fig, axes = plt.subplots(len(results), 3, figsize=(6.6, 2.25 * len(results)))
    for r, (label, maps, probs) in enumerate(results):
        for c, (stem, k) in enumerate(zip(cols, pos, strict=True)):
            ax = axes[r, c]
            img = np.asarray(Image.open(IMG / f"{stem}.png").convert("L").resize((512, 512)))
            sal = np.array(Image.fromarray(maps[k]).resize((512, 512), Image.BILINEAR))
            ax.imshow(img, cmap="gray")
            ax.imshow(sal / (sal.max() or 1), cmap="jet", alpha=0.42)
            for (x, y, w, h), kind in bx[stem]:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.2, ec=COLOR[kind]))
            m = mask(bx[stem], maps[k].shape[0])
            ratio = (maps[k][m].sum() / (maps[k].sum() or 1e-12)) / max(m.mean(), 1e-6)
            ax.set_xlabel(f"ratio {ratio:.1f} · p(TB) {probs[k]:.2f}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title("active lesion" if c < 2 else "inactive lesion only", fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=170)
    fig.savefig(out.with_suffix(".pdf"))


def main() -> None:
    args = sys.argv[1:]
    is_compact = args[0] == "--compact"
    if is_compact:
        args = args[1:]
    out = Path(args[0])
    specs = [s.split("=", 1) for s in args[1:]]
    bx = boxes()
    available = {p.stem for p in IMG.glob("*.png")}
    stems = pick(bx, available)
    if is_compact:
        results = [(label, *maps_for(Path(run).expanduser(), stems)) for label, run in specs]
        compact(out, stems, results, bx)
        print("images:", stems)
        return
    results = [(label, *maps_for(Path(run).expanduser(), stems)) for label, run in specs]
    ncol = 1 + len(results)
    fig, axes = plt.subplots(len(stems), ncol, figsize=(2.3 * ncol, 2.35 * len(stems)))
    for r, stem in enumerate(stems):
        img = np.asarray(Image.open(IMG / f"{stem}.png").convert("L").resize((512, 512)))
        kinds = {k for _, k in bx[stem]}
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            if c > 0:
                label, maps, probs = results[c - 1]
                sal = np.array(Image.fromarray(maps[r]).resize((512, 512), Image.BILINEAR))
                ax.imshow(sal / (sal.max() or 1), cmap="jet", alpha=0.42)
                m = mask(bx[stem], maps[r].shape[0])
                ratio = (maps[r][m].sum() / (maps[r].sum() or 1e-12)) / max(m.mean(), 1e-6)
                ax.set_xlabel(f"ratio {ratio:.1f} · p(TB) {probs[r]:.2f}", fontsize=7)
            for (x, y, w, h), k in bx[stem]:
                ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.2, ec=COLOR[k]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("Image and boxes" if c == 0 else results[c - 1][0], fontsize=8)
            if c == 0:
                ax.set_ylabel("active" if "active" in kinds else "inactive only", fontsize=8)
    fig.suptitle("Grad-CAM for TB on TBX11K, Shenzhen-trained models (cyan: active lesion, "
                 "yellow: inactive lesion)", fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out.with_suffix(".png"), dpi=170)
    fig.savefig(out.with_suffix(".pdf"))
    print("images:", stems)


if __name__ == "__main__":
    main()
