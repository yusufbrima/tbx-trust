"""Figures for the paper, generated from the experiment outputs (never edited by hand).

Usage: python scripts/reporting/make_figures.py OUT_DIR
Writes (PDF + PNG preview): external_auroc, operating_points, val_vs_external, saliency_vs_auroc.

Arms with no finished seed are skipped (the foundation-model runs may still be in progress), so
the script can be re-run at any time; the console lists what each figure contains.

Operating points use the plan's endpoint definition for every arm: the threshold that reaches
90 % sensitivity on the source validation set (``select_threshold(method="sensitivity")``),
applied unchanged to each external site. It is recomputed from ``predictions_val.csv`` because
the NLM-protocol runs stored a Youden threshold.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tbmxai.evaluation.threshold import select_threshold

R = Path("~/tbmxai_runs").expanduser()

# Initialisation family -> colour (reference palette, categorical slots 1-3, fixed order).
INIT_COLOR = {"ImageNet": "#2a78d6", "MIMIC-CXR": "#eb6834", "Public CXR FM": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WHO_SENS, WHO_SPEC = 0.90, 0.70
WIDTH = 6.3  # inches; figures are included at \linewidth


@dataclass(frozen=True)
class Arm:
    label: str  # row label in the figures
    init: str  # key of INIT_COLOR
    frozen: bool
    runs: dict[str, str]  # panel key -> glob of seed run folders

    def seeds(self, panel: str) -> list[Path]:
        pattern = self.runs.get(panel)
        if pattern is None:
            return []
        return [
            Path(p).parent for p in sorted(glob.glob(str(R / pattern / "predictions_test.csv")))
        ]


# Panel key -> (title, site in predictions_test.csv)
PANELS = {
    "sz_mont": ("Shenzhen $\\rightarrow$ Montgomery", "montgomery"),
    "sz_tbx": ("Shenzhen $\\rightarrow$ TBX11K", "tbx11k"),
    "tbx_sz": ("TBX11K $\\rightarrow$ Shenzhen", "shenzhen"),
}

ARMS = [
    Arm(
        "DenseNet-121, ImageNet",
        "ImageNet",
        False,
        {
            "sz_mont": "nlm_tb_tuned/cxr_densenet_seeds/seed*",
            "sz_tbx": "multisite/cxr_from_shenzhen/seed*",
            "tbx_sz": "multisite/cxr_from_tbx11k/seed*",
        },
    ),
    Arm(
        "DenseNet-121, MIMIC",
        "MIMIC-CXR",
        False,
        {"sz_mont": "transfer/nlm_densenet_mimicpt/seed*"},
    ),
    Arm(
        "DenseNet-121, MIMIC, frozen",
        "MIMIC-CXR",
        True,
        {"sz_mont": "transfer/nlm_densenet_mimicpt_frozen/seed*"},
    ),
    Arm(
        "ConvNeXt-T, ImageNet",
        "ImageNet",
        False,
        {
            "sz_mont": "nlm_tb_tuned/cxr_convnext_seeds/seed*",
            "sz_tbx": "multisite/cxr_convnext_from_shenzhen/seed[1-3]",
            "tbx_sz": "multisite_retuned/imagenet/seed*",
        },
    ),
    Arm(
        "ConvNeXt-T, MIMIC",
        "MIMIC-CXR",
        False,
        {
            "sz_mont": "transfer/nlm_convnext_mimicpt/seed*",
            "sz_tbx": "transfer/multisite_convnext_mimicpt_from_shenzhen/seed[1-3]",
            "tbx_sz": "multisite_retuned/mimic/seed*",
        },
    ),
    Arm(
        "ConvNeXt-T, MIMIC, frozen",
        "MIMIC-CXR",
        True,
        {"sz_mont": "transfer/nlm_convnext_mimicpt_frozen/seed*"},
    ),
    Arm(
        "TorchXRayVision",
        "Public CXR FM",
        False,
        {
            "sz_mont": "transfer/nlm_xrv/seed*",
            "sz_tbx": "transfer/multisite_xrv_from_shenzhen/seed*",
        },
    ),
    Arm(
        "TorchXRayVision, frozen",
        "Public CXR FM",
        True,
        {
            "sz_mont": "transfer/nlm_xrv_frozen/seed*",
            "sz_tbx": "transfer/multisite_xrv_frozen_from_shenzhen/seed*",
        },
    ),
    Arm(
        "RAD-DINO",
        "Public CXR FM",
        False,
        {
            "sz_mont": "transfer/nlm_raddino/seed*",
            "sz_tbx": "transfer/multisite_raddino_from_shenzhen/seed*",
        },
    ),
    Arm(
        "RAD-DINO, frozen",
        "Public CXR FM",
        True,
        {
            "sz_mont": "transfer/nlm_raddino_frozen/seed*",
            "sz_tbx": "transfer/multisite_raddino_frozen_from_shenzhen/seed*",
        },
    ),
]

# Saliency analyses (Grad-CAM on Montgomery, lung masks), every image-only model analysed:
# model label -> (short name, init, frozen, training data/protocol, label offset in points).
SAL_SZ, SAL_MS, SAL_TBX = "Shenzhen (NLM protocol)", "Shenzhen (multi-site protocol)", "TBX11K"
SALIENCY = {
    "Shenzhen→Montgomery: CXR DenseNet-121 (tuned)": (
        "DenseNet",
        "ImageNet",
        False,
        SAL_SZ,
        (6, -3),
    ),
    "Shenzhen→Montgomery: CXR ConvNeXt-T": ("ConvNeXt", "ImageNet", False, SAL_SZ, (-6, 5)),
    "Shenzhen→Montgomery: ConvNeXt-T MIMIC-pretrained": (
        "ConvNeXt",
        "MIMIC-CXR",
        False,
        SAL_SZ,
        (-4, 6),
    ),
    "Shenzhen→Montgomery: ConvNeXt-T MIMIC-pretrained, frozen": (
        "ConvNeXt frozen",
        "MIMIC-CXR",
        True,
        SAL_SZ,
        (-7, -6),
    ),
    "Multi-site: DenseNet-121 trained on Shenzhen": (
        "DenseNet",
        "ImageNet",
        False,
        SAL_MS,
        (6, -3),
    ),
    "Multi-site: ConvNeXt-T trained on Shenzhen": ("ConvNeXt", "ImageNet", False, SAL_MS, (6, -8)),
    "Multi-site: ConvNeXt-T MIMIC-pretrained, trained on Shenzhen": (
        "ConvNeXt",
        "MIMIC-CXR",
        False,
        SAL_MS,
        (6, -3),
    ),
    "Multi-site: DenseNet-121 trained on TBX11K": ("DenseNet", "ImageNet", False, SAL_TBX, (6, -3)),
    "Multi-site: ConvNeXt-T trained on TBX11K": ("ConvNeXt", "ImageNet", False, SAL_TBX, (-6, 4)),
    "Multi-site: ConvNeXt-T MIMIC-pretrained, trained on TBX11K": (
        "ConvNeXt",
        "MIMIC-CXR",
        False,
        SAL_TBX,
        (6, -3),
    ),
}
SAL_MARK = {SAL_SZ: "o", SAL_MS: "s", SAL_TBX: "^"}
SALIENCY_DIRS = ["saliency_lungs", "saliency_lungs_transfer", "saliency_lungs_convnext_multisite"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def site_preds(run: Path, site: str, split: str = "test") -> pd.DataFrame:
    d = pd.read_csv(run / f"predictions_{split}.csv")
    if split == "test" and "site" in d:
        d = d[d["site"] == site]
    d = d.assign(p=d[[c for c in d if c.startswith("prob_")][-1]])
    return d[["y_true", "p"]]


def per_seed(arm: Arm, panel: str) -> pd.DataFrame:
    """One row per seed: external AUROC, validation AUROC, deployed sensitivity / specificity."""
    site = PANELS[panel][1]
    rows = []
    for run in arm.seeds(panel):
        t, v = site_preds(run, site), site_preds(run, site, "val")
        thr = select_threshold(
            v["y_true"].to_numpy(),
            v["p"].to_numpy(),
            method="sensitivity",
            target_sensitivity=WHO_SENS,
        )
        pos, y = t["p"] >= thr, t["y_true"] == 1
        rows.append(
            {
                "seed": run.name,
                "auroc": roc_auc_score(t["y_true"], t["p"]),
                "val_auroc": roc_auc_score(v["y_true"], v["p"]),
                "sens": (pos & y).sum() / y.sum(),
                "spec": (~pos & ~y).sum() / (~y).sum(),
            }
        )
    return pd.DataFrame(rows)


def collect() -> pd.DataFrame:
    frames = []
    for i, arm in enumerate(ARMS):
        for panel in PANELS:
            d = per_seed(arm, panel)
            if len(d):
                frames.append(
                    d.assign(row=i, arm=arm.label, init=arm.init, frozen=arm.frozen, panel=panel)
                )
    return pd.concat(frames, ignore_index=True)


def marker_kw(init: str, frozen: bool, size: float) -> dict:
    c = INIT_COLOR[init]
    return {
        "marker": "o",
        "s": size,
        "facecolors": "white" if frozen else c,
        "edgecolors": c,
        "linewidths": 1.0 if frozen else 0.5,
        "zorder": 3,
    }


def legend_handles() -> list[Line2D]:
    h = [
        Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", markersize=6, label=k)
        for k, c in INIT_COLOR.items()
    ]
    h.append(
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            markersize=6,
            label="frozen encoder",
        )
    )
    return h


def row_axis(ax: plt.Axes, show_labels: bool) -> None:
    ax.set_ylim(len(ARMS) - 0.5, -0.5)
    ax.set_yticks(range(len(ARMS)), [a.label for a in ARMS])
    ax.tick_params(axis="y", length=0, labelleft=show_labels)
    for i in range(len(ARMS)):
        ax.axhline(i, color=GRID, lw=0.5, zorder=0)
    for b in (2.5, 5.5):  # DenseNet | ConvNeXt | public FMs
        ax.axhline(b, color=MUTED, lw=0.5, zorder=0)


def dot_row(ax: plt.Axes, d: pd.DataFrame, col: str) -> None:
    rng = np.random.default_rng(0)
    for (row, init, frozen), g in d.groupby(["row", "init", "frozen"]):
        jitter = rng.uniform(-0.12, 0.12, len(g))
        ax.scatter(
            g[col], row + jitter, **{**marker_kw(init, frozen, 9), "alpha": 0.55, "zorder": 2}
        )
        ax.scatter([g[col].mean()], [row], **marker_kw(init, frozen, 34))


def fig_external_auroc(df: pd.DataFrame, out: Path) -> None:
    panels = [p for p in PANELS if (df["panel"] == p).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 3.0), sharey=True)
    for j, (ax, p) in enumerate(zip(np.atleast_1d(axes), panels, strict=True)):
        row_axis(ax, j == 0)
        ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
        dot_row(ax, df[df["panel"] == p], "auroc")
        ax.set_xlim(0.45, 1.0)
        ax.set_title(PANELS[p][0])
        ax.set_xlabel("External AUROC")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.6, 0.0),
    )
    save(fig, out / "external_auroc")


def fig_operating_points(df: pd.DataFrame, out: Path) -> None:
    panels = [p for p in ("sz_mont", "sz_tbx") if (df["panel"] == p).any()]
    fig, axes = plt.subplots(
        1, 2 * len(panels), figsize=(WIDTH, 3.0), sharey=True, gridspec_kw={"wspace": 0.12}
    )
    for j, p in enumerate(panels):
        d = df[df["panel"] == p]
        for k, (col, target, name) in enumerate(
            (("sens", WHO_SENS, "Sensitivity"), ("spec", WHO_SPEC, "Specificity"))
        ):
            ax = axes[2 * j + k]
            row_axis(ax, j == 0 and k == 0)
            ax.axvspan(target, 1.0, color="#f1f0ec", zorder=0)
            ax.axvline(target, color=MUTED, lw=0.6, zorder=1)
            dot_row(ax, d, col)
            ax.set_xlim(0.0 if col == "spec" else 0.5, 1.0)
            ax.set_xticks([0.2, 0.4, 0.6, 0.8] if col == "spec" else [0.6, 0.7, 0.8, 0.9])
            ax.set_xlabel(f"{name} (target {target:.0%})")
            if k == 0:
                ax.set_title(PANELS[p][0], loc="left")
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.6, 0.0),
    )
    save(fig, out / "operating_points")


def fig_val_vs_external(df: pd.DataFrame, out: Path) -> None:
    panels = [p for p in ("sz_mont", "sz_tbx", "tbx_sz") if (df["panel"] == p).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 2.4))
    for ax, p in zip(np.atleast_1d(axes), panels, strict=True):
        d = df[df["panel"] == p]
        for (init, frozen), g in d.groupby(["init", "frozen"]):
            ax.scatter(g["val_auroc"], g["auroc"], **marker_kw(init, frozen, 16))
        rho = spearmanr(d["val_auroc"], d["auroc"]).statistic
        ax.text(
            0.04,
            0.96,
            f"Spearman $\\rho$ = {rho:.2f}\n{len(d)} seed runs",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            color=MUTED,
        )
        ax.set_title(PANELS[p][0])
        ax.set_xlabel("Source validation AUROC")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("External AUROC")
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout(w_pad=1.5)
    save(fig, out / "val_vs_external")


def saliency_points() -> pd.DataFrame:
    runs = pd.concat([pd.read_csv(R / "analysis" / d / "per_run.csv") for d in SALIENCY_DIRS])
    runs = runs[runs["model"].isin(SALIENCY)]
    info = {}
    for d in SALIENCY_DIRS:
        for line in pd.read_json(R / "analysis" / d / "info.json")["runs"]:
            label, pattern = line.split("=", 1)
            info[label] = Path(pattern).parent
    rows = []
    for _, r in runs.iterrows():
        t = site_preds(info[r["model"]] / r["run"], "montgomery")
        rows.append(
            {
                "model": r["model"],
                "lung_ratio": r["lung_ratio"],
                "auroc": roc_auc_score(t["y_true"], t["p"]),
            }
        )
    return pd.DataFrame(rows)


def fig_saliency(out: Path) -> None:
    d = saliency_points()
    fig, ax = plt.subplots(figsize=(WIDTH * 0.66, 2.8))
    for model, g in d.groupby("model"):
        name, init, frozen, train, offset = SALIENCY[model]
        kw = {**marker_kw(init, frozen, 10), "marker": SAL_MARK[train]}
        ax.scatter(g["lung_ratio"], g["auroc"], **{**kw, "alpha": 0.45, "zorder": 2})
        xy = (g["lung_ratio"].mean(), g["auroc"].mean())
        ax.scatter(*xy, **{**kw, "s": 42})
        ax.annotate(
            name,
            xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=6,
            color=MUTED,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
        )
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.text(
        0.98,
        0.985,
        "no lung\npreference",
        fontsize=6,
        color=MUTED,
        ha="right",
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    means = d.groupby("model")[["lung_ratio", "auroc"]].mean()
    rho = spearmanr(means["lung_ratio"], means["auroc"]).statistic
    ax.text(
        0.02,
        0.03,
        f"Spearman $\\rho$ = {rho:.2f} over {len(means)} model means",
        transform=ax.transAxes,
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_xlabel("Grad-CAM lung ratio on Montgomery (energy share / lung-area share)")
    ax.set_ylabel("Montgomery AUROC")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    h = [
        *legend_handles()[:2],
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            markersize=6,
            label="frozen encoder",
        ),
    ]
    h += [
        Line2D([], [], marker=m, ls="", color=MUTED, markersize=5, label=f"trained on {t}")
        for t, m in SAL_MARK.items()
    ]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6.5)
    save(fig, out / "saliency_vs_auroc")


def fig_recalibration(out: Path) -> None:
    """P(meets the WHO minimum) after local recalibration, by confirmed local TB cases.

    recalibration: threshold at the 5th percentile of the k local TB scores (intended
    sensitivity 95 %); population sensitivity from the site's TB score density, specificity on all
    the site's non-TB images; single runs (the deployed artefact), averaged.
    """
    f = R / "analysis" / "recalibration" / "summary.csv"
    if not f.exists():
        print("skip recalibration figure: run scripts/analysis/recalibration.py first")
        return
    d = pd.read_csv(f)
    d = d[(d["rule"] == "q05") & (d["evaluation"] == "density")]
    fam = {"ImageNet": "ImageNet", "MIMIC": "MIMIC-CXR", "Public FM": "Public CXR FM"}
    show = [
        "ConvNeXt-T, ImageNet",
        "ConvNeXt-T, MIMIC",
        "TorchXRayVision",
        "RAD-DINO",
        "RAD-DINO, frozen",
        "RAD-DINO, retuned",
    ]
    ref = d[d["model"] == "Perfect discrimination (reference)"]
    fig, axes = plt.subplots(
        1, 2, figsize=(WIDTH, 2.5), sharey=True, gridspec_kw={"width_ratios": [1, 1.6]}
    )
    for ax, site in zip(axes, ("montgomery", "tbx11k"), strict=True):
        ends: list[list] = []
        kmax = 40 if site == "montgomery" else 100
        ref_s = ref[ref["k_cases"] <= kmax]
        ax.plot(
            ref_s["k_cases"],
            ref_s["p_who"],
            color=MUTED,
            lw=1,
            ls=(0, (1, 2)),
            label="perfect discrimination",
        )
        for m in show:
            g = d[(d["site"] == site) & (d["model"] == m)].sort_values("k_cases")
            if g.empty:
                continue
            c = INIT_COLOR[fam[g["family"].iloc[0]]]
            ls = (0, (4, 2)) if "frozen" in m else ((0, (1, 1)) if "retuned" in m else "-")
            ax.plot(g["k_cases"], g["p_who"], color=c, lw=1.4, ls=ls, marker="o", ms=3, label=m)
            ends.append([float(g["p_who"].iloc[-1]), m])
        ends.sort()
        for i in range(1, len(ends)):  # keep end labels at least 0.06 apart
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
        ends.append([float(ref_s["p_who"].iloc[-1]) + 0.03, "perfect discrimination"])
        for y, m in ends:
            ax.annotate(
                m,
                (kmax, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=MUTED,
                va="center",
            )
        ax.set_title(SITE_TITLE[site], loc="left")
        ax.set_xticks([10, 20, 30, 40] if site == "montgomery" else [10, 20, 40, 60, 80, 100])
        ax.set_xlim(8, kmax * 1.45)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("P(meets WHO minimum)")
    fig.supxlabel(
        "Confirmed local TB cases used to set the threshold (consecutive patients at 5% = 20x)",
        fontsize=7.5,
        y=-0.05,
    )
    save(fig, out / "recalibration")


SITE_TITLE = {
    "montgomery": "Shenzhen $\\rightarrow$ Montgomery",
    "tbx11k": "Shenzhen $\\rightarrow$ TBX11K",
}


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    style()
    df = collect()
    summary = df.groupby(["panel", "arm", "init", "frozen"], sort=False).agg(
        seeds=("auroc", "size"),
        auroc=("auroc", "mean"),
        sens=("sens", "mean"),
        spec=("spec", "mean"),
    )
    print(summary.round(3).to_string())
    fig_external_auroc(df, out)
    fig_operating_points(df, out)
    fig_val_vs_external(df, out)
    fig_saliency(out)
    fig_recalibration(out)
    print("wrote", sorted(p.name for p in out.glob("*.pdf")))


if __name__ == "__main__":
    main()
