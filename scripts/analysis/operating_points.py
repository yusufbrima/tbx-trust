"""Operating-point, programme and contrast analyses for the paper.

From saved predictions only (no retraining). Writes to OUT_DIR:

  operating_points.csv  per arm x site x population: every single run as the deployed artefact
                        (threshold = highest score reaching 90 % sensitivity on that run's
                        source-validation scores; the seed ensemble gives AUROC only, as seeds use
                        different validation splits), with Clopper-Pearson 95 % CIs; whether the WHO
                        minimum (90/70) and optimal (95/80) triage targets are met on the point
                        estimate and on the CI lower bounds; the oracle (whole-site) specificity at
                        90 % sensitivity; and the split of the specificity shortfall into
                        discrimination (oracle below 70 %) and threshold placement (run vs oracle).
  programme.csv         per arm x site, at 1 / 5 / 10 % prevalence: Xpert referrals per 1,000
                        screened, tests per TB case detected, cases missed per 1,000, cartridge cost
                        per case detected (US$7.97), for every single run.
  contrasts.csv         every C1-C7 contrast, seed-paired: mean over seeds of the per-seed AUROC
                        difference (single-model estimand) with a test-patient bootstrap CI, and a
                        training-variance sensitivity analysis (seed as a random effect, t on k - 1
                        df) with its own verdict; Holm within C1, C2, C5 for both.
  tbx11k_classes.csv    TBX11K AUROC of TB vs healthy and TB vs sick-but-non-TB, and TB split by the
                        TBX11K box categories (active / obsolete-only / uncertain-only).

Populations: "all"; "adults" (Montgomery/Shenzhen without the <18 age band; WHO's CAD recommendation
covers people aged >= 15); "active TB" (TBX11K positives restricted to images with an
ActiveTuberculosis box; obsolete-only and uncertain-only TB images removed).

Usage: python scripts/analysis/operating_points.py OUT_DIR [B]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score, roc_curve

from tbmxai.evaluation.threshold import select_threshold

R = Path("~/tbmxai_runs").expanduser()
TBX_ANN = Path("~/data/tbx11k/TBX11K/TBX11K/annotations/json").expanduser()
CARTRIDGE_USD = 7.97

# arm label -> {site: run glob}; protocol: NLM (Montgomery, 5 seeds), multi-site (3 seeds)
ARMS = {
    "DenseNet-121, ImageNet": {
        "montgomery": "nlm_tb_tuned/cxr_densenet_seeds/seed*",
        "tbx11k": "multisite/cxr_from_shenzhen/seed*",
    },
    "DenseNet-121, MIMIC": {"montgomery": "transfer/nlm_densenet_mimicpt/seed*"},
    "DenseNet-121, MIMIC, frozen": {"montgomery": "transfer/nlm_densenet_mimicpt_frozen/seed*"},
    "ConvNeXt-T, ImageNet": {
        "montgomery": "nlm_tb_tuned/cxr_convnext_seeds/seed*",
        "tbx11k": "multisite/cxr_convnext_from_shenzhen/seed[1-3]",
    },
    "ConvNeXt-T, MIMIC": {
        "montgomery": "transfer/nlm_convnext_mimicpt/seed*",
        "tbx11k": "transfer/multisite_convnext_mimicpt_from_shenzhen/seed[1-3]",
    },
    "ConvNeXt-T, MIMIC, frozen": {"montgomery": "transfer/nlm_convnext_mimicpt_frozen/seed*"},
    "TorchXRayVision": {
        "montgomery": "transfer/nlm_xrv/seed*",
        "tbx11k": "transfer/multisite_xrv_from_shenzhen/seed*",
    },
    "TorchXRayVision, frozen": {
        "montgomery": "transfer/nlm_xrv_frozen/seed*",
        "tbx11k": "transfer/multisite_xrv_frozen_from_shenzhen/seed*",
    },
    "RAD-DINO (pre-specified)": {
        "montgomery": "transfer/nlm_raddino/seed*",
        "tbx11k": "transfer/multisite_raddino_from_shenzhen/seed*",
    },
    "RAD-DINO, frozen": {
        "montgomery": "transfer/nlm_raddino_frozen/seed*",
        "tbx11k": "transfer/multisite_raddino_frozen_from_shenzhen/seed*",
    },
    "RAD-DINO (retuned)": {
        "montgomery": "transfer/raddino_lr/nlm_*/seed*",
        "tbx11k": "transfer/raddino_lr/multisite_*/seed*",
    },
    # trained on TBX11K
    "DenseNet-121, ImageNet [TBX11K-trained]": {"shenzhen": "multisite/cxr_from_tbx11k/seed*"},
    # ConvNeXt-T TBX11K-trained arms: TBX11K-only re-tune (settings selected on TBX11K validation)
    "ConvNeXt-T, ImageNet [TBX11K-trained]": {"shenzhen": "multisite_retuned/imagenet/seed*"},
    "ConvNeXt-T, MIMIC [TBX11K-trained]": {"shenzhen": "multisite_retuned/mimic/seed*"},
}


def cp(k: int, n: int) -> tuple[float, float]:
    """Clopper-Pearson exact 95 % interval."""
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def read(run: Path, split: str) -> pd.DataFrame:
    d = pd.read_csv(run / f"predictions_{split}.csv")
    d["p"] = d[[c for c in d if c.startswith("prob_")][-1]]
    return d


def tbx_tb_kind() -> dict[str, str]:
    """TBX11K TB image stem -> active / obsolete / uncertain (from the box categories)."""
    kind: dict[str, set[int]] = {}
    for f in ("TBX11K_train.json", "TBX11K_val.json"):
        d = json.loads((TBX_ANN / f).read_text())
        name = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
        for a in d["annotations"]:
            kind.setdefault(name[a["image_id"]], set()).add(a["category_id"])
    out = {}
    for stem, cats in kind.items():
        out[stem] = "active" if 1 in cats else ("obsolete" if cats == {2} else "uncertain")
    return out


def population(d: pd.DataFrame, site: str, pop: str, kinds: dict[str, str]) -> pd.DataFrame:
    if pop == "adults":
        return (
            d[d["age_group"] != "<18"]
            if "age_group" in d and d["age_group"].notna().any()
            else d.iloc[0:0]
        )
    if pop == "active TB":
        if site != "tbx11k":
            return d.iloc[0:0]
        stem = d["patient_id"].astype(str).str.removeprefix("tbx11k_")
        keep = (d["y_true"] == 0) | (stem.map(kinds) == "active")
        return d[keep]
    return d


def rates(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pos = p >= thr
    tp, fn = int((pos & (y == 1)).sum()), int((~pos & (y == 1)).sum())
    tn, fp = int((~pos & (y == 0)).sum()), int((pos & (y == 0)).sum())
    se_lo, se_hi = cp(tp, tp + fn)
    sp_lo, sp_hi = cp(tn, tn + fp)
    return {
        "sens": tp / (tp + fn),
        "sens_lo": se_lo,
        "sens_hi": se_hi,
        "spec": tn / (tn + fp),
        "spec_lo": sp_lo,
        "spec_hi": sp_hi,
        "n_tb": tp + fn,
        "n_non_tb": tn + fp,
    }


def oracle_spec(y: np.ndarray, p: np.ndarray, target: float = 0.9) -> float:
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    return float((1 - fpr[tpr >= target - 1e-12]).max())


def verdicts(r: dict[str, float]) -> dict[str, bool]:
    return {
        "who_min": r["sens"] >= 0.9 and r["spec"] >= 0.7,
        "who_min_ci": r["sens_lo"] >= 0.9 and r["spec_lo"] >= 0.7,
        "who_opt": r["sens"] >= 0.95 and r["spec"] >= 0.8,
    }


def ensemble(runs: list[Path], site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tests, vals = [], []
    for run in runs:
        t, v = read(run, "test"), read(run, "val")
        if "site" in t:
            t = t[t["site"] == site]
        tests.append(t.set_index("sample_id"))
        vals.append(v.set_index("sample_id")["p"])
    t = tests[0].drop(columns=["p"]).copy()
    t["p"] = pd.concat([x["p"] for x in tests], axis=1).mean(axis=1)
    v = vals[0].to_frame("p")
    v["p"] = pd.concat(vals, axis=1).mean(axis=1)
    v["y_true"] = read(runs[0], "val").set_index("sample_id")["y_true"]
    return t.reset_index(), v.reset_index()


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    out.mkdir(parents=True, exist_ok=True)
    kinds = tbx_tb_kind()
    rows, prog, tbx = [], [], []
    for arm, sites in ARMS.items():
        for site, pattern in sites.items():
            runs = [
                Path(p).parent for p in sorted(glob.glob(str(R / pattern / "predictions_test.csv")))
            ]
            if not runs:
                continue
            # Seeds draw different validation splits, so a seed ensemble has no valid source
            # threshold. The deployed artefact is therefore a single run (each seed is a model a
            # site could deploy, with its own validation-set threshold); the ensemble gives AUROC.
            ens_t, _ = ensemble(runs, site)
            for pop in ("all", "adults", "active TB"):
                e = population(ens_t, site, pop, kinds)
                if e.empty or e["y_true"].nunique() < 2:
                    continue
                y, p = e["y_true"].to_numpy(int), e["p"].to_numpy(float)
                rows.append(
                    {
                        "arm": arm,
                        "site": site,
                        "population": pop,
                        "unit": "ensemble",
                        "run": f"{len(runs)} seeds",
                        "auroc": roc_auc_score(y, p),
                        "oracle_spec90": oracle_spec(y, p),
                    }
                )
                for run in runs:
                    t, v = read(run, "test"), read(run, "val")
                    if "site" in t:
                        t = t[t["site"] == site]
                    t = population(t, site, pop, kinds)
                    thr = select_threshold(
                        v["y_true"].to_numpy(),
                        v["p"].to_numpy(),
                        method="sensitivity",
                        target_sensitivity=0.9,
                    )
                    yy, pp = t["y_true"].to_numpy(int), t["p"].to_numpy(float)
                    rr = rates(yy, pp, thr)
                    ora = oracle_spec(yy, pp)
                    # spec available at the sensitivity the run actually reached (on its own ROC)
                    ora_at = oracle_spec(yy, pp, target=rr["sens"])
                    rows.append(
                        {
                            "arm": arm,
                            "site": site,
                            "population": pop,
                            "unit": "run",
                            "run": run.name,
                            "auroc": roc_auc_score(yy, pp),
                            **rr,
                            **verdicts(rr),
                            "oracle_spec90": ora,
                            "sens_shift": rr["sens"] - 0.9,
                            "spec_cost_of_shift": ora - rr["spec"],
                            "spec_gap_at_reached_sens": ora_at - rr["spec"],
                        }
                    )
                    if pop == "all":
                        for prev in (0.01, 0.05, 0.10):
                            for neg_spec, scen in ((rr["spec"], "site negatives"),):
                                ref = 1000 * (prev * rr["sens"] + (1 - prev) * (1 - neg_spec))
                                det = 1000 * prev * rr["sens"]
                                prog.append(
                                    {
                                        "arm": arm,
                                        "site": site,
                                        "run": run.name,
                                        "prevalence": prev,
                                        "scenario": scen,
                                        "sens": rr["sens"],
                                        "spec": neg_spec,
                                        "referrals_per_1000": ref,
                                        "cases_detected_per_1000": det,
                                        "tests_per_case": ref / det if det else np.nan,
                                        "missed_per_1000": 1000 * prev * (1 - rr["sens"]),
                                        "cartridge_usd_per_case": CARTRIDGE_USD * ref / det
                                        if det
                                        else np.nan,
                                    }
                                )
                        if site == "tbx11k":
                            cls = t["tbx_class"].astype(str).to_numpy()
                            stem = (
                                t["patient_id"]
                                .astype(str)
                                .str.removeprefix("tbx11k_")
                                .map(kinds)
                                .fillna("")
                                .to_numpy()
                            )
                            rec = {"arm": arm, "run": run.name}
                            for neg in ("healthy", "sick_non_tb"):
                                m = (yy == 1) | (cls == neg)
                                rec[f"auroc_tb_vs_{neg}"] = roc_auc_score(yy[m], pp[m])
                                rec[f"flagged_{neg}"] = float((pp[cls == neg] >= thr).mean())
                            for k in ("active", "obsolete"):
                                sel = (yy == 1) & (stem == k)
                                rec[f"n_tb_{k}"] = int(sel.sum())
                                rec[f"flagged_tb_{k}"] = float((pp[sel] >= thr).mean())
                            # sick-but-non-TB specificity: the triage-relevant negatives
                            rec["spec_sick_non_tb"] = 1 - rec["flagged_sick_non_tb"]
                            tbx.append(rec)
            print(f"done {arm} / {site}", flush=True)
    pd.DataFrame(rows).to_csv(out / "operating_points.csv", index=False)
    pd.DataFrame(prog).to_csv(out / "programme.csv", index=False)
    pd.DataFrame(tbx).to_csv(out / "tbx11k_classes.csv", index=False)
    contrasts(out, b)


def contrasts(out: Path, b: int) -> None:
    """Seed-paired contrasts: seed s of arm A and seed s of arm B share the same source train /
    validation split, so the estimand is the mean over seeds of the per-seed AUROC difference
    (a single deployable model, not an ensemble). The 95 % CI resamples test patients (the same
    resample for every seed); the per-seed differences and how many favour B are reported too."""
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_arms import COMPARISONS

    rng = np.random.default_rng(0)
    rows = []
    for cid, family, question, pa, pb, site in COMPARISONS:
        if site is None:
            continue  # comparisons without a TB test site are not part of this analysis
        mats, ys = [], None
        if cid == "C1" and site == "tbx11k":
            # the two C1 arms on TBX11K were trained with seeds 4-10 as well, for this
            # training-variance analysis only; the pre-specified analysis keeps seeds 1-3
            pa, pb = pa.replace("seed[1-3]", "seed*"), pb.replace("seed[1-3]", "seed*")
        for pattern in (pa, pb):
            frames = []
            runs = glob.glob(str(R / pattern))
            for run in sorted(runs, key=lambda r: int(Path(r).name.removeprefix("seed"))):
                d = read(Path(run), "test")
                if "site" in d:
                    d = d[d["site"] == site]
                frames.append(d.set_index("sample_id").sort_index())
            ys = frames[0]["y_true"].to_numpy(int)
            mats.append(np.column_stack([f["p"].to_numpy(float) for f in frames]))
        k = min(mats[0].shape[1], mats[1].shape[1])
        per_seed = [
            roc_auc_score(ys, mats[1][:, j]) - roc_auc_score(ys, mats[0][:, j]) for j in range(k)
        ]
        point = float(np.mean(per_seed))
        n, diffs = len(ys), []
        for _ in range(b):
            idx = rng.integers(0, n, n)
            yb = ys[idx]
            if yb.min() == yb.max():
                continue
            diffs.append(
                np.mean(
                    [
                        roc_auc_score(yb, mats[1][idx, j]) - roc_auc_score(yb, mats[0][idx, j])
                        for j in range(k)
                    ]
                )
            )
        diffs = np.array(diffs)
        p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
        p = max(p, 1 / len(diffs))  # a bootstrap p cannot be smaller than 1 / B
        # Sensitivity analysis with training variance (seed as a random effect): the variance of
        # the mean difference adds the between-seed variance of the per-seed differences / k to the
        # test-patient variance; t with k - 1 degrees of freedom.
        se_pat = float(diffs.std(ddof=1))
        sd_seed = float(np.std(per_seed, ddof=1)) if k > 1 else np.nan
        se_tot = float(np.sqrt(se_pat**2 + sd_seed**2 / k))
        tq = float(student_t.ppf(0.975, k - 1))
        lo_t, hi_t = point - tq * se_tot, point + tq * se_tot
        p_t = float(2 * student_t.sf(abs(point) / se_tot, k - 1))
        rows.append(
            {
                "id": cid,
                "family": family,
                "question": question,
                "site": site,
                "arm_a": pa.split("/")[-2],
                "arm_b": pb.split("/")[-2],
                "seeds": k,
                "diff": point,
                "lower": float(np.quantile(diffs, 0.025)),
                "upper": float(np.quantile(diffs, 0.975)),
                "p_seed_paired": p,
                "sd_seed": sd_seed,
                "lower_tv": lo_t,
                "upper_tv": hi_t,
                "p_tv": p_t,
                "verdict_tv": "B" if lo_t > 0 else ("A" if hi_t < 0 else "n.s."),
                "seeds_favour_b": int(sum(d > 0 for d in per_seed)),
                "per_seed": " ".join(f"{d:+.3f}" for d in per_seed),
            }
        )
        print(f"contrast {cid} {site}: {point:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    conf = df["family"] == "confirmatory"
    p = df.loc[conf, "p_seed_paired"].to_numpy()
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    df["p_holm_paper_a"] = np.nan
    df.loc[conf, "p_holm_paper_a"] = adj
    pt = df.loc[conf, "p_tv"].to_numpy()
    order, adj_t, run = np.argsort(pt), np.empty(len(pt)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(pt) - rank) * pt[i]))
        adj_t[i] = run
    df["p_holm_tv"] = np.nan
    df.loc[conf, "p_holm_tv"] = adj_t
    df.to_csv(out / "contrasts.csv", index=False)


if __name__ == "__main__":
    main()
