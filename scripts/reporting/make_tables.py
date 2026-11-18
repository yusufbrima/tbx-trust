"""LaTeX tables for the paper, generated from the experiment outputs (never typed by hand).

Each table is written as a standalone .tex file into OUT_DIR.

Usage: python scripts/reporting/make_tables.py OUT_DIR [NAME ...]   (default: every table)
Writes:
  datasets.tex     sites: role, setting, sampling, capture, reference standard, unit and counts
  decisions.tex    pre-specified hypotheses and exploratory references: pre-specified verdict and the
                   two-level (seed, patient) bootstrap interval
  operating.tex    deployed operating points of single runs (plan threshold): mean and range,
                   sensitivity error, oracle specificity at 90 %, runs meeting the WHO minimum
  programme.tex    referrals, tests per case and missed cases per 1,000 screened at 1 % and 5 %
  tbx11k.tex       TBX11K by image class: TB vs healthy, TB vs sick-but-non-TB, flags by TB box type
  subsets.tex      adults-only Montgomery and active-TB-only TBX11K operating points
  pretraining.tex  MIMIC CheXpert pretraining: validation AUROC per finding, both backbones
  mimic.tex        MIMIC TB site per patient (scripts/analysis/mimic_tb_patient.py)
  lesions.tex      lesion-level Grad-CAM against TBX11K boxes (scripts/analysis/saliency_boxes.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

D = Path("~/data/tbmxai").expanduser()
R = Path("~/tbmxai_runs").expanduser()
A = R / "analysis" / "operating_points"
SITE = {"montgomery": "Montgomery", "tbx11k": "TBX11K", "shenzhen": "Shenzhen"}
ARM = {
    "cxr_convnext_seeds": "ConvNeXt-T, ImageNet",
    "cxr_densenet_seeds": "DenseNet-121, ImageNet",
    "nlm_convnext_mimicpt": "ConvNeXt-T, MIMIC",
    "nlm_convnext_mimicpt_frozen": "ConvNeXt-T, MIMIC, frozen",
    "nlm_densenet_mimicpt": "DenseNet-121, MIMIC",
    "nlm_densenet_mimicpt_frozen": "DenseNet-121, MIMIC, frozen",
    "cxr_convnext_from_shenzhen": "ConvNeXt-T, ImageNet",
    "multisite_convnext_mimicpt_from_shenzhen": "ConvNeXt-T, MIMIC",
    "cxr_convnext_from_tbx11k": "ConvNeXt-T, ImageNet",
    "multisite_convnext_mimicpt_from_tbx11k": "ConvNeXt-T, MIMIC",
    "imagenet": "ConvNeXt-T, ImageNet",  # multisite_retuned/imagenet (TBX11K-only re-tune)
    "mimic": "ConvNeXt-T, MIMIC",  # multisite_retuned/mimic
    "nlm_xrv": "TorchXRayVision",
    "multisite_xrv_from_shenzhen": "TorchXRayVision",
    "nlm_raddino": "RAD-DINO",
    "multisite_raddino_from_shenzhen": "RAD-DINO",
    "nlm_raddino_frozen": "RAD-DINO, frozen",
    "multisite_raddino_frozen_from_shenzhen": "RAD-DINO, frozen",
}


def tex(s: object) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def pct(v: float) -> str:
    return f"{100 * v:.0f}"


def pval(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def datasets_table() -> str:
    nlm = json.loads((D / "nlm_tb" / "build_info.json").read_text())["per_site"]
    tbx = pd.read_csv(D / "tbx11k" / "labels.csv")
    mt = json.loads((D / "mimic_tb_site" / "build_info.json").read_text())
    nj = json.loads((D / "naija_cxr" / "build_info.json").read_text())
    pm = json.loads((R / "pretrain" / "mimic_chexpert_convnext_tiny" / "metrics.json").read_text())[
        "split_counts"
    ]
    rows = [
        (
            "Shenzhen (China)",
            "source; external for TBX11K-trained models",
            "TB hospital, case--control",
            "digital PA",
            "radiologist reading",
            "image",
            nlm["shenzhen"]["tb"] + nlm["shenzhen"]["normal"],
            nlm["shenzhen"]["tb"],
        ),
        (
            "Montgomery (USA)",
            "external",
            "county TB programme, case--control",
            "digital PA",
            "radiologist reading",
            "image",
            nlm["montgomery"]["tb"] + nlm["montgomery"]["normal"],
            nlm["montgomery"]["tb"],
        ),
        (
            "TBX11K (China)",
            "source; external for Shenzhen-trained models",
            "curated: healthy / sick non-TB / TB",
            "digital",
            "microbiology as reported by the dataset's authors (results not released); radiologist TB boxes (active, latent)",
            "image",
            len(tbx),
            int((tbx["tb"] == "tb").sum()),
        ),
        (
            "MIMIC-CXR TB site (USA)",
            "external, exploratory",
            "tertiary hospital, mostly inpatient",
            "digital PA and AP portable",
            "TB episode from codes, culture or treatment",
            f"patient ({mt['patients']:,})",
            mt["images"],
            mt["tb_images"],
        ),
        (
            "Naija-CXR (Nigeria)",
            "external, exploratory",
            "teaching-hospital radiology archive",
            "smartphone photographs of films",
            "radiologist impression",
            "patient",
            nj["patients"],
            nj["n_tb"],
        ),
        (
            "MIMIC-CXR pretraining",
            "pretraining only",
            "tertiary hospital",
            "digital",
            "CheXpert labels from reports",
            f"patient ({pm['train']['patients'] + pm['val']['patients']:,})",
            pm["train"]["images"] + pm["val"]["images"],
            None,
        ),
    ]
    body = "\n".join(
        f"  {tex(n)} & {tex(role)} & {tex(setting)} & {tex(cap)} & {tex(ref)} & {tex(unit)} & \\num{{{img}}} & "
        f"{'--' if tb is None else f'\\num{{{tb}}}'} \\\\"
        for n, role, setting, cap, ref, unit, img, tb in rows
    )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Sites. TB counts are images except for Naija-CXR (patients). Shenzhen, Montgomery and "
        "TBX11K carry no patient identifiers, so one image counts as one patient. MIMIC TB site: 65 patients "
        "with a TB episode (patient-level prevalence 1.6\\%). Pretraining: 50,850 training and 7,302 "
        "validation images; the 4,000 test patients and all TB-episode patients outside the test set were "
        "excluded.}\n\\label{tab:datasets}\n"
        "{\\scriptsize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{p{2.2cm}p{2.2cm}p{2.3cm}p{1.7cm}p{2.4cm}p{1.3cm}rr}\n"
        "\\toprule\nSite & Role & Setting and sampling & Capture & Reference standard & Unit & Images & TB \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def decisions_table() -> str:
    reg = pd.read_csv(R / "analysis" / "comparisons" / "decision_table.csv")
    reg = reg[reg["id"] != "C6"].reset_index(drop=True)
    two = read(A / "contrasts.csv").reset_index(drop=True)
    verdict = {"B better": "B", "A better": "A", "no clear difference": "n.s."}
    lines = []
    for (_, r), (_, t) in zip(reg.iterrows(), two.iterrows(), strict=True):
        assert r["id"] == t["id"] and r["site"] == t["site"]
        train = "TBX11K" if r["id"] == "C5" else "Shenzhen"
        holm = "--" if pd.isna(t["p_holm_tv"]) else pval(t["p_holm_tv"])
        v_tv = t["verdict_tv"] if pd.isna(t["p_holm_tv"]) or t["p_holm_tv"] < 0.05 else "n.s."
        fam = {"confirmatory": "exposed", "supporting": "supp.", "exploratory": "expl."}[r["family"]]
        if r["family"] == "confirmatory" and r["site"] == "tbx11k":
            fam = "indep."  # the only confirmatory site no development decision touched
        lines.append(
            f"  {r['id']} ({fam}) & {train} $\\rightarrow$ {SITE[r['site']]} & {tex(ARM.get(r['arm_a'], r['arm_a']))} & "
            f"{tex(ARM.get(r['arm_b'], r['arm_b']))} & {r['auroc_a']:.3f} & {r['auroc_b']:.3f} & "
            f"{r['diff_b_minus_a']:+.3f} [{r['diff_lower']:+.3f}, {r['diff_upper']:+.3f}] & {verdict[r['decision']]} & "
            f"{t['diff']:+.3f} [{t['lower_tv']:+.3f}, {t['upper_tv']:+.3f}] & {int(t['seeds_favour_b'])}/{int(t['seeds'])} & "
            f"{holm} & {v_tv} \\\\"
        )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Pre-specified hypotheses and exploratory foundation-model references (C7, added after the plan "
        "was written). indep.: confirmatory, on the site no development decision touched; exposed: pre-specified, "
        "on a site that development decisions had seen; supp.: supporting; expl.: exploratory. Pre-specified analysis (primary): seed-ensemble "
        "AUROCs, test-patient bootstrap 95\\% CI and the pre-specified verdict. Training-variance analysis "
        "(sensitivity analysis; Section~\\ref{sec:methods}): mean seed-paired difference, CI with between-seed "
        "variance, seeds favouring B, Holm $p$ within C1, C2, C5, and its own verdict; C1 on TBX11K: ten seed pairs. C5 arms use settings "
        "re-tuned on TBX11K validation only.}\n\\label{tab:decisions}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llllcccccccc}\n\\toprule\n"
        " & & & & \\multicolumn{4}{c}{Pre-specified (ensemble)} & \\multicolumn{4}{c}{Training variance (single model)} \\\\\n"
        "\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}\n"
        "ID & Train $\\rightarrow$ test & Arm A & Arm B & A & B & $\\Delta$ [95\\% CI] & Verdict & "
        "$\\Delta$ [95\\% CI] & Seeds & Holm $p$ & Verdict \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def read(path: Path) -> pd.DataFrame:
    """Read an analysis CSV, mapping the older arm label "(registered)" to "(pre-specified)"."""
    return pd.read_csv(path).replace(r"\(registered\)", "(pre-specified)", regex=True)


def run_summary(o: pd.DataFrame, population: str) -> pd.DataFrame:
    """Per arm x site: ensemble AUROC (ranking) and the single runs' deployed operating points."""
    e = o[(o["unit"] == "ensemble") & (o["population"] == population)]
    r = o[(o["unit"] == "run") & (o["population"] == population)]
    g = r.groupby(["arm", "site"], sort=False)
    s = g.agg(
        runs=("run", "size"),
        sens=("sens", "mean"), sens_min=("sens", "min"), sens_max=("sens", "max"),
        spec=("spec", "mean"), spec_min=("spec", "min"), spec_max=("spec", "max"),
        oracle=("oracle_spec90", "mean"), cost=("spec_cost_of_shift", "mean"),
        run_auroc=("auroc", "mean"),
        who=("who_min", "sum"), who_ci=("who_min_ci", "sum"),
        n_tb=("n_tb", "first"), n_non_tb=("n_non_tb", "first"),
    ).reset_index()  # fmt: skip
    return s.merge(e[["arm", "site", "auroc"]], on=["arm", "site"], how="left")


def operating_table() -> str:
    s = run_summary(read(A / "operating_points.csv"), "all")
    lines = []
    for _, r in s.iterrows():
        bf = (lambda x: f"\\textbf{{{x}}}") if r["who_ci"] > 0 else (lambda x: x)
        lines.append(
            f"  {bf(tex(r['arm']))} & {bf(SITE[r['site']])} & {r['run_auroc']:.3f} & {r['auroc']:.3f} & "
            f"{bf(pct(r['sens']))} ({pct(r['sens_min'])}--{pct(r['sens_max'])}) & {100 * (r['sens'] - 0.9):+.0f} & "
            f"{bf(pct(r['spec']))} ({pct(r['spec_min'])}--{pct(r['spec_max'])}) & {pct(r['oracle'])} & "
            f"{100 * r['cost']:+.0f} & {int(r['who'])}/{int(r['runs'])} & "
            f"{bf(str(int(r['who_ci'])) + '/' + str(int(r['runs'])))} \\\\"
        )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Deployed operating points of single runs, each at its own source threshold (90\\% "
        "sensitivity on its validation split). The single run is the estimand: run AUROC, sensitivity and "
        "specificity are means over the same runs (range, \\%); the ensemble AUROC is shown for reference and is not "
        "a deployed model. Sens.\\ error: delivered minus intended sensitivity; oracle: best specificity at "
        "90\\% sensitivity on the site's ROC; spec.\\ cost: oracle minus delivered specificity (points). "
        "Runs meeting the WHO minimum (90\\%/70\\%) on the point estimate and on both exact 95\\% lower "
        "bounds. Bold: the only arm and site with runs meeting the target on both bounds.}\n"
        "\\label{tab:operating}\n{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}"
        "\\begin{tabular}{llccccccccc}\n\\toprule\n"
        " & & \\multicolumn{2}{c}{AUROC} & & & & & & \\multicolumn{2}{c}{Runs meeting} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){10-11}\n"
        "Model & Site & Run & Ens. & Sensitivity & Sens.\\ error & Specificity & Oracle spec. & Spec.\\ cost & "
        "Point & Bounds \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def programme_table() -> str:
    p = read(A / "programme.csv")
    p = p[p["prevalence"].isin([0.01, 0.05]) & p["site"].isin(["montgomery", "tbx11k"])]
    num = ["referrals_per_1000", "tests_per_case", "missed_per_1000", "cartridge_usd_per_case"]
    m = p.groupby(["arm", "site", "prevalence"], sort=False)[num].mean().reset_index()
    lines = []
    for (arm, site), g in m.groupby(["arm", "site"], sort=False):
        a, b = g[g["prevalence"] == 0.01].iloc[0], g[g["prevalence"] == 0.05].iloc[0]
        lines.append(
            f"  {tex(arm)} & {SITE[site]} & {a['referrals_per_1000']:.0f} & {a['tests_per_case']:.0f} & "
            f"{a['cartridge_usd_per_case']:.0f} & "
            f"{b['referrals_per_1000']:.0f} & {b['tests_per_case']:.1f} & {b['missed_per_1000']:.1f} & "
            f"{b['cartridge_usd_per_case']:.0f} \\\\"
        )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{What the deployed operating points of Table~\\ref{tab:operating} would mean for a programme, per "
        "1,000 people screened at 1\\% (community screening) and 5\\% (facility triage) TB prevalence: Xpert "
        "referrals, tests per TB case detected, cases missed (5\\%), and cartridge cost per case detected at each prevalence (US\\$7.97 "
        "per cartridge). Every quantity is computed for each run and then averaged over runs, so a mean of ratios such as tests per case "
        "is not the ratio of the mean referrals to the mean cases detected. Sensitivity and specificity are carried from "
        "case--control sites without a verifiable bacteriological reference, so these are illustrations, not programme estimates.}\n"
        "\\label{tab:programme}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{llccccccc}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{1\\% prevalence} & \\multicolumn{4}{c}{5\\% prevalence} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\n"
        "Model & Site & Referrals & Tests/case & US\\$/case & Referrals & Tests/case & Missed & US\\$/case \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def tbx11k_table() -> str:
    t = read(A / "tbx11k_classes.csv")
    t = t.groupby("arm", sort=False).mean(numeric_only=True).reset_index()  # mean over runs
    lines = [
        f"  {tex(r['arm'])} & {r['auroc_tb_vs_healthy']:.3f} & {r['auroc_tb_vs_sick_non_tb']:.3f} & "
        f"{pct(r['flagged_healthy'])} & {pct(r['flagged_sick_non_tb'])} & {pct(r['flagged_tb_active'])} & "
        f"{pct(r['flagged_tb_obsolete'])} \\\\"
        for _, r in t.iterrows()
    ]
    n_act, n_obs = int(t["n_tb_active"].iloc[0]), int(t["n_tb_obsolete"].iloc[0])
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{Shenzhen-trained models on TBX11K by image class (means over single runs): AUROC of TB against "
        f"healthy and against sick-but-non-TB images, and the share flagged (\\%) at the deployed threshold, for healthy (3,800), "
        f"sick-but-non-TB (3,662), active TB ({n_act}) and latent TB only ({n_obs}) images; the one remaining TB image has no lesion box and "
        f"counts only as TB.}}\n"
        "\\label{tab:tbx11k}\n{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{AUROC, TB vs} & \\multicolumn{4}{c}{Flagged as TB (\\%)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
        "Model & healthy & sick non-TB & Healthy & Sick non-TB & Active TB & Latent TB \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table}\n"
    )


def subsets_table() -> str:
    o = read(A / "operating_points.csv")
    parts = [run_summary(o, pop).assign(population=pop) for pop in ("adults", "active TB")]
    s = pd.concat(parts, ignore_index=True)
    lines = [
        f"  {tex(r['arm'])} & {SITE[r['site']]} & {tex(r['population'])} & {int(r['n_tb'])} & {int(r['n_non_tb'])} & "
        f"{r['auroc']:.3f} & {pct(r['sens'])} ({pct(r['sens_min'])}--{pct(r['sens_max'])}) & "
        f"{pct(r['spec'])} ({pct(r['spec_min'])}--{pct(r['spec_max'])}) & {pct(r['oracle'])} & "
        f"{int(r['who'])}/{int(r['runs'])} & {int(r['who_ci'])}/{int(r['runs'])} \\\\"
        for _, r in s.iterrows()
    ]
    return (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{Sensitivity analyses of Table~\\ref{tab:operating}: Montgomery without people younger than 18 "
        "(WHO's CAD recommendation covers people aged 15 or older; ages are recorded in bands), and TBX11K with TB "
        "positives restricted to images that carry an active-TB lesion box. Single runs as in "
        "Table~\\ref{tab:operating}: mean (range) in \\%, and runs meeting the WHO minimum on the point estimate "
        "and on both lower bounds.}\n\\label{tab:subsets}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{lllcccccccc}\n\\toprule\n"
        "Model & Site & Population & TB & Non-TB & AUROC & Sensitivity & Specificity & Oracle spec. & "
        "Point & Bounds \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def pretraining_table() -> str:
    m = {
        b: json.loads((R / "pretrain" / f"mimic_chexpert_{b}" / "metrics.json").read_text())
        for b in ("densenet121", "convnext_tiny")
    }
    d, c = m["densenet121"], m["convnext_tiny"]
    lines = [
        f"  {f} & {100 * d['train_prevalence'][f]:.1f} & {d['best_val_auroc'][f]:.3f} & "
        f"{c['best_val_auroc'][f]:.3f} \\\\"
        for f in d["findings"]
    ]
    lines.append(
        f"  \\midrule\n  Mean & & {d['best_val_mean_auroc']:.3f} & {c['best_val_mean_auroc']:.3f} \\\\"
    )
    n = d["split_counts"]
    return (
        "\\begin{table}[h]\n\\centering\n"
        f"\\caption{{MIMIC-CXR pretraining: validation AUROC per CheXpert finding ({n['train']['images']:,} "
        f"training images from {n['train']['patients']:,} patients, {n['val']['images']:,} validation images from "
        f"{n['val']['patients']:,} patients; {d['image']['size']}-pixel input). Prevalence: training share "
        "with the finding positive (uncertain labels ignored).}\n\\label{tab:pretraining}\n"
        "{\\footnotesize\\begin{tabular}{lccc}\n\\toprule\n"
        "Finding & Prevalence (\\%) & DenseNet-121 & ConvNeXt-T \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def mimic_table() -> str:
    m = R / "analysis" / "mimic_tb_patient"
    s = pd.read_csv(m / "summary.csv")
    info = json.loads((m / "cohort.json").read_text())
    f = pd.read_csv(m / "findings.csv")
    nofind = f[f["finding"] == "No Finding"].groupby("arm")["flag_rate"].mean()

    def rng(r: pd.Series, k: str) -> str:
        return f"{pct(r[k + '_mean'])} ({pct(r[k + '_min'])}--{pct(r[k + '_max'])})"

    lines = [
        f"  {tex(r['arm'])}{'$^\\dagger$' if r['leaky'] else ''} & {r['auroc_ens_pulm_neg']:.3f} "
        f"[{r['lo']:.2f}, {r['hi']:.2f}] & {r['auroc_pulm_ruleout_mean']:.3f} & {rng(r, 'sens_pulm')} & "
        f"{rng(r, 'flag_neg')} & {rng(r, 'flag_ruleout')} & {pct(nofind[r['arm']])} \\\\"
        for _, r in s.sort_values(["leaky", "arm"]).iterrows()
    ]
    return (
        "\\begin{table*}[t]\n\\centering\n"
        f"\\caption{{MIMIC TB site, one film per patient: {info['tb_patients_pulmonary']} patients with pulmonary TB "
        f"(first film no later than 7 days after treatment start) against {info['negative_patients']:,} patients "
        f"without TB, and {info['ruleout_patients']} TB rule-out patients (AFB smear or culture ordered, all "
        "negative). AUROC: seed ensemble with a stratified bootstrap 95\\% CI; rule-out: mean single-run AUROC "
        "against the rule-out patients. Rates (\\%): single runs at their own plan threshold, mean (range over "
        "seeds). No finding: share flagged among negative films the report labels as showing no finding. "
        "$^\\dagger$Pretrained on MIMIC-CXR, which includes these patients (not leakage-free).}\n"
        "\\label{tab:mimic}\n{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Model & AUROC & Rule-out AUROC & Sensitivity & Flagged, no TB & Flagged, rule-out & No finding \\\\"
        "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def saliency_boxes_table() -> str:
    t = pd.read_csv(R / "analysis" / "saliency_boxes" / "summary.csv")
    t = t[t["kind"].isin(["active", "obsolete"])]
    w = t.pivot_table(index="model", columns="kind", values=["box_ratio", "pointing", "flagged"])
    n = t.groupby("kind")["images"].first()
    lines = [
        f"  {tex(m)} & {r[('box_ratio', 'active')]:.1f} & {r[('box_ratio', 'obsolete')]:.1f} & "
        f"{pct(r[('pointing', 'active')])} & {pct(r[('pointing', 'obsolete')])} & "
        f"{pct(r[('flagged', 'active')])} & {pct(r[('flagged', 'obsolete')])} \\\\"
        for m, r in w.iterrows()
    ]
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Lesion-level Grad-CAM of Shenzhen-trained models on TBX11K's TB images, against the "
        f"radiologists' boxes ({n['active']} images with an active lesion, {n['obsolete']} with latent lesions "
        "only; means over runs). Ratio: share of saliency inside the boxes divided by the boxes' share of the "
        "image (1 = no preference). Pointing: saliency peak inside a box (16-pixel margin), \\%. Flagged: share "
        "flagged at the deployed threshold, \\%.}\n\\label{tab:lesions}\n"
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Ratio} & \\multicolumn{2}{c}{Pointing} & \\multicolumn{2}{c}{Flagged} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Model & active & latent & active & latent & active & latent \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    for name, fn in (
        ("datasets", datasets_table),
        ("decisions", decisions_table),
        ("operating", operating_table),
        ("programme", programme_table),
        ("tbx11k", tbx11k_table),
        ("subsets", subsets_table),
        ("pretraining", pretraining_table),
        ("mimic", mimic_table),
        ("lesions", saliency_boxes_table),
    ):
        if not only or name in only:
            (out / f"{name}.tex").write_text(fn())
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
