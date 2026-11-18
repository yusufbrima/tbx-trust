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
  shared.tex       shared-split ensembles on the internal and external test sets
  recalchecks.tex  checks on the threshold recalibration simulation
  noshift.tex      what a model with no site shift would deliver (scripts/analysis/portability_reference.py)
  sick.tex         TBX11K: active TB against sick-but-non-TB images, per arm
  symptomatic.tex  programme scenario with symptomatic negatives
  rankrules.tex    sample sizes for a sensitivity assurance of 90 %, by order-statistic rule
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

D = Path("~/data/tbmxai").expanduser()
R = Path("~/tbmxai_runs").expanduser()
A = R / "analysis" / "operating_points"
REF = R / "analysis" / "portability_reference"
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
            "outpatient clinics of a TB hospital, case--control",
            "digital, frontal (PA; paediatric films AP)",
            "clinical reading released with the images",
            "image",
            nlm["shenzhen"]["tb"] + nlm["shenzhen"]["normal"],
            nlm["shenzhen"]["tb"],
        ),
        (
            "Montgomery (USA)",
            "external",
            "county TB screening programme, case--control",
            "digital PA",
            "clinical reading released with the images",
            "image",
            nlm["montgomery"]["tb"] + nlm["montgomery"]["normal"],
            nlm["montgomery"]["tb"],
        ),
        (
            "TBX11K (China)",
            "source; external for Shenzhen-trained models",
            "curated: healthy / sick non-TB / TB",
            "digital",
            "microbiology as reported by the dataset's authors (results not released; it cannot confirm inactive disease); radiologist TB boxes (active, inactive)",
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
        "with a TB episode (patient-level prevalence 1.6\\%; 0.9\\% for the 37 with pulmonary TB in the primary analysis). Pretraining: 50,850 training and 7,302 "
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
        "was written). indep.: confirmatory, on the site that informed no hyperparameter or architecture choice for these arms; exposed: pre-specified, "
        "on a site that development decisions had seen; supp.: supporting; expl.: exploratory. Pre-specified analysis (primary): seed-ensemble "
        "AUROCs, test-patient bootstrap 95\\% CI and the pre-specified verdict. Training-variance analysis "
        "(sensitivity analysis; Section~\\ref{sec:methods}): mean seed-paired difference, CI with between-seed "
        "variance (Welch--Satterthwaite $t$), seeds favouring B and Holm $p$ over the five C1, C2 and C5 contrasts. Its verdict uses that Holm $p$ for those contrasts and the interval for the others. C1 on TBX11K: ten seed pairs. C5 arms use settings "
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
    ns = read(REF / "arm_no_shift.csv")
    out_of_range = {(r["arm"], r["site"]): bool(r["outside"]) for _, r in ns.iterrows()}
    lines = []
    for _, r in s.iterrows():
        mark = "$^\\dagger$" if out_of_range[(r["arm"], r["site"])] else ""
        lines.append(
            f"  {tex(r['arm'])} & {SITE[r['site']]} & {r['run_auroc']:.3f} & {r['auroc']:.3f} & "
            f"{pct(r['sens'])}{mark} ({pct(r['sens_min'])}--{pct(r['sens_max'])}) & {100 * (r['sens'] - 0.9):+.0f} & "
            f"{pct(r['spec'])} ({pct(r['spec_min'])}--{pct(r['spec_max'])}) & {pct(r['oracle'])} & "
            f"{100 * r['cost']:+.0f} & {int(r['who'])}/{int(r['runs'])} & {int(r['who_ci'])}/{int(r['runs'])} \\\\"
        )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Deployed operating points of single runs, each at its own source threshold (90\\% "
        "sensitivity on its validation split). The single run is the estimand: run AUROC, sensitivity and "
        "specificity are means over the same runs (range, \\%); the ensemble AUROC is shown for reference and is not "
        "a deployed model. Sens.\\ error: delivered minus intended sensitivity; oracle: best specificity at "
        "90\\% sensitivity on the site's ROC; spec.\\ cost: oracle minus delivered specificity (points). "
        "Runs meeting the WHO minimum (90\\%/70\\%) on the point estimate and on both exact 95\\% lower "
        "bounds. $^\\dagger$Mean sensitivity outside the range expected with no site shift (\\tableref{tab:noshift}). Montgomery and "
        "TBX11K rows come from separate training runs. Runs of an arm share seed-specific splits with other arms. The re-tuned "
        "RAD-DINO arm was added after the pre-specified arm had been scored externally.}\n"
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
    p = p[p["prevalence"].isin([0.01, 0.10]) & p["site"].isin(["montgomery", "tbx11k"])]
    num = ["referrals_per_1000", "tests_per_case", "missed_per_1000", "cartridge_usd_per_case"]
    m = p.groupby(["arm", "site", "prevalence"], sort=False)[num].mean().reset_index()

    def cells(r: pd.Series) -> str:
        return (
            f"{r['referrals_per_1000']:.0f} & {r['tests_per_case']:.1f} & {r['missed_per_1000']:.1f} & "
            f"{r['cartridge_usd_per_case']:.0f}"
        )

    lines = []
    for (arm, site), g in m.groupby(["arm", "site"], sort=False):
        a, b = g[g["prevalence"] == 0.01].iloc[0], g[g["prevalence"] == 0.10].iloc[0]
        lines.append(f"  {tex(arm)} & {SITE[site]} & {cells(a)} & {cells(b)} \\\\")
    lines.append(
        f"  \\midrule\n  Test everyone & -- & 1000 & 100.0 & 0.0 & {100 * 7.97:.0f} & 1000 & 10.0 & 0.0 & {10 * 7.97:.0f} \\\\"
    )
    return (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{What the deployed operating points of Table~\\ref{tab:operating} would mean for a programme, per "
        "1,000 people at 1\\% (community screening) and 10\\% (facility triage) TB prevalence: Xpert referrals, tests "
        "per TB case detected, cases missed, and cartridge cost per case detected (US\\$7.97 per cartridge; radiography, "
        "staffing and specimen transport are excluded). Every quantity is computed for each run and then averaged over "
        "runs, so a mean of ratios such as tests per case is not the ratio of the mean referrals to the mean cases "
        "detected. Montgomery's negatives are normal films and TBX11K's are half healthy, so referrals are understated "
        "for symptomatic people (\\tableref{tab:symptomatic}). These are illustrations, not programme estimates.}\n"
        "\\label{tab:programme}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{llcccccccc}\n\\toprule\n"
        " & & \\multicolumn{4}{c}{1\\% prevalence} & \\multicolumn{4}{c}{10\\% prevalence} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
        "Model & Site & Referrals & Tests/case & Missed & US\\$/case & Referrals & Tests/case & Missed & US\\$/case \\\\\n\\midrule\n"
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
        f"sick-but-non-TB (3,662), active TB ({n_act}) and inactive TB only ({n_obs}) images. Inactive TB is the class TBX11K labels latent: old or "
        f"healed disease on the radiograph. The one remaining TB image has no lesion box and counts only as TB.}}\n"
        "\\label{tab:tbx11k}\n{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}"
        "\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{AUROC, TB vs} & \\multicolumn{4}{c}{Flagged as TB (\\%)} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\n"
        "Model & healthy & sick non-TB & Healthy & Sick non-TB & Active TB & Inactive TB \\\\\n\\midrule\n"
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
        "Model & Site & Population & TB & Non-TB & AUROC (ens.) & Sensitivity & Specificity & Oracle spec. & "
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
        f"(first episode film, no later than 7 days after treatment start in the {info['tb_patients_with_treatment_start']} of "
        f"{info['tb_patients_with_eligible_film']} TB patients with a recorded start) against {info['negative_patients']:,} patients "
        f"without TB, and {info['ruleout_patients']} AFB-tested patients with no \\emph{{M.~tuberculosis}} isolated (they may include "
        "other mycobacteria; ethambutol, used to date treatment, is also given for other mycobacteria). AUROC: seed ensemble with a stratified bootstrap 95\\% CI; rule-out: mean single-run AUROC "
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
        f"radiologists' boxes ({n['active']} images with an active lesion, {n['obsolete']} with inactive lesions "
        "only; means over runs). Ratio: share of saliency inside the boxes divided by the boxes' share of the "
        "whole image. Boxes lie in the lungs, so a map spread evenly over the lungs would score about 3.9 if the lungs "
        "cover a quarter of the image, as on Montgomery; the ratio therefore does not show lesion localisation. Pointing: saliency peak inside a box (16-pixel margin), \\%. Flagged: share "
        "flagged at the deployed threshold, \\%.}\n\\label{tab:lesions}\n"
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lcccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Ratio} & \\multicolumn{2}{c}{Pointing} & \\multicolumn{2}{c}{Flagged} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Model & active & inactive & active & inactive & active & inactive \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def shared_table() -> str:
    s = pd.read_csv(R / "analysis" / "shared_split" / "summary.csv")
    sites = (("shenzhen (internal)", "Shenzhen (internal)"), ("montgomery", "Montgomery"), ("tbx11k", "TBX11K"))
    lines = []
    for arm in s["arm"].unique():
        for key, name in sites:
            x = s[(s["arm"] == arm) & (s["site"].str.lower() == key)].iloc[0]
            lines.append(
                f"  {tex(arm)} & {name} & {x['auroc']:.3f} & {pct(x['sens'])} [{pct(x['sens_lo'])}] & "
                f"{pct(x['spec'])} [{pct(x['spec_lo'])}] & {pct(x['oracle_spec90'])} & "
                f"{'yes' if x['who_point'] else 'no'} & {'yes' if x['who_bounds'] else 'no'} \\\\"
            )
    return (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{Shared-split sensitivity analysis: six arms retrained with five seeds on one Shenzhen train / validation / "
        "internal-test split (68 TB, 65 non-TB internal test images), so the seed ensemble has genuine validation scores. "
        "Ensemble threshold: 90\\% sensitivity on the averaged validation scores, applied unchanged. Sensitivity and "
        "specificity in \\% [exact 95\\% lower bound]; oracle: best specificity at 90\\% sensitivity on the site's ROC; WHO "
        "minimum on the point estimate and on both lower bounds. The internal test images may have informed the "
        "hyperparameter searches, which used another split of Shenzhen.}\n\\label{tab:shared}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{llcccccc}\n\\toprule\n"
        "Ensemble & Test set & AUROC & Sensitivity & Specificity & Oracle spec. & WHO (point) & WHO (bounds) \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def recalchecks_table() -> str:
    d = R / "analysis" / "recalibration_checks"
    m, th = pd.read_csv(d / "misspecification.csv"), pd.read_csv(d / "theory.csv").set_index("k_cases")
    modes = ("kde_x1", "empirical", "held-out", "kde_x0.5", "kde_x2")
    lines = []
    for k in (20, 40, 80):
        v = [m[(m["k_cases"] == k) & (m["mode"] == c)]["p_sens90"].mean() for c in modes]
        lines.append(
            f"  P(sens.\\ $\\geq$ 90\\%) & {k} & {th.loc[k, 'p_sens90_lower']:.2f}--{th.loc[k, 'p_sens90_upper']:.2f} & "
            + " & ".join(f"{x:.2f}" for x in v) + " \\\\"
        )
    for name, label in (("RAD-DINO, retuned", "RAD-DINO re-tuned"), ("RAD-DINO", "RAD-DINO pre-spec.")):
        for k in (20, 80):
            v = [m[(m["model"] == name) & (m["k_cases"] == k) & (m["mode"] == c)]["p_who"].mean() for c in modes]
            lines.append(f"  P(WHO), {label} & {k} & -- & " + " & ".join(f"{x:.2f}" for x in v) + " \\\\")
    return (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{Checks on the threshold recalibration simulation (TBX11K, 5th-percentile rule, 1{,}000 draws per run, "
        "averaged over runs; Monte Carlo error about $\\pm$0.01, so 0.92 here and 0.91 in the main 2{,}000-draw analysis agree). "
        "Exact: the order-statistic bounds, valid for any continuous score distribution, so these rows largely confirm a "
        "theorem; P(WHO), which depends on the score distribution, is the informative check. Density: the kernel density of "
        "the main analysis; empirical: calibration cases resampled from the site's TB scores; held-out: calibrate on $k$ TB "
        "images and judge on the rest; half and double: the kernel bandwidth. P(sens.\\ $\\geq$ 90\\%) is averaged over all "
        "models; P(WHO) is shown for the two fine-tuned RAD-DINO arms, the models that reach it most often.}\n"
        "\\label{tab:recalchecks}\n{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{lcccccccc}\n\\toprule\n"
        "Quantity & $k$ & Exact & Density & Empirical & Held-out & Half bw & Double bw \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )


def noshift_table() -> str:
    n = pd.read_csv(REF / "no_shift.csv")
    a = read(REF / "arm_no_shift.csv")
    lines = []
    for _, r in n.iterrows():
        g = a[a["site"] == r["site"]]
        train = "TBX11K" if r["site"] == "shenzhen" else "Shenzhen"
        lines.append(
            f"  {train} $\\rightarrow$ {SITE[r['site']]} & {int(r['val_tb'])} & {int(r['n_tb'])} / {int(r['n_non_tb']):,} & "
            f"{pct(r['sens_lo'])}--{pct(r['sens_hi'])} & {pct(r['p_point'])} & {pct(r['p_bounds'])} & "
            f"{int(g['outside'].sum())} of {len(g)} \\\\"
        )
    return (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{What a model with no site shift would deliver. The threshold is the score that reaches 90\\% sensitivity on "
        "the validation TB cases, so its population sensitivity follows the order-statistic law of \\sectionref{sec:methods}; test "
        "counts are binomial, and true specificity is set to 90\\%. Range: central 95\\% of delivered sensitivity for one run. "
        "Meets WHO: chance that such a run meets the minimum on the point estimate and on both exact lower bounds. Last column: "
        "arms whose mean delivered sensitivity lies outside the corresponding range for the mean of their runs "
        "(marked in Table~\\ref{tab:operating}).}\n\\label{tab:noshift}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{lcccccc}\n\\toprule\n"
        "Train $\\rightarrow$ test & Validation TB & Test TB / non-TB & Sensitivity range (\\%) & Meets WHO, point (\\%) & "
        "Meets WHO, bounds (\\%) & Arms outside \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def sick_table() -> str:
    d = read(REF / "sick_negatives.csv")
    g = d.groupby("arm", sort=False)
    lines = [
        f"  {tex(arm)} & {pct(x['sens'].mean())} ({pct(x['sens'].min())}--{pct(x['sens'].max())}) & "
        f"{pct(x['spec'].mean())} ({pct(x['spec'].min())}--{pct(x['spec'].max())}) & "
        f"{int(x['meets_point'].sum())}/{len(x)} & {int(x['meets_bounds'].sum())}/{len(x)} \\\\"
        for arm, x in g
    ]
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Shenzhen-trained models on TBX11K at the deployed threshold, with active TB (660 images) as the positives and "
        "sick-but-non-TB images (3,662) as the negatives: mean (range) over runs in \\%, and runs meeting the WHO minimum on the "
        "point estimate and on both exact lower bounds.}\n\\label{tab:sick}\n"
        "{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{lcccc}\n\\toprule\n"
        "Model & Sensitivity & Specificity & Point & Bounds \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table}\n"
    )


def symptomatic_table() -> str:
    d = read(REF / "symptomatic.csv")
    m = d.groupby(["arm", "prevalence"], sort=False)[["referrals", "tests_per_case", "missed", "saved"]].mean().reset_index()
    lines = []
    for arm, g in m.groupby("arm", sort=False):
        a, b = g[g["prevalence"] == 0.10].iloc[0], g[g["prevalence"] == 0.20].iloc[0]
        lines.append(
            f"  {tex(arm)} & {a['referrals']:.0f} & {a['tests_per_case']:.1f} & {a['missed']:.1f} & {pct(a['saved'])} & "
            f"{b['referrals']:.0f} & {b['tests_per_case']:.1f} & {b['missed']:.1f} & {pct(b['saved'])} \\\\"
        )
    lines.append("  \\midrule\n  Test everyone & 1000 & 10.0 & 0.0 & 0 & 1000 & 5.0 & 0.0 & 0 \\\\")
    return (
        "\\begin{table*}[h]\n\\centering\n"
        "\\caption{Symptomatic triage scenario on TBX11K, per 1,000 people tested: sensitivity from active TB images and "
        "specificity from sick-but-non-TB and inactive TB images, means over runs. Saved: cartridges saved against testing "
        "everyone. Missed cases exclude Xpert's own false negatives. An illustration from case--control data.}\n"
        "\\label{tab:symptomatic}\n{\\resizebox{\\linewidth}{!}{\\footnotesize\\begin{tabular}{lcccccccc}\n\\toprule\n"
        " & \\multicolumn{4}{c}{10\\% prevalence} & \\multicolumn{4}{c}{20\\% prevalence} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "Model & Referrals & Tests/case & Missed & Saved (\\%) & Referrals & Tests/case & Missed & Saved (\\%) \\\\\n\\midrule\n"
        + "\n".join(lines)
        + "\n\\bottomrule\n\\end{tabular}}}\n\\end{table*}\n"
    )


def rankrules_table() -> str:
    d = pd.read_csv(REF / "rank_rules.csv")
    lines = [
        f"  {int(r['rank'])} & {int(r['k'])} & {r['assurance']:.3f} & {int(r['tested_at_20pct']):,} & "
        f"{int(r['tested_at_10pct']):,} & {int(r['tested_at_5pct']):,} \\\\"
        for _, r in d.iterrows()
    ]
    return (
        "\\begin{table}[h]\n\\centering\n"
        "\\caption{Smallest number $k$ of exchangeable confirmed TB cases that gives at least a 90\\% chance of reaching 90\\% "
        "population sensitivity when the threshold is the $m$-th lowest local score, and the consecutive people tested to find "
        "them at 20\\%, 10\\% and 5\\% prevalence. A lower rank needs fewer cases but sets a lower threshold, so it costs "
        "specificity.}\n\\label{tab:rankrules}\n{\\footnotesize\\begin{tabular}{cccccc}\n\\toprule\n"
        "$m$ & $k$ & Assurance & 20\\% & 10\\% & 5\\% \\\\\n\\midrule\n"
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
        ("shared", shared_table),
        ("recalchecks", recalchecks_table),
        ("noshift", noshift_table),
        ("sick", sick_table),
        ("symptomatic", symptomatic_table),
        ("rankrules", rankrules_table),
    ):
        if not only or name in only:
            (out / f"{name}.tex").write_text(fn())
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
