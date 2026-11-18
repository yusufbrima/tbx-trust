# MIMIC-IV + MIMIC-CXR development dataset

This guide builds the **20,000-patient MIMIC subset** used in the paper for two purposes: supervised
pretraining of the image encoders on CheXpert findings, and the exploratory MIMIC TB test site.

Every step is a short script in `scripts/mimic/`. Settings live in **one file** you edit once,
so you never need to copy long commands.

## 0. Prerequisites

1. A **credentialed PhysioNet account** that has signed the data use agreements for:
   [MIMIC-IV](https://physionet.org/content/mimiciv/),
   [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/),
   [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) and
   [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) (for the reports).
2. `wget`, `uv`, and about **50 GB** of free disk space: roughly 10 GB of tables and 20–30 GB of
   images for 5k patients.
3. Open `scripts/mimic/env.sh` in an editor and set:
   - `PHYSIONET_USER`: your PhysioNet username
   - `MIMIC_DIR`: where the raw downloads go (default `~/data/physionet`)
   - `TBMXAI_MIMIC_OUT`: where the built subset goes (default `~/data/tbmxai/mimic_subset`)
   - `TBMXAI_OUTPUT_ROOT`: where experiment runs go (default `~/tbmxai_runs`)
   - optionally `MIMIC_N_PATIENTS` (default 20000) and the dataset versions
   - `MIMIC_MAX_SIDE` (default 1024): each CXR is shrunk to this longest side right after it
     is downloaded (chunks of `MIMIC_DOWNLOAD_CHUNK` = 2000 images), so ~82k images take
     ~20 GB instead of ~150 GB. Set it empty to keep the originals.

None of these locations need to be inside the repository.

Optional: store your PhysioNet login once so download scripts don't ask for a password
(saved in `~/.netrc`, readable only by you; delete the `physionet.org` line to revoke):

```bash
bash scripts/mimic/setup_login.sh
```

## 1. Dry run (optional, no credentials, ~1 minute)

```bash
bash scripts/mimic/00_dry_run.sh
```

This generates a *fake* MIMIC tree with the real file layout and runs selection and build on it.

## 2. Download the tables (~10 GB; asks for your PhysioNet password once)

```bash
bash scripts/mimic/01_download_tables.sh
bash scripts/mimic/status.sh          # shows which tables are present
```

| Dataset | Files |
|---|---|
| MIMIC-IV `hosp/` | patients, admissions, diagnoses_icd, labevents, d_labitems, omr, prescriptions, microbiologyevents, transfers, services, procedures_icd |
| MIMIC-IV `icu/` | icustays, chartevents, d_items |
| MIMIC-IV-ED `ed/` | edstays, triage, vitalsign, diagnosis |
| MIMIC-CXR-JPG | mimic-cxr-2.0.0-{metadata, split, chexpert, negbio}.csv.gz |
| MIMIC-CXR | mimic-cxr-reports.zip |

Downloads are resumable: re-run the script if it is interrupted.

## 3. Select the patient subset

`02_select.sh` needs only 8 small files (under ~100 MB): `patients`, `admissions`,
`diagnoses_icd`, ED `edstays` + `diagnosis`, and the CXR-JPG `metadata`, `split` and
`chexpert` CSVs. To run selection while the large tables are still downloading, fetch just
these in a second terminal (both scripts skip files that are already complete):

```bash
bash scripts/mimic/01a_download_select_inputs.sh
```


```bash
bash scripts/mimic/02_select.sh
```

What it does:
- keeps frontal CXRs (PA/AP);
- links each study to the hospital admission or ED stay during which it was taken;
- derives all labels;
- samples patients (seeded): every patient positive for the rare targets (`tb_episode`, `tb`,
  `hiv_pneumonia`)
  is included first, then the rest is filled randomly.

It writes `selection/image_urls.txt`, which lists only the subset's JPGs.

Enrichment raises prevalence compared with the source population. Both prevalences are recorded
in `selection/selection_info.json`.

## 4. Download only the selected images (~20–30 GB)

```bash
bash scripts/mimic/03_download_images.sh
```

## 5. Build tbmxai-ready tables

```bash
bash scripts/mimic/04_build.sh
```

Output in `$TBMXAI_MIMIC_OUT` is a [standard dataset](DATASETS.md): one file per
modality, all linked by `patient_id` + `study_id`. The MIMIC configs use it with
`data: {dataset: ...}`.

| File | Grain | Content |
|---|---|---|
| `dataset.yaml` | – | Label columns and classes, prediction time (`study_time`), image root |
| `labels.parquet` | study | `label_*` targets + subgroup metadata (sex, age, race, insurance, prior HIV, encounter type, official split) |
| `tabular.parquet` | study | EHR features (built, not used in this paper) |
| `images.parquet` | image | `image_path` (relative to the MIMIC-CXR-JPG root), view position, `present` flag |
| `reports.parquet` | study | `report_text` (FINDINGS + IMPRESSION) and `report_full` |
| `feature_dictionary.csv` | feature | Group, type, coverage and description of every feature |
| `build_info.json` | – | Counts, label prevalence, the full build config |

## One patient split for every stage (`dev_split`)

`02_select.sh` writes `selection/dev_split.csv`: a fixed, stratified patient-level split
(70 % train / 10 % val / 20 % test; `selection.dev_split` in `configs/mimic/build.yaml`;
`tbmxai mimic dev-split` creates it for an older selection). The build copies it into
`labels.csv` as `dev_split`, and every MIMIC config uses it
(`split: {name: predefined, params: {column: dev_split}}`). Supervised pretraining trains on
`train`, early-stops on `val` and never sees `test`, so downstream test patients are unseen by
every stage. (MIMIC-CXR's official split is unsuitable here: its validate/test sets hold ~1 %
of patients.) 20,000-patient selection: 14,000 / 2,000 / 4,000 patients.

## Supervised pretraining (MIMIC as the learner)

```bash
bash scripts/mimic/06_pretrain.sh                 # densenet121 and convnext_tiny
```

Trains an ImageNet-initialised backbone on every frontal image of the selection to predict the
14 CheXpert findings (multi-label; uncertain labels masked, unmentioned = negative), on the
shared `dev_split` (train; early stopping on val mean AUROC; test never used). Patients
positive for `tb_episode` are excluded so MIMIC can later serve as an external TB test site.
Needs only the selection and the images (not `04_build`). Output per backbone in
`$TBMXAI_OUTPUT_ROOT/pretrain/<name>/`: `backbone.pt` (use as `checkpoint:` of a `timm_vision`
encoder with `pretrained: false`), `metrics.json` (per-finding validation AUROC),
`history.csv`, `manifest.csv`, `last.pt` (resume with `--resume`).

## The MIMIC TB test site

```bash
uv run python scripts/analysis/mimic_tb_site.py build                       # 65 TB-episode patients + dev_split test patients
uv run python scripts/analysis/mimic_tb_site.py score OUT "LABEL=RUN_GLOB" ...  # score Shenzhen-trained runs
uv run python scripts/analysis/mimic_tb_patient.py OUT                       # one film per patient, rule-out set
```

TB-episode patients are excluded from pretraining, so the site is unseen by the MIMIC-pretrained
encoders; public foundation models trained on MIMIC-CXR are not leakage-free on it.

## How many TB cases MIMIC has (v3.1, CXR-JPG 2.1.0)

| TB case definition | Patients with a frontal CXR |
|---|---|
| `tb`: CXR during a TB-coded encounter | 44 |
| `tb_episode`: ±30 days of TB-coded encounter, *M. tuberculosis* culture or pyrazinamide | 65 |
| Any CXR at any time in a patient ever TB-coded (ceiling; mostly unrelated images) | 77 |

185 patients have an active-TB code, but only 77 have any chest X-ray in MIMIC-CXR. MIMIC is
therefore too small to *train* TB models; use it as an external TB test site and train on
public TB sets.

## Faster image download on Google Cloud

PhysioNet mirrors MIMIC-CXR-JPG in a requester-pays GCS bucket. From a GCP VM this is far
faster than HTTPS (~50 images/min over HTTPS here). One-time setup: add your Google account at
physionet.org/settings/cloud, request GCS access on the MIMIC-CXR-JPG page, and run
`gcloud auth login` with that account. Then:

```bash
GCP_BILLING_PROJECT=my-project bash scripts/mimic/03b_download_images_gcs.sh   # instead of 03_*.sh
```

It copies only the selected images (same folder layout as wget), shrinks each 2,000-image chunk
(`MIMIC_MAX_SIDE`) and is resumable. Transfer is billed to the billing project.

