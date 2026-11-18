# Bringing your own data: the standard dataset format

Any dataset can be used in tbmxai **without writing code**. Build one table per modality,
all linkable by a key, and add a short `dataset.yaml`. Dataset adapters, such as the MIMIC
builder, write this same format.

- [Layout](#layout)
- [Keys and linking](#keys-and-linking)
- [What goes in each file](#what-goes-in-each-file)
- [dataset.yaml](#datasetyaml)
- [Image folders without a table](#image-folders-without-a-table)
- [Recipes: raw data → standard dataset without code](#recipes-raw-data--standard-dataset-without-code)
- [Pooling datasets (multi-site)](#pooling-datasets-multi-site)
- [Datasets without patient IDs](#datasets-without-patient-ids)
- [Validate before use](#validate-before-use)
- [Using a dataset in an experiment](#using-a-dataset-in-an-experiment)
- [Avoiding leakage: your responsibility](#avoiding-leakage-your-responsibility)

## Layout

```text
my_dataset/
├── dataset.yaml   # name, label columns and their classes
├── labels.csv     # patient_id, [study_id], label column(s), metadata (sex, age, site, …)
├── tabular.csv    # patient_id, [study_id], EHR / demographic features        (optional)
├── images.csv     # patient_id, [study_id], image_path, [image metadata]      (optional)
└── reports.csv    # patient_id, [study_id], report_text                       (optional)
```

- The file names are fixed. `labels.csv` is required, plus at least one modality file.
- Any combination of modalities works: EHR only, images only, reports only, or all three.
- Each file may be `.csv`, or `.parquet` for large tables (e.g. `tabular.parquet`).
- Image files can live anywhere (PNG, JPEG, TIFF or DICOM). `image_path` is relative to the
  dataset folder, or to `image_root` in `dataset.yaml`, or absolute.

Start from a template that has the right headers:

```bash
tbmxai dataset init my_dataset            # --modality tabular|images|reports to choose
```

## Keys and linking

Two key columns, with fixed names, link the files:

| Column | Required | Meaning |
|---|---|---|
| `patient_id` | in every file | The person. It is the unit of splitting: a patient is never in two splits. |
| `study_id` | optional | A finer key: study, visit or encounter. Use it when labels or features differ per study. |

Two files are joined on `patient_id` + `study_id` when both have `study_id`, otherwise on
`patient_id` alone. For example:

- **One row per patient everywhere:** use `patient_id` only.
- **Several X-rays per patient, one label per patient:** `images.csv` has several rows per
  `patient_id`; each image becomes a sample with the patient's label and EHR.
- **Several visits per patient, a label per visit:** put `study_id` in `labels.csv` and in
  the files that are per visit (`images.csv`, `tabular.csv`). A file without `study_id`
  (e.g. patient-level demographics) applies to all of the patient's visits.

Each row of the finest file you load (usually `images.csv`) becomes one **sample**. The
validator rejects links that would multiply rows (both files having several rows per key).

## What goes in each file

A column's role is decided by the file it is in. There are no lists of columns to maintain.

| File | Columns besides the keys |
|---|---|
| `labels.csv` | The label columns declared in `dataset.yaml`. **Every other column is metadata**: available for subgroups, splits and the prediction time, never a model input. |
| `tabular.csv` | **Every column is a model input.** Leave out anything that must not be an input. Optional `feature_time`: when the features were recorded (checked against the prediction time; not an input). |
| `images.csv` | `image_path`. Other columns (e.g. `view`) are image metadata. |
| `reports.csv` | `report_text`. Other columns are ignored. |

- A column can be in both `tabular.csv` and `labels.csv`, e.g. `age`: it is loaded once and
  used both as an input and for subgroups.
- Label values may be text or numbers. Integer values read back as floats (`1.0`) still
  match a class `'1'`.
- Label columns may appear only in `labels.csv`; the validator reports an error otherwise.

## dataset.yaml

```yaml
format_version: '2.0'
name: my_tb_cohort
description: Adults screened for TB at site X, 2019–2023
labels:                            # label columns of labels.csv
  tb_status: {classes: [negative, positive], description: Culture-confirmed TB}
  hiv:       {classes: ['0', '1']}
time_column: visit_date            # optional: prediction time in labels.csv
patient_ids: provided              # 'assumed' when patient_id is not a real patient (see below)
image_root: /data/cxr              # optional: base for relative image_path (default: this folder)
```

- **Classes** are listed in encoding order, with the positive class **last** for binary
  targets.
- `time_column` enables temporal splits, index-encounter selection and the `feature_time`
  check.
- `~` and `${ENV_VAR}` are expanded in `image_root`.

## Image folders without a table

If your images are just files in folders, `index-images` writes `images.csv` for you. It
takes the patient ID from wherever the collection keeps it:

```bash
tbmxai dataset index-images /raw/shenzhen my_dataset --patient-id-from filename \
    --id-pattern '(?P<id>CHNCXR_\d+)'
```

| `--patient-id-from` | Patient ID comes from | Extra options | Example |
|---|---|---|---|
| `filename` | Regex on the file name (without extension) | `--id-pattern` | `CHNCXR_0001_0.png` → `CHNCXR_0001` |
| `path` | Regex on the path relative to the folder | `--id-pattern` | `p10/p10000032/s5041/x.jpg` with `p(?P<id>\d{8})/` |
| `dicom` | DICOM `PatientID`; `StudyInstanceUID` becomes `study_id` | none | Hospital PACS exports |
| `table` | A lookup table mapping image file → patient | `--lookup`, `--lookup-image-column`, `--lookup-patient-column` | NIH ChestX-ray14, TBX11K |
| `none` | Nothing: each image is its own patient | requires `patient_ids: assumed` | Folder-organised Kaggle collections |

More options:
- `--study-pattern`: regex on the relative path giving `study_id`.
- `--label-from-folder LABEL`: the image's parent folder (`Normal/`, `Tuberculosis/`) is the
  label. `labels.csv` is written too, one row per image, and `dataset.yaml` if it does not
  exist yet.
- `--overwrite`: replace existing files.

The result is an ordinary `images.csv`; you can edit it or add columns. Any other source
format (a hospital export, a spreadsheet) is converted the same way: a short script that
writes the CSVs.

## Recipes: raw data → standard dataset without code

When a dataset ships as image folders, side-car text files and tables, describe the
conversion once in a **recipe** and build it with one command. The same command runs a
registered Python adapter when a recipe is not enough (MIMIC):

```bash
tbmxai dataset build configs/datasets/nlm_tb.yaml ~/data/nlm_tb ~/data/tbmxai/nlm_tb
tbmxai dataset build configs/datasets/tbx11k.yaml ~/data/tbx11k ~/data/tbmxai/tbx11k
tbmxai dataset build mimic RAW ~/data/tbmxai/mimic_subset -p config=configs/mimic/build.yaml
tbmxai dataset adapters          # registered adapters and recipe parsers
```

Every build writes the standard tables, `dataset.yaml`, `build_info.json` (recipe hash,
counts, missing values, dropped duplicates) and validates the result (`validation.csv`).

A recipe has five parts: `records` (what one sample is), `fields` (how each column is
derived), `tables` (which fields go where), `labels` (classes, positive last) and a header.

```yaml
name: tbx11k
patient_ids: assumed                 # provided | assumed (no real patient ids)
archives: ["TBX11K.zip"]             # extracted once into RAW/<zip stem>/

records:
  files: ["**/imgs/health/*.png", "**/imgs/sick/*.png", "**/imgs/tb/*.png"]
  sort_by: patient_id
  drop_duplicate_images: true        # keep one record per near-identical image

fields:                              # evaluated in order
  patient_id: {from: stem, format: "tbx11k_{}"}
  tb:         {from: parent, map: {health: non_tb, sick: non_tb, tb: tb}}
  site:       {from: const, value: tbx11k}
  split:      {from: list, lists: {train: TBX11K/TBX11K/lists/TBX11K_train.txt}}

tables:
  labels: [tb, site, split]          # patient_id (+ study_id) are added automatically
  images: []                         # image_path is added automatically for file records

labels:
  tb: {classes: [non_tb, tb]}
```

| Source (`from`) | Value |
|---|---|
| `filename`, `stem`, `path` | The file's name, name without suffix, path relative to RAW |
| `parent`, `parent:N` | The containing folder's name (N levels up) |
| `column:X` | Column `X`, when `records: {table: file.csv}` (one record per row) |
| `field:X` | A field defined earlier |
| `const` | `value` |
| `file` | Text of a side-car file: files matching `glob`, indexed by stem, looked up by `lookup_by` |
| `table` | Column `column` of another table (`path`), indexed by `key`, looked up by `lookup_by` |
| `list` | Name of the list file (`lists: {train: …, val: …}`) whose line the record path ends with |

Transforms, in order: `regex` (named group `id`, else group 1), `parse` (a registered parser
returning a value or a dict of new fields, e.g. `nlm_reading`), `map` (+ `unmapped: error |
missing | keep`), `format`, `cast` (`str | float | int`), `bins` + `labels` (left-closed),
`default`. Register parsers in Python with `tbmxai.datasets.register_parser`.

YAML pitfalls: quote map keys such as `"1"`, `"yes"`, `"no"`, `"on"`, `"off"` and `"null"`;
unquoted, YAML turns them into numbers, booleans or null.

`configs/datasets/nlm_tb.yaml` produces byte-identical output to the original Python
converter; a test keeps it that way.

## Pooling datasets (multi-site)

```bash
tbmxai dataset merge ~/data/tbmxai/tb_multisite ~/data/tbmxai/nlm_tb ~/data/tbmxai/tbx11k \
    --label tb=non_tb,tb --rename tb:normal=non_tb --name tb_multisite
```

- Tables are concatenated; `labels.csv` gains a `source` column. Columns an input lacks
  stay empty (TBX11K has no age or sex); image paths become absolute.
- Label values are harmonised with `--rename` and must then match `--label` exactly.
- Patient ids must not collide across inputs (prefix them in a recipe, e.g. `format`).
- Validation fingerprints **every** image by default, so near-identical images shared by
  two datasets are reported before they can leak across sites.

## Datasets without patient IDs

Some public collections give no patient information. They can still be used:

```bash
tbmxai dataset index-images /raw/tb_kaggle tb_kaggle --patient-id-from none --label-from-folder label
```

This writes `images.csv`, `labels.csv` and a `dataset.yaml` with `patient_ids: assumed`.
Each image is then treated as its own patient, which means **patient-level splitting cannot
be guaranteed**: the same person may appear in both training and test data. tbmxai never
hides this:
- the validator reports it;
- the cohort is flagged (`patient_ids_assumed`);
- the data-quality report, `report.md` and `report.html` all carry a warning.

The validator also looks for near-identical images belonging to different "patients"
(possible duplicates).

## Validate before use

```bash
tbmxai dataset validate my_dataset                  # decodes a random sample of 500 images
tbmxai dataset validate my_dataset --check-images 0 # decode every image
```

| Checks | Level |
|---|---|
| `dataset.yaml` schema; `labels.csv` and one modality file present and readable | error |
| `patient_id` in every file; `image_path` / `report_text` present | error |
| Duplicate keys in `labels.csv` or `tabular.csv`; links that would multiply rows | error |
| Label values match the declared classes; label columns outside `labels.csv` | error |
| Features recorded after the prediction time (`feature_time` > `time_column`) | error |
| Image files missing, or sampled images that cannot be decoded | error / warning |
| Rows that don't link to a `labels.csv` row | warning |
| Features that alone separate a label almost perfectly (possible leakage) | warning |
| ID-like or entirely empty features; tiny classes | warning |
| Near-identical images belonging to different patients | warning |
| Report-derived labels combined with report text; patient IDs assumed | warning |

The command exits with an error code if any errors are found, so it can gate a pipeline.

## Using a dataset in an experiment

```yaml
data:
  dataset: /data/my_tb_cohort        # folder with dataset.yaml
  label_column: tb_status            # optional if the dataset has a single label
  use: [tabular, images]             # optional: which files to load (default: all present)
view:
  modalities: [imaging, tabular]
  metadata_columns: [sex, age, site]
```

`use` controls the sample unit: loading `images` gives one sample per image, even for an
EHR-only model. That keeps EHR, imaging and multimodal experiments on identical samples.
Each run fingerprints `dataset.yaml` and every data file in `run_info.json`.

## Avoiding leakage: your responsibility

The format can carry times and the validator flags obvious problems, but only you can
decide what was **known at the moment of prediction**:

- `tabular.csv` must contain only data recorded **before** the prediction time: no
  discharge diagnoses, results that came back later, or treatment given because of the
  outcome.
- Labels derived from a report (e.g. NLP labels) must not be predicted from that same
  report text.
- If one patient has several samples, give them the same `patient_id`. Otherwise a patient
  can land in both training and test.
