# Shenzhen + Montgomery TB chest X-ray sets

Two small public TB datasets from the U.S. National Library of Medicine (Jaeger et al.,
*Quant Imaging Med Surg* 2014;4(6):475–477). No login is needed.

| Set | Site | Images | TB / normal | Download |
|---|---|---|---|---|
| Shenzhen | Shenzhen No.3 People's Hospital, China | 662 | 336 / 326 | `ChinaSet_AllFiles.zip`, 3.8 GB |
| Montgomery | Montgomery County TB screening, USA | 138 | 58 / 80 | `NLM-MontgomeryCXRSet.zip`, 0.6 GB |

Each image comes with a clinical reading: sex, age and the radiologist's finding.

## Workflow

Settings (download folder, dataset folder, run folder) are in `scripts/nlm_tb/env.sh`.

```bash
bash scripts/nlm_tb/01_download.sh   # both zips from NLM Open-i; resumable; integrity-checked
bash scripts/nlm_tb/02_build.sh      # extract, convert, validate, cache resized images
bash scripts/nlm_tb/03_train.sh      # three models + statistical comparison
```

`02_build.sh` runs `tbmxai dataset build configs/datasets/nlm_tb.yaml RAW OUT` (a recipe; the
older `tbmxai dataset nlm-tb RAW OUT` gives identical output), which writes a
[standard dataset](DATASETS.md):

| File | Content |
|---|---|
| `labels.csv` | `patient_id`, `tb` (normal / tb, from the file-name suffix), `site`, `sex`, `age`, `age_group` |
| `tabular.csv` | `age`, `sex_male`: the model inputs of the demographic and fusion models |
| `images.csv` | `image_path`, relative to the raw download folder (`image_root`) |
| `readings.csv` | The radiologist's finding text, for reference only. It is **not loaded**: it states the diagnosis, so using it would leak the label. |

Lung masks in the Montgomery archive are ignored.

## Patient IDs

The sets give one ID per image and no patient identifiers, so the dataset is written with
`patient_ids: assumed`. Every report says that patient-level splitting cannot be guaranteed.
This matters little for the main design: test patients come from a different site
(Montgomery) than training patients (Shenzhen).

## Experiments

All three configs train on Shenzhen (80 % train, 20 % validation) and test on **all of
Montgomery** with the `external` splitter:

```yaml
split:
  name: external
  params: {column: site, test_values: montgomery, val_size: 0.2}
```

| Config | Model |
|---|---|
| `configs/nlm_tb/demographics_lr.yaml` | Logistic regression on age and sex: a baseline for how much the image adds |
| `configs/nlm_tb/cxr_densenet.yaml` | DenseNet-121 (ImageNet-pretrained) on the CXR |
| `configs/nlm_tb/multimodal.yaml` | DenseNet-121 + age/sex MLP, late fusion (one head per input), modality ablation at test time |

Every config loads the same files, so all models see identical samples and the same
Montgomery test set. `03_train.sh` therefore compares them directly with `tbmxai compare`
(DeLong, McNemar, paired bootstrap).

## Hyperparameter search

```bash
bash scripts/tmux_job.sh tune bash scripts/nlm_tb/04_tune.sh
```

1. **Optuna search** for the CXR and fusion models (30 trials each, median pruning):
   backbone (DenseNet-121, ResNet-50, ConvNeXt-Tiny), image size, learning rate, weight decay,
   backbone learning-rate multiplier, head dropout; for fusion also the fusion type
   (`late`, `gated`, `concat`) and age/sex dropout. Trials train on Shenzhen-train and are
   scored on Shenzhen-validation; Montgomery is used once, after refitting the best settings.
2. **Five seeds** of each best setting (`tbmxai cv --folds 0 --seed …`). Each seed re-draws the
   Shenzhen train/validation split and the initialisation, which shows how much of a
   difference between models is run-to-run noise.
3. **Comparison** of the tuned models with the demographics baseline on Montgomery.

All three configs choose the decision threshold on Shenzhen validation (Youden's J) and early-stop on
validation loss. The fusion config defaults to `late` fusion: in the first run, `gated` fusion
learned to ignore age and sex entirely (identical results with and without them).

Caveats:
- With 138 test images, confidence intervals are wide.
- Montgomery is a screening population, while Shenzhen is hospital-based. A drop in
  performance from validation to test is expected and is the point of external validation.
