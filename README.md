# An Empirical Investigation of the Trustworthiness of Tuberculosis Chest Radiograph AI Across Unseen Sites

Code, configurations and analysis scripts for the preprint by Yusuf Brima (Osnabrück University).
The manuscript itself is not part of this repository.
Everything in the paper's tables and figures is regenerated from the public (or credentialed) raw data
by the scripts below. No patient-level data are included in this repository.

The paper separates three properties of TB chest radiograph models at sites they have not seen:
**discrimination** (external AUROC), **decision portability** (whether a threshold fixed on the source
site delivers its intended operating point elsewhere) and **adaptability** (how many confirmed local TB
cases a site needs to reset the threshold).

## Install

```bash
uv sync                                    # exact environment from uv.lock (Python 3.12)
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run pytest --basetemp ~/.cache/tbmxai-pytest
```

The `tbmxai` package (`src/`) is a configuration-driven framework: every model, splitter, metric and
explainer is a registered component named in a YAML config (`docs/CONFIGURATION.md`, `docs/EVALUATION.md`).

## Data (not redistributed)

| Site | Access | Build |
|---|---|---|
| Shenzhen, Montgomery | U.S. National Library of Medicine (public) | `scripts/nlm_tb/01_download.sh`, `02_build.sh` |
| TBX11K | public, CC BY 4.0 | `scripts/multisite/00_download_tbx11k.sh`, `01_build.sh` |
| MIMIC-IV, MIMIC-CXR-JPG | PhysioNet, credentialed access | `scripts/mimic/01*` → `02_select.sh` → `03*` → `04_build.sh` |
| Naija-CXR | Zenodo 10.5281/zenodo.21781842 | `scripts/analysis/naija_site.py build` |

Data locations default to `~/data` and runs to `~/tbmxai_runs` (overridable; see each folder's
`env.sh` and `docs/DATASETS.md`). Naija-CXR images carry burned-in patient identifiers; the code never
displays or exports them. The MIMIC subset used in the paper is fixed by seed and selection settings
(`configs/mimic/build.yaml`, `MIMIC_N_PATIENTS=20000`).

## Repository layout

| Folder | Contents |
|---|---|
| `src/tbmxai/` | the framework (data, models, evaluation, saliency) |
| `configs/` | one YAML per experiment arm (`nlm_tb/`, `multisite/`, `transfer/`, `pretrain/`, `datasets/`, `mimic/`) |
| `scripts/nlm_tb/`, `scripts/multisite/`, `scripts/mimic/` | download and build each data source; training entry points |
| `scripts/experiments/` | the training runs behind the paper, one script per experiment |
| `scripts/analysis/` | analyses of the trained runs (contrasts, operating points, recalibration, saliency, sites) |
| `scripts/reporting/` | the paper's tables (LaTeX) and figures (PDF) from the analysis outputs |
| `tests/` | unit and integration tests (`uv run pytest`) |

## Experiments

| What | Config(s) | Command |
|---|---|---|
| MIMIC CheXpert pretraining, MIMIC-initialised arms | `configs/pretrain/`, `configs/transfer/*mimicpt*.yaml` | `scripts/experiments/mimic_pretraining_and_transfer.sh` |
| Shenzhen → Montgomery arms | `configs/nlm_tb/`, `configs/transfer/nlm_*.yaml` | `uv run tbmxai cv CONFIG --folds 0 --seed 1 ... --seed 5` |
| Shenzhen → TBX11K, TBX11K → Shenzhen | `configs/multisite/`, `configs/transfer/multisite_*.yaml` | as above, 3 seeds (10 for the C1 arms: `scripts/experiments/c1_extra_seeds.sh`) |
| Foundation-model references | `configs/transfer/*xrv*.yaml`, `*raddino*.yaml` | `scripts/experiments/foundation_models.sh` |
| RAD-DINO learning-rate selection | `configs/transfer/raddino_lr/` | `scripts/experiments/raddino_learning_rate.sh`, then `raddino_retuned_scoring.sh` |
| TBX11K-only re-tune (C5) | `configs/multisite/tune_convnext_tbx11k.yaml` | `scripts/experiments/tbx11k_retune.sh` |
| Internal split, shared split | `configs/transfer/internal*/` | `scripts/experiments/internal_split.sh`, `shared_split.sh` |
| Saliency (lung masks, lesion boxes) | — | `scripts/experiments/saliency_lesions.sh`, `saliency_foundation_models.sh` |
| Naija-CXR scoring | — | `scripts/experiments/naija_scoring.sh` |

Each script runs its steps in order and logs `OK` / `FAILED` per step; `scripts/tmux_job.sh NAME CMD...`
runs one in a detached tmux session with a log file. Run one GPU job at a time.

## Analyses → paper

| Paper | Script |
|---|---|
| Table 2 (pre-specified contrasts, training-variance analysis) | `scripts/analysis/compare_arms.py`, `operating_points.py` |
| Table 3, operating-point, subset, programme and TBX11K-class tables | `scripts/analysis/operating_points.py` → `scripts/reporting/make_tables.py` |
| Figure 2, recalibration checks | `scripts/analysis/recalibration.py`, `recalibration_checks.py` |
| MIMIC TB site, one film per patient | `scripts/analysis/mimic_tb_site.py`, `mimic_tb_patient.py` |
| Naija-CXR flag rates | `scripts/analysis/naija_site.py` |
| Saliency: lung masks, lesion boxes, overlays | `scripts/analysis/saliency_lungs.py`, `saliency_boxes.py`, `scripts/reporting/saliency_overlays.py` |
| Shared-split sensitivity analysis | `scripts/analysis/score_site.py`, `shared_split.py` |
| Figures | `scripts/reporting/make_figures.py` (writes PDFs into the directory you pass) |

Each analysis script takes its output folder as `OUT_DIR`. The reporting scripts read from
`~/tbmxai_runs/analysis/`: `comparisons/` (`compare_arms.py`), `operating_points/`, `recalibration/`,
`mimic_tb_patient/`, `saliency_boxes/` and the saliency-lung folders named in `make_figures.py`.

## Notes

- Seeds, thresholds and calibration are fitted on source data only; test sites are never used for any
  choice. Thresholds reach 90 % sensitivity on each run's source validation split.
- AI assistance: Claude (Anthropic), through Claude Code, assisted with writing and running analysis
  code and with drafting the manuscript; see the paper's statement on the use of AI tools.

## License

Apache License 2.0 (`LICENSE`). If you use this code, please cite the preprint.
