# Multi-site TB external validation (Shenzhen, Montgomery, TBX11K)

Three public TB chest X-ray sources, pooled into one standard dataset with a `site` column:

| Site | Images | non-TB / TB | Notes |
|---|---|---|---|
| Shenzhen (China, hospital) | 662 | 326 / 336 | age, sex; no patient ids |
| Montgomery (USA, screening) | 138 | 80 / 58 | age, sex; no patient ids |
| TBX11K (China) | 8,262 | 7,462 / 800 | healthy, sick-non-TB, TB; 512 × 512; no age/sex |

TBX11K uses only its own labelled train + val images. Excluded: `imgs/extra/` (copies of
Montgomery and Shenzhen, which would leak across sites), the unlabelled challenge test set,
and one image from each of 138 duplicate pairs (126 pixel-identical, all "sick but non-TB";
41 pairs straddle TBX11K's official train/val lists). A full fingerprint scan of the pooled
9,062 images finds no near-duplicates across or within sites.

## Workflow

```bash
bash scripts/nlm_tb/01_download.sh              # Shenzhen + Montgomery (NLM Open-i)
bash scripts/multisite/00_download_tbx11k.sh    # TBX11K (authors' Google Drive, 3.3 GB)
bash scripts/multisite/01_build.sh              # recipes → merge (+ duplicate scan) → image cache
bash scripts/multisite/02_train.sh              # both directions × seeds (MULTISITE_SEEDS)
```

## Experiments

| Config | Train / validate | External test |
|---|---|---|
| `configs/multisite/cxr_from_shenzhen.yaml` | Shenzhen (80 / 20 %) | Montgomery + TBX11K |
| `configs/multisite/cxr_from_tbx11k.yaml` | TBX11K (85 / 15 %) | Shenzhen + Montgomery |

- Image-only DenseNet-121 (TBX11K has no age or sex).
- The decision threshold is fixed on the source validation split to reach **90 % sensitivity**
  (WHO target product profile for TB triage) and applied unchanged to each external site.
  Compare each site's test `sensitivity` with 0.90, and `spec_at_sens90` with the TPP's 0.70.
- Per-site results: subgroup analysis by `site` (`tables/subgroups.csv`, report).
- Seeds re-draw the source train/validation split and the initialisation
  (`tbmxai cv --folds 0 --seed …`).

Caveats: none of the sets has patient identifiers (`patient_ids: assumed`); TB definitions
differ (NLM: radiological TB manifestation; TBX11K: active or latent TB with boxes).
