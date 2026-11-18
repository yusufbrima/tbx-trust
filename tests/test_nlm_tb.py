"""Shenzhen + Montgomery converter, external (site) split and repo configs on a fake layout."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tbmxai import ExperimentConfig, Pipeline
from tbmxai.data.dataset_format import DatasetSpec, validate_dataset
from tbmxai.datasets.nlm_tb import build_nlm_tb, parse_reading


@pytest.mark.parametrize(
    ("text", "sex", "age"),
    [
        ("male 35yrs\nnormal\n", "M", 35.0),
        ("Female   38yrs\nPTB in the left upper field", "F", 38.0),
        ("Male 7month\nnormal", "M", 0.58),
        ("femal 32yrs\nnormal\n", "F", 32.0),
        ("female24yrs\nnormal", "F", 24.0),
        ("male35yrs  \t\nPTB in the left upper field \t", "M", 35.0),
        ("male 42\nnormal", "M", 42.0),
        ("Patient's Sex: O \nPatient's Age: 005Y\nnormal\n", None, 5.0),
        ("Patient's Sex: M\nPatient's Age: 040Y\nnormal", "M", 40.0),
        ("Patient's Sex: F\nPatient's Age: 062Y\nleft upper lobe cavity", "F", 62.0),
        ("", None, None),
    ],
)
def test_parse_reading(text, sex, age) -> None:
    parsed = parse_reading(text)
    assert parsed["sex"] == sex and parsed["age"] == age
    assert "Sex" not in parsed["reading"] and "yrs" not in parsed["reading"]


def _fake_nlm(raw, n_shenzhen=40, n_montgomery=20, seed=0):
    rng = np.random.default_rng(seed)
    sets = [
        ("ChinaSet_AllFiles/ChinaSet_AllFiles", "CHNCXR", n_shenzhen),
        ("NLM-MontgomeryCXRSet/MontgomerySet", "MCUCXR", n_montgomery),
    ]
    for folder, prefix, n in sets:
        for i in range(1, n + 1):
            tb = int(i % 2 == 0)
            stem = f"{prefix}_{i:04d}_{tb}"
            img = rng.integers(0, 80, (48, 48)).astype(np.uint8)
            if tb:
                img[8:24, 8:24] += 120  # learnable "lesion"
            for sub in (
                "CXR_png",
                "ManualMask/leftMask",
            ):  # masks share file names: must be ignored
                (raw / folder / sub).mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(raw / folder / sub / f"{stem}.png")
            (raw / folder / "ClinicalReadings").mkdir(exist_ok=True)
            age, finding = 20 + i, "PTB in the right upper field" if tb else "normal"
            reading = (
                f"male {age}yrs\n{finding}\n"
                if prefix == "CHNCXR"
                else f"Patient's Sex: F\nPatient's Age: {age:03d}Y\n{finding}\n"
            )
            (raw / folder / "ClinicalReadings" / f"{stem}.txt").write_text(reading)


def test_build_and_validate(tmp_path) -> None:
    _fake_nlm(tmp_path / "raw")
    info = build_nlm_tb(tmp_path / "raw", tmp_path / "ds")
    assert info["n_images"] == 60 and info["unparsed_age"] == 0
    assert info["per_site"] == {
        "montgomery": {"normal": 10, "tb": 10},
        "shenzhen": {"normal": 20, "tb": 20},
    }
    labels = pd.read_csv(tmp_path / "ds" / "labels.csv")
    assert set(labels["sex"]) == {"M", "F"} and "reading" not in labels
    assert "reading" not in pd.read_csv(
        tmp_path / "ds" / "tabular.csv"
    )  # finding text never an input
    images = pd.read_csv(tmp_path / "ds" / "images.csv")
    assert images["image_path"].str.contains("/CXR_png/").all()
    spec = DatasetSpec.load(tmp_path / "ds")
    assert spec.patient_ids == "assumed" and spec.roles()["features"] == ["age", "sex_male"]
    report = validate_dataset(tmp_path / "ds", check_images=None)
    assert report.ok, report.to_frame()


def _cfg(name, tmp_path, monkeypatch, **updates):
    monkeypatch.setenv("TBMXAI_NLM_TB", str(tmp_path / "ds"))
    monkeypatch.setenv("TBMXAI_OUTPUT_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("TBMXAI_IMAGE_CACHE", str(tmp_path / "cache"))
    cfg = ExperimentConfig.from_yaml(f"configs/nlm_tb/{name}.yaml")
    small = {
        "evaluation": cfg.evaluation.model_copy(update={"n_bootstrap": 20}),
        "subgroups": cfg.subgroups.model_copy(update={"min_size": 5}),
        **updates,
    }
    return cfg.model_copy(update=small)


def test_external_split_demographics(tmp_path, monkeypatch) -> None:
    _fake_nlm(tmp_path / "raw")
    build_nlm_tb(tmp_path / "raw", tmp_path / "ds")
    cfg = _cfg("demographics_lr", tmp_path, monkeypatch)
    pipe = Pipeline(cfg)
    view = pipe.harmonize(pipe.ingest())
    splits = pipe.split(view)
    assert set(splits.test.metadata["site"]) == {"montgomery"} and len(splits.test) == 20
    assert set(splits.train.metadata["site"]) == set(splits.val.metadata["site"]) == {"shenzhen"}
    result = pipe.run(tmp_path / "run")
    assert result.split_sizes["test"] == 20
    assert (
        "patient-level splitting cannot be guaranteed"
        in (tmp_path / "run" / "report.md").read_text()
    )


def test_cxr_config_runs(tmp_path, monkeypatch) -> None:
    _fake_nlm(tmp_path / "raw")
    build_nlm_tb(tmp_path / "raw", tmp_path / "ds")
    cfg = _cfg("multimodal", tmp_path, monkeypatch)
    params = {**cfg.model.params, "image": {"size": 32, "in_chans": 1}}
    params["encoders"] = {
        **params["encoders"],
        "imaging": {"name": "timm_vision", "params": {"backbone": "resnet18", "pretrained": False}},
    }
    cfg = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(update={"params": params}),
            "training": cfg.training.model_copy(
                update={
                    "epochs": 1,
                    "num_workers": 0,
                    "device": "cpu",
                    "precision": "32",
                    "batch_size": 8,
                }
            ),
            "explain": cfg.explain.model_copy(
                update={"methods": [], "samples": cfg.explain.samples}
            ),
        }
    )
    result = Pipeline(cfg).run(tmp_path / "mm")
    assert result.split_sizes["test"] == 20 and "auroc" in result.evaluations["test"].metrics


def test_external_split_internal_test(tmp_path, monkeypatch) -> None:
    from tbmxai.data.splits import ExternalSplitter

    _fake_nlm(tmp_path / "raw")
    build_nlm_tb(tmp_path / "raw", tmp_path / "ds")
    pipe = Pipeline(_cfg("demographics_lr", tmp_path, monkeypatch))
    view = pipe.harmonize(pipe.ingest())
    splitter = ExternalSplitter(
        column="site", test_values="montgomery", val_size=0.2, internal_test_size=0.25
    )
    splits = splitter.split(view, seed=3)
    sites = splits.test.metadata["site"]
    n_src = int((view.metadata["site"] == "shenzhen").sum())
    assert (sites == "montgomery").sum() == 20 and 0 < (sites == "shenzhen").sum() < n_src
    assert set(splits.train.metadata["site"]) == {"shenzhen"}
    splits.check_patient_disjoint()
    assert len(splits.train) + len(splits.val) + (sites == "shenzhen").sum() == n_src
    again = splitter.split(view, seed=3)
    assert list(again.test.patient_ids) == list(splits.test.patient_ids)
    other = splitter.split(view, seed=4)
    assert list(other.val.patient_ids) != list(splits.val.patient_ids)
    shared = ExternalSplitter(
        column="site", test_values="montgomery", val_size=0.2, internal_test_size=0.25, split_seed=0
    )
    a, b = shared.split(view, seed=1), shared.split(view, seed=2)
    assert list(a.val.patient_ids) == list(b.val.patient_ids)
    assert list(a.test.patient_ids) == list(b.test.patient_ids)
