"""Dataset recipes, the adapter registry and dataset merging."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from test_nlm_tb import _fake_nlm
from typer.testing import CliRunner

from tbmxai.cli import app
from tbmxai.data.dataset_format import DatasetSpec, validate_dataset
from tbmxai.datasets.adapters import ADAPTERS, PARSERS, register_parser, resolve_adapter
from tbmxai.datasets.merge import merge_datasets
from tbmxai.datasets.recipe import Recipe

REPO = Path(__file__).resolve().parents[1]


def test_nlm_recipe_matches_reference_converter(tmp_path) -> None:
    """configs/datasets/nlm_tb.yaml reproduces the Python converter byte for byte."""
    _fake_nlm(tmp_path / "raw")
    ADAPTERS.build("nlm_tb_legacy").build(tmp_path / "raw", tmp_path / "legacy")
    info = resolve_adapter(str(REPO / "configs/datasets/nlm_tb.yaml")).build(
        tmp_path / "raw", tmp_path / "recipe"
    )
    for name in ("labels.csv", "tabular.csv", "images.csv", "readings.csv", "dataset.yaml"):
        assert (tmp_path / "recipe" / name).read_bytes() == (
            tmp_path / "legacy" / name
        ).read_bytes(), name
    assert info["n_records"] == 60 and info["recipe_sha256"]
    assert validate_dataset(tmp_path / "recipe", check_images=10).ok


def _png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(value)
    Image.fromarray(rng.integers(0, 255, (32, 32)).astype(np.uint8)).save(path)


def _fake_tbx(raw: Path) -> None:
    root = raw / "TBX" / "imgs"
    for folder, prefix, n in (("health", "h", 6), ("sick", "s", 4), ("tb", "tb", 4)):
        for i in range(n):
            _png(root / folder / f"{prefix}{i:04d}.png", hash((folder, i)) % 10_000)
    # an exact duplicate (same label) and a conflicting duplicate (different label)
    Image.open(root / "sick" / "s0000.png").save(root / "sick" / "s0099.png")
    Image.open(root / "health" / "h0000.png").save(root / "tb" / "tb0099.png")
    (raw / "TBX" / "extra").mkdir()
    _png(raw / "TBX" / "extra" / "x.png", 1)  # must not be picked up
    (raw / "TBX" / ".hidden.png").write_bytes(b"")
    lists = raw / "TBX" / "lists"
    lists.mkdir()
    (lists / "train.txt").write_text("health/h0000.png\nhealth/h0001.png\nsick/s0000.png\n")
    (lists / "val.txt").write_text("tb/tb0000.png\n")
    pd.DataFrame({"image": [f"h{i:04d}" for i in range(6)], "age": range(30, 36)}).to_csv(
        raw / "TBX" / "meta.csv", index=False
    )


TBX_RECIPE = {
    "name": "fake_tbx",
    "patient_ids": "assumed",
    "records": {
        "files": ["**/imgs/health/*.png", "**/imgs/sick/*.png", "**/imgs/tb/*.png"],
        "sort_by": "patient_id",
        "drop_duplicate_images": True,
    },
    "fields": {
        "patient_id": {"from": "stem", "format": "tbx_{}"},
        "cls": {"from": "parent", "map": {"health": "healthy", "sick": "sick", "tb": "tb"}},
        "tb": {"from": "field:cls", "map": {"tb": "tb"}, "unmapped": "missing", "default": "no"},
        "split": {
            "from": "list",
            "lists": {"train": "TBX/lists/train.txt", "val": "TBX/lists/val.txt"},
        },
        "age": {"from": "table", "path": "TBX/meta.csv", "key": "image", "column": "age"},
        "number": {"from": "stem", "regex": r"(\d+)$", "cast": "int"},
        "age_group": {"from": "field:age", "bins": [0, 33, 200], "labels": ["<33", "33+"]},
    },
    "tables": {"labels": ["tb", "cls", "split", "age_group"], "tabular": ["age", "number"]},
    "labels": {"tb": {"classes": ["no", "tb"]}, "cls": {"classes": ["healthy", "sick", "tb"]}},
}


def test_recipe_sources_transforms_and_duplicates(tmp_path) -> None:
    _fake_tbx(tmp_path / "raw")
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(TBX_RECIPE, sort_keys=False))
    info = Recipe.load(tmp_path / "r.yaml").build(tmp_path / "raw", tmp_path / "out")
    labels = pd.read_csv(tmp_path / "out" / "labels.csv").set_index("patient_id")
    tab = pd.read_csv(tmp_path / "out" / "tabular.csv").set_index("patient_id")
    images = pd.read_csv(tmp_path / "out" / "images.csv")

    # 14 images + 2 duplicates; s0099 dropped (duplicate), h0000 + tb0099 dropped (label conflict)
    assert len(labels) == 13 and "tbx_s0099" not in labels.index
    assert {"tbx_h0000", "tbx_tb0099"}.isdisjoint(labels.index)
    reasons = sorted(d["reason"] for d in info["duplicate_images_dropped"])
    assert reasons == ["duplicate", "label conflict"]
    assert not images["image_path"].str.contains("extra|hidden").any()

    assert labels.loc["tbx_tb0001", "tb"] == "tb" and labels.loc["tbx_h0001", "tb"] == "no"
    assert (
        labels.loc["tbx_h0001", "split"] == "train" and labels.loc["tbx_tb0000", "split"] == "val"
    )
    assert pd.isna(labels.loc["tbx_tb0001", "split"])
    assert tab.loc["tbx_h0001", "age"] == 31 and pd.isna(tab.loc["tbx_s0001", "age"])
    assert tab.loc["tbx_tb0003", "number"] == 3
    assert labels.loc["tbx_h0005", "age_group"] == "33+"
    assert list(labels.index) == sorted(labels.index)

    spec = DatasetSpec.load(tmp_path / "out")
    assert spec.patient_ids == "assumed" and spec.images_root == (tmp_path / "raw").resolve()
    assert validate_dataset(tmp_path / "out", check_images=None).ok


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fields": {"cls": {"from": "parent", "map": {"health": "healthy"}}}}, "not in map"),
        ({"labels": {"tb": {"classes": ["no", "yes"]}}}, "not in classes"),
        ({"fields": {"x": {"from": "nowhere"}}}, "unknown source"),
        ({"fields": {"x": {"from": "file"}}}, "needs `glob`"),
        ({"fields": {"x": {"from": "field:later"}, "later": {"from": "stem"}}}, "before it"),
        ({"fields": {"x": {"from": "stem", "parse": "nope"}}}, "unknown parser"),
        ({"fields": {"x": {"from": "stem", "cast": "float"}}}, "cannot cast"),
        ({"records": {"files": "*.png", "table": "t.csv"}}, "exactly one"),
    ],
)
def test_recipe_errors(tmp_path, change, message) -> None:
    _fake_tbx(tmp_path / "raw")
    recipe = {k: (dict(v) if isinstance(v, dict) else v) for k, v in TBX_RECIPE.items()}
    for key, value in change.items():
        if key == "fields":
            recipe["fields"] = {**recipe["fields"], **value}
        else:
            recipe[key] = {**recipe[key], **value} if key == "labels" else value
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    with pytest.raises(ValueError, match=message):
        Recipe.load(tmp_path / "r.yaml").build(tmp_path / "raw", tmp_path / "out")


def test_table_records_and_custom_parser(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {"pid": ["a", "b", "c"], "dx": ["TB", "none", "TB"], "note": ["x;61", "y;45", "z;30"]}
    ).to_csv(raw / "meta.csv", index=False)

    @register_parser("semicolon_age")
    def _age(text: str) -> dict:
        return {"age": float(text.split(";")[1])}

    try:
        (tmp_path / "r.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": "tab",
                    "records": {"table": "meta.csv"},
                    "fields": {
                        "patient_id": {"from": "column:pid"},
                        "tb": {"from": "column:dx", "map": {"TB": "tb", "none": "no"}},
                        "parsed": {"from": "column:note", "parse": "semicolon_age"},
                    },
                    "tables": {"labels": ["tb"], "tabular": ["age"]},
                    "labels": {"tb": {"classes": ["no", "tb"]}},
                }
            )
        )
        Recipe.load(tmp_path / "r.yaml").build(raw, tmp_path / "out")
    finally:
        PARSERS.pop("semicolon_age")
    assert pd.read_csv(tmp_path / "out" / "tabular.csv")["age"].tolist() == [61.0, 45.0, 30.0]
    assert not (tmp_path / "out" / "images.csv").exists()


def test_merge_and_cli(tmp_path) -> None:
    _fake_nlm(tmp_path / "nlm_raw")
    _fake_tbx(tmp_path / "tbx_raw")
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(TBX_RECIPE, sort_keys=False))
    runner = CliRunner()
    for source, raw, out in (
        (str(REPO / "configs/datasets/nlm_tb.yaml"), "nlm_raw", "nlm"),
        (str(tmp_path / "r.yaml"), "tbx_raw", "tbx"),
    ):
        res = runner.invoke(
            app, ["dataset", "build", source, str(tmp_path / raw), str(tmp_path / out)]
        )
        assert res.exit_code == 0, res.output
    with pytest.raises(ValueError, match="not in"):  # NLM 'normal' must be harmonised
        merge_datasets([tmp_path / "nlm", tmp_path / "tbx"], tmp_path / "m", {"tb": ["no", "tb"]})
    res = runner.invoke(
        app,
        [
            "dataset",
            "merge",
            *(str(tmp_path / d) for d in ("m", "nlm", "tbx")),
            *("-l", "tb=no,tb", "-r", "tb:normal=no", "--name", "pooled"),
        ],
    )
    assert res.exit_code == 0, res.output
    labels = pd.read_csv(tmp_path / "m" / "labels.csv")
    assert set(labels["source"]) == {"nlm_tb", "fake_tbx"} and set(labels["tb"]) == {"no", "tb"}
    images = pd.read_csv(tmp_path / "m" / "images.csv")
    assert images["image_path"].map(lambda p: Path(p).is_absolute()).all()
    spec = DatasetSpec.load(tmp_path / "m")
    assert spec.name == "pooled" and spec.patient_ids == "assumed" and list(spec.labels) == ["tb"]
    res = runner.invoke(app, ["dataset", "adapters"])
    assert "nlm_tb_legacy" in res.output and "nlm_reading" in res.output
