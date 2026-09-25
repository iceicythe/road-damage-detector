import json
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from road_damage.data import convert_box, grouped_split, prepare


def test_coordinate_conventions_and_invalid_boxes():
    assert convert_box([1, 1, 100, 50], 100, 50, "voc1") == [0.5, 0.5, 1.0, 1.0]
    assert convert_box([0, 0, 100, 50], 100, 50, "zero") == [0.5, 0.5, 1.0, 1.0]
    for box in ([1, 1, 101, 50], [20, 20, 10, 10], [0, 0, 100, 50]):
        with pytest.raises(ValueError):
            convert_box(box, 100, 50, "voc1")


def test_transitive_duplicate_and_route_grouping():
    records = [{"sha256": "same", "source_group": "route_a"},
               {"sha256": "same", "source_group": "route_b"},
               {"sha256": "different", "source_group": "route_b"}]
    records += [{"sha256": str(i)} for i in range(12)]
    result = grouped_split(records, 42)
    assert len({r["split"] for r in result[:3]}) == 1
    assert {r["split"] for r in result} == {"train", "val", "test"}


def make_sample(raw, i, category="D00"):
    base = raw / "Japan" / "train"
    images = base / "images"
    annotations = base / "annotations" / "xmls"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 50), (i * 20, 10, 20)).save(images / f"sample_{i}.jpg")
    root = ET.Element("annotation")
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "100"
    ET.SubElement(size, "height").text = "50"
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = category
    box = ET.SubElement(obj, "bndbox")
    for key, val in zip(["xmin", "ymin", "xmax", "ymax"], [1, 1, 100, 50]):
        ET.SubElement(box, key).text = str(val)
    ET.ElementTree(root).write(annotations / f"sample_{i}.xml")


def test_preparation_quarantines_unknown_and_missing_annotations(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    for i in range(8):
        make_sample(raw, i, "D00" if i < 7 else "UNKNOWN")
    Image.new("RGB", (100, 50), "white").save(raw / "Japan/train/images/unlabelled.jpg")
    prepare(raw, output, "voc1", preview_count=2)
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    assert audit["accepted"] == 7
    assert audit["rejected"] == 2
    assert len(list((output / "preview").glob("*.jpg"))) == 2
    for label in output.glob("labels/*/*.txt"):
        assert label.read_text().strip() == "0 0.50000000 0.50000000 1.00000000 1.00000000"
    manifests = [json.loads(line) for p in output.glob("splits/*.jsonl") for line in p.read_text().splitlines()]
    assert len(manifests) == 7
    assert all((output / r["prepared_image"]).exists() for r in manifests)
    with pytest.raises(ValueError, match="输出目录已存在"):
        prepare(raw, output, "voc1")

