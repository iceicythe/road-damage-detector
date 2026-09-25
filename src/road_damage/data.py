"""Conservative RDD2022 VOC conversion. Suspicious samples are quarantined in the report."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET

from .common import CLASSES, NAMES, sha256, write_json


def convert_box(values, width, height, origin):
    """Convert inclusive VOC coordinates, or explicit zero-based xyxy, to YOLO."""
    x1, y1, x2, y2 = map(float, values)
    if origin == "voc1":
        x1 -= 1
        y1 -= 1
    if not all(math.isfinite(x) for x in [x1, y1, x2, y2]):
        raise ValueError("nonfinite_box")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("invalid_or_out_of_bounds_box")
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def grouped_split(records, seed, train_ratio=0.7, val_ratio=0.15):
    """Keep both exact duplicates and optional source groups in the same partition."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        keys = [("hash", row["sha256"])]
        if row.get("source_group"):
            keys.append(("source", row["source_group"]))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    chunks = list(groups.values())
    if len(chunks) < 3:
        raise ValueError("至少需要 3 个独立数据组才能划分 train/val/test。")
    random.Random(seed).shuffle(chunks)
    # Approximate ratios by group count: never split a source group to hit a ratio.
    train_n = max(1, min(len(chunks) - 2, int(len(chunks) * train_ratio)))
    val_n = max(1, min(len(chunks) - train_n - 1, int(len(chunks) * val_ratio)))
    for group_index, chunk in enumerate(chunks):
        split = "train" if group_index < train_n else "val" if group_index < train_n + val_n else "test"
        for i in chunk:
            records[i]["split"] = split
            records[i]["split_group"] = f"group_{group_index:06d}"
    return records


def prepare(raw, output, origin, seed=42, groups_file=None, preview_count=24):
    from PIL import Image, ImageDraw
    import yaml

    if not raw.is_dir():
        raise ValueError(f"原始数据目录不存在：{raw}")
    if output.exists():
        raise ValueError(f"输出目录已存在，避免覆盖请改用新目录：{output}")
    groups = {}
    if groups_file:
        with groups_file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["image"].replace("\\", "/")
                if key in groups or not row["group"].strip():
                    raise ValueError("分组文件包含重复路径或空 group。")
                groups[key] = row["group"].strip()
    xmls = sorted(raw.rglob("train/annotations/xmls/*.xml"))
    if not xmls:
        raise ValueError("未找到 */train/annotations/xmls/*.xml，请按 README 解压 RDD2022。")
    records, issues, annotated_images = [], [], set()
    for xml in xmls:
        image = xml.parents[2] / "images" / (xml.stem + ".jpg")
        annotated_images.add(image.resolve())
        relative = image.relative_to(raw).as_posix()
        if groups_file and relative not in groups:
            raise ValueError(f"分组文件缺少图片：{relative}")
        try:
            root = ET.parse(xml).getroot()
            with Image.open(image) as im:
                im.load()
                width, height = im.size
            if int(root.findtext("size/width", "0")) != width or int(root.findtext("size/height", "0")) != height:
                raise ValueError("image_xml_size_mismatch")
            labels = []
            for obj in root.findall("object"):
                name = obj.findtext("name", "").strip()
                if name not in CLASSES:
                    raise ValueError(f"unknown_class:{name}")
                if obj.findtext("difficult", "0").strip() == "1":
                    raise ValueError("difficult_object_requires_review")
                values = [obj.findtext(f"bndbox/{key}", "") for key in ["xmin", "ymin", "xmax", "ymax"]]
                labels.append([CLASSES[name], *convert_box(values, width, height, origin)])
            records.append({"image": relative, "annotation": xml.relative_to(raw).as_posix(),
                            "sha256": sha256(image), "annotation_sha256": sha256(xml),
                            "width": width, "height": height, "labels": labels,
                            "source_group": groups.get(relative), "source": xml.parents[3].name})
        except (OSError, ValueError, ET.ParseError) as exc:
            issues.append({"image": relative, "annotation": xml.relative_to(raw).as_posix(), "reason": str(exc)})
    for image in sorted(raw.rglob("train/images/*.jpg")):
        if image.resolve() not in annotated_images:
            issues.append({"image": image.relative_to(raw).as_posix(), "reason": "missing_annotation"})
    records = grouped_split(records, seed)
    output.mkdir(parents=True)
    counts = {}
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        selected = [r for r in records if r["split"] == split]
        category = Counter(CLASSES_INV[int(label[0])] for r in selected for label in r["labels"])
        counts[split] = {"images": len(selected), "groups": len({r["split_group"] for r in selected}),
                         "background_images": sum(not r["labels"] for r in selected),
                         "boxes_by_class": {key: category[key] for key in CLASSES},
                         "sources": dict(Counter(r["source"] for r in selected))}
    for row in records:
        stem = hashlib.sha256(row["image"].encode()).hexdigest()[:16] + "_" + Path(row["image"]).stem
        image_target = output / "images" / row["split"] / (stem + ".jpg")
        shutil.copy2(raw / row["image"], image_target)
        row["prepared_image"] = image_target.relative_to(output).as_posix()
        label_text = "".join(str(label[0]) + " " + " ".join(f"{v:.8f}" for v in label[1:]) + "\n" for label in row["labels"])
        (output / "labels" / row["split"] / (stem + ".txt")).write_text(label_text, encoding="utf-8")
    (output / "splits").mkdir()
    for split in counts:
        lines = [json.dumps(r, ensure_ascii=False) for r in records if r["split"] == split]
        (output / "splits" / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "dataset.yaml").write_text(yaml.safe_dump({"path": str(output.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES))}, sort_keys=False), encoding="utf-8")
    notes = ["精确重复图片与显式来源组不会跨集合；未自动检测近重复。",
             "按组数近似 70/15/15；未执行类别分层，请检查每类覆盖后冻结划分。",
             "缺标注、未知类别、困难目标及异常坐标整张隔离，不自动视为背景。"]
    if not groups_file:
        notes.append("未提供道路/序列分组，除精确重复外采用图片级随机划分；不能声称道路场景独立。")
    report = {"seed": seed, "coordinate_origin": origin, "raw_root": str(raw),
              "split_counts": counts, "accepted": len(records), "rejected": len(issues),
              "exact_duplicate_copies": len(records) - len({r["sha256"] for r in records}),
              "notes": notes, "issues": issues}
    write_json(output / "audit.json", report)
    preview = output / "preview"
    preview.mkdir()
    for i, row in enumerate(random.Random(seed).sample(records, min(preview_count, len(records)))):
        with Image.open(raw / row["image"]) as original:
            image = original.convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, x, y, w, h in row["labels"]:
            left, top = (x-w/2)*image.width, (y-h/2)*image.height
            draw.rectangle([left, top, (x+w/2)*image.width, (y+h/2)*image.height], outline="red", width=2)
            draw.text((left, top), CLASSES_INV[class_id], fill="red")
        image.thumbnail((1200, 1200))
        image.save(preview / f"{i:03d}_{Path(row['image']).stem}.jpg")
    print(json.dumps(report, ensure_ascii=False, indent=2))


CLASSES_INV = {value: key for key, value in CLASSES.items()}

