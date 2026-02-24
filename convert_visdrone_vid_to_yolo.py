#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert VisDrone-VID (10-column annotations) to YOLO detection format.

VisDrone-VID annotation format (10 columns):
[0] frame_index (1-based)
[1] target_id
[2] bbox_left
[3] bbox_top
[4] bbox_width
[5] bbox_height
[6] score (VID train/val often 1)
[7] object_category
[8] truncation
[9] occlusion

Output YOLO label line:
<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image

# -----------------------------
# VisDrone category mapping
# -----------------------------
# Common VisDrone categories (object_category):
# 0: ignored regions
# 1: pedestrian
# 2: people
# 3: bicycle
# 4: car
# 5: van
# 6: truck
# 7: tricycle
# 8: awning-tricycle
# 9: bus
# 10: motor
#
# You can modify this mapping for your task.
VISDRONE_TO_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VisDrone-VID annotations (10 columns) to YOLO detection format."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path containing VisDrone2019-VID-train / VisDrone2019-VID-val",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output YOLO dataset directory",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output images/train|val (default: create hardlinks if possible, else copy)",
    )
    parser.add_argument(
        "--only-cats",
        type=str,
        default="",
        help="Comma-separated VisDrone category ids to keep, e.g. '1,2' (empty = keep all valid categories)",
    )
    parser.add_argument(
        "--drop-trunc-ge",
        type=float,
        default=None,
        help="Drop boxes with truncation >= this value (optional)",
    )
    parser.add_argument(
        "--drop-occl-ge",
        type=float,
        default=None,
        help="Drop boxes with occlusion >= this value (optional)",
    )
    parser.add_argument(
        "--min-box",
        type=float,
        default=2.0,
        help="Drop boxes with width/height < min-box pixels",
    )
    return parser.parse_args()


def safe_link_or_copy(src: Path, dst: Path, force_copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        # Try hard link first (fast, no extra storage)
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def yolo_box_from_xywh(left, top, w, h, img_w, img_h):
    # Clip to image boundary
    x1 = max(0.0, left)
    y1 = max(0.0, top)
    x2 = min(float(img_w), left + w)
    y2 = min(float(img_h), top + h)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return (
        xc / img_w,
        yc / img_h,
        bw / img_w,
        bh / img_h,
    )


def build_class_mapping(only_cats_set):
    """
    Return:
      keep_categories: sorted list of VisDrone cat ids
      visdrone_cat_to_yolo_id: dict
      yolo_names: list
    """
    valid = sorted(VISDRONE_TO_NAME.keys())
    if only_cats_set:
        keep = sorted([c for c in valid if c in only_cats_set])
    else:
        keep = valid

    vis2yolo = {cat: i for i, cat in enumerate(keep)}
    names = [VISDRONE_TO_NAME[c] for c in keep]
    return keep, vis2yolo, names


def parse_10col_line(line: str):
    """
    Parse one annotation line with 10 columns:
    frame, target_id, left, top, width, height, score, category, truncation, occlusion
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 10:
        return None

    try:
        frame_idx = int(float(parts[0]))
        # target_id = int(float(parts[1]))  # not used for pure detection
        left = float(parts[2])
        top = float(parts[3])
        width = float(parts[4])
        height = float(parts[5])
        # score = float(parts[6])           # usually 1 in train/val
        category = int(float(parts[7]))
        truncation = float(parts[8])
        occlusion = float(parts[9])
    except ValueError:
        return None

    return {
        "frame_idx": frame_idx,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "category": category,
        "truncation": truncation,
        "occlusion": occlusion,
    }


def process_split(
    split_name: str,  # "train" or "val"
    src_dir: Path,    # VisDrone2019-VID-train or ...-val
    out_dir: Path,
    keep_categories,
    vis2yolo,
    copy_images: bool,
    drop_trunc_ge,
    drop_occl_ge,
    min_box,
):
    """
    Expected VisDrone-VID layout:
      src_dir/
        sequences/<seq_name>/<0000001.jpg ...>
        annotations/<seq_name>.txt
    """
    seq_root = src_dir / "sequences"
    ann_root = src_dir / "annotations"

    if not seq_root.exists():
        raise FileNotFoundError(f"Missing sequences folder: {seq_root}")
    if not ann_root.exists():
        raise FileNotFoundError(f"Missing annotations folder: {ann_root}")

    out_img_root = out_dir / "images" / split_name
    out_lbl_root = out_dir / "labels" / split_name
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_lbl_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([p for p in seq_root.iterdir() if p.is_dir()])

    total_images = 0
    total_boxes = 0
    total_kept_boxes = 0

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        ann_file = ann_root / f"{seq_name}.txt"
        if not ann_file.exists():
            # Skip sequence if no annotation file
            continue

        # Group detections by frame_index
        frame_to_objs = defaultdict(list)
        with ann_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = parse_10col_line(line)
                if obj is None:
                    continue
                frame_to_objs[obj["frame_idx"]].append(obj)

        # Iterate images in this sequence
        img_files = sorted(
            [p for p in seq_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )

        for img_p in img_files:
            total_images += 1
            # frame number from file name e.g. "0000001.jpg" -> 1
            try:
                frame_idx = int(img_p.stem)
            except ValueError:
                # fallback: skip weird name
                continue

            # unique output name: <seq_name>_<frame7>.jpg
            out_stem = f"{seq_name}_{img_p.stem}"
            out_img_p = out_img_root / f"{out_stem}{img_p.suffix.lower()}"
            out_lbl_p = out_lbl_root / f"{out_stem}.txt"

            safe_link_or_copy(img_p, out_img_p, force_copy=copy_images)

            # get image size
            with Image.open(img_p) as im:
                img_w, img_h = im.size

            yolo_lines = []
            objs = frame_to_objs.get(frame_idx, [])
            total_boxes += len(objs)

            for o in objs:
                cat = o["category"]
                if cat not in keep_categories:
                    continue
                if o["width"] < min_box or o["height"] < min_box:
                    continue
                if drop_trunc_ge is not None and o["truncation"] >= drop_trunc_ge:
                    continue
                if drop_occl_ge is not None and o["occlusion"] >= drop_occl_ge:
                    continue

                box = yolo_box_from_xywh(
                    o["left"], o["top"], o["width"], o["height"], img_w, img_h
                )
                if box is None:
                    continue

                cls = vis2yolo[cat]
                xc, yc, bw, bh = box
                yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                total_kept_boxes += 1

            # write label file (can be empty if no objects kept)
            out_lbl_p.write_text("\n".join(yolo_lines), encoding="utf-8")

    return {
        "split": split_name,
        "images": total_images,
        "boxes_total": total_boxes,
        "boxes_kept": total_kept_boxes,
    }


def write_data_yaml(out_dir: Path, names):
    yaml_text = (
        f"path: {out_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/val\n\n"
        f"names:\n"
    )
    for i, n in enumerate(names):
        yaml_text += f"  {i}: {n}\n"

    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    args = parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # parse only-cats
    only_cats_set = set()
    if args.only_cats.strip():
        for x in args.only_cats.split(","):
            x = x.strip()
            if x:
                only_cats_set.add(int(x))

    keep_categories, vis2yolo, yolo_names = build_class_mapping(only_cats_set)

    train_src = root / "VisDrone2019-VID-train"
    val_src = root / "VisDrone2019-VID-val"

    print("== Class mapping ==")
    for c in keep_categories:
        print(f"VisDrone cat {c} ({VISDRONE_TO_NAME[c]}) -> YOLO class {vis2yolo[c]}")
    print()

    train_stat = process_split(
        split_name="train",
        src_dir=train_src,
        out_dir=out_dir,
        keep_categories=keep_categories,
        vis2yolo=vis2yolo,
        copy_images=args.copy_images,
        drop_trunc_ge=args.drop_trunc_ge,
        drop_occl_ge=args.drop_occl_ge,
        min_box=args.min_box,
    )

    val_stat = process_split(
        split_name="val",
        src_dir=val_src,
        out_dir=out_dir,
        keep_categories=keep_categories,
        vis2yolo=vis2yolo,
        copy_images=args.copy_images,
        drop_trunc_ge=args.drop_trunc_ge,
        drop_occl_ge=args.drop_occl_ge,
        min_box=args.min_box,
    )

    write_data_yaml(out_dir, yolo_names)

    print("== Done ==")
    print(train_stat)
    print(val_stat)
    print(f"Output: {out_dir.resolve()}")
    print(f"data.yaml: {(out_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
