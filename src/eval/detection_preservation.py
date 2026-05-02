"""Offline detection-preservation eval — run YOLO-World twice, compare.

Answers the question the benchmark actually cares about: *after we
chewed on the frame with H.264 + optical-flow masking + adaptive CRF,
does the detector still see what it saw on the untouched original?*

This is deliberately offline. The online hot path is already doing
inference; re-running it there for "did we preserve it?" would double
the GPU cost and distort the very timings we are trying to measure.

Inputs
------
``--run-dir`` points at a session produced by the camera with
``--save-artifacts``. The directory must contain:

  - ``original.mp4`` — the raw BGR frames before masking / encoding.
  - ``decoded_adaptive.mp4`` — the frames the server actually sees after
    encode/decode.
  - ``session_config.json`` — used to recover the class prompts.

Method
------
For each matched frame pair (by frame index):

  1. Run YOLO-World on the original → "ref" detection set.
  2. Run YOLO-World on the decoded_adaptive → "cand" detection set.
  3. Greedy-match cand→ref by IoU ≥ ``--iou-threshold``. Multiple
     candidates may exist — each ref is matched at most once.
  4. Record per-frame ``preserved_fraction = matched / len(ref)`` with
     the convention ``1.0`` when there is nothing to preserve.

Output
------
Writes ``<run-dir>/detection_preservation.json``. ``summarize_run()``
picks this up automatically and folds it into summary.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


log = logging.getLogger("echostream.detpres")


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a: (N,4), b: (M,4) → (N,M)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _greedy_match(iou: np.ndarray, threshold: float
                  ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy IoU matching. Returns (matches, unmatched_ref, unmatched_cand).

    `matches` is a list of (ref_idx, cand_idx, iou) tuples.
    """
    if iou.size == 0:
        return [], list(range(iou.shape[0])), list(range(iou.shape[1]))
    matches: List[Tuple[int, int, float]] = []
    used_ref: set[int] = set()
    used_cand: set[int] = set()
    # Sort all (ref, cand) pairs by IoU desc, pick greedily.
    n_ref, n_cand = iou.shape
    flat = [
        (float(iou[r, c]), r, c)
        for r in range(n_ref) for c in range(n_cand)
        if iou[r, c] >= threshold
    ]
    flat.sort(reverse=True)
    for i, r, c in flat:
        if r in used_ref or c in used_cand:
            continue
        used_ref.add(r)
        used_cand.add(c)
        matches.append((r, c, i))
    unmatched_ref = [r for r in range(n_ref) if r not in used_ref]
    unmatched_cand = [c for c in range(n_cand) if c not in used_cand]
    return matches, unmatched_ref, unmatched_cand


def _boxes_from_result(result, score_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (xyxy, conf, cls) from an Ultralytics Results object."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
    clses = boxes.cls.detach().cpu().numpy().astype(np.int32)
    keep = confs >= float(score_threshold)
    return xyxy[keep], confs[keep], clses[keep]


def _open_pair(run_dir: Path):
    import cv2
    original = cv2.VideoCapture(str(run_dir / "original.mp4"))
    adaptive = cv2.VideoCapture(str(run_dir / "decoded_adaptive.mp4"))
    if not original.isOpened():
        raise FileNotFoundError(f"missing original.mp4 in {run_dir}")
    if not adaptive.isOpened():
        raise FileNotFoundError(f"missing decoded_adaptive.mp4 in {run_dir}")
    return original, adaptive


def _load_classes(run_dir: Path, fallback: List[str]) -> List[str]:
    cfg_path = run_dir / "session_config.json"
    if not cfg_path.exists():
        return fallback
    try:
        cfg = json.loads(cfg_path.read_text())
        classes = cfg.get("classes") or []
        return list(classes) if classes else fallback
    except Exception as e:
        log.warning("session_config.json unreadable: %s", e)
        return fallback


def evaluate(run_dir: str, model_path: str = "yolov8s-world.pt",
             device: str = "auto", iou_threshold: float = 0.5,
             score_threshold: float = 0.25,
             max_frames: Optional[int] = None,
             stride: int = 1,
             classes_override: Optional[List[str]] = None) -> Dict[str, object]:
    run_dir_p = Path(run_dir)
    original_cap, adaptive_cap = _open_pair(run_dir_p)
    classes = classes_override or _load_classes(run_dir_p, fallback=["object"])

    # Importing detection lazily keeps --help fast.
    from src.inference.detection import YoloWorldDetector
    det = YoloWorldDetector(model_path=model_path, device=device,
                            conf_threshold=score_threshold)
    det.set_classes(classes)
    det.warmup()

    import cv2
    preserved_fractions: List[float] = []
    per_frame_iou: List[float] = []
    per_frame_recall: List[float] = []
    per_frame_precision: List[float] = []
    per_class: Dict[int, Dict[str, int]] = {}
    frames_with_regression = 0
    frames_with_zero_ref = 0
    frames_compared = 0
    t_start = time.perf_counter()

    frame_idx = -1
    try:
        while True:
            ok_o, frame_o = original_cap.read()
            ok_a, frame_a = adaptive_cap.read()
            if not ok_o or not ok_a or frame_o is None or frame_a is None:
                break
            frame_idx += 1
            if max_frames is not None and frames_compared >= max_frames:
                break
            if stride > 1 and (frame_idx % stride) != 0:
                continue

            ref_xyxy, _ref_conf, ref_cls = _boxes_from_result(
                det._model.predict(frame_o, verbose=False, device=det.device,
                                   conf=score_threshold)[0],
                score_threshold,
            )
            cand_xyxy, _cand_conf, cand_cls = _boxes_from_result(
                det._model.predict(frame_a, verbose=False, device=det.device,
                                   conf=score_threshold)[0],
                score_threshold,
            )

            iou = _iou_xyxy(ref_xyxy, cand_xyxy)
            matches, unmatched_ref, unmatched_cand = _greedy_match(
                iou, iou_threshold,
            )

            n_ref = ref_xyxy.shape[0]
            n_cand = cand_xyxy.shape[0]
            n_match = len(matches)

            if n_ref == 0:
                # Nothing to preserve → perfect score, but precision
                # still meaningful if the candidate hallucinated.
                preserved_fractions.append(1.0)
                per_frame_recall.append(1.0)
                per_frame_precision.append(1.0 if n_cand == 0 else 0.0)
                frames_with_zero_ref += 1
            else:
                pres = n_match / float(n_ref)
                preserved_fractions.append(pres)
                per_frame_recall.append(pres)
                per_frame_precision.append(
                    n_match / float(n_cand) if n_cand > 0 else 0.0
                )
                if pres < 1.0:
                    frames_with_regression += 1
                # Per-class breakdown (ref side).
                for r, _c, _i in matches:
                    cls_id = int(ref_cls[r])
                    d = per_class.setdefault(
                        cls_id, {"matched": 0, "ref_total": 0})
                    d["matched"] += 1
                for r in range(n_ref):
                    cls_id = int(ref_cls[r])
                    d = per_class.setdefault(
                        cls_id, {"matched": 0, "ref_total": 0})
                    d["ref_total"] += 1

            if matches:
                per_frame_iou.append(
                    float(np.mean([m[2] for m in matches]))
                )
            frames_compared += 1
    finally:
        original_cap.release()
        adaptive_cap.release()

    wall = time.perf_counter() - t_start

    def _safe_mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    per_class_out: Dict[str, Dict[str, float]] = {}
    for cls_id, d in per_class.items():
        name = classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)
        rate = (d["matched"] / d["ref_total"]) if d["ref_total"] > 0 else 0.0
        per_class_out[name] = {
            "matched": int(d["matched"]),
            "ref_total": int(d["ref_total"]),
            "preserved_fraction": float(rate),
        }

    out = {
        "run_dir": str(run_dir_p),
        "model": model_path,
        "device": det.device,
        "classes": classes,
        "iou_threshold": float(iou_threshold),
        "score_threshold": float(score_threshold),
        "stride": int(stride),
        "frames_compared": int(frames_compared),
        "frames_with_zero_ref": int(frames_with_zero_ref),
        "frames_with_regression": int(frames_with_regression),
        "mean_preserved_fraction": _safe_mean(preserved_fractions),
        "mean_recall": _safe_mean(per_frame_recall),
        "mean_precision": _safe_mean(per_frame_precision),
        "mean_matched_iou": _safe_mean(per_frame_iou),
        "per_class": per_class_out,
        "eval_wall_sec": round(wall, 3),
    }
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline detection-preservation eval for an EchoStream run.",
    )
    p.add_argument("--run-dir", required=True,
                   help="Session directory produced by --save-artifacts.")
    p.add_argument("--model", default="yolov8s-world.pt")
    p.add_argument("--device", default="auto",
                   choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.25,
                   help="Per-box confidence filter applied to both videos.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap the number of frames compared.")
    p.add_argument("--stride", type=int, default=1,
                   help="Only compare every Nth frame (useful for long runs).")
    p.add_argument("--classes", default=None,
                   help="Override class list (comma-separated). "
                        "Defaults to session_config.json.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    classes_override: Optional[List[str]] = None
    if args.classes:
        classes_override = [c.strip() for c in args.classes.split(",") if c.strip()]

    result = evaluate(
        run_dir=args.run_dir, model_path=args.model, device=args.device,
        iou_threshold=args.iou_threshold, score_threshold=args.score_threshold,
        max_frames=args.max_frames, stride=args.stride,
        classes_override=classes_override,
    )
    out_path = Path(args.run_dir) / "detection_preservation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info(
        "detection preservation: frames=%d preserved=%.3f recall=%.3f "
        "precision=%.3f mean_iou=%.3f regressions=%d → %s",
        result["frames_compared"],
        result["mean_preserved_fraction"],
        result["mean_recall"],
        result["mean_precision"],
        result["mean_matched_iou"],
        result["frames_with_regression"],
        out_path,
    )


if __name__ == "__main__":
    main()
