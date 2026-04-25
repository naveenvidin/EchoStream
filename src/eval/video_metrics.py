"""Bitrate / bandwidth / stability evaluation helpers.

Two sources of truth for bytes-per-second:

1. **Live, authoritative:** the H.264 backend reports exact encoded packet
   bytes; the camera accumulates them in metrics.csv (`encoded_bytes`
   column). Summing that column gives the bits actually sent on the wire.
2. **File-size based:** mp4 container size / duration. This includes
   container overhead, so it is slightly higher than (1) for the adaptive
   stream, and is the *only* source for reference streams like
   `original.mp4` / `masked.mp4` that were written by `cv2.VideoWriter`.

`summarize_run(run_dir)` folds both together, then layers on:
- Latency percentiles (p50/p95) for processing_time, server decode, server
  inference, and end-to-end per-frame encoded-bytes.
- Stability counters: CRF transitions observed, zero-detection ratio,
  mean/std ROI occupancy.

These extras make two runs meaningfully comparable — average bitrate
alone hides tail latency and controller chatter.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


@dataclass
class BitrateSummary:
    file_path: str
    size_bytes: int
    duration_sec: float
    avg_bitrate_bps: float


@dataclass
class RunSummary:
    run_dir: str
    classes: List[str] = field(default_factory=list)
    frames_processed: int = 0
    wall_clock_sec: float = 0.0

    # File-size bitrates for each saved artifact
    original: Optional[BitrateSummary] = None
    masked: Optional[BitrateSummary] = None
    adaptive: Optional[BitrateSummary] = None
    recorded_input: Optional[BitrateSummary] = None

    # Authoritative pipeline stats (from metrics.csv)
    adaptive_encoded_bytes_live: int = 0
    adaptive_avg_bitrate_bps_live: float = 0.0
    masked_vs_original_savings_pct: Optional[float] = None
    adaptive_vs_original_savings_pct: Optional[float] = None

    # Time-series derived (means)
    mean_roi_ratio: Optional[float] = None
    std_roi_ratio: Optional[float] = None
    mean_conf_metric: Optional[float] = None
    total_detections: int = 0

    # Distribution stats for the per-frame conf_metric (mean/median/min/p5/p95
    # plus fraction-of-frames-below-threshold). Powers the "Detector
    # confidence over time" panel in the Streamlit evaluator — the whole
    # point of the adaptive pipeline is preserving these numbers under
    # compression, so they get their own block instead of just `mean_*`.
    confidence_summary: Dict[str, Any] = field(default_factory=dict)

    # Stability / control-loop quality
    zero_detection_ratio: Optional[float] = None
    crf_transition_count: int = 0
    restart_count: int = 0

    # Pipeline health counters (from PipelineCounters.summary_dict merged in)
    health: Dict[str, Any] = field(default_factory=dict)

    # Detection preservation (populated by offline eval if available)
    detection_preservation: Optional[Dict[str, Any]] = None

    # Latency percentiles (milliseconds unless noted)
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ── File-size based bitrate ──────────────────────────────────────────────────

def probe_duration_ffprobe(path: str) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet",
             "-select_streams", "v:0",
             "-show_entries", "stream=duration",
             "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode("utf-8").strip()
        return float(out) if out else None
    except Exception:
        return None


def probe_duration_cv2(path: str) -> Optional[float]:
    """Pure-Python fallback using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return (frames / fps) if fps > 0 else None
    finally:
        cap.release()


def bitrate_of_file(path: str) -> Optional[BitrateSummary]:
    if not os.path.isfile(path):
        return None
    size_bytes = os.path.getsize(path)
    duration = probe_duration_ffprobe(path) or probe_duration_cv2(path)
    if not duration or duration <= 0:
        return BitrateSummary(
            file_path=str(path), size_bytes=size_bytes,
            duration_sec=0.0, avg_bitrate_bps=0.0,
        )
    return BitrateSummary(
        file_path=str(path), size_bytes=size_bytes,
        duration_sec=float(duration),
        avg_bitrate_bps=(size_bytes * 8) / float(duration),
    )


# ── Percentile helper ────────────────────────────────────────────────────────

def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _pct_block(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0,
                "p99": 0.0, "max": 0.0}
    sv = sorted(vals)
    return {
        "count": len(sv),
        "mean": float(mean(sv)),
        "p50": _percentile(sv, 50),
        "p95": _percentile(sv, 95),
        "p99": _percentile(sv, 99),
        "max": float(sv[-1]),
    }


# ── Live-stream stats (from metrics.csv) ─────────────────────────────────────

def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def accumulate_live_stats(metrics_csv: str) -> Dict[str, Any]:
    """Walk the per-frame CSV and return aggregate pipeline stats."""
    stats: Dict[str, Any] = {
        "frames": 0,
        "encoded_bytes_total": 0,
        "wall_clock_sec": 0.0,
        "mean_roi_ratio": None,
        "std_roi_ratio": None,
        "mean_conf_metric": None,
        "total_detections": 0,
        "zero_detection_ratio": None,
        "crf_transition_count": 0,
        "restart_count": 0,
        "percentiles": {},
    }
    if not os.path.isfile(metrics_csv):
        return stats

    import csv
    rois: List[float] = []
    confs: List[float] = []
    proc_ms: List[float] = []
    infer_us: List[float] = []
    decode_us: List[float] = []
    enc_bytes: List[float] = []
    send_ms_list: List[float] = []
    recv_ms_list: List[float] = []
    flow_ms_list: List[float] = []
    encode_ms_list: List[float] = []
    e2e_ms_list: List[float] = []
    zero_det_frames = 0
    crf_transitions = 0
    restart_cnt = 0
    det_total = 0
    total_bytes = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    frames = 0

    with open(metrics_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            frames += 1
            b = _to_int(row.get("encoded_bytes")) or 0
            total_bytes += b
            enc_bytes.append(float(b))

            roi = _to_float(row.get("roi_ratio"))
            if roi is not None:
                rois.append(roi)
            conf = _to_float(row.get("conf_metric"))
            if conf is not None:
                confs.append(conf)
            d = _to_int(row.get("num_detections")) or 0
            det_total += d
            if d == 0:
                zero_det_frames += 1

            pm = _to_float(row.get("processing_time_ms"))
            if pm is not None:
                proc_ms.append(pm)
            iu = _to_float(row.get("infer_us_server"))
            if iu is not None:
                infer_us.append(iu)
            du = _to_float(row.get("decode_us_server"))
            if du is not None:
                decode_us.append(du)
            sm = _to_float(row.get("send_ms"))
            if sm is not None:
                send_ms_list.append(sm)
            rm = _to_float(row.get("recv_ms"))
            if rm is not None:
                recv_ms_list.append(rm)
            fm = _to_float(row.get("flow_ms"))
            if fm is not None:
                flow_ms_list.append(fm)
            em = _to_float(row.get("encode_ms"))
            if em is not None:
                encode_ms_list.append(em)
            e2e = _to_float(row.get("end_to_end_loop_ms"))
            if e2e is not None:
                e2e_ms_list.append(e2e)

            ts = _to_float(row.get("timestamp_sec"))
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            ct = _to_int(row.get("crf_transition_count"))
            if ct is not None:
                crf_transitions = max(crf_transitions, ct)
            rc = _to_int(row.get("restart_count"))
            if rc is not None:
                restart_cnt = max(restart_cnt, rc)

    stats["frames"] = frames
    stats["encoded_bytes_total"] = total_bytes
    if first_ts is not None and last_ts is not None and last_ts > first_ts:
        stats["wall_clock_sec"] = last_ts - first_ts
    if rois:
        stats["mean_roi_ratio"] = float(mean(rois))
        stats["std_roi_ratio"] = float(pstdev(rois)) if len(rois) > 1 else 0.0
    if confs:
        stats["mean_conf_metric"] = float(mean(confs))
        sv = sorted(confs)
        n = len(sv)
        mid = n // 2
        median = float(sv[mid] if n % 2 else 0.5 * (sv[mid - 1] + sv[mid]))
        stats["confidence_summary"] = {
            "count": n,
            "mean": float(mean(confs)),
            "median": median,
            "min": float(sv[0]),
            "max": float(sv[-1]),
            "p5": _percentile(sv, 5),
            "p95": _percentile(sv, 95),
            "frac_below_0_3": sum(1 for x in confs if x < 0.3) / n,
            "frac_below_0_5": sum(1 for x in confs if x < 0.5) / n,
        }
    else:
        stats["confidence_summary"] = {}
    stats["total_detections"] = det_total
    if frames:
        stats["zero_detection_ratio"] = zero_det_frames / float(frames)
    stats["crf_transition_count"] = crf_transitions
    stats["restart_count"] = restart_cnt
    stats["percentiles"] = {
        "processing_time_ms": _pct_block(proc_ms),
        "infer_us_server": _pct_block(infer_us),
        "decode_us_server": _pct_block(decode_us),
        "encoded_bytes": _pct_block(enc_bytes),
        "send_ms": _pct_block(send_ms_list),
        "recv_ms": _pct_block(recv_ms_list),
        "flow_ms": _pct_block(flow_ms_list),
        "encode_ms": _pct_block(encode_ms_list),
        "end_to_end_loop_ms": _pct_block(e2e_ms_list),
    }
    return stats


# ── Run-level summary ────────────────────────────────────────────────────────

def _savings_pct(candidate: Optional[BitrateSummary],
                 baseline: Optional[BitrateSummary]) -> Optional[float]:
    if not candidate or not baseline or baseline.size_bytes <= 0:
        return None
    saved = baseline.size_bytes - candidate.size_bytes
    return 100.0 * saved / baseline.size_bytes


def summarize_run(run_dir: str) -> RunSummary:
    run_dir_p = Path(run_dir)
    original = bitrate_of_file(str(run_dir_p / "original.mp4"))
    masked = bitrate_of_file(str(run_dir_p / "masked.mp4"))
    adaptive = bitrate_of_file(str(run_dir_p / "decoded_adaptive.mp4"))

    # Recorded-input mp4 is written by RawInputRecorder when --record-input
    # was used. It lives at run_dir/raw_recorded_input.mp4 by default, but
    # we also honour the path stored in session_config.json if present.
    rec_path = run_dir_p / "raw_recorded_input.mp4"
    recorded_input = bitrate_of_file(str(rec_path)) if rec_path.exists() else None

    live = accumulate_live_stats(str(run_dir_p / "metrics.csv"))
    wall = float(live["wall_clock_sec"] or 0.0)
    bytes_live = int(live["encoded_bytes_total"] or 0)
    live_bps = (bytes_live * 8 / wall) if wall > 0 else 0.0

    classes: List[str] = []
    health: Dict[str, Any] = {}
    cfg_path = run_dir_p / "session_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            classes = list(cfg.get("classes") or [])
            # session_config is also where the live PipelineCounters
            # dump lands if the camera side writes it there. No-op if
            # the key is missing.
            health = dict(cfg.get("pipeline_health") or {})
            rec_cfg = cfg.get("recorded_input") or {}
            if recorded_input is None and rec_cfg.get("file_path"):
                # Follow explicit path from session config if default
                # location is empty (user passed --record-input elsewhere).
                recorded_input = bitrate_of_file(rec_cfg["file_path"])
        except Exception:
            pass

    # Detection preservation is written to a sibling JSON by the
    # offline eval (src/eval/detection_preservation.py). Fold it in
    # opportunistically so summary.json is a single point of truth.
    det_pres: Optional[Dict[str, Any]] = None
    det_path = run_dir_p / "detection_preservation.json"
    if det_path.exists():
        try:
            det_pres = json.loads(det_path.read_text())
        except Exception:
            det_pres = None

    return RunSummary(
        run_dir=str(run_dir_p),
        classes=classes,
        frames_processed=int(live["frames"] or 0),
        wall_clock_sec=wall,
        original=original,
        masked=masked,
        adaptive=adaptive,
        recorded_input=recorded_input,
        adaptive_encoded_bytes_live=bytes_live,
        adaptive_avg_bitrate_bps_live=live_bps,
        masked_vs_original_savings_pct=_savings_pct(masked, original),
        adaptive_vs_original_savings_pct=_savings_pct(adaptive, original),
        mean_roi_ratio=live.get("mean_roi_ratio"),
        std_roi_ratio=live.get("std_roi_ratio"),
        mean_conf_metric=live.get("mean_conf_metric"),
        confidence_summary=live.get("confidence_summary") or {},
        total_detections=int(live["total_detections"] or 0),
        zero_detection_ratio=live.get("zero_detection_ratio"),
        crf_transition_count=int(live["crf_transition_count"] or 0),
        restart_count=int(live["restart_count"] or 0),
        health=health,
        detection_preservation=det_pres,
        percentiles=live.get("percentiles") or {},
    )


def write_summary(run_dir: str) -> RunSummary:
    summary = summarize_run(run_dir)
    out = Path(run_dir) / "summary.json"
    out.write_text(summary.to_json())
    return summary
