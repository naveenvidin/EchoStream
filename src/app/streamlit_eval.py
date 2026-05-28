"""Streamlit evaluation UI for EchoStream benchmark runs.

Point it at one or more directories produced by
`camera_h264 --save-artifacts --output-dir <dir>`. Each run is expected
to contain at minimum:

  <dir>/original.mp4
  <dir>/masked.mp4
  <dir>/decoded_adaptive.mp4
  <dir>/metrics.csv
  <dir>/session_config.json
  <dir>/summary.json              (optional — regenerated if missing)

Run (single run):
    streamlit run app/streamlit_eval.py -- --run-dir runs/benchmark_001

Run (multi-run A/B compare, comma-separated):
    streamlit run app/streamlit_eval.py -- \\
        --run-dir runs/adaptive_baseline,runs/adaptive_optimized

The `--` before `--run-dir` is required so Streamlit forwards the
argument to this script instead of consuming it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _trim_warmup(df, skip_first: int):
    """Drop the first `skip_first` rows; protects bandwidth charts from the
    startup outlier caused by the first metrics.csv row using dt ≈ 1e-6 s.
    See artifacts.py: the very first frame computes inst_bps with a near-zero
    dt, producing a multi-million-kbps spike that crushes the chart's y-axis.
    """
    if df is None or skip_first <= 0 or len(df) == 0:
        return df
    n = min(int(skip_first), max(0, len(df) - 1))
    return df.iloc[n:].reset_index(drop=True) if n else df


def _robust_ymax(series, percentile: float = 99.0, pad: float = 1.10) -> Optional[float]:
    """Percentile-clipped upper bound for a numeric series."""
    try:
        import numpy as np
        vals = np.asarray(series, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.percentile(vals, percentile) * pad)
    except Exception:
        return None


def _parse_run_dirs() -> List[str]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--run-dir", default=None)
    args, _ = p.parse_known_args()
    raw = args.run_dir or os.environ.get("ECHOSTREAM_RUN_DIR", "")
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        try:
            # repo root = parents[2] when app/streamlit_eval.py is the script
            here = Path(__file__).resolve()
            sys.path.insert(0, str(here.parents[1]))
            from src.eval.video_metrics import write_summary
            write_summary(str(run_dir))
        except Exception as e:
            return {"error": f"summary generation failed: {e!r}"}
    try:
        return json.loads(summary_path.read_text())
    except Exception as e:
        return {"error": f"summary.json unreadable: {e!r}"}


def _fmt_bitrate(bps) -> str:
    if not bps:
        return "—"
    try:
        bps = float(bps)
    except Exception:
        return "—"
    if bps >= 1_000_000:
        return f"{bps/1_000_000:.2f} Mbps"
    return f"{bps/1_000:.1f} kbps"


def _fmt_bytes(n) -> str:
    try:
        n = int(n)
    except Exception:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_ms(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "—"
    if x >= 1000:
        return f"{x/1000:.2f} s"
    return f"{x:.1f} ms"


def _fmt_us(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "—"
    if x >= 1_000_000:
        return f"{x/1_000_000:.2f} s"
    if x >= 1_000:
        return f"{x/1_000:.1f} ms"
    return f"{x:.0f} µs"


def _fmt_pct(v, decimals: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def _fmt_ratio_pct(v, decimals: int = 1) -> str:
    """`v` already in 0-100 percent (e.g. from savings fields)."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}%"
    except Exception:
        return "—"


def _short_sha(sha) -> str:
    if not sha:
        return "—"
    s = str(sha)
    return s[:12] if len(s) > 12 else s


# ── Confidence-over-time helpers ─────────────────────────────────────────────

def _pick_x_axis(df, prefer: str = "timestamp_sec"):
    """Choose an x-axis column from a metrics DataFrame.

    Prefers a real timestamp (timestamp_sec / source_timestamp_sec) so the
    plot is legible in seconds, falls back to frame_index, and finally to
    the row index if nothing else is present.
    """
    candidates = []
    if prefer == "timestamp_sec":
        candidates = ["timestamp_sec", "source_timestamp_sec", "frame_index"]
    else:
        candidates = ["frame_index", "timestamp_sec", "source_timestamp_sec"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _video_duration_sec(path: Path) -> Optional[float]:
    """Return video duration, preferring ffprobe and falling back to OpenCV."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode("utf-8").strip()
        return float(out) if out else None
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return (frames / fps) if fps > 0 else None
    except Exception:
        return None
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _packet_bandwidth_series(path: Path, window_sec: float) -> Optional[List[Dict[str, float]]]:
    """Build a rolling kbps time series from video packet sizes via ffprobe."""
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "packet=pts_time,dts_time,size",
                "-of", "csv=p=0",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).decode("utf-8", errors="replace")
    except Exception:
        return None

    packets = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
            size = int(float(parts[-1]))
        except ValueError:
            continue
        packets.append((ts, size))
    if not packets:
        return None

    packets.sort(key=lambda item: item[0])
    series = []
    left = 0
    bytes_in_window = 0
    win = max(float(window_sec), 0.1)
    for right, (ts, size) in enumerate(packets):
        bytes_in_window += size
        while left <= right and packets[left][0] < ts - win:
            bytes_in_window -= packets[left][1]
            left += 1
        elapsed = max(ts - packets[left][0], win if right == left else 0.001)
        series.append({
            "time_sec": float(ts),
            "kbps": (bytes_in_window * 8.0) / max(elapsed, 0.001) / 1000.0,
        })
    return series


def _flat_bandwidth_series(path: Path, window_sec: float) -> Optional[List[Dict[str, float]]]:
    """Fallback when packet-level data is unavailable."""
    if not path.exists():
        return None
    duration = _video_duration_sec(path)
    if not duration or duration <= 0:
        return None
    kbps = (path.stat().st_size * 8.0) / duration / 1000.0
    step = max(float(window_sec), 1.0)
    points = []
    t = 0.0
    while t <= duration:
        points.append({"time_sec": t, "kbps": kbps})
        t += step
    if not points or points[-1]["time_sec"] < duration:
        points.append({"time_sec": duration, "kbps": kbps})
    return points


def _render_raw_masked_bandwidth_section(st, run_dir: Path, summary: dict,
                                         key_prefix: str = ""):
    """Top-level raw-vs-masked bandwidth timeline."""
    raw_path = run_dir / "original.mp4"
    masked_path = run_dir / "masked.mp4"
    if not raw_path.exists() or not masked_path.exists():
        st.info("original.mp4 and masked.mp4 are required for raw vs masked bandwidth.")
        return

    st.subheader("Raw vs Masked Bandwidth Usage Over Time")
    st.caption(
        "Raw video bandwidth is the baseline demand. Masked video bandwidth "
        "shows the effect of ROI masking and blur-based compression preparation. "
        "The gap between the curves indicates bandwidth savings over time."
    )

    ctrl, _spacer = st.columns([1, 3])
    window_sec = ctrl.slider(
        "Rolling window (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key=f"raw_masked_bw_window_{key_prefix}",
    )

    raw_series = (
        _packet_bandwidth_series(raw_path, window_sec)
        or _flat_bandwidth_series(raw_path, window_sec)
    )
    masked_series = (
        _packet_bandwidth_series(masked_path, window_sec)
        or _flat_bandwidth_series(masked_path, window_sec)
    )
    if not raw_series or not masked_series:
        st.warning("Could not derive raw/masked bandwidth over time from the videos.")
        return

    avg_raw = (summary.get("original") or {}).get("avg_bitrate_bps") or 0
    avg_masked = (summary.get("masked") or {}).get("avg_bitrate_bps") or 0
    savings = None
    if avg_raw:
        savings = 100.0 * (float(avg_raw) - float(avg_masked or 0)) / float(avg_raw)

    m1, m2, m3 = st.columns(3)
    m1.metric("Average raw bandwidth", _fmt_bitrate(avg_raw))
    m2.metric("Average masked bandwidth", _fmt_bitrate(avg_masked))
    m3.metric("Average savings", f"{savings:.1f}%" if savings is not None else "??")

    try:
        import altair as alt
        import pandas as pd
        plot_df = pd.concat([
            pd.DataFrame(raw_series).assign(stream="Raw video"),
            pd.DataFrame(masked_series).assign(stream="Masked video"),
        ], ignore_index=True)
        chart = (
            alt.Chart(plot_df)
            .mark_line()
            .encode(
                x=alt.X("time_sec:Q", title="Time (s)"),
                y=alt.Y("kbps:Q", title="Bandwidth (kbps)", scale=alt.Scale(zero=False)),
                color=alt.Color("stream:N", title="Stream"),
                tooltip=[
                    alt.Tooltip("stream:N", title="Stream"),
                    alt.Tooltip("time_sec:Q", title="Time (s)", format=".2f"),
                    alt.Tooltip("kbps:Q", title="Bandwidth (kbps)", format=".1f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        import pandas as pd
        raw_df = pd.DataFrame(raw_series).set_index("time_sec")
        masked_df = pd.DataFrame(masked_series).set_index("time_sec")
        chart_df = pd.DataFrame({
            "Raw video": raw_df["kbps"],
            "Masked video": masked_df["kbps"],
        })
        chart_df.index.name = "Time (s)"
        st.line_chart(chart_df)


def _render_top_confidence_over_time(st, df, summary, key_prefix: str = ""):
    """Prominent, top-of-page confidence-over-time chart.

    Pulls `conf_metric` from metrics.csv (the actual confidence score
    returned by the server for each segment) and plots it against time.
    Same data as the deeper confidence panel below — duplicated up top so
    the two headline tracking signals (confidence + bandwidth) are
    visible without scrolling, for both single-run and replay analysis.
    """
    if "conf_metric" not in df.columns:
        st.info("conf_metric column missing from metrics.csv — "
                "confidence-over-time chart unavailable.")
        return

    st.subheader("Confidence score over time")
    st.caption(
        "Per-frame detector confidence reported by the server. Higher is "
        "better. For live runs this tracks how detection quality moves "
        "as the scene changes; for replay runs it shows how the adaptive "
        "encoder's CRF choices affected detection on the same input."
    )

    conf_summary = summary.get("confidence_summary") or {}
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Mean", f"{float(conf_summary.get('mean') or 0):.3f}")
    cs2.metric("Median", f"{float(conf_summary.get('median') or 0):.3f}")
    cs3.metric(
        "p5 / p95",
        f"{float(conf_summary.get('p5') or 0):.3f} / "
        f"{float(conf_summary.get('p95') or 0):.3f}",
    )
    cs4.metric("Frames < 0.5",
               _fmt_pct(conf_summary.get("frac_below_0_5")))

    cc1, cc2 = st.columns([1, 3])
    smooth_win = cc1.slider(
        "Smoothing (frames)",
        min_value=1, max_value=101, value=15, step=2,
        key=f"top_conf_smooth_{key_prefix}",
    )
    show_raw = cc2.checkbox(
        "Show raw curve alongside smoothed",
        value=True, key=f"top_conf_show_raw_{key_prefix}",
    )

    x_col = _pick_x_axis(df, prefer="timestamp_sec")
    if x_col is None:
        st.warning("No timestamp_sec or frame_index column to plot against.")
        return

    import pandas as pd
    series = {
        "smoothed": (
            df["conf_metric"].astype(float)
              .rolling(window=int(smooth_win), min_periods=1, center=True)
              .mean()
              .values
        ),
    }
    if show_raw:
        series["raw"] = df["conf_metric"].astype(float).values
    chart_df = pd.DataFrame(series, index=df[x_col].values)
    chart_df.index.name = (
        "Time (s)" if x_col in ("timestamp_sec", "source_timestamp_sec")
        else "Frame index"
    )
    st.line_chart(chart_df)


def _render_top_bandwidth_over_time(st, df, summary, key_prefix: str = ""):
    """Prominent, top-of-page bandwidth-consumption-over-time chart.

    Uses `instantaneous_bitrate_bps` and `cumulative_encoded_bytes`
    from metrics.csv — the actual on-wire bandwidth produced by the
    adaptive encoder, not just raw vs masked video sizes. This is the
    quantity the camera/server link actually consumes, so it is the
    headline cost signal for live and replay runs alike.
    """
    if "instantaneous_bitrate_bps" not in df.columns:
        st.info("instantaneous_bitrate_bps column missing from "
                "metrics.csv — bandwidth-over-time chart unavailable.")
        return

    st.subheader("Bandwidth consumption over time")
    st.caption(
        "Actual on-wire bandwidth produced by the adaptive encoder, "
        "in kbps. Peaks line up with high-information segments; troughs "
        "line up with idle / heavily masked segments. The cumulative "
        "panel shows total bytes sent so far — useful for sanity-checking "
        "long replays against expected data budgets."
    )

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Live avg bitrate",
               _fmt_bitrate(summary.get("adaptive_avg_bitrate_bps_live")))
    bm2.metric("Live total bytes",
               _fmt_bytes(summary.get("adaptive_encoded_bytes_live")))
    bm3.metric("Adaptive vs original savings",
               f"{(summary.get('adaptive_vs_original_savings_pct') or 0):.1f}%")
    bm4.metric("Frames",
               int(summary.get("frames_processed") or 0))

    cc1, cc2, cc3 = st.columns([1, 1, 2])
    smooth_win = cc1.slider(
        "Smoothing (frames)",
        min_value=1, max_value=101, value=15, step=2,
        key=f"top_bw_smooth_{key_prefix}",
    )
    skip_first = cc2.slider(
        "Skip first samples",
        min_value=0, max_value=20, value=1, step=1,
        key=f"top_bw_skip_{key_prefix}",
        help="Drops the warmup row(s); the first metrics.csv row has an "
             "inflated bitrate from a ~µs dt at startup.",
    )
    show_raw = cc3.checkbox(
        "Show raw curve alongside smoothed",
        value=False, key=f"top_bw_show_raw_{key_prefix}",
        help="Per-frame bitrate is noisy — smoothed view is usually clearer.",
    )

    df_plot = _trim_warmup(df, skip_first)
    x_col = _pick_x_axis(df_plot, prefer="timestamp_sec")
    if x_col is None or len(df_plot) == 0:
        st.warning("No timestamp_sec or frame_index column to plot against.")
        return

    import pandas as pd
    bitrate_kbps = df_plot["instantaneous_bitrate_bps"].astype(float) / 1000.0
    smoothed = bitrate_kbps.rolling(
        window=int(smooth_win), min_periods=1, center=True
    ).mean()
    series = {"smoothed (kbps)": smoothed.values}
    if show_raw:
        series["raw (kbps)"] = bitrate_kbps.values
    chart_df = pd.DataFrame(series, index=df_plot[x_col].values)
    chart_df.index.name = (
        "Time (s)" if x_col in ("timestamp_sec", "source_timestamp_sec")
        else "Frame index"
    )
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.line_chart(chart_df)
    with right_col:
        st.empty()

    if "cumulative_encoded_bytes" in df.columns:
        cum_kb = (
            df["cumulative_encoded_bytes"].astype(float) / 1024.0
        )
        cum_df = pd.DataFrame(
            {"cumulative (KB)": cum_kb.values},
            index=df[x_col].values,
        )
        cum_df.index.name = chart_df.index.name
        st.markdown("**Cumulative bandwidth consumed**")
        st.line_chart(cum_df)


def _render_baseline_vs_echostream(st, run_dir: Path, df, key_prefix: str = ""):
    """Headline A/B chart: baseline (raw + fixed CRF) vs EchoStream (masked + adaptive).

    Reads `baseline_metrics.csv` if present (written by camera_h264.py when
    baseline_enabled and --save-artifacts are both on). Overlays its bandwidth
    and confidence curves against the masked side from `metrics.csv`. The
    intended takeaway: confidence drops only a little, bandwidth savings are
    much larger.

    Warmup rows are trimmed and the y-axis is percentile-clipped because the
    first row of metrics.csv reports inst_bps divided by dt ≈ 1e-6 s
    (artifact of the producer's per-row dt) — a multi-million-kbps point
    that would otherwise flatten the chart.
    """
    baseline_csv = run_dir / "baseline_metrics.csv"
    if not baseline_csv.exists() or df is None or len(df) == 0:
        return

    import pandas as pd
    try:
        bdf = pd.read_csv(baseline_csv)
    except Exception as e:
        st.warning(f"baseline_metrics.csv unreadable: {e}")
        return
    if len(bdf) == 0:
        return

    st.markdown("### Baseline vs StreamSense — headline A/B")
    st.caption(
        "Same source frames sent through two parallel paths: **baseline** "
        "(unmasked + fixed CRF) and **EchoStream** (motion-masked + adaptive CRF). "
        "Look for: confidence curves close together, bandwidth curves far apart."
    )

    # ── display controls ────────────────────────────────────────────────────
    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 1])
    smooth = cc1.slider(
        "Smoothing (samples)",
        min_value=1, max_value=101, value=15, step=2,
        key=f"ab_smooth_{key_prefix}",
    )
    skip_first = cc2.slider(
        "Skip first samples",
        min_value=0, max_value=20, value=1, step=1,
        key=f"ab_skip_{key_prefix}",
        help="Drops the warmup row(s); the first metrics.csv row has an "
             "inflated bitrate from a ~µs dt at startup.",
    )
    auto_refresh = cc3.checkbox(
        "Live (auto-refresh)", value=False,
        key=f"ab_live_{key_prefix}",
        help="Re-read CSVs every few seconds for a near-real-time view.",
    )
    refresh_sec = cc4.slider(
        "Refresh interval (s)",
        min_value=1, max_value=30, value=3, step=1,
        key=f"ab_refresh_{key_prefix}",
        disabled=not auto_refresh,
    )

    df_plot = _trim_warmup(df, skip_first)
    bdf_plot = _trim_warmup(bdf, skip_first)

    # Defensive: coerce every numeric column the A/B panel touches so a
    # malformed cell (e.g. "False" landing in encoded_bytes from a column-
    # misaligned row written by concurrent CSV writers) becomes NaN rather
    # than crashing .mean() / .astype(float) downstream. Clean runs are
    # unaffected — pd.to_numeric on already-numeric columns is a no-op.
    _AB_NUMERIC_COLS = (
        "conf_metric", "encoded_bytes", "fps", "instantaneous_bitrate_bps",
        "cumulative_encoded_bytes", "crf", "timestamp_sec",
    )
    def _coerce_numeric(frame):
        if frame is None or len(frame) == 0:
            return frame
        frame = frame.copy()
        for col in _AB_NUMERIC_COLS:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame
    df_plot = _coerce_numeric(df_plot)
    bdf_plot = _coerce_numeric(bdf_plot)

    # EchoStream's `instantaneous_bitrate_bps` column was historically inflated
    # ~30-80× because artifacts.py divided by wall-clock dt between log_frame()
    # calls (which collapses to ~1 ms inside a GOP burst) instead of the
    # per-frame period 1/fps. Recompute kbps here from encoded_bytes * 8 * fps
    # so legacy CSVs render correctly. New CSVs (post-fix) report the same
    # value in their inst_bps column, so this stays consistent either way.
    def _echo_kbps(frame):
        if (
            "encoded_bytes" in frame.columns
            and "fps" in frame.columns
            and len(frame)
        ):
            return (
                pd.to_numeric(frame["encoded_bytes"], errors="coerce")
                * 8.0
                * pd.to_numeric(frame["fps"], errors="coerce")
                / 1000.0
            )
        if "instantaneous_bitrate_bps" in frame.columns:
            return pd.to_numeric(frame["instantaneous_bitrate_bps"], errors="coerce") / 1000.0
        return None

    es_kbps_series = _echo_kbps(df_plot)
    # Baseline writer in camera_h264.py is already correct (segment-duration
    # based), so its column is trustworthy.
    bl_kbps_series = (
        pd.to_numeric(bdf_plot["instantaneous_bitrate_bps"], errors="coerce") / 1000.0
        if "instantaneous_bitrate_bps" in bdf_plot.columns and len(bdf_plot) else None
    )

    # ── headline summary cards (computed on trimmed data) ────────────────────
    es_mean_conf = (
        float(df_plot["conf_metric"].mean())
        if "conf_metric" in df_plot.columns and len(df_plot) else 0.0
    )
    bl_mean_conf = (
        float(bdf_plot["conf_metric"].mean())
        if "conf_metric" in bdf_plot.columns and len(bdf_plot) else 0.0
    )
    es_avg_bps = (
        float(es_kbps_series.mean()) * 1000.0
        if es_kbps_series is not None and len(es_kbps_series) else 0.0
    )
    bl_avg_bps = (
        float(bl_kbps_series.mean()) * 1000.0
        if bl_kbps_series is not None and len(bl_kbps_series) else 0.0
    )
    conf_delta_pct = (
        (es_mean_conf - bl_mean_conf) / bl_mean_conf * 100.0
        if bl_mean_conf > 1e-6 else 0.0
    )
    bw_saved_pct = (
        (bl_avg_bps - es_avg_bps) / bl_avg_bps * 100.0
        if bl_avg_bps > 1e-6 else 0.0
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline mean conf", f"{bl_mean_conf:.3f}")
    m2.metric("StreamSense mean conf", f"{es_mean_conf:.3f}",
              f"{conf_delta_pct:+.1f}%")
    m3.metric("Baseline avg bandwidth", _fmt_bitrate(bl_avg_bps))
    m4.metric("StreamSense avg bandwidth", _fmt_bitrate(es_avg_bps),
              f"saved {bw_saved_pct:.1f}%")

    # ── combined dual-axis chart: bandwidth (left) + confidence (right) ──────
    have_bw = (
        es_kbps_series is not None
        and bl_kbps_series is not None
        and "timestamp_sec" in df_plot.columns
        and "timestamp_sec" in bdf_plot.columns
        and len(df_plot) and len(bdf_plot)
    )
    have_conf = (
        "conf_metric" in df_plot.columns
        and "timestamp_sec" in df_plot.columns
        and "conf_metric" in bdf_plot.columns
        and "timestamp_sec" in bdf_plot.columns
        and len(df_plot) and len(bdf_plot)
    )

    if have_bw or have_conf:
        st.markdown(
            "**Bandwidth vs Confidence over time — baseline vs StreamSense**"
        )
        st.caption(
            "Left axis = bandwidth (kbps), right axis = confidence (0–1). "
            "Each of the four lines has its own color (see legend). "
            "Win condition: bandwidth lines far apart, confidence lines close together."
        )

        # Color mapping per series. Distinct hues for all four lines so the
        # chart is readable at a glance without relying on line styling.
        COLORS = {
            "Baseline bandwidth":    "#d62728",  # red
            "StreamSense bandwidth": "#1f77b4",  # blue
            "Baseline confidence":   "#ff7f0e",  # orange
            "StreamSense confidence":"#2ca02c",  # green
        }
        SERIES_DOMAIN = list(COLORS.keys())
        SERIES_RANGE = [COLORS[s] for s in SERIES_DOMAIN]

        # Bandwidth: use field name `kbps` (own field, never shared with conf
        # — this is what makes `resolve_scale(y='independent')` actually keep
        # the two y-axes separate; sharing a field name silently merges the
        # scales and collapses confidence onto the bandwidth axis).
        bw_df = None
        bw_ymax = None
        if have_bw:
            es_bw = pd.DataFrame({
                "time_sec": df_plot["timestamp_sec"].astype(float).values,
                "kbps": es_kbps_series.values,
                "series": "StreamSense bandwidth",
            })
            bl_bw = pd.DataFrame({
                "time_sec": bdf_plot["timestamp_sec"].astype(float).values,
                "kbps": bl_kbps_series.values,
                "series": "Baseline bandwidth",
            })
            if smooth > 1:
                es_bw["kbps"] = (
                    es_bw["kbps"].rolling(window=int(smooth), min_periods=1, center=True).mean()
                )
                bl_bw["kbps"] = (
                    bl_bw["kbps"].rolling(window=int(smooth), min_periods=1, center=True).mean()
                )
            bw_df = pd.concat([bl_bw, es_bw], ignore_index=True)
            bw_ymax = _robust_ymax(bw_df["kbps"], percentile=99.0, pad=1.10)

        # Confidence: own field name `conf`, kept distinct from `kbps`.
        conf_df = None
        if have_conf:
            es_c = pd.DataFrame({
                "time_sec": df_plot["timestamp_sec"].astype(float).values,
                "conf": df_plot["conf_metric"].astype(float).values,
                "series": "StreamSense confidence",
            })
            bl_c = pd.DataFrame({
                "time_sec": bdf_plot["timestamp_sec"].astype(float).values,
                "conf": bdf_plot["conf_metric"].astype(float).values,
                "series": "Baseline confidence",
            })
            if smooth > 1:
                es_c["conf"] = (
                    es_c["conf"].rolling(window=int(smooth), min_periods=1, center=True).mean()
                )
                bl_c["conf"] = (
                    bl_c["conf"].rolling(window=int(smooth), min_periods=1, center=True).mean()
                )
            conf_df = pd.concat([bl_c, es_c], ignore_index=True)

        try:
            import altair as alt
            # One shared color scale across both layers — Altair will merge
            # them into a single legend with all four series.
            series_color = alt.Color(
                "series:N", title="Series",
                scale=alt.Scale(domain=SERIES_DOMAIN, range=SERIES_RANGE),
                legend=alt.Legend(orient="top", columns=4),
            )

            layers = []
            if bw_df is not None:
                bw_scale = (
                    alt.Scale(zero=True) if bw_ymax is None
                    else alt.Scale(domain=[0, bw_ymax], clamp=True)
                )
                bw_layer = (
                    alt.Chart(bw_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (s)"),
                        y=alt.Y(
                            "kbps:Q",
                            title="Bandwidth (kbps)",
                            scale=bw_scale,
                            axis=alt.Axis(orient="left"),
                        ),
                        color=series_color,
                        tooltip=[
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("time_sec:Q", title="Time (s)", format=".2f"),
                            alt.Tooltip("kbps:Q", title="kbps", format=".1f"),
                        ],
                    )
                )
                layers.append(bw_layer)

            if conf_df is not None:
                conf_layer = (
                    alt.Chart(conf_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (s)"),
                        y=alt.Y(
                            "conf:Q",
                            title="Confidence score (0–1)",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(orient="right"),
                        ),
                        color=series_color,
                        tooltip=[
                            alt.Tooltip("series:N", title="Series"),
                            alt.Tooltip("time_sec:Q", title="Time (s)", format=".2f"),
                            alt.Tooltip("conf:Q", title="Confidence", format=".3f"),
                        ],
                    )
                )
                layers.append(conf_layer)

            if layers:
                chart = (
                    alt.layer(*layers)
                    .resolve_scale(y="independent")
                    .properties(height=380)
                )
                st.altair_chart(chart, use_container_width=True)
        except Exception:
            # Fallback path (Altair missing). Keep two simple charts so the
            # panel never blanks out in low-dep environments.
            if bw_df is not None:
                st.markdown("_Bandwidth (kbps)_")
                st.line_chart(
                    bw_df.pivot_table(index="time_sec", columns="series", values="kbps")
                         .sort_index()
                )
            if conf_df is not None:
                st.markdown("_Confidence (0–1)_")
                st.line_chart(
                    conf_df.pivot_table(index="time_sec", columns="series", values="conf")
                          .sort_index()
                )

        # Bandwidth-only companion chart (single y-axis, no confidence).
        # Reuses the same kbps series as the combined chart so values match;
        # relabels "Baseline bandwidth" → "Raw bandwidth" for poster framing.
        if bw_df is not None and len(bw_df):
            st.markdown("**Raw vs StreamSense bandwidth over time**")
            st.caption(
                "Bandwidth only — no confidence overlay. Wider gap between "
                "the two lines = more bandwidth saved by StreamSense."
            )
            bw_only_df = bw_df.copy()
            bw_only_df["series"] = bw_only_df["series"].replace(
                {"Baseline bandwidth": "Raw bandwidth"}
            )
            BW_ONLY_COLORS = {
                "Raw bandwidth":         "#d62728",  # red
                "StreamSense bandwidth": "#1f77b4",  # blue
            }
            bw_only_domain = list(BW_ONLY_COLORS.keys())
            bw_only_range = [BW_ONLY_COLORS[s] for s in bw_only_domain]
            try:
                import altair as alt
                bw_only_scale = (
                    alt.Scale(zero=True) if bw_ymax is None
                    else alt.Scale(domain=[0, bw_ymax], clamp=True)
                )
                bw_only_chart = (
                    alt.Chart(bw_only_df)
                    .mark_line(strokeWidth=2)
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (s)"),
                        y=alt.Y("kbps:Q", title="Bandwidth (kbps)", scale=bw_only_scale),
                        color=alt.Color(
                            "series:N", title="Stream",
                            scale=alt.Scale(domain=bw_only_domain, range=bw_only_range),
                            legend=alt.Legend(orient="top"),
                        ),
                        tooltip=[
                            alt.Tooltip("series:N", title="Stream"),
                            alt.Tooltip("time_sec:Q", title="Time (s)", format=".2f"),
                            alt.Tooltip("kbps:Q", title="kbps", format=".1f"),
                        ],
                    )
                    .properties(width=900, height=300)
                )
                st.altair_chart(bw_only_chart, use_container_width=False)
            except Exception:
                st.line_chart(
                    bw_only_df.pivot_table(index="time_sec", columns="series", values="kbps")
                              .sort_index()
                )

    if auto_refresh:
        time.sleep(float(refresh_sec))
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass


def _render_confidence_section(st, df, summary, key_prefix: str = ""):
    """Confidence-over-time panel with raw/smoothed curves, threshold
    overlay, summary stats, and adjacent compression-context charts."""
    if "conf_metric" not in df.columns:
        st.info("conf_metric column missing from metrics.csv — "
                "confidence panel unavailable.")
        return

    st.subheader("Detector confidence over time")
    st.caption(
        "Confidence is the headline quality signal — bitrate savings only "
        "matter if conf_metric stays high. Watch for drops that line up "
        "with CRF rises or ROI ratio dips."
    )

    conf_summary = summary.get("confidence_summary") or {}
    cs1, cs2, cs3, cs4, cs5 = st.columns(5)
    cs1.metric("Mean", f"{float(conf_summary.get('mean') or 0):.3f}")
    cs2.metric("Median", f"{float(conf_summary.get('median') or 0):.3f}")
    cs3.metric(
        "Min / Max",
        f"{float(conf_summary.get('min') or 0):.3f} / "
        f"{float(conf_summary.get('max') or 0):.3f}",
    )
    cs4.metric(
        "p5 / p95",
        f"{float(conf_summary.get('p5') or 0):.3f} / "
        f"{float(conf_summary.get('p95') or 0):.3f}",
    )
    cs5.metric(
        "Frames < 0.5",
        _fmt_pct(conf_summary.get("frac_below_0_5")),
        f"< 0.3: {_fmt_pct(conf_summary.get('frac_below_0_3'))}",
    )

    has_ts = ("timestamp_sec" in df.columns
              or "source_timestamp_sec" in df.columns)
    cc1, cc2, cc3 = st.columns([1, 2, 2])
    x_pref = cc1.radio(
        "X axis",
        ["time (s)", "frame index"],
        horizontal=True,
        index=0 if has_ts else 1,
        key=f"xaxis_{key_prefix}",
    )
    smooth_win = cc2.slider(
        "Smoothing window (frames)",
        min_value=1, max_value=101, value=15, step=2,
        help="Rolling-mean window. 1 = raw only.",
        key=f"smooth_{key_prefix}",
    )
    threshold = cc3.slider(
        "Low-confidence threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key=f"thresh_{key_prefix}",
    )

    x_col = _pick_x_axis(
        df, prefer="timestamp_sec" if x_pref == "time (s)" else "frame_index")
    if x_col is None:
        st.warning("No timestamp_sec or frame_index column to plot against.")
        return

    import pandas as pd
    chart_df = pd.DataFrame({
        "raw": df["conf_metric"].astype(float).values,
        "smoothed": (
            df["conf_metric"].astype(float)
              .rolling(window=int(smooth_win), min_periods=1, center=True)
              .mean()
              .values
        ),
        "threshold": float(threshold),
    }, index=df[x_col].values)
    chart_df.index.name = x_col
    st.line_chart(chart_df)

    frac_below = float((df["conf_metric"].astype(float) < threshold).mean())
    st.caption(
        f"{frac_below*100:.1f}% of frames fall below the {threshold:.2f} "
        f"threshold (live recompute on this run)."
    )

    overlay_cols = [c for c in ("crf", "instantaneous_bitrate_bps", "roi_ratio")
                    if c in df.columns]
    if overlay_cols:
        with st.expander(
            "Compression context — visually correlate confidence drops "
            "with CRF / bitrate / ROI",
            expanded=False,
        ):
            tabs = st.tabs(overlay_cols)
            for tab, col in zip(tabs, overlay_cols):
                with tab:
                    st.line_chart(df.set_index(x_col)[[col]])


def _render_bitrate_vs_conf_section(st, df, key_prefix: str = ""):
    """One chart, dual y-axes: bitrate (kbps) on the left, detector
    confidence (0–1) on the right. Compression-vs-quality tradeoff at a
    glance — if conf stays flat while bitrate falls, the pipeline is doing
    its job; if conf drops every time bitrate drops, compression is too
    aggressive.
    """
    if ("conf_metric" not in df.columns
            or "instantaneous_bitrate_bps" not in df.columns):
        return

    st.subheader("Bitrate vs Detector Confidence Over Time")
    st.caption(
        "Use this chart to see whether bitrate reductions are causing "
        "confidence drops. If confidence stays stable while bitrate falls, "
        "that is a good result — the adaptive pipeline is preserving "
        "quality. If confidence drops sharply whenever bitrate drops, "
        "compression may be too aggressive."
    )

    cc1, cc2, cc3 = st.columns([1, 1, 2])
    smooth_conf = cc1.checkbox(
        "Smooth confidence", value=True, key=f"bvc_sc_{key_prefix}")
    smooth_bw = cc2.checkbox(
        "Smooth bitrate", value=True, key=f"bvc_sb_{key_prefix}",
        help="Per-frame instantaneous bitrate is noisy — smoothing makes "
             "the trend readable.",
    )
    smooth_win = cc3.slider(
        "Smoothing window (frames)",
        min_value=1, max_value=101, value=15, step=2,
        key=f"bvc_w_{key_prefix}",
    )

    x_col = _pick_x_axis(df, prefer="timestamp_sec")
    if x_col is None:
        st.warning("No timestamp_sec or frame_index column to plot against.")
        return
    x_title = ("Time (s)"
               if x_col in ("timestamp_sec", "source_timestamp_sec")
               else "Frame index")

    import pandas as pd
    bitrate_kbps = df["instantaneous_bitrate_bps"].astype(float) / 1000.0
    conf = df["conf_metric"].astype(float)
    if smooth_bw and smooth_win > 1:
        bitrate_kbps = bitrate_kbps.rolling(
            window=int(smooth_win), min_periods=1, center=True).mean()
    if smooth_conf and smooth_win > 1:
        conf = conf.rolling(
            window=int(smooth_win), min_periods=1, center=True).mean()

    plot_df = pd.DataFrame({
        x_col: df[x_col].astype(float).values,
        "bitrate_kbps": bitrate_kbps.values,
        "conf": conf.values,
    })

    try:
        import altair as alt
    except ImportError:
        # Altair ships with Streamlit, so this is unlikely; fall back to
        # a single-axis chart so the panel never blanks out.
        st.warning("altair not available — falling back to a single-axis chart.")
        st.line_chart(plot_df.set_index(x_col))
        return

    bw_color = "#1f77b4"
    cf_color = "#ff7f0e"

    base = alt.Chart(plot_df).encode(
        x=alt.X(f"{x_col}:Q", title=x_title),
    )

    bitrate_line = base.mark_line(color=bw_color).encode(
        y=alt.Y(
            "bitrate_kbps:Q",
            axis=alt.Axis(title="Bitrate (kbps)", titleColor=bw_color),
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip(f"{x_col}:Q", title=x_title, format=".2f"),
            alt.Tooltip("bitrate_kbps:Q",
                        title="Bitrate (kbps)", format=".1f"),
        ],
    )

    conf_line = base.mark_line(color=cf_color).encode(
        y=alt.Y(
            "conf:Q",
            axis=alt.Axis(title="Confidence (0–1)", titleColor=cf_color),
            scale=alt.Scale(domain=[0, 1]),
        ),
        tooltip=[
            alt.Tooltip(f"{x_col}:Q", title=x_title, format=".2f"),
            alt.Tooltip("conf:Q", title="Confidence", format=".3f"),
        ],
    )

    chart = (
        alt.layer(bitrate_line, conf_line)
        .resolve_scale(y="independent")
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)


# ── Single-run rendering ─────────────────────────────────────────────────────

def _render_single_run(st, run_dir: Path):
    st.caption(f"📁 {run_dir}")

    if not run_dir.is_dir():
        st.error(f"Run directory not found: {run_dir}")
        return

    cfg_path = run_dir / "session_config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception as e:
            st.warning(f"session_config.json unreadable: {e}")

    classes = cfg.get("classes") or []
    with st.expander("Session config", expanded=False):
        st.json(cfg)
    if classes:
        st.markdown("**Prompted classes:** " + ", ".join(f"`{c}`" for c in classes))

    summary = _load_summary(run_dir)
    if "error" in summary:
        st.warning(summary["error"])

    orig = summary.get("original") or {}
    masked = summary.get("masked") or {}
    adaptive = summary.get("adaptive") or {}
    pct = summary.get("percentiles") or {}
    metrics_csv = run_dir / "metrics.csv"
    df = None
    if metrics_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
        except Exception as e:
            st.warning(f"metrics.csv unreadable: {e}")

    st.subheader("Bitrate & bandwidth")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original avg bitrate",
              _fmt_bitrate(orig.get("avg_bitrate_bps")),
              _fmt_bytes(orig.get("size_bytes")))
    c2.metric("Masked avg bitrate",
              _fmt_bitrate(masked.get("avg_bitrate_bps")),
              _fmt_bytes(masked.get("size_bytes")))
    c3.metric("Adaptive avg bitrate",
              _fmt_bitrate(adaptive.get("avg_bitrate_bps")),
              _fmt_bytes(adaptive.get("size_bytes")))
    c4.metric("Adaptive (live on-wire)",
              _fmt_bitrate(summary.get("adaptive_avg_bitrate_bps_live")),
              _fmt_bytes(summary.get("adaptive_encoded_bytes_live")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Masked vs original savings",
              f"{(summary.get('masked_vs_original_savings_pct') or 0):.1f}%")
    c6.metric("Adaptive vs original savings",
              f"{(summary.get('adaptive_vs_original_savings_pct') or 0):.1f}%")
    c7.metric("Mean ROI ratio",
              f"{(summary.get('mean_roi_ratio') or 0):.3f}")
    c8.metric("Mean confidence",
              f"{(summary.get('mean_conf_metric') or 0):.3f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Frames processed", summary.get("frames_processed") or 0)
    c10.metric("Total detections", summary.get("total_detections") or 0)
    zdr = summary.get("zero_detection_ratio")
    c11.metric("Zero-detection frames",
               f"{(zdr or 0) * 100:.1f}%" if zdr is not None else "—")
    c12.metric("CRF transitions",
               summary.get("crf_transition_count") or 0,
               f"{summary.get('restart_count') or 0} encoder restarts")

    # Headline tracking signals: confidence + actual bandwidth over time.
    # Promoted to the top so live and replay runs both surface these
    # without scrolling. Detailed panels remain further down.
    #
    # When a `baseline_metrics.csv` is present in the run dir (camera was run
    # with baseline_enabled + --save-artifacts), the baseline-vs-EchoStream
    # A/B overlay is shown first; it answers the headline question directly.
    _render_baseline_vs_echostream(
        st, run_dir, df, key_prefix=f"ab_{run_dir.name}")
    if df is not None and len(df) > 0:
        st.markdown("### Headline tracking — confidence & bandwidth over time")
        _render_top_confidence_over_time(
            st, df, summary, key_prefix=f"top_{run_dir.name}")
        _render_top_bandwidth_over_time(
            st, df, summary, key_prefix=f"top_{run_dir.name}")

    _render_raw_masked_bandwidth_section(
        st, run_dir, summary, key_prefix=run_dir.name)
    if df is not None and len(df) > 0:
        _render_bitrate_vs_conf_section(
            st, df, key_prefix=run_dir.name)

    # ── Pipeline health (capture / processing / encode / response / artifact)
    health = summary.get("health") or {}
    if health:
        st.subheader("Pipeline health")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric(
            "Frames captured",
            int(health.get("total_frames_captured") or 0),
            f"{int(health.get('total_capture_attempt') or 0)} attempts",
        )
        h2.metric(
            "Capture drop rate",
            _fmt_pct(health.get("capture_drop_rate")),
            f"{int(health.get('total_capture_drops') or 0)} dropped",
        )
        h3.metric(
            "Processing skip rate",
            _fmt_pct(health.get("processing_skip_rate")),
            f"{int(health.get('total_processing_skips') or 0)} skipped",
        )
        h4.metric(
            "Encode zero-packet rate",
            _fmt_pct(health.get("encode_zero_packet_rate")),
            f"{int(health.get('total_encode_zero_packet') or 0)} / "
            f"{int(health.get('total_frames_encoded') or 0)} encoded",
        )
        h5, h6, h7, h8 = st.columns(4)
        h5.metric(
            "Response timeouts",
            int(health.get("total_response_timeouts") or 0),
            _fmt_pct(health.get("response_timeout_rate"))
            + " of expected",
        )
        h6.metric(
            "Stale responses",
            int(health.get("total_stale_responses") or 0),
            f"invalid: {int(health.get('total_invalid_responses') or 0)}",
        )
        h7.metric(
            "Artifact drops",
            int(health.get("total_artifact_drops") or 0),
            _fmt_pct(health.get("artifact_drop_rate"))
            + " of enqueued",
        )
        h8.metric(
            "Long loop gaps ≥100ms",
            int(health.get("long_loop_gap_count") or 0),
            f"max {_fmt_ms(health.get('max_loop_gap_ms'))}",
        )
        h9, h10, h11, h12 = st.columns(4)
        expected = health.get("expected_fps") or 0
        def _fps_delta(actual):
            try:
                a = float(actual); e = float(expected)
                if e <= 0:
                    return None
                return f"{(a - e):+.1f} vs expected {e:.1f}"
            except Exception:
                return None
        h9.metric("Observed input fps",
                  f"{float(health.get('observed_input_fps') or 0):.1f}",
                  _fps_delta(health.get("observed_input_fps")))
        h10.metric("Observed encoded fps",
                   f"{float(health.get('observed_encoded_fps') or 0):.1f}",
                   _fps_delta(health.get("observed_encoded_fps")))
        h11.metric("Observed response fps",
                   f"{float(health.get('observed_response_fps') or 0):.1f}",
                   _fps_delta(health.get("observed_response_fps")))
        h12.metric("Mean loop gap",
                   _fmt_ms(health.get("mean_loop_gap_ms")))

    # ── Detection preservation (offline eval)
    det_pres = summary.get("detection_preservation") or {}
    if det_pres:
        st.subheader("Detection preservation (offline eval)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Preserved fraction",
                  _fmt_pct(det_pres.get("mean_preserved_fraction")))
        d2.metric("Mean recall",
                  _fmt_pct(det_pres.get("mean_recall")))
        d3.metric("Mean precision",
                  _fmt_pct(det_pres.get("mean_precision")))
        d4.metric("Mean matched IoU",
                  f"{float(det_pres.get('mean_matched_iou') or 0):.3f}")
        st.caption(
            f"Compared {int(det_pres.get('frames_compared') or 0)} frames "
            f"(regressions: {int(det_pres.get('frames_with_regression') or 0)}, "
            f"zero-ref: {int(det_pres.get('frames_with_zero_ref') or 0)}, "
            f"IoU threshold: {det_pres.get('iou_threshold')})"
        )
        per_class = det_pres.get("per_class") or {}
        if per_class:
            with st.expander("Per-class preservation"):
                try:
                    import pandas as pd
                    rows = [
                        {
                            "class": k,
                            "matched": v.get("matched", 0),
                            "ref_total": v.get("ref_total", 0),
                            "preserved": _fmt_pct(v.get("preserved_fraction")),
                        }
                        for k, v in per_class.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                except Exception:
                    st.json(per_class)
    else:
        st.caption(
            "detection_preservation.json not found — run "
            "`python -m src.eval.detection_preservation --run-dir <this dir>` "
            "to populate."
        )

    # ── Recorded-input sidecar (reproducibility anchor)
    rec_cfg = (cfg.get("recorded_input") or {}) if cfg else {}
    rec_bitrate = summary.get("recorded_input") or {}
    if rec_cfg or rec_bitrate:
        st.subheader("Recorded raw input")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Frames",
                  int(rec_cfg.get("frame_count") or 0),
                  f"fps {rec_cfg.get('capture_fps', '—')}")
        r2.metric("Duration",
                  f"{float(rec_cfg.get('duration_sec') or 0):.2f} s")
        r3.metric("File size",
                  _fmt_bytes(rec_cfg.get("size_bytes")),
                  _fmt_bitrate(rec_bitrate.get("avg_bitrate_bps")))
        r4.metric("SHA-256",
                  _short_sha(rec_cfg.get("sha256")),
                  (rec_cfg.get("file_path") or "").split("/")[-1] or "—")
        st.caption(
            "Replay with "
            f"`--input {rec_cfg.get('file_path', '<path>')}` "
            "for deterministic comparison runs."
        )

    # ── Environment / reproducibility trail
    env = (cfg.get("environment") or {}) if cfg else {}
    if env:
        with st.expander("Environment & reproducibility", expanded=False):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Git SHA", _short_sha(env.get("git_commit_sha")),
                      "dirty" if env.get("git_is_dirty") else "clean")
            e2.metric("ffmpeg", env.get("ffmpeg_version") or "—")
            e3.metric("Input sha256",
                      _short_sha(env.get("input_video_sha256")),
                      env.get("input_video_path") or "—")
            e4.metric("Working dir",
                      (env.get("cwd") or "—").split("/")[-1] or "—")
            st.json(env)

    st.subheader("Latency percentiles")
    proc = pct.get("processing_time_ms") or {}
    infu = pct.get("infer_us_server") or {}
    decu = pct.get("decode_us_server") or {}
    enc = pct.get("encoded_bytes") or {}
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Frame p50",
              _fmt_ms(proc.get("p50")),
              f"p95 {_fmt_ms(proc.get('p95'))}")
    l2.metric("Server infer p50",
              _fmt_us(infu.get("p50")),
              f"p95 {_fmt_us(infu.get('p95'))}")
    l3.metric("Server decode p50",
              _fmt_us(decu.get("p50")),
              f"p95 {_fmt_us(decu.get('p95'))}")
    l4.metric("Encoded bytes p50",
              _fmt_bytes(enc.get("p50")),
              f"p95 {_fmt_bytes(enc.get('p95'))}")

    send = pct.get("send_ms") or {}
    recv = pct.get("recv_ms") or {}
    flow = pct.get("flow_ms") or {}
    e2e = pct.get("end_to_end_loop_ms") or {}
    l5, l6, l7, l8 = st.columns(4)
    l5.metric("Flow p50",
              _fmt_ms(flow.get("p50")),
              f"p95 {_fmt_ms(flow.get('p95'))}")
    l6.metric("Send p50",
              _fmt_ms(send.get("p50")),
              f"p95 {_fmt_ms(send.get('p95'))}")
    l7.metric("Recv p50",
              _fmt_ms(recv.get("p50")),
              f"p95 {_fmt_ms(recv.get('p95'))}")
    l8.metric("End-to-end loop p50",
              _fmt_ms(e2e.get("p50")),
              f"p95 {_fmt_ms(e2e.get('p95'))}")

    st.subheader("Video outputs")
    v1, v2, v3 = st.columns(3)
    for col, label, name in (
        (v1, "Original", "original.mp4"),
        (v2, "Masked", "masked.mp4"),
        (v3, "Decoded adaptive", "decoded_adaptive.mp4"),
    ):
        col.markdown(f"**{label}**")
        path = run_dir / name
        if path.exists():
            col.video(str(path))
            col.caption(_fmt_bytes(path.stat().st_size))
        else:
            col.info(f"{name} missing")

    if metrics_csv.exists():
        if df is not None and len(df) > 0:
            _render_confidence_section(
                st, df, summary, key_prefix=run_dir.name)

            st.subheader("Time series")

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**CRF over time**")
                st.line_chart(df.set_index("timestamp_sec")[["crf"]])
                st.markdown("**ROI ratio over time**")
                st.line_chart(df.set_index("timestamp_sec")[["roi_ratio"]])
            with colB:
                st.markdown("**Instantaneous bitrate (bps)**")
                st.line_chart(
                    df.set_index("timestamp_sec")[["instantaneous_bitrate_bps"]]
                )
                if "num_detections" in df.columns:
                    st.markdown("**Detections per frame**")
                    st.line_chart(
                        df.set_index("timestamp_sec")[["num_detections"]]
                    )

            if "processing_time_ms" in df.columns:
                st.markdown("**Per-frame processing time (ms)**")
                st.line_chart(
                    df.set_index("timestamp_sec")[["processing_time_ms"]]
                )

            st.markdown("**Cumulative encoded bytes**")
            st.line_chart(
                df.set_index("timestamp_sec")[["cumulative_encoded_bytes"]]
            )

            st.subheader("Per-frame metrics (head)")
            st.dataframe(df.head(50), use_container_width=True)
    else:
        st.info("metrics.csv missing — time-series views unavailable.")


# ── Multi-run comparison ─────────────────────────────────────────────────────

def _render_comparison(st, run_dirs: List[Path]):
    import pandas as pd

    st.subheader("Run comparison")

    rows = []
    for rd in run_dirs:
        s = _load_summary(rd)
        if "error" in s:
            st.warning(f"{rd.name}: {s['error']}")
            continue
        pct = s.get("percentiles") or {}
        proc = pct.get("processing_time_ms") or {}
        infu = pct.get("infer_us_server") or {}
        e2e = pct.get("end_to_end_loop_ms") or {}
        health = s.get("health") or {}
        det_pres = s.get("detection_preservation") or {}
        conf_sum = s.get("confidence_summary") or {}
        rows.append({
            "run": rd.name,
            "frames": s.get("frames_processed") or 0,
            "live_bitrate_kbps": round(
                (s.get("adaptive_avg_bitrate_bps_live") or 0) / 1000.0, 1),
            "adaptive_vs_original_%": round(
                s.get("adaptive_vs_original_savings_pct") or 0, 1),
            "obs_input_fps": round(health.get("observed_input_fps") or 0, 1),
            "obs_encoded_fps": round(health.get("observed_encoded_fps") or 0, 1),
            "obs_response_fps": round(health.get("observed_response_fps") or 0, 1),
            "capture_drop_%": round(
                (health.get("capture_drop_rate") or 0) * 100, 2),
            "proc_skip_%": round(
                (health.get("processing_skip_rate") or 0) * 100, 2),
            "resp_timeout_%": round(
                (health.get("response_timeout_rate") or 0) * 100, 2),
            "stale_resp": int(health.get("total_stale_responses") or 0),
            "artifact_drops": int(health.get("total_artifact_drops") or 0),
            "long_loop_gaps": int(health.get("long_loop_gap_count") or 0),
            "preserved_%": round(
                (det_pres.get("mean_preserved_fraction") or 0) * 100, 1)
                if det_pres else None,
            "preservation_recall_%": round(
                (det_pres.get("mean_recall") or 0) * 100, 1)
                if det_pres else None,
            "mean_roi": round(s.get("mean_roi_ratio") or 0, 3),
            "mean_conf": round(s.get("mean_conf_metric") or 0, 3),
            "median_conf": round(conf_sum.get("median") or 0, 3),
            "p5_conf": round(conf_sum.get("p5") or 0, 3),
            "min_conf": round(conf_sum.get("min") or 0, 3),
            "frac_conf<0.5_%": round(
                (conf_sum.get("frac_below_0_5") or 0) * 100, 1),
            "zero_det_%": round((s.get("zero_detection_ratio") or 0) * 100, 1),
            "crf_transitions": s.get("crf_transition_count") or 0,
            "proc_p50_ms": round(proc.get("p50") or 0, 1),
            "proc_p95_ms": round(proc.get("p95") or 0, 1),
            "e2e_p95_ms": round(e2e.get("p95") or 0, 1),
            "infer_p95_ms": round((infu.get("p95") or 0) / 1000.0, 1),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Overlay time-series (confidence + CRF + live bitrate) across runs.
    st.subheader("Time-series overlay")

    conf_frames = {}
    crf_frames = {}
    bw_frames = {}
    roi_frames = {}
    per_run_dfs = {}
    for rd in run_dirs:
        csv_path = rd / "metrics.csv"
        if not csv_path.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        x_col = _pick_x_axis(df, prefer="timestamp_sec")
        if x_col is None:
            continue
        df = df.set_index(x_col)
        per_run_dfs[rd.name] = df
        if "conf_metric" in df.columns:
            conf_frames[rd.name] = df["conf_metric"].astype(float)
        if "crf" in df.columns:
            crf_frames[rd.name] = df["crf"]
        if "instantaneous_bitrate_bps" in df.columns:
            bw_frames[rd.name] = df["instantaneous_bitrate_bps"]
        if "roi_ratio" in df.columns:
            roi_frames[rd.name] = df["roi_ratio"]

    if conf_frames or bw_frames:
        st.markdown("### Headline tracking — confidence & bandwidth over time")
        st.caption(
            "Cross-run overlay of the two headline signals. Use this to "
            "compare how each run's confidence and on-wire bandwidth "
            "evolve on the same replay input or across live sessions."
        )

    if bw_frames:
        import pandas as pd
        st.markdown("**Bandwidth consumption over time (kbps, overlay)**")
        bw_df_kbps = pd.DataFrame(
            {name: s.astype(float) / 1000.0 for name, s in bw_frames.items()}
        )
        oc1, _ = st.columns([1, 3])
        bw_smooth = oc1.slider(
            "Bandwidth smoothing (frames)",
            min_value=1, max_value=101, value=15, step=2,
            key="cmp_bw_smooth",
        )
        if bw_smooth > 1:
            bw_df_kbps = bw_df_kbps.rolling(
                window=int(bw_smooth), min_periods=1, center=True
            ).mean()
        st.line_chart(bw_df_kbps)

    if conf_frames:
        import pandas as pd
        st.markdown("**Detector confidence (overlay)**")
        st.caption(
            "If one run preserves confidence noticeably better, its curve "
            "sits higher and is steadier. A smoothed view makes the gap "
            "easier to read than the raw per-frame jitter."
        )
        oc1, oc2 = st.columns([1, 3])
        smooth_mode = oc1.radio(
            "Curve",
            ["smoothed", "raw"],
            horizontal=True,
            key="cmp_conf_mode",
        )
        smooth_win = oc2.slider(
            "Smoothing window (frames)",
            min_value=1, max_value=101, value=15, step=2,
            key="cmp_conf_smooth",
        )
        conf_df = pd.DataFrame(conf_frames)
        if smooth_mode == "smoothed":
            conf_df = conf_df.rolling(
                window=int(smooth_win), min_periods=1, center=True
            ).mean()
        st.line_chart(conf_df)

    if crf_frames:
        import pandas as pd
        st.markdown("**CRF**")
        st.line_chart(pd.DataFrame(crf_frames))
    if bw_frames:
        import pandas as pd
        st.markdown("**Instantaneous bitrate (bps)**")
        st.line_chart(pd.DataFrame(bw_frames))
    if roi_frames:
        import pandas as pd
        st.markdown("**ROI ratio**")
        st.line_chart(pd.DataFrame(roi_frames))


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    import streamlit as st

    st.set_page_config(page_title="EchoStream Evaluator",
                       page_icon="🎥", layout="wide")
    st.title("EchoStream — Benchmark Evaluator")

    run_dirs_raw = _parse_run_dirs()
    if not run_dirs_raw:
        typed = st.text_input(
            "Run directory (comma-separate multiple runs to compare)",
            value="runs/benchmark_001",
            help="Path(s) produced by camera_h264.py --save-artifacts",
        )
        run_dirs_raw = [s.strip() for s in typed.split(",") if s.strip()]

    run_dirs = [Path(s).resolve() for s in run_dirs_raw]
    if len(run_dirs) > 1:
        _render_comparison(st, run_dirs)
        st.divider()
        tabs = st.tabs([d.name for d in run_dirs])
        for tab, rd in zip(tabs, run_dirs):
            with tab:
                _render_single_run(st, rd)
    elif run_dirs:
        _render_single_run(st, run_dirs[0])
    else:
        st.info("Provide at least one --run-dir to continue.")


main()
