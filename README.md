# EchoStream — Machine-Centric Video Streaming for AI Inference

Two-process adaptive H.264 pipeline. A **camera/edge node** captures or
replays video, masks it with optical flow, encodes with `libx264` (via a
persistent `ffmpeg` subprocess), and streams to an **edge-server node**
running **YOLO-World** open-vocabulary detection. The server returns a
confidence metric + detection heatmap + boxes; the camera uses that
feedback to adjust CRF and refine the next frame's mask.

```
python -m venv venv
source venv/bin/activate   # Windows (bash): source venv/Scripts/activate
pip install -r requirements.txt
```

> **Protocol v3.** Camera and server are upgraded together. A v2 side
> cannot talk to a v3 side — the handshake will reject it. See
> §"Wire protocol" below.


Artifacts land in `runs/test/` (videos, metrics.csv, summary.json,
recorded-input sidecar JSON with SHA-256). The camera logs
`preview=ON  record_input=<path>  save_artifacts=ON` at startup so you
can confirm all three are active. Everything below is variations on
this flow.

## 1. Start the server with YOLO-World

```bash
python -m src.inference.server_h264 \
    --model yolov8s-world.pt \
    --classes "person,wallet,bed" \
    --device auto
```

The first connecting camera will send its own prompted class list in a
handshake — that overrides `--classes` for the session. So you can leave
the server running and change prompts per-run from the camera side.

| flag               | default                | description                               |
|--------------------|------------------------|-------------------------------------------|
| `--model`          | `yolov8s-world.pt`     | YOLO-World checkpoint                     |
| `--classes`        | `person`               | fallback prompts if handshake is missing  |
| `--device`         | `auto`                 | `auto` / `cuda` / `mps` / `cpu`           |
| `--port`           | `9999`                 | TCP port                                  |
| `--conf-threshold` | `0.05`                 | detector confidence floor                 |
| `--show-window`    | off                    | render YOLO annotations server-side       |

## 2. Live webcam run + fixed-video recording

```bash
python -m src.streaming.camera_h264 \
    --input 0 --classes "person,wallet,bed" \
    --save-artifacts --output-dir runs/live_001 \
    --record-input runs/live_001/raw_recorded_input.mp4 \
    --response-timeout-sec 2.0
```

The live side-by-side preview window shows during recording; press `q`
in it to exit cleanly. For headless/CI runs add `--no-preview`.

The `--record-input` tap runs *before* masking, so the MP4 reflects what
the camera actually saw. A sidecar `<mp4>.json` is written alongside it
with fps, frame count, duration, `input_source`, and the MP4's SHA-256 —
enough to prove two runs replayed the exact same bytes.

Key flags (new in v3):

| flag                            | purpose                                             |
|---------------------------------|-----------------------------------------------------|
| `--record-input <path>`         | tap post-resize BGR into MP4 (webcam only)          |
| `--record-input-fps <fps>`      | override container fps (default = probed)           |
| `--record-input-max-frames <N>` | cap recording length                                |
| `--response-timeout-sec <sec>`  | give up on server reply; counts as a timeout        |
| `--save-artifacts`              | write videos + metrics.csv + summary.json           |
| `--output-dir <dir>`            | where artifacts go                                  |
| `--loop-video`                  | rewind file input on EOF                            |

`ECHOSTREAM_PROACTIVE=1` (env var) enables per-GOP CRF planning.

## 3. Deterministic replay of the recorded video

```bash
python -m src.streaming.camera_h264 \
    --input runs/live_001/raw_recorded_input.mp4 \
    --classes "person,wallet,bed" \
    --save-artifacts --output-dir runs/replay_001
```

Feeds the exact same bytes through masking → encode → send → detect.
Ideal A/B baseline: run twice with different controller settings and
compare.

## 4. Offline detection-preservation eval

Runs YOLO-World over `original.mp4` vs `decoded_adaptive.mp4` and
reports preserved fraction, recall, precision, mean matched IoU, and a
per-class breakdown. Results land in `detection_preservation.json`,
which `summary.json` automatically folds in on next read.

```bash
python -m src.eval.detection_preservation \
    --run-dir runs/live_001 \
    --model yolov8s-world.pt \
    --iou-threshold 0.5 --stride 3
```

`--stride 3` compares every 3rd frame (useful for long runs). `--help`
for the full flag list.

## 5. Streamlit evaluator (single run or A/B)

```bash
# single run
streamlit run app/streamlit_eval.py -- --run-dir runs/live_001

# A/B comparison
streamlit run app/streamlit_eval.py -- --run-dir runs/live_001,runs/replay_001
```

(The `--` is required so Streamlit forwards `--run-dir` to the app.)

The UI surfaces: bitrate / savings, latency percentiles (frame, send,
recv, flow, e2e), pipeline-health block (capture/processing/encode/
response/artifact counters, observed FPS, long-loop-gap count),
detection preservation (if the offline eval ran), recorded-input
metadata (frames, SHA-256, replay hint), and a reproducibility expander
(git SHA, ffmpeg version, input sha256).

## What the run directory contains

```
runs/<name>/
  original.mp4               raw input, resized to pipeline resolution
  masked.mp4                 optical-flow-masked frames fed to the encoder
  decoded_adaptive.mp4       locally-decoded H.264 round-trip output
  raw_recorded_input.mp4     webcam tap, only when --record-input was set
  raw_recorded_input.json    sidecar: fps, duration, frame count, sha256
  metrics.csv                per-frame stats (see below)
  session_config.json        CLI args, determinism, environment, health
  summary.json               aggregated run summary, written on exit
  detection_preservation.json  populated by the offline eval
```

**metrics.csv** columns (grouped):

- Identifiers: `frame_index, sequence_id, timestamp_sec,
  source_timestamp_sec, loop_timestamp_sec, input_source, width, height, fps`
- Health (running totals — rate = running / frame_index):
  `capture_ok, capture_drop_count_running, processing_skip_count_running,
  encode_zero_packet_count_running, response_timeout_count_running,
  stale_response_count_running, artifact_drop_count_running`
- Observed FPS (10-s rolling window):
  `expected_fps, observed_input_fps, observed_encoded_fps, observed_response_fps`
- Scene / codec state: `roi_ratio, conf_metric, crf, encoded_bytes,
  cumulative_encoded_bytes, instantaneous_bitrate_bps, average_bitrate_bps,
  num_detections, detected_classes, restart_count, crf_transition_count,
  proactive_mode, processing_time_ms`
- Per-phase timings: `flow_ms, encode_ms, send_ms, recv_ms, parse_ms,
  end_to_end_loop_ms`
- Server-reported: `decode_us_server, infer_us_server`

## Pipeline-health counters

Frame drops are *not* a single number — each subsystem owns a rate:

| counter                      | meaning                                              |
|------------------------------|------------------------------------------------------|
| `capture_drop_rate`          | `cap.read()` returned nothing                        |
| `processing_skip_rate`       | resize/flow-prep failed                              |
| `encode_zero_packet_rate`    | ffmpeg consumed a frame but emitted no NALs          |
| `response_timeout_rate`      | no server reply within `--response-timeout-sec`      |
| `stale_response_rate`        | response seq below watermark (stale)                 |
| `artifact_drop_rate`         | mp4 writer fell behind disk                          |
| `long_loop_gap_count`        | outer loop ticks gapped >100 ms (pause indicator)    |
| `observed_input/encoded/response_fps` | 10-s rolling actual FPS per stage           |

A healthy run has all six rates in the noise floor; when a number
spikes, you know exactly which subsystem is leaking time.

## Wire protocol (v3)

See `src/inference/protocol.py`. Summary:

- **Handshake** (camera→server, once): `!HHH protocol_version, heat_w, heat_h`,
  then `!I num_classes`, then per class `!H name_len` + utf-8 bytes.
- **Encoded video** (camera→server, per packet): `!QI payload_len, sequence_id`
  + H.264 NAL bytes (`src/codec/wire.py`).
- **Detection response** (server→camera, per packet):
  `!fIIIIII metric, heat_w, heat_h, num_boxes, decode_us, infer_us, sequence_id`,
  then `heat_w*heat_h` uint8 heatmap bytes, then `num_boxes × !fffffI` —
  one `(x1, y1, x2, y2, conf, cls_idx)` per detection.

The `sequence_id` round-trip lets the camera detect stale, invalid, or
timed-out responses. **Breaking change from v2** — both wire framing and
response header grew; update both sides together.

## Architecture

- `src/streaming/camera_h264.py` — camera entrypoint. Input source
  (webcam or mp4), optical-flow mask, H.264 encode, TCP send, server
  reply handling, dashboard, artifact + recorder writing.
- `src/inference/server_h264.py` — server entrypoint. Handshake, YOLO
  prompts, H.264 decode, inference, reply.
- `src/inference/detection.py` — `YoloWorldDetector` wrapper.
- `src/inference/protocol.py` — shared v3 handshake + response framing.
- `src/codec/h264_backend.py` — H.264 backend + proactive CRF planner.
- `src/codec/wire.py` — v3 TCP framing with sequence_id.
- `src/optical_flow/*` — DIS flow + affine warp + ROI fusion.
- `src/eval/artifacts.py` — mp4/CSV/JSON writers (async worker).
- `src/eval/pipeline_counters.py` — six-group health counters.
- `src/eval/recorded_input.py` — fixed-video recorder + sidecar.
- `src/eval/environment.py` — git SHA / ffmpeg version / input sha256.
- `src/eval/detection_preservation.py` — offline preservation eval.
- `src/eval/video_metrics.py` — summary statistics (ffprobe fallback).
- `src/utils/reproducibility.py` — `set_seeds(seed, strict)`.
- `app/streamlit_eval.py` — Streamlit dashboard (single + A/B).
