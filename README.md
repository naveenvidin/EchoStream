# EchoStream Main Compare

This folder keeps the compare branch's original streaming path:

- camera captures or replays video
- optical-flow masking prepares the frames
- `SegmentEncoder` encodes each segment with `ffmpeg`/`libx264`
- camera sends `[4-byte payload length][H.264 segment bytes]`
- server decodes with its existing `ffmpeg` H.264 decoder
- server sends back one 4-byte confidence float
- camera maps confidence to CRF for future segments

YOLO-World and evaluator support were added on top of that path. The newer
`EchoStream` v3 handshake / sequence-aware codec protocol was not ported here.

## Setup

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
this worked for darius
```python -m venv venv                              
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `streamlit` is not recognized, run it through Python:

```powershell
python -m streamlit --version
```

## Split deployment (Pi camera + GPU server)

The pipeline is split across two machines for real deployment. Two
config files are provided so each side can be launched without editing
shared defaults:

- `configs/server_gpu.json` — GPU/inference machine (server side)
- `configs/camera_pi.json` — Raspberry Pi (camera/edge side)

Edit `configs/camera_pi.json` and set `server_ip` to the GPU server's
reachable address before launching the Pi. Pipeline logic (detection,
masking, adaptive encoding, protocol, replay, artifacts, feedback) is
unchanged — only deployment defaults differ.

**On the GPU server:**

```bash
python -m src.inference.server_h264 --config configs/server_gpu.json
```

**On the Raspberry Pi:**

```bash
python -m src.streaming.camera_h264 --config configs/camera_pi.json
```

Add `--no-preview` on the Pi if needed (already the default in
`camera_pi.json`). The legacy single-machine commands below still work
unchanged via `configs/default.json`.

## 1. Start the server with YOLO-World

```powershell
python -m src.inference.server_h264 --model yolov8s-world.pt --classes "person,wallet,bed" --device auto --show-window
```

Server flags (each overrides the matching key in the JSON config; if
omitted, the config value is used):

| flag | default | description |
| --- | --- | --- |
| `--config` | `configs/default.json` | JSON config file |
| `--model` | from config | YOLO-World checkpoint path |
| `--classes` | from config | prompted class list, e.g. `person,wallet,bed` |
| `--device` | from config | `auto`, `cuda`, `mps`, or `cpu` |
| `--conf-threshold` | from config | detector confidence floor |
| `--nms-iou` | from config | NMS IoU threshold |
| `--tracker` | from config | `kalman` or `none` |
| `--host` | `0.0.0.0` | bind host |
| `--port` | from config | TCP port |
| `--width` / `--height` | from config | decoded frame size |
| `--show-window` | off | show server-side detections |
| `--save-artifacts` | off | write server-side artifacts to `--output-dir` |
| `--output-dir <dir>` | `runs/server_<ts>` | server artifact directory |

When `--save-artifacts` is on, the server writes:

```text
<output-dir>/
  decoded.mp4           decoded H.264 frames as the server received them
  annotated.mp4         decoded frames with detection boxes drawn
  server_metrics.csv    per-frame: num_boxes, conf_min/max/mean, infer_ms
  server_config.json    model, classes, device, host, port, fps, client_addr
```

Camera-side artifacts (under the camera's own `--output-dir`) are
unchanged and complement these — both can be saved during the same run
on their respective machines.

The compare protocol does not send prompted classes from camera to server, so use
the same `--classes` value on both commands for clean artifact metadata.

## 2. Live webcam run + recording

```powershell
python -m src.streaming.camera_h264 --input 0 --classes "person,wallet,bed" --save-artifacts --output-dir runs/live_001 --record-input runs/live_001/raw_recorded_input.mp4 --response-timeout-sec 2.0
```

The live preview now shows two panels:

- left: raw camera feed
- right: masked / ROI-prepared stream

Press `q` in the preview window to exit. For headless runs, add `--no-preview`.

`--record-input` writes the resized raw webcam input before masking, so the file
can be replayed later for deterministic comparisons. File inputs are already
reproducible, so `--record-input` is ignored for replay runs.

Camera flags:

| flag | purpose |
| --- | --- |
| `--input <0|path>` | webcam index or video file |
| `--classes <list>` | class list saved in artifacts; match server prompts |
| `--save-artifacts` | write videos, `metrics.csv`, and `summary.json` |
| `--output-dir <dir>` | run output directory |
| `--record-input <path>` | record raw webcam input for replay |
| `--record-input-fps <fps>` | override recorded-input container fps |
| `--record-input-max-frames <N>` | cap raw input recording length |
| `--loop-video` | rewind file input at EOF |
| `--max-frames <N>` | stop after N frames |
| `--response-timeout-sec <sec>` | accepted for CLI compatibility; feedback uses the existing listener thread |
| `--no-preview` | disable the OpenCV preview window |

## 3. Deterministic replay

```powershell
python -m src.streaming.camera_h264 --input runs/live_001/raw_recorded_input.mp4 --classes "person,wallet,bed" --save-artifacts --output-dir runs/replay_001
```

Start the server first, using the same prompted classes:

```powershell
python -m src.inference.server_h264 --model yolov8s-world.pt --classes "person,wallet,bed" --device auto
```

## 4. Offline detection-preservation eval

This optional step compares detections on `original.mp4` and
`decoded_adaptive.mp4`, then writes `detection_preservation.json`.

```powershell
python -m src.eval.detection_preservation --run-dir runs/live_001 --model yolov8s-world.pt --classes "person,wallet,bed" --iou-threshold 0.5 --stride 3
```

## 5. Streamlit evaluator

Single run:

```powershell
python -m streamlit run src/app/streamlit_eval.py -- --run-dir runs/live_001
```

A/B comparison:

```powershell
python -m streamlit run src/app/streamlit_eval.py -- --run-dir runs/live_001,runs/replay_001
```

The top of the dashboard shows:

1. summary bitrate / savings metrics
2. **Headline tracking — confidence & bandwidth over time** (prominent
   per-frame charts pulled from `metrics.csv`: detector confidence and
   actual on-wire kbps, both with smoothing controls; works for live,
   replay, and single-run analysis)
3. Raw vs Masked Bandwidth Usage Over Time
4. Bitrate vs Detector Confidence Over Time

The multi-run comparison view also gains a headline overlay of both
signals across runs.

Lower sections include detector confidence detail, time-series charts, video
outputs, latency percentiles, pipeline-health counters, recorded-input metadata,
and detection-preservation results when available.

## Run directory contents

```text
runs/<name>/
  original.mp4                 raw input, resized to pipeline resolution
  masked.mp4                   optical-flow-masked frames sent to the encoder
  decoded_adaptive.mp4         local decode of the encoded H.264 segment artifacts
  raw_recorded_input.mp4       webcam tap, only when --record-input was set
  raw_recorded_input.json      sidecar: fps, duration, frame count, sha256
  metrics.csv                  per-frame stats
  session_config.json          run config and final health counters
  summary.json                 aggregated run summary
  detection_preservation.json  optional offline eval result
```

Because this compare path only receives a 4-byte confidence response from the
server, live metrics do not include per-frame detection boxes/classes or
server-side decode/inference timings. Those fields may be blank/default in
`metrics.csv` and `summary.json`.

## Compare wire protocol

This folder intentionally keeps the simpler compare protocol:

- camera to server: `!I payload_len` followed by one H.264 segment
- server to camera: `!f confidence`

There is no camera/server handshake, heatmap response, detection-box response,
or sequence-id round trip in this folder.

## Architecture

- `src/streaming/camera_h264.py` - camera entrypoint, input/replay, optical-flow masking, segment encoding, TCP send, confidence listener, preview, artifacts, recording.
- `src/inference/server_h264.py` - server entrypoint, length-prefixed H.264 receive, ffmpeg decode, YOLO-World inference, confidence reply.
- `src/inference/detection.py` - YOLO-World wrapper and prompted class parsing.
- `src/optical_flow/*` - optical-flow masking support.
- `src/eval/artifacts.py` - video/CSV/session writers.
- `src/eval/pipeline_counters.py` - capture/processing/encode/response/artifact health counters.
- `src/eval/recorded_input.py` - raw input recorder and sidecar metadata.
- `src/eval/detection_preservation.py` - optional offline preservation eval.
- `src/eval/video_metrics.py` - summary statistics.
- `src/app/streamlit_eval.py` - Streamlit dashboard.
