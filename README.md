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

## 1. Start the server with YOLO-World

```powershell
python -m src.inference.server_h264 --model yolov8s-world.pt --classes "person,wallet,bed" --device auto --show-window
```

Server flags:

| flag | default | description |
| --- | --- | --- |
| `--model` | `yolov8s-world.pt` | YOLO-World checkpoint |
| `--classes` | `person` | prompted class list, e.g. `person,wallet,bed` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--port` | `9999` | TCP port |
| `--conf-threshold` | `0.05` | detector confidence floor |
| `--show-window` | off | show server-side detections |

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
2. Raw vs Masked Bandwidth Usage Over Time
3. Bitrate vs Detector Confidence Over Time

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
