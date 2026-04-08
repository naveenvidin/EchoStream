# Machine-Centric Video Streaming Optimization for AI Inference Pipelines

Real-time adaptive video streaming pipeline that uses a learned neural codec
(CompressAI bmshj2018_hyperprior) with I/P-frame GOP structure, motion-based
ROI masking, and AI-confidence-driven bitrate adaptation.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Start the inference server first
python server.py

# Then start the camera/encoder node
python camera.py
```

Press `q` in the dashboard window to stop.

## Architecture

- **camera.py** — Captures webcam frames, applies motion masking, encodes with
  the neural codec, and streams to the server. Adapts compression quality based
  on the server's AI confidence score.
- **server.py** — Receives and decodes the neural codec bitstream, runs AI
  inference, and sends a confidence metric back to the camera node.
- **neural_codec.py** — DCVC-RT-style learned video codec using CompressAI.
  Supports I-frame (full) and P-frame (residual) coding with 8 quality levels.
- **motion_masker.py** — Optical-flow-based ROI detection to focus encoding on
  regions with motion.

## Configuration (camera.py)

| Variable      | Default     | Description                                   |
|---------------|-------------|-----------------------------------------------|
| `FIXED_CRF`  | `None`      | Set to a value (e.g. 30) to disable adaptive CRF |
| `GOP_SIZE`    | `10`        | Frames between I-frames                       |
| `CODEC_DEVICE`| `None`     | `None` = auto-detect (CUDA > CPU)             |

## Known Issues

- **OOM on CPU**: The adaptive CRF previously caused multiple codec models to
  accumulate in memory. Fixed by evicting old models on quality switch. If still
  tight on RAM (<16GB), set `FIXED_CRF` to avoid model reloading.