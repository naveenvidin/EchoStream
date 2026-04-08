import struct

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from compressai.zoo import bmshj2018_hyperprior


I_FRAME = 0
P_FRAME = 1


class NeuralCodec:

    def __init__(self, quality=3, device=None, gop_size=10):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gop_size = gop_size
        self._frame_counter = 0
        self._reference_frame = None
        self._models = {}
        self._current_quality = None
        self._load_model(quality)

    def _load_model(self, quality):
        quality = max(1, min(8, quality))
        if quality not in self._models:
            for q in list(self._models.keys()):
                del self._models[q]
            net = bmshj2018_hyperprior(quality=quality, pretrained=True)
            net = net.eval().to(self.device)
            self._models[quality] = net
        self._current_quality = quality

    @property
    def model(self):
        return self._models[self._current_quality]

    def set_quality(self, quality):
        quality = max(1, min(8, quality))
        if quality != self._current_quality:
            self._load_model(quality)
            self.reset_gop()

    @torch.no_grad()
    def encode(self, frame_bgr):
        x = self._bgr_to_tensor(frame_bgr)
        h, w = x.shape[2], x.shape[3]
        x = self._pad(x)

        frame_type = I_FRAME if self._should_send_iframe() else P_FRAME

        if frame_type == P_FRAME and self._reference_frame is not None and self._reference_frame.shape == x.shape:
            residual = (x - self._reference_frame) * 0.5 + 0.5
            residual = residual.clamp(0.0, 1.0)
            out = self.model(residual)
            x_hat = out["x_hat"].clamp(0.0, 1.0)
            self._reference_frame = (
                self._reference_frame + (x_hat - 0.5) * 2.0
            ).clamp(0.0, 1.0)
            result_tensor = self._reference_frame[:, :, :h, :w]
        else:
            frame_type = I_FRAME
            out = self.model(x)
            x_hat = out["x_hat"].clamp(0.0, 1.0)
            self._reference_frame = x_hat
            result_tensor = x_hat[:, :, :h, :w]

        self._frame_counter += 1

        recon_bgr = self._tensor_to_bgr(result_tensor)
        jpeg_quality = self._quality_to_jpeg(self._current_quality)
        ok, encoded = cv2.imencode(
            ".jpg", recon_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encoding of reconstructed frame failed")

        jpeg_bytes = encoded.tobytes()

        header = struct.pack("!BBHHI", frame_type, self._current_quality, h, w, len(jpeg_bytes))
        return header + jpeg_bytes, frame_type

    @torch.no_grad()
    def decode(self, bitstream):
        hdr_size = struct.calcsize("!BBHHI")
        frame_type, quality, h, w, jpeg_len = struct.unpack(
            "!BBHHI", bitstream[:hdr_size]
        )
        jpeg_data = bitstream[hdr_size:hdr_size + jpeg_len]

        arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise RuntimeError("JPEG decoding failed")

        x = self._bgr_to_tensor(frame_bgr)
        if frame_type == P_FRAME and self._reference_frame is not None:
            self._reference_frame = x
        else:
            self._reference_frame = x

        return frame_bgr

    def _should_send_iframe(self):
        return self._frame_counter % self.gop_size == 0

    def reset_gop(self):
        self._frame_counter = 0
        self._reference_frame = None

    def _bgr_to_tensor(self, frame_bgr):
        rgb = frame_bgr[:, :, ::-1].copy()
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        return x.to(self.device)

    @staticmethod
    def _tensor_to_bgr(x):
        rgb = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return rgb[:, :, ::-1].copy()

    @staticmethod
    def _pad(x):
        _, _, h, w = x.shape
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    @staticmethod
    def crf_to_quality(crf_val):
        q = 8 - int(crf_val / 51.0 * 7)
        return max(1, min(8, q))

    @staticmethod
    def _quality_to_jpeg(quality):
        return int(30 + (quality - 1) * (65 / 7))
