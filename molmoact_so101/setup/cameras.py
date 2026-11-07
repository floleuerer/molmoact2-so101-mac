"""
cameras.py — platform-independent OpenCV camera capture.

OpenCVCamera opens either
  * a local camera by index (AVFoundation on macOS, V4L2 on Linux), or
  * a network stream by URL (e.g. a Raspberry Pi streaming with rpicam-vid:
    tcp://raspberrypi.local:8888, rtsp://..., http://.../stream.mjpg).

A background thread keeps only the newest frame, so read() never returns a
stale buffered frame — essential for network streams, where OpenCV would
otherwise queue frames and add seconds of latency.

Frames are normalised to the model's input geometry (default 640×480): if the
camera delivers another aspect ratio (e.g. a 16:9 webcam), the
frame is centre-cropped to 4:3 before resizing so objects are not distorted.
"""
import os
import platform
import re
import threading
import time

import cv2
import numpy as np

IS_MAC   = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

FLIP_CHOICES = ["v", "h", "180", "none"]
_FLIP_CODES  = {"v": 0, "h": 1, "180": -1}

# Low-latency FFmpeg options for network streams. Must be set before the
# first VideoCapture is created; setdefault so the user can override.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|framedrop;1",
)


def mac_video_devices():
    """macOS: [(opencv_index, name, unique_id)] in OpenCV's AVFoundation order.

    OpenCV numbers AVFoundation devices sorted by uniqueID — NOT the order
    ffmpeg / System Information show (that order changes between runs).
    """
    if not IS_MAC:
        return []
    import AVFoundation as AVF
    devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
    devs.sort(key=lambda d: str(d.uniqueID()))
    return [(i, str(d.localizedName()).replace("\xa0", " "), str(d.uniqueID()))
            for i, d in enumerate(devs)]


def parse_source(spec):
    """Turn a camera spec into something cv2.VideoCapture accepts.

    '3'                     → 3 (OpenCV index)
    'tcp://pi.local:8888'   → URL, returned as-is
    'C920' / 'Innomaker'    → macOS: index of the camera whose name contains
                              this text (case-insensitive). Indices shift when
                              cameras are re-plugged; names don't.
    """
    if isinstance(spec, int):
        return spec
    spec = str(spec).strip()
    if re.fullmatch(r"\d+", spec):
        return int(spec)
    if "://" in spec or spec.startswith("/"):
        return spec
    if IS_MAC:
        devices = mac_video_devices()
        hits = [(i, n) for i, n, _ in devices if spec.lower() in n.lower()]
        if len(hits) == 1:
            print(f"[Cameras] '{spec}' → index {hits[0][0]} ({hits[0][1]})")
            return hits[0][0]
        listing = ", ".join(f"[{i}] {n}" for i, n, _ in devices) or "none"
        what = "matches several cameras" if hits else "matches no camera"
        raise ValueError(f"Camera name '{spec}' {what}. Connected: {listing}")
    raise ValueError(f"Camera '{spec}' is neither an index nor a URL "
                     "(name lookup is only supported on macOS).")


def _backend_for(source):
    if isinstance(source, str):
        return cv2.CAP_FFMPEG
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    if IS_LINUX:
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def fit_to(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to width:height aspect, then resize to (width, height)."""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    target = width / height
    if abs(w / h - target) > 1e-3:
        if w / h > target:                       # too wide → crop sides
            new_w = int(round(h * target))
            x0 = (w - new_w) // 2
            img = img[:, x0:x0 + new_w]
        else:                                    # too tall → crop top/bottom
            new_h = int(round(w / target))
            y0 = (h - new_h) // 2
            img = img[y0:y0 + new_h]
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


class OpenCVCamera:
    """Threaded latest-frame grabber for a camera index or stream URL."""

    def __init__(self, source, *, name="camera", width=640, height=480,
                 fps=30, flip="none", open_timeout=10.0):
        self.source    = parse_source(source)
        self.name      = name
        self.width     = width
        self.height    = height
        self.flip_code = _FLIP_CODES.get(flip)
        self._frame    = None
        self._frame_t  = 0.0
        self._lock     = threading.Lock()
        self._stop     = threading.Event()

        if IS_LINUX and isinstance(self.source, int):
            dev = f"/dev/video{self.source}"
            if not os.path.exists(dev):
                present = sorted(p for p in os.listdir("/dev") if p.startswith("video"))
                raise FileNotFoundError(
                    f"[{name}] {dev} does not exist. Available: {present}. "
                    "Check with `v4l2-ctl --list-devices`."
                )

        cap = cv2.VideoCapture(self.source, _backend_for(self.source))
        t0 = time.time()
        while not cap.isOpened() and time.time() - t0 < open_timeout:
            time.sleep(0.25)
            cap.open(self.source, _backend_for(self.source))
        if not cap.isOpened():
            hint = ""
            if IS_MAC and isinstance(self.source, int):
                hint = (" On macOS, grant your terminal camera access in System "
                        "Settings → Privacy & Security → Camera, then run "
                        "`uv run cameras.py` to find the right index.")
            raise RuntimeError(f"[{name}] could not open camera {self.source!r}.{hint}")

        if isinstance(self.source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        print(f"[{name}] opened {self.source!r} "
              f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} capture → "
              f"{width}x{height})")

        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"{name}-grabber")
        self._thread.start()

    def _loop(self):
        fails = 0
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok or img is None:
                fails += 1
                if fails % 50 == 1:
                    print(f"[{self.name}] read failed (x{fails})")
                time.sleep(0.02)
                continue
            fails = 0
            img = fit_to(img, self.width, self.height)
            if self.flip_code is not None:
                img = cv2.flip(img, self.flip_code)
            with self._lock:
                self._frame   = img
                self._frame_t = time.monotonic()

    def read(self, max_age_s: float = 1.0):
        """Newest frame (BGR), or None if nothing fresh has arrived."""
        with self._lock:
            if self._frame is None or time.monotonic() - self._frame_t > max_age_s:
                return None
            return self._frame

    # RealSenseCapture-compatible names so either can be the scene camera.
    get_latest_color = read

    def close(self):
        if self._cap is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()
        self._cap = None

    release = close
