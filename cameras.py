"""
cameras.py — find the wrist and scene cameras.

Probes OpenCV camera indices (and optionally stream URLs), saves one snapshot
per working camera, and optionally shows a live preview, so you can tell which
index is the wrist camera and which is the scene camera.

    uv run cameras.py                       # probe indices 0..7, save snapshots
    uv run cameras.py --show                # + live preview of every camera
    uv run cameras.py --url tcp://raspberrypi.local:8888   # also test a stream
    uv run cameras.py --focus Innomaker     # live focus aid for one camera

Then pass the indices / URL to inference.py:
    uv run inference.py --wrist-cam <idx|url> --scene-cam <idx> --prompt "..."
"""
import argparse
import os
import time

import cv2
import numpy as np

from molmoact_so101.setup.cameras import IS_MAC, OpenCVCamera, mac_video_devices


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-index", type=int, default=8,
                   help="Linux: probe camera indices 0..max-index-1.")
    p.add_argument("--url", action="append", default=[],
                   help="Also test this stream URL (repeatable).")
    p.add_argument("--out", default="camera_snapshots",
                   help="Directory for one snapshot per camera.")
    p.add_argument("--show", action="store_true",
                   help="Show a live preview window per camera (press Q to quit).")
    p.add_argument("--focus", metavar="CAM",
                   help="Live focus aid for one camera (index, name or URL): "
                        "full-resolution view with a sharpness score. Turn the "
                        "lens until the score peaks.")
    args = p.parse_args()

    if args.focus is not None:
        return focus_aid(args.focus)

    if IS_MAC:
        devices = mac_video_devices()
        print("Cameras (OpenCV index → name). You can pass either to inference.py;")
        print("names are safer because indices shift when cameras are re-plugged.")
        for i, n, _ in devices:
            print(f"  [{i}] {n}")
        print()
        indices = [i for i, _, _ in devices]
    else:
        indices = list(range(args.max_index))

    sources = indices + args.url
    cams = {}
    for src in sources:
        try:
            cam = OpenCVCamera(src, name=f"cam {src}", open_timeout=2.0 if isinstance(src, int) else 10.0)
        except Exception as e:
            if isinstance(src, str):
                print(f"[cam {src}] FAILED: {e}")
            continue
        cams[src] = cam

    if not cams:
        print("No cameras opened.")
        if IS_MAC:
            print("On macOS, allow your terminal app camera access: System Settings → "
                  "Privacy & Security → Camera, then restart the terminal.")
        return

    os.makedirs(args.out, exist_ok=True)
    print("\nWaiting for frames...")
    deadline = time.time() + 6.0
    pending = set(cams)
    while pending and time.time() < deadline:
        for src in list(pending):
            img = cams[src].read()
            if img is not None and (np.mean(img) > 3.0 or time.time() > deadline - 0.5):
                safe = str(src).replace("/", "_").replace(":", "_")
                path = os.path.join(args.out, f"cam_{safe}.jpg")
                cv2.imwrite(path, img)
                dark = "   (very dark — lens covered?)" if np.mean(img) <= 3.0 else ""
                print(f"  {src!r:>40}  →  {path}{dark}")
                pending.discard(src)
        time.sleep(0.1)
    for src in pending:
        print(f"  {src!r:>40}  →  opened but no frames")

    if args.show:
        print("\nLive preview — press Q in a window to quit.")
        try:
            while True:
                for src, cam in cams.items():
                    img = cam.read()
                    if img is not None:
                        cv2.imshow(f"camera {src}", img)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()

    for cam in cams.values():
        cam.close()


def focus_aid(source):
    """Live view at native resolution with a centre-region sharpness score."""
    cam = OpenCVCamera(source, name="focus", width=1280, height=960)
    best = 0.0
    print("Turn the lens until the score peaks (best is kept as a reference). "
          "Q / Esc to quit, R to reset best.")
    try:
        while True:
            img = cam.read()
            if img is None:
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    break
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
            score = float(cv2.Laplacian(roi, cv2.CV_64F).var())
            best = max(best, score)
            view = img.copy()
            cv2.rectangle(view, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 2)
            bar = int((w - 40) * min(score / max(best, 1e-6), 1.0))
            cv2.rectangle(view, (20, h - 50), (20 + bar, h - 25), (0, 200, 0), -1)
            cv2.putText(view, f"sharpness {score:7.0f}   best {best:7.0f}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.imshow(f"focus {source}", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                best = 0.0
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
