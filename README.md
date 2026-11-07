# MolmoAct2 on SO-101 — macOS / Apple Silicon edition

Zero-shot robot manipulation with [MolmoAct2](https://huggingface.co/allenai/MolmoAct2-SO100_101) from AI2 on a [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm, **running locally on a Mac with Apple Silicon**: no NVIDIA GPU, no RealSense, just ordinary USB webcams. You give it a natural-language prompt; it predicts joint targets from camera images and the current joint state. No training, no demonstrations.

> **Fork notice.** This is a macOS-focused fork of [irenegracekp/molmoact2-so101](https://github.com/irenegracekp/molmoact2-so101), which targets Linux + CUDA + RealSense. See it in action in [the original author's X post](https://twitter.com/irenekarrot/status/2053966402986647692), and read AI2's announcement at [allenai.org/blog/molmoact2](https://allenai.org/blog/molmoact2).
>
> What this fork adds:
> - Runs on **Apple Silicon (MPS)** in bfloat16; device is auto-detected
> - **Any USB webcam** as scene or wrist camera, selected **by name** (`--scene-cam C920`), plus network streams
> - **`--exec-mode sync`** for devices where one prediction takes longer than the 1 s chunk it produces (the default off CUDA)
> - **`--scene-only` without a wrist camera**
> - `cameras.py`: camera finder, live preview and **focus aid**
> - **uv** for dependency management (`pyproject.toml` + `uv.lock`)
>
> Linux + CUDA + RealSense still works. See [Linux / CUDA](#linux--cuda).

---

## Hardware

| Part | Notes |
|---|---|
| Mac with Apple Silicon | Tested: M4 Pro, 48 GB, macOS 15. The model uses ~11 GB of unified memory in bfloat16; 32 GB+ RAM recommended. |
| SO-101 follower arm | [TheRobotStudio SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), **with its own power supply** (USB alone does not power the servos) |
| Scene camera | Any USB webcam, e.g. Logitech C920 (tested), Logitech Brio or Razer Kiyo |
| Wrist camera (optional) | Any USB webcam, e.g. Innomaker U20CAM-1080p (tested) |
| Disk | ~22 GB for the model weights |

---

## Quickstart (macOS)

```bash
# 1. Install (uv fetches Python 3.12 if needed)
brew install uv            # if you don't have it
uv sync

# 2. Find the cameras
uv run cameras.py          # lists cameras by name, saves a snapshot of each to camera_snapshots/

# 3. Calibrate the arm (once; arm plugged in and powered; port is auto-detected)
uv run calibrate.py

# 4. Dry run: model runs, arm stays still
uv run inference.py --scene-cam C920 --scene-only --show --dry-run \
    --prompt "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."

# 5. Real run (start gentle)
uv run inference.py --scene-cam C920 --scene-only --show --max-step-deg 5 \
    --prompt "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl."
```

The first run downloads the model (~22 GB) from Hugging Face. Press `Ctrl+C` **once** to stop: the program finishes the current inference and then switches off torque, so **support the arm if it's raised**.

---

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/); Python 3.12 is required (LeRobot 0.5.x needs ≥ 3.12).

```bash
uv sync
```

Run everything through `uv run …`; there's no need to activate the venv.

| Package | Pin | Why |
|---|---|---|
| `lerobot[feetech]` | `==0.5.1` | Joint-angle convention, see [Calibration](#calibration). The `feetech` extra installs the servo SDK; without it `inference.py` silently falls back to a simulated arm. |
| `transformers` | `>=4.52.0` | MolmoAct2's processor imports `transformers.video_utils`. |
| `torch` | PyPI wheel | Includes the MPS backend on macOS. |
| `pyobjc-framework-avfoundation` | macOS only | Selecting cameras by name. |

**macOS notes**
- **Camera permission:** the first time, macOS asks whether your terminal may use the camera. If you dismissed it: System Settings → Privacy & Security → Camera → enable your terminal app, then restart it.
- If your shell sets `UV_ENV_FILE=.env`, `uv run` fails when `.env` doesn't exist. Fix it with `touch .env` (it's gitignored).
- `objc: Class AVFFrameReceiver is implemented in both …` at startup is harmless: OpenCV and PyAV both bundle libavdevice, but camera capture doesn't use it.

---

## Cameras

```bash
uv run cameras.py                     # name ↔ index mapping + one snapshot per camera
uv run cameras.py --show              # live preview of every camera
uv run cameras.py --focus Innomaker   # focus aid: live view with a sharpness score
uv run cameras.py --url rtsp://camera.local:8554/stream   # test a network stream
```

`--scene-cam` / `--wrist-cam` accept:
- a **name substring** such as `C920`, `Innomaker` or `Brio` (recommended). macOS renumbers cameras whenever something is re-plugged, and the order shown by ffmpeg / System Information is *not* OpenCV's order. Names are resolved to the right index at startup.
- an **OpenCV index** such as `1`
- a **stream URL** (`rtsp://`, `tcp://`, `http://`) such as `rtsp://camera.local:8554/stream`. A background thread keeps only the newest frame, so stream buffering doesn't add latency.

Every frame is centre-cropped to 4:3 and resized to 640×480 (the training resolution), so 16:9 webcams, camera-as-webcam apps or an iPhone via Continuity Camera are fine.

**Wrist orientation:** `--wrist-flip` defaults to `180` (the original author's camera is mounted upside down). If your wrist image looks upside down in `--show`, use `--wrist-flip none`. The focus aid always shows the unrotated image.

**Focusing the wrist camera:** run `uv run cameras.py --focus <name>`, hold something textured in the yellow box at grasp distance (5–15 cm from the gripper), and turn the lens until the score peaks. Keys: `R` resets the best score, `Q`/`Esc` quits. Lock the lens afterwards; arm vibration will slowly shift it.

---

## Calibration

**Power first.** The arm's power supply must be connected; USB only powers the controller board. Without servo power, LeRobot reports `Missing motor IDs: 1 … 6` / `found motor list: {}`.

```bash
uv run calibrate.py                                  # auto-detects the arm's port
uv run calibrate.py --port /dev/tty.usbmodemXXXX     # or give it explicitly
```

`calibrate.py` runs LeRobot's calibration wizard with the settings `inference.py` expects (`--robot.type=so101_follower --robot.id=so_follower`).

### Finding the arm's port

Auto-detection (in `calibrate.py` and `inference.py`) works when exactly **one** arm is connected over USB. If it reports several ports, or you want to check by hand:

```bash
ls /dev/tty.usbmodem*        # macOS: one entry per connected arm, e.g. /dev/tty.usbmodem58760435711
ls /dev/ttyACM*              # Linux
uv run lerobot-find-port     # interactive: unplug the arm's USB when asked; the port that disappears is the arm
```

- **No port at all** → the USB cable isn't connected, or it's a charge-only cable.
- **Several ports** → usually a leader arm or another USB-serial device. Unplug it, or pass `--port` / `--follower-port` explicitly.
- The name stays the same as long as you use the same controller board, so you can keep it in your shell history.
- Use the `/dev/tty.*` name on macOS, not `/dev/cu.*`.

### The calibration wizard

1. Move every joint to the middle of its range and press Enter.
2. Move **every joint end to end**, including the **gripper fully open and fully closed**, then press Enter.

The result is written to `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so_follower.json`, which `inference.py` reads (robot id `so_follower`). Sanity check: each joint except `wrist_roll` should cover roughly 160–220°, and the gripper roughly 100–130°. A gripper range of ~40° means it wasn't opened fully, and the gripper won't open far enough to grasp.

`configs/so_follower.json` is the original author's arm, kept for reference only. Don't use it for your arm.

### ⚠️ LeRobot version gotcha

MolmoAct2 was trained on LeRobot datasets using the v2.1 joint-angle convention; LeRobot 0.5.x calibrates in the v3.0 convention. The default `--joint-offsets 0,90,90,0,0,0` and `--joint-signs 1,-1,1,1,1,1` apply the [official conversion](https://huggingface.co/docs/lerobot/backwardcomp), but only for calibrations made with **exactly lerobot 0.5.1** (pinned in `uv.lock`). Any other version can make **the arm slam into the table on startup**.

---

## Running

```bash
uv run inference.py --scene-cam C920 [--wrist-cam Innomaker | --scene-only] --prompt "..." [options]
```

### How it runs on a Mac

On an M4 Pro one prediction takes **~2.2 s** (the first ~3–4 s) and yields a 30-step chunk, i.e. 1 s of motion. Because prediction is slower than execution, the default is **`--exec-mode sync`**: observe → predict (arm holds still) → play the chunk → repeat. Expect stop-and-go motion. The async temporal-ensembling mode from the original repo needs inference under ~1 s (a CUDA GPU).

Each chunk logs how far it wants to move:
```
[Producer] 2180 ms  id=4  Δstart=[...]  Δend=[...]  max|Δ|=23.4°
```
If `max|Δ|` stays at a few degrees chunk after chunk, the model is predicting "stay still". See [Getting the model to move](#getting-the-model-to-move).

### Key options

| Flag | Default | Description |
|---|---|---|
| `--prompt` | *(required)* | Natural-language task instruction |
| `--scene-cam` | *(required on macOS)* | Scene camera: name, index or URL (`realsense` on Linux) |
| `--wrist-cam` | *(required unless `--scene-only`)* | Wrist camera: name, index or URL |
| `--scene-only` | off | Feed the scene image into both model inputs; no wrist camera needed |
| `--wrist-flip` / `--scene-flip` | `180` / `none` | `none`, `180`, `v` (vertical) or `h` (mirror) |
| `--follower-port` | `auto` | Picks the only `/dev/tty.usbmodem*`, see [Finding the arm's port](#finding-the-arms-port) |
| `--device` | `auto` | `cuda` → `mps` → `cpu` |
| `--dtype` | `bfloat16` | `float32` also fits in 48 GB, but is ~2× slower |
| `--exec-mode` | `auto` | `sync` off CUDA, `async` on CUDA |
| `--actions-per-chunk` | 30 (all) | Execute only the first N steps of each chunk before re-observing |
| `--max-step-deg` | `15.0` | Per-tick joint motion cap; use ~5 for first tests |
| `--joint-min` / `--joint-max` | none | Hard per-joint limits in arm frame, e.g. `none,-65,none,none,none,none` |
| `--num-steps` | `10` | Flow-solver iterations |
| `--dry-run` | off | Run the model and log actions; the arm does not move |
| `--show` | off | Camera preview windows (`Q` to quit) |
| `--save-frames-dir` | off | Save the exact images fed to the model |

### Getting the model to move

MolmoAct2 is used zero-shot here, so how closely your scene resembles its training data decides almost everything. The training data (community SO-100/101 LeRobot datasets) looks like this: **bright, even light; a light, plain table; chunky colourful objects (fruit, bowls, cubes); a side view at table height and/or a top-down view showing the whole arm and the objects.** In practice:

1. **Light**: bright and even, not dim evening light.
2. **Table**: put white paper or a light board under the workspace; clear the background.
3. **Camera**: show the **whole arm and the objects**, either from the side at table height (arm on one side, objects in front of it) or from above.
4. **Objects**: chunky and colourful. Thin items like pens are hard.
5. **Prompt**: phrase it like the training tasks: `"Move the arm towards the <object>, grasp it, lift it up, and drop it into the <container>."`
6. **Use `--dry-run`** while adjusting the scene, and drop it once `max|Δ|` regularly exceeds ~15–20°.

Demonstrations often begin with a short pause, so a chunk predicted from a resting arm frequently has its motion only in the second half. Seeing nothing at all for a few chunks, while the official sample does move, is a sign the scene is too far from the training data.

To check the model itself works on your Mac, run the official sample from the [model card](https://huggingface.co/allenai/MolmoAct2-SO100_101). It should predict 15–45° of shoulder/elbow motion within a chunk.

---

## Linux / CUDA

The original Linux setup still works:

```bash
uv sync --extra realsense        # adds pyrealsense2
uv run inference.py --follower-port /dev/ttyACM0 --wrist-cam 8 \
    --prompt "pick up the lemon and drop it in the red bowl"
```

- `torch` comes from the PyTorch CUDA 12.8 index (change the `pytorch-cuda` URL in `pyproject.toml` and `uv lock` for another CUDA version).
- Defaults on Linux: `--scene-cam realsense`, `--wrist-cam 8`, `--exec-mode async` (~700 ms per chunk on an RTX 3080-class GPU), with optional `--cuda-graph`.
- The RealSense D455 needs a USB 3 data cable; on USB 2.1 it runs at 15 fps.
- On Linux the wrist camera's white balance is locked via `v4l2-ctl` (`sudo apt install v4l-utils`), and an adaptive brightness controller runs (disable it with `--no-wrist-ae`).

---

## Known limitations

- **Zero-shot is fragile.** A new arm, room, camera angle and lighting are all out of distribution. If it won't work reliably, the next step is recording demonstrations and fine-tuning.
- **Wrist images are out of distribution.** MolmoAct2-SO100_101 was trained on two third-person views, so `--scene-only` is often as good or better.
- **Stop-and-go on Mac.** Inference (~2.2 s) is slower than a chunk (1 s), so the arm pauses between chunks.
- **No depth.** RGB only (`enable_depth_reasoning=False`).
- **Single arm, no teleop fallback.**

---

## Troubleshooting

**`Missing motor IDs: 1 … 6` / `found motor list: {}`**
The servos have no power. Connect the arm's power supply (USB alone is not enough) and check the 3-pin cable from the board to the base servo.

**Arm jerks to one position at startup, then a motor stops responding**
Calibration mismatch, see the [LeRobot version gotcha](#️-lerobot-version-gotcha). Power-cycle the arm and recalibrate with the pinned lerobot.

**Camera can't be opened / shows black**
Check the camera permission (above), run `uv run cameras.py`, and pass the camera **by name**. A camera that shows a nearly black image is usually covered or facing something dark.

**`Camera name 'X' matches no camera`**
The camera isn't connected or is named differently. The error lists the connected cameras.

**Arm never moves; `max|Δ|` stays at a few degrees**
The model predicts "stay still". See [Getting the model to move](#getting-the-model-to-move).

**Arm never moves; log says `no active chunks` / `inference is slower than the chunk duration`**
You're in `--exec-mode async` on a slow device. Use `sync` (the default off CUDA).

**Gripper barely opens**
The gripper calibration range is too small. Recalibrate and open the gripper fully during the range step.

**`ImportError: cannot import name 'Teleoperator'` / wrong lerobot**
Run `uv sync` to restore the pinned versions from `uv.lock`.

---

## Credits

- Original project: [irenegracekp/molmoact2-so101](https://github.com/irenegracekp/molmoact2-so101) (MIT)
- Model: [allenai/MolmoAct2-SO100_101](https://huggingface.co/allenai/MolmoAct2-SO100_101) by AI2
- Arm: [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100) by TheRobotStudio, driven via [LeRobot](https://github.com/huggingface/lerobot)
