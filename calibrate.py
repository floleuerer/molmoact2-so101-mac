"""
calibrate.py — run LeRobot's SO-101 follower calibration with the settings
inference.py expects (robot type so101_follower, id so_follower), and the
serial port detected automatically.

    uv run calibrate.py                          # auto-detect the port
    uv run calibrate.py --port /dev/tty.usbmodemXXXX

The arm's power supply must be connected (USB alone doesn't power the servos).
"""
import argparse
import os
import sys

from molmoact_so101.setup.ports import resolve_follower_port


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="auto",
                   help="Serial port of the follower arm (default: auto-detect).")
    p.add_argument("--id", default="so_follower",
                   help="Calibration id. inference.py uses 'so_follower'.")
    args = p.parse_args()

    port = resolve_follower_port(args.port)
    cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_calibrate",
           "--robot.type=so101_follower", f"--robot.id={args.id}",
           f"--robot.port={port}"]
    print("[Calibrate]", " ".join(cmd[2:]), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
