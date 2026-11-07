"""ports.py — locate the SO-101 follower's serial port."""
import glob
import platform

_PATTERN = "/dev/tty.usbmodem*" if platform.system() == "Darwin" else "/dev/ttyACM*"


def list_serial_ports():
    return sorted(glob.glob(_PATTERN))


def resolve_follower_port(port: str = "auto") -> str:
    """'auto' → the only matching USB serial port; otherwise return `port` unchanged."""
    if port != "auto":
        return port
    found = list_serial_ports()
    if len(found) == 1:
        print(f"[Port] Follower port: {found[0]}")
        return found[0]
    if not found:
        raise SystemExit(
            f"No arm found ({_PATTERN} matched nothing). Is the arm's USB cable "
            "plugged in? (Servo power is separate — it's needed too, but doesn't "
            "create the port.)"
        )
    raise SystemExit(
        f"Several serial ports found: {found}. Unplug the other device(s) (e.g. a "
        "leader arm) or pass the port explicitly. `uv run lerobot-find-port` "
        "identifies the right one."
    )
