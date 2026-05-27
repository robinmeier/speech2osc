#!/usr/bin/env python3
"""
speech2osc.py  —  whisper.cpp stream → OSC → Max/MSP
Offline, real-time speech transcription on Apple Silicon.

Single command: python speech2osc.py
First run builds whisper.cpp, downloads the model, and lets you pick your
audio input. Config is saved so subsequent runs go straight to transcription.

Max patch minimal setup:
  [udpreceive 7400]
  |
  [OSC-route /transcript]
  |
  [print]

Examples:
  python speech2osc.py                   # first run: setup + pick device
  python speech2osc.py --port 9000       # different OSC port
  python speech2osc.py --model small     # higher accuracy
  python speech2osc.py --language fr     # force French
  python speech2osc.py --reset-device    # re-pick audio input
  python speech2osc.py --list-devices    # list SDL2 capture devices and exit
  python speech2osc.py --test-osc        # send a test OSC message and exit
  python speech2osc.py --setup           # force reinstall / rebuild
"""

import os
import sys

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Re-launch under a managed uv venv so all imports are available.
# The venv lives at ~/.speech2osc/venv and is created automatically.

_BASE_DIR  = os.path.expanduser("~/.speech2osc")
_VENV_DIR  = os.path.join(_BASE_DIR, "venv")
_VENV_PY   = os.path.join(_VENV_DIR, "bin", "python")
_DEPS      = ["python-osc", "pysdl2", "pysdl2-dll"]


def _bootstrap() -> None:
    import shutil
    import subprocess

    print("\033[36m[bootstrap] Configuring Python environment…\033[0m", flush=True)
    os.makedirs(_BASE_DIR, exist_ok=True)

    # Ensure uv is available
    if not shutil.which("uv"):
        if not shutil.which("brew"):
            print("\033[31mHomebrew not found — install from https://brew.sh\033[0m")
            sys.exit(1)
        print("\033[36m[bootstrap] Installing uv via Homebrew…\033[0m", flush=True)
        subprocess.run(["brew", "install", "uv"], check=True)

    # Create venv once
    if not os.path.exists(_VENV_PY):
        print("\033[36m[bootstrap] Creating venv…\033[0m", flush=True)
        subprocess.run(["uv", "venv", _VENV_DIR, "--python", "python3"], check=True)

    # Install / sync Python deps
    subprocess.run(
        ["uv", "pip", "install", "--python", _VENV_PY, "--quiet"] + _DEPS,
        check=True,
    )

    print("\033[36m[bootstrap] Re-launching under managed venv…\033[0m", flush=True)
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)
    sys.exit(0)  # unreachable, silences type checkers


if os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    _bootstrap()

# ─────────────────────────────────────────────────────────────────────────────
# All imports below are safe — we are now inside the managed venv.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path


BASE       = Path(_BASE_DIR)
WHISPER    = BASE / "whisper.cpp"
MODELS     = WHISPER / "models"
CONFIG     = BASE / "config.json"

MODEL_URLS = {
    "tiny":   "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
    "base":   "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
    "small":  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
    "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
}
MODEL_MB   = {"tiny": 75, "base": 142, "small": 466, "medium": 1532}

_ANSI      = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_TIMESTAMP = re.compile(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]")
_SPECIAL   = re.compile(r"\[(?:BLANK_AUDIO|MUSIC|NOISE|_[A-Z_]+_)\]")


# ── Colour helpers ────────────────────────────────────────────────────────────

def _c(msg: str, code: str) -> str:
    return f"\033[{code}m{msg}\033[0m"

def info(msg: str)  -> None: print(_c(msg, "36"),  flush=True)
def ok(msg: str)    -> None: print(_c(msg, "32"),  flush=True)
def warn(msg: str)  -> None: print(_c(msg, "33"),  flush=True)
def err(msg: str)   -> None: print(_c(msg, "31"),  flush=True)
def bold(msg: str)  -> None: print(_c(msg, "1"),   flush=True)


# ── System dependencies ───────────────────────────────────────────────────────

def _run(*cmd: str, cwd: Path | None = None) -> None:
    subprocess.run(list(cmd), check=True, cwd=cwd)


def ensure_brew_deps() -> None:
    import shutil
    if not shutil.which("brew"):
        err("Homebrew not found — install from https://brew.sh")
        sys.exit(1)
    missing = [
        p for p in ("cmake", "git", "sdl2")
        if subprocess.run(["brew", "list", p], capture_output=True).returncode != 0
    ]
    if missing:
        info(f"brew install {' '.join(missing)}")
        _run("brew", "install", *missing)


# ── whisper.cpp ───────────────────────────────────────────────────────────────

def _stream_binary() -> Path | None:
    # Prefer whisper-stream; plain 'stream' is a deprecated shim that exits immediately.
    for rel in ("build/bin/whisper-stream", "build/bin/stream", "build/stream"):
        p = WHISPER / rel
        if p.exists():
            return p
    return None


def _ncpu() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())


def _clone() -> None:
    if WHISPER.exists():
        ok("whisper.cpp already cloned.")
        return
    info("Cloning whisper.cpp…")
    _run("git", "clone", "https://github.com/ggerganov/whisper.cpp", str(WHISPER))


def _build() -> Path:
    info("Building whisper.cpp with Metal + SDL2…")
    _run(
        "cmake", "-B", "build",
        "-DGGML_METAL=1",
        "-DWHISPER_SDL2=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        cwd=WHISPER,
    )
    _run(
        "cmake", "--build", "build", "--config", "Release", f"-j{_ncpu()}",
        cwd=WHISPER,
    )
    b = _stream_binary()
    if not b:
        err("Build succeeded but stream binary not found — check CMake output.")
        sys.exit(1)
    ok(f"Binary: {b}")
    return b


def _download_model(model: str) -> Path:
    path = MODELS / f"ggml-{model}.bin"
    if path.exists():
        ok(f"Model ggml-{model}.bin already present.")
        return path
    info(f"Downloading ggml-{model}.bin (~{MODEL_MB[model]} MB)…")
    MODELS.mkdir(parents=True, exist_ok=True)

    def _prog(count: int, block: int, total: int) -> None:
        pct = min(100, int(count * block * 100 / total))
        print(f"\r  {pct}%  ", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URLS[model], path, _prog)
    print()
    ok(f"Saved to {path}")
    return path


def setup(model: str = "base") -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    ensure_brew_deps()
    _clone()
    if not _stream_binary():
        _build()
    else:
        ok("Stream binary already built.")
    _download_model(model)
    ok("Setup complete.")


# ── Audio device selection ────────────────────────────────────────────────────

def _sdl2_lib_hint() -> None:
    """Help pysdl2 find Homebrew's SDL2 when pysdl2-dll doesn't cover ARM64."""
    if os.environ.get("PYSDL2_DLL_PATH"):
        return
    for p in ("/opt/homebrew/lib/libSDL2.dylib", "/usr/local/lib/libSDL2.dylib"):
        if os.path.exists(p):
            os.environ["PYSDL2_DLL_PATH"] = p
            return


def list_sdl2_devices() -> list[tuple[int, str]]:
    """Return [(sdl2_index, device_name), …] for all SDL2 capture devices."""
    import ctypes
    import sdl2

    _sdl2_lib_hint()
    if sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO) != 0:
        raise RuntimeError(sdl2.SDL_GetError().decode())
    try:
        n = sdl2.SDL_GetNumAudioDevices(1)  # 1 = capture
        return [
            (
                i,
                ctypes.cast(
                    sdl2.SDL_GetAudioDeviceName(i, 1), ctypes.c_char_p
                ).value.decode("utf-8", errors="replace"),
            )
            for i in range(n)
        ]
    finally:
        sdl2.SDL_Quit()


def pick_device(saved: int | None = None) -> tuple[int, str]:
    try:
        devices = list_sdl2_devices()
    except Exception as e:
        warn(f"SDL2 device listing failed: {e}")
        warn("Falling back to manual index entry.")
        idx = int(input("SDL2 capture device index (0 = system default): ").strip())
        return idx, f"device #{idx}"

    if not devices:
        err("No audio capture devices found.")
        sys.exit(1)

    bold("\nAudio input devices (SDL2 indices):")
    for idx, name in devices:
        tag = "  ← saved" if saved is not None and idx == saved else ""
        print(f"  [{idx}]  {name}{tag}")
    print()

    hi = len(devices) - 1
    prompt = f"Device number [0–{hi}]"
    if saved is not None:
        prompt += f"  (Enter = keep [{saved}])"
    prompt += ":  "

    while True:
        raw = input(prompt).strip()
        if raw == "" and saved is not None:
            name = devices[saved][1] if saved <= hi else f"device #{saved}"
            return saved, name
        if raw.isdigit() and 0 <= int(raw) <= hi:
            i = int(raw)
            return i, devices[i][1]
        print(f"  Please enter a number between 0 and {hi}.")


# ── Config ────────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return {}


def save_cfg(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, indent=2))


# ── Transcription loop ────────────────────────────────────────────────────────

def _p_cores() -> int:
    """Number of performance cores on Apple Silicon; falls back to half of total."""
    try:
        return int(
            subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
        )
    except subprocess.CalledProcessError:
        return max(1, _ncpu() // 2)


def _clean(raw: str) -> str:
    text = _ANSI.sub("", raw)
    text = _TIMESTAMP.sub("", text)
    text = _SPECIAL.sub("", text)
    return text.strip()


def run_transcription(
    binary: Path,
    model_path: Path,
    device_idx: int,
    *,
    host: str,
    port: int,
    address: str,
    language: str,
    threads: int,
) -> None:
    from pythonosc.udp_client import SimpleUDPClient

    client = SimpleUDPClient(host, port)
    if threads == 0:
        threads = _p_cores()

    lang = "auto" if language == "auto" else language

    cmd = [
        str(binary),
        "-m",          str(model_path),
        "-c",          str(device_idx),
        "-l",          lang,
        "-t",          str(threads),
        "--step",      "500",   # process every 500 ms
        "--length",    "5000",  # 5-second sliding window
        "--keep",      "200",   # overlap from previous window
        "--vad-thold", "0.6",   # silence / speech energy threshold
        "--freq-thold","100",   # high-pass cutoff (Hz) to ignore rumble
        "--no-fallback",        # don't retry with higher temperature
    ]

    sep = "─" * 54
    bold(f"\n{sep}")
    info(f"  Device   : [{device_idx}]")
    info(f"  Model    : {model_path.name}")
    info(f"  Language : {lang}")
    info(f"  Threads  : {threads} P-cores")
    info(f"  OSC      : {host}:{port}  {address}")
    bold(sep)
    print()
    print(f"  Max:  [udpreceive {port}] → [OSC-route {address}] → [print]")
    print()
    ok("Listening…  Ctrl+C to stop\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Use newline="\n" so that \r inside lines is preserved (not translated).
        # whisper.cpp stream uses \r to overwrite partial results in-terminal;
        # the committed segment is always the text after the last \r on each \n-line.
        reader = io.TextIOWrapper(
            proc.stdout,
            encoding="utf-8",
            errors="replace",
            newline="\n",
        )
        for raw_line in reader:
            segment = raw_line.split("\r")[-1] if "\r" in raw_line else raw_line
            text = _clean(segment)
            if text:
                ok(f"→  {text}")
                client.send_message(address, text)
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        proc.wait()
        warn("\nStopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="whisper.cpp stream → OSC → Max/MSP  (Apple Silicon, fully offline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--setup",        action="store_true",  help="Force reinstall / rebuild")
    ap.add_argument("--model",        default="base",       choices=list(MODEL_URLS),
                    help="Whisper model size (default: base)")
    ap.add_argument("--host",         default="127.0.0.1",  help="OSC target host (default: 127.0.0.1)")
    ap.add_argument("--port",         type=int, default=7400, help="OSC UDP port (default: 7400)")
    ap.add_argument("--address",      default="/transcript", help="OSC address (default: /transcript)")
    ap.add_argument("--language",     default="auto",        help="Language code or 'auto' (default: auto)")
    ap.add_argument("--threads",      type=int, default=0,   help="CPU threads — 0 = auto-detect P-cores")
    ap.add_argument("--device",       type=int, default=None,help="SDL2 capture device index — skip picker")
    ap.add_argument("--reset-device", action="store_true",   help="Re-run the interactive device picker")
    ap.add_argument("--list-devices", action="store_true",   help="Print SDL2 capture devices and exit")
    ap.add_argument("--test-osc",     action="store_true",   help="Send a test OSC message and exit")
    args = ap.parse_args()

    cfg = load_cfg()

    # ── --list-devices ────────────────────────────────────────────────────────
    if args.list_devices:
        try:
            devs = list_sdl2_devices()
        except Exception as e:
            err(f"Could not list devices: {e}")
            sys.exit(1)
        bold("SDL2 capture devices:")
        for i, name in devs:
            print(f"  [{i}]  {name}")
        return

    # ── --test-osc ────────────────────────────────────────────────────────────
    if args.test_osc:
        from pythonosc.udp_client import SimpleUDPClient
        SimpleUDPClient(args.host, args.port).send_message(
            args.address, "speech2osc test message"
        )
        ok(f"Sent test message → {args.host}:{args.port}{args.address}")
        return

    # ── Setup (auto or forced) ─────────────────────────────────────────────────
    model_path = MODELS / f"ggml-{args.model}.bin"
    if args.setup or not _stream_binary() or not model_path.exists():
        setup(args.model)

    binary = _stream_binary()
    if not binary:
        err("Stream binary not found. Run:  python speech2osc.py --setup")
        sys.exit(1)

    model_path = MODELS / f"ggml-{args.model}.bin"
    if not model_path.exists():
        model_path = _download_model(args.model)

    # ── Device selection ───────────────────────────────────────────────────────
    if args.device is not None:
        device_idx  = args.device
        device_name = f"device #{device_idx}"
    elif not args.reset_device and "device_index" in cfg:
        device_idx  = cfg["device_index"]
        device_name = cfg.get("device_name", f"device #{device_idx}")
        ok(f"Using saved device: [{device_idx}]  {device_name}")
        info("  Run with --reset-device to change.")
    else:
        device_idx, device_name = pick_device(cfg.get("device_index"))
        cfg.update({"device_index": device_idx, "device_name": device_name})
        save_cfg(cfg)

    run_transcription(
        binary,
        model_path,
        device_idx,
        host=args.host,
        port=args.port,
        address=args.address,
        language=args.language,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
