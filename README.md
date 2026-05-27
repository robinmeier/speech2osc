# speech2osc

Offline, real-time speech transcription on Apple Silicon → Max/MSP via OSC.

**One command.** Automatic setup. No cloud. ~0.5–2 s latency. Fully offline.

```bash
python speech2osc.py
```

First run: installs dependencies, builds whisper.cpp with Metal acceleration, downloads the model, and lets you pick your audio input. Subsequent runs go straight to transcription.

---

## What it does

- Captures audio from any input device (built-in mic, USB audio interface, etc.)
- Transcribes in real-time using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) stream binary (Metal-accelerated on Apple Silicon)
- Sends each finalized utterance as an OSC message over UDP
- Receives it in Max/MSP with `[udpreceive]` and `[OSC-route]`

**Why:** Co-locate speech-to-text and text-to-speech on the same machine for a real voice loop. No latency, no cloud, no privacy concerns.

---

## Setup

**Requirements:**
- macOS with Apple Silicon (M1, M2, M3, M4) or Intel
- Homebrew (`brew.sh`)
- Python 3.9+

**First run:**
```bash
python speech2osc.py
```

This will:
1. Install `uv` (Python package manager) via Homebrew
2. Create a managed Python venv at `~/.speech2osc/venv`
3. Install Python dependencies (`python-osc`, `pysdl2`, `pysdl2-dll`)
4. Clone and build whisper.cpp with Metal + SDL2 support
5. Download the base model (~142 MB)
6. Show you a list of audio input devices and let you pick one

Config is saved, so next runs are instant.

---

## Usage

**Basic transcription:**
```bash
python speech2osc.py
```

**Common options:**

```bash
# Re-pick your audio input device
python speech2osc.py --reset-device

# Higher accuracy (slower, ~1 s latency)
python speech2osc.py --model small

# Different OSC target
python speech2osc.py --port 9000 --address /speech

# Force a language (e.g., French)
python speech2osc.py --language fr

# Test OSC connectivity to Max
python speech2osc.py --test-osc

# Force rebuild / reinstall everything
python speech2osc.py --setup
```

Full help:
```bash
python speech2osc.py --help
```

---

## Max/MSP setup

Minimal patch to receive transcriptions:

```
[udpreceive 7400]
|
[OSC-route /transcript]
|
[print]
```

Or with `[textedit]` or `[message]` to capture the output.

**OSC message format:**
- Default port: `7400`
- Default address: `/transcript`
- Message content: the transcribed text (string)

Example output:
```
/transcript "Hello world"
/transcript "How are you"
```

---

## How it works

```
Audio input (any device)
    ↓
whisper.cpp stream binary (-DGGML_METAL=1, Metal acceleration)
    ↓ stdout (finalized transcriptions)
Python wrapper (parse & clean output)
    ↓
python-osc (UDP)
    ↓
Max [udpreceive] → [OSC-route]
```

**Latency:** ~0.5–2 seconds end-to-end on Apple Silicon. Depends on model size and audio length.

**Languages:** Auto-detect or lock to a specific language.

**Models:**
- `tiny` (39M) — fastest, lowest accuracy
- `base` (74M) — **default**, good balance
- `small` (244M) — better accuracy
- `medium` (769M) — highest accuracy, slow

---

## Where data lives

Everything is isolated in `~/.speech2osc/`:

```
~/.speech2osc/
├── venv/                    # Python virtual environment
├── whisper.cpp/             # whisper.cpp source + build + models
│   ├── build/               # Compiled binaries
│   └── models/              # Downloaded ggml-*.bin files
└── config.json              # Saved device preference
```

**To remove everything:**
```bash
rm -rf ~/.speech2osc/
```

**To remove just models:**
```bash
rm ~/.speech2osc/whisper.cpp/models/ggml-*.bin
```

---

## Troubleshooting

**"Stream binary not found"**
- Run `python speech2osc.py --setup` to force rebuild.

**"No audio capture devices found"**
- Check that your audio device is recognized by the system.
- Run `python speech2osc.py --list-devices` to list SDL2 capture devices.
- Try `--reset-device` to re-pick.

**"ModuleNotFoundError: No module named 'pythonosc'"**
- The venv wasn't created properly. Try: `rm -rf ~/.speech2osc/venv` then `python speech2osc.py`.

**Audio goes silent or transcription stops**
- Press Ctrl+C and re-run. The script should reconnect to your device.

**"WARNING: The binary 'stream' is deprecated"**
- Update the script to the latest version (this should be fixed).

---

## Performance

On Apple Silicon with the base model:
- **Latency:** 0.5–1.5 s (speech → transcription → OSC message)
- **CPU:** ~20–30% (one performance core)
- **Memory:** ~200–300 MB
- **Accuracy:** ~90% for English

Larger models are slower but more accurate; smaller models are faster but less accurate.

---

## License

MIT. Use freely.

---

## Credits

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — OpenAI's Whisper in C++, optimized for CPU/Metal
- [python-osc](https://github.com/attwad/python-osc) — OSC client
- [pysdl2](https://github.com/marcusva/py-sdl2) — SDL2 bindings for audio device listing

---

## Contributing

Issues, PRs, and feedback welcome. Star if you find it useful!
