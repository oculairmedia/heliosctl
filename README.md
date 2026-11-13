# heliosctl

Audio-reactive ILDA laser controller, real-time DSP engine, and offline ILDA inspector for the **Bitlasers Helios Laser DAC** over USB on Linux.

Built for live performance and music-synchronized laser projection with strict galvo scanner safety enforcement, zero flicker (45–50 Hz refresh rates), and a non-blocking local Unix-socket control plane for real-time parameter tuning.

---

## Architecture

`heliosctl` separates physical hardware/protocol safety from audio DSP and visual geometry generation:

```
┌────────────────────────────────────────────────────────┐
│                      heliosctl                         │
│   (CLI entrypoint: devices, inspect, preview, audio)   │
└───────────────┬────────────────────────┬───────────────┘
                │                        │
     ┌──────────▼──────────┐  ┌──────────▼──────────┐
     │      Audio DSP      │  │     Live Control    │
     │      (liquid)       │  │    (live_control)   │
     │  - NumPy/SciPy STFT │  │  - Non-blocking IPC │
     │  - Multi-band flux  │  │  - tune, stop, renew│
     │  - 6s Comb PLL sync │  │  - Session watchdog │
     │  - Macro-activity   │  └─────────────────────┘
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  Geometric Shaders  │
     │  - Concentric liquid│
     │  - Directional wave │
     │  - Spectral color   │
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │    Scanner Safety   │
     │  - 20kpps / <500pts │  --> 45-50 Hz flicker-free
     │  - Segment < 225    │  --> Prevents galvo strain
     │  - Curvature < 15   │  --> Prevents mirror whip
     │  - Blanking jumps   │  --> Zero transit lines
     └──────────┬──────────┘
                │
     ┌──────────▼──────────┐
     │  Helios C++ ctypes  │
     │  (libHeliosDacAPI)  │
     │  - SINGLE_MODE = 2  │  --> Fails dark on exit
     │  - Safe Stop/Close  │
     └─────────────────────┘
```

---

## Hardware & Scanner Safety Rules

1. **Galvo Scanner Limits (20 kpps):**
   - **Point Budget:** Frames are capped at **$\le$ 500 points total** to guarantee **$\ge$ 40 Hz refresh rate**. Above 600 points, galvo mirrors visibly flicker at 20 kpps.
   - **Segment Distance:** Maximum adjacent point distance is strictly capped at **$< 225$ ILDA units** to protect scanner mirrors from mechanical overshoot and thermal strain.
   - **Curvature:** Second-difference curvature is capped at **$< 15$ ILDA units**.
   - **Safety Envelope:** Coordinates are constrained to $|x| \le 16000$ and $|y| \le 13000$.
2. **Blanking Transitions:**
   - The laser diode is completely blanked (`r=0, g=0, b=0, i=0, blank=True`) during transit jumps between separate shapes (e.g. between the center sphere and outer expanding radio rings), ensuring zero drag streaks on the wall.
3. **Fail-Dark DAC Lifecycle:**
   - Frames are submitted with `HELIOS_FLAG_SINGLE_MODE = 2`. If playback stops or the process terminates, the DAC **never loops the last frame** (preventing static hot spots).
   - `SIGINT` and `SIGTERM` always trigger explicit `sdk.stop(0)` and `sdk.close()` calls.

---

## Hardware Setup (Linux)

### 1. USB udev Rule
Allow non-root user access to the Helios USB DAC:

```bash
printf '%s\n' 'SUBSYSTEM=="usb", ATTR{idVendor}=="1209", ATTR{idProduct}=="e500", TAG+="uaccess"' | \
  sudo tee /etc/udev/rules.d/70-helios.rules
sudo udevadm control --reload-rules
```
Unplug and reconnect the Helios USB cable.

### 2. Microphone Configuration
For live audio-reactivity via PipeWire / ALSA, ensure the capture preamp is not railed. On Surface Pro 7 or similar internal mics, set clean unclipped gain:

```bash
amixer -c 0 cset numid=8 2,2    # Internal Mic Boost = +20 dB (not +30 dB saturation)
amixer -c 0 cset numid=6 45,45   # Capture Volume = +16.5 dB
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/oculairmedia/heliosctl.git
cd heliosctl

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x heliosctl
```

### Running the Live Visualizer

Start the 5-minute living concentric radio-wave visualizer:

```bash
./heliosctl audio --live-control --pattern liquid --gain 3 --base-size 1.0
```

### Live Control & Real-Time Tuning

From another terminal (or an agent/script) while the laser is running:

```bash
./heliosctl status
./heliosctl tune --gain 2.5 --ripple-depth .08 --frequency 6
./heliosctl renew 300    # replace the deadline with 300 s from now (not needed when --duration is omitted)
./heliosctl stop         # cleanly shuts off laser output
```

### Playing & Inspecting ILDA Files

```bash
# Inspect an ILDA Format 5 file offline
./heliosctl inspect tests/rotating-orbit.ild

# Render offline PNG or GIF preview
./heliosctl preview tests/rotating-orbit.ild --out preview.gif

# Play an ILDA file for a fixed duration (seconds)
./heliosctl play tests/rotating-orbit.ild --duration 10
```

---

## Running Tests

All 50 tests run completely offline without requiring hardware access:

```bash
python3 -m unittest discover -s tests -q
```

---

## License

MIT © 2026 [Oculair Media](https://oculair.ca) / Emmanuel Umukoro
