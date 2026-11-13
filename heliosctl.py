#!/usr/bin/env python3
"""Small, safety-bounded Helios ILDA Format 5 utility."""
import argparse
import ctypes
import math
import os
import random
import signal
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MAX_FILE = 64 * 1024 * 1024
MAX_FRAMES = 10000
MAX_POINTS = 100000
MAX_TOTAL_POINTS = 2_000_000
MAX_PPS = 40000


class CliError(Exception):
    pass


@dataclass
class Frame:
    index: int
    points: list


def parse_ilda5(path):
    p = Path(path)
    size = p.stat().st_size
    if size > MAX_FILE:
        raise CliError(f"file exceeds {MAX_FILE} byte limit")
    data = p.read_bytes()
    pos = 0
    frames = []
    expected_total = None
    total_points = 0
    ended = False
    while pos < len(data):
        if len(data) - pos < 32:
            raise CliError("truncated ILDA header")
        h = data[pos:pos + 32]
        if h[:4] != b"ILDA":
            raise CliError(f"invalid ILDA header at byte {pos}")
        fmt = h[7]
        if fmt != 5:
            raise CliError(f"unsupported ILDA format {fmt}; only Format 5 true-color indexed frames are supported")
        count, frame_num, total, projector, reserved = struct.unpack(">HHHBB", h[24:32])
        pos += 32
        if reserved != 0:
            raise CliError("nonzero reserved header byte")
        if count == 0:
            if frame_num or total or projector:
                raise CliError("malformed end-of-frames marker")
            ended = True
            if pos != len(data):
                raise CliError("trailing bytes after end-of-frames marker")
            break
        if len(frames) >= MAX_FRAMES:
            raise CliError("frame count exceeds safety limit")
        if count > MAX_POINTS or total_points + count > MAX_TOTAL_POINTS:
            raise CliError("point count exceeds safety limit")
        if total and (frame_num >= total or total > MAX_FRAMES):
            raise CliError("invalid frame numbering/count")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise CliError("inconsistent declared frame count")
        nbytes = count * 8
        if len(data) - pos < nbytes:
            raise CliError("truncated ILDA frame points")
        points = []
        for i in range(count):
            x, y, status, b, g, r = struct.unpack_from(">hhBBBB", data, pos + 8 * i)
            if status & 0x3F:
                raise CliError("unsupported nonzero status bits in Format 5 point")
            blank = bool(status & 0x40)
            last = bool(status & 0x80)
            if last != (i == count - 1):
                raise CliError("invalid last-point marker")
            if blank and (r or g or b):
                raise CliError("blanked point has nonzero color data")
            points.append((x, y, 0 if blank else r, 0 if blank else g, 0 if blank else b, blank))
        pos += nbytes
        total_points += count
        frames.append(Frame(frame_num, points))
    if not ended:
        raise CliError("missing ILDA end-of-frames marker")
    if not frames:
        raise CliError("file contains no supported Format 5 frames")
    if expected_total not in (None, 0, len(frames)):
        raise CliError("declared frame count does not match file")
    return frames, total_points


def describe(path):
    frames, points = parse_ilda5(path)
    print(f"ILDA Format 5: {len(frames)} frames, {points} points, {Path(path).stat().st_size} bytes")
    for frame in frames:
        lit = sum(not p[5] for p in frame.points)
        print(f"  frame {frame.index}: {len(frame.points)} points ({lit} lit, {len(frame.points)-lit} blank)")


def preview(path, out):
    frames, _ = parse_ilda5(path)
    dest = Path(out)
    suffix = dest.suffix.lower()
    if suffix not in (".png", ".gif"):
        raise CliError("preview output must end in .png or .gif")
    try:
        from PIL import Image, ImageDraw
        images = []
        for frame in frames:
            im = Image.new("RGB", (800, 800), (8, 18, 30))
            d = ImageDraw.Draw(im)
            d.ellipse((85, 85, 715, 715), outline=(24, 50, 67))
            previous = None
            for x, y, r, g, b, blank in frame.points:
                pixel = (400 + int(x * 0.009), 400 - int(y * 0.009))
                if not blank and previous is not None:
                    d.line((previous, pixel), fill=(r, g, b), width=2)
                previous = None if blank else pixel
            images.append(im)
        if suffix == ".gif":
            images[0].save(dest, save_all=True, append_images=images[1:], duration=100, loop=0, optimize=True)
        else:
            images[0].save(dest)
    except ImportError:
        import shutil
        import subprocess
        from html import escape
        renderer = shutil.which("magick") or shutil.which("convert")
        if not renderer:
            raise CliError("preview needs Pillow or ImageMagick; neither is installed")
        svg_files = []
        try:
            selected = frames if suffix == ".gif" else frames[:1]
            for fi, frame in enumerate(selected):
                elems = ['<rect width="800" height="800" fill="#08121e"/>', '<circle cx="400" cy="400" r="315" fill="none" stroke="#183243"/>']
                previous = None
                for x, y, r, g, b, blank in frame.points:
                    point = (400 + int(x * 0.009), 400 - int(y * 0.009))
                    if not blank and previous is not None:
                        color = f"#{r:02x}{g:02x}{b:02x}"
                        elems.append(f'<line x1="{previous[0]}" y1="{previous[1]}" x2="{point[0]}" y2="{point[1]}" stroke="{color}" stroke-width="2"/>')
                    previous = None if blank else point
                svg = '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800">' + ''.join(elems) + '</svg>'
                tmp = dest.with_name(f".{dest.stem}-{fi:05d}.svg")
                tmp.write_text(svg, encoding="utf-8")
                svg_files.append(tmp)
            subprocess.run([renderer, *(map(str, svg_files)), str(dest)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as e:
            raise CliError(f"preview renderer failed: {e}") from e
        finally:
            for tmp in svg_files:
                tmp.unlink(missing_ok=True)
    print(f"Wrote preview: {dest}")


class HeliosSDK:
    SINGLE_MODE = 1 << 1  # HELIOS_FLAGS_SINGLE_MODE from the official SDK header.

    def __init__(self, library):
        self.lib = ctypes.CDLL(library)
        self.lib.OpenDevicesOnlyUsb.restype = ctypes.c_int
        self.lib.GetStatus.argtypes = [ctypes.c_uint]
        self.lib.GetStatus.restype = ctypes.c_int
        self.lib.WriteFrame.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_int]
        self.lib.WriteFrame.restype = ctypes.c_int
        self.lib.Stop.argtypes = [ctypes.c_uint]
        self.lib.Stop.restype = ctypes.c_int
        self.lib.CloseDevices.restype = ctypes.c_int

    def open(self):
        return self.lib.OpenDevicesOnlyUsb()

    def status(self, device):
        return self.lib.GetStatus(device)

    def write(self, device, pps, flags, array, count):
        return self.lib.WriteFrame(device, pps, flags, ctypes.cast(array, ctypes.c_void_p), count)

    def stop(self, device):
        return self.lib.Stop(device)

    def close(self):
        return self.lib.CloseDevices()


def prepare_frames(frames, pps):
    # Convert every input frame to native Helios points. Grouping into >=20ms
    # playback groups is only done for file playback cadence; live frames return
    # one native frame per input frame so shape and point count are preserved.
    converted = []
    for frame in frames:
        group = list(frame.points)
        if len(group) > 4095:
            raise CliError("frame exceeds Helios 4095-point limit")
        array = (HeliosPoint * len(group))()
        for i, (x, y, r, g, b, blank) in enumerate(group):
            if not (-32768 <= x <= 32767 and -32768 <= y <= 32767):
                raise CliError("coordinate outside signed ILDA16 range")
            array[i] = HeliosPoint((x + 32768) >> 4, (y + 32768) >> 4,
                                  0 if blank else r, 0 if blank else g,
                                  0 if blank else b, 0 if blank else 255)
        converted.append(array)
    return converted


def group_file_frames(frames, pps):
    # File playback only: group >=20ms of original samples into SDK frames.
    target = math.ceil(pps * 0.020)
    groups, pending = [], []
    for frame in frames:
        pending.extend(frame.points)
        if len(pending) >= target:
            groups.append(pending)
            pending = []
    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)
    out = []
    for group in groups:
        if len(group) > 4095:
            raise CliError("grouped frame exceeds Helios 4095-point limit")
        if len(group) / pps < 0.010:
            need = math.ceil(pps * 0.010) - len(group)
            if need > 0:
                group = group + [(0, 0, 0, 0, 0, True)] * need
        array = (HeliosPoint * len(group))()
        for i, (x, y, r, g, b, blank) in enumerate(group):
            if not (-32768 <= x <= 32767 and -32768 <= y <= 32767):
                raise CliError("coordinate outside signed ILDA16 range")
            array[i] = HeliosPoint((x + 32768) >> 4, (y + 32768) >> 4,
                                  0 if blank else r, 0 if blank else g,
                                  0 if blank else b, 0 if blank else 255)
        out.append(array)
    return out


def run_play(args, sdk_factory=HeliosSDK, clock=time.monotonic, sleeper=time.sleep, live_frames=None):
    # duration=None means run until stopped (Ctrl-C/SIGTERM or `heliosctl stop`).
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise CliError("duration must be finite and > 0 seconds (omit for unbounded)")
    if args.pps < 1000 or args.pps > MAX_PPS:
        raise CliError(f"pps must be between 1000 and {MAX_PPS}")
    if not math.isfinite(args.ready_timeout) or args.ready_timeout <= 0 or args.ready_timeout > 30:
        raise CliError("ready-timeout must be finite, > 0 and <= 30 seconds")
    if live_frames is None:
        frames, _ = parse_ilda5(args.file)
        frames = group_file_frames(frames, args.pps)
    else:
        frames = live_frames
    sdk = sdk_factory(args.library)
    opened = False
    devices = 0
    interrupted = False
    previous_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    cleanup_errors = []
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    try:
        for s in previous_handlers:
            signal.signal(s, interrupt)
    except ValueError as exc:
        raise CliError("playback must run on the main thread to handle signals safely") from exc
    try:
        devices = sdk.open()
        if devices < 0:
            raise CliError(f"OpenDevicesOnlyUsb failed ({devices})")
        opened = True
        if devices != 1:
            raise CliError(f"requires exactly one isolated USB DAC; opened {devices}. SDK discovery/close affects all discovered DACs")
        deadline = math.inf if args.duration is None else clock() + args.duration
        control = getattr(args, '_control', None)
        if control:
            control.deadline = deadline
        frame_index = 0
        while clock() < deadline:
            if control:
                control.poll(clock())
                deadline = control.deadline
                if control.stopping:
                    break
                frames.gain = control.params['gain']
                frames.base_size = control.params['base-size']
                frames.expansion = control.params['expansion']
                frames.ripple.depth = control.params['ripple-depth']
                frames.ripple.frequency = control.params['frequency']
            wait_until = min(deadline, clock() + args.ready_timeout)
            status = 0
            while clock() < wait_until:
                if control:
                    control.poll(clock())
                    deadline = control.deadline
                    if control.stopping or clock() >= deadline:
                        break
                status = sdk.status(0)
                if status < 0:
                    raise CliError(f"GetStatus failed ({status})")
                if status > 1:
                    raise CliError(f"GetStatus returned unexpected value ({status})")
                if status == 1:
                    break
                sleeper(min(0.01, max(0.0, wait_until - clock())))
            else:
                if clock() >= deadline:
                    break
                raise CliError("DAC readiness timeout")
            if control and control.stopping:
                break
            points = frames[frame_index]
            if clock() >= deadline:
                break
            result = sdk.write(0, args.pps, HeliosSDK.SINGLE_MODE, points, len(points))
            if result != 1:
                raise CliError(f"WriteFrame failed ({result})")
            frame_index = (frame_index + 1) % len(frames)
            if clock() >= deadline:
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for s in previous_handlers:
            signal.signal(s, signal.SIG_IGN)
        if opened:
            for device in range(min(devices, 1)):
                try:
                    result = sdk.stop(device)
                    if result != 1:
                        cleanup_errors.append(f"Stop({device}) failed ({result})")
                except Exception as e:
                    cleanup_errors.append(f"Stop({device}) failed: {e}")
            try:
                result = sdk.close()
                if result != 1:
                    cleanup_errors.append(f"CloseDevices failed ({result})")
            except Exception as e:
                cleanup_errors.append(f"CloseDevices failed: {e}")
        for s, handler in previous_handlers.items():
            signal.signal(s, handler)
        if cleanup_errors:
            for error in cleanup_errors:
                print(f"warning: {error}", file=sys.stderr)
            raise CliError("playback cleanup failed: " + "; ".join(cleanup_errors))
    if interrupted:
        print("Interrupted; requested Stop and closed SDK devices.", file=sys.stderr)


class HeliosPoint(ctypes.Structure):
    # Matches HeliosDac.h: uint16 x/y followed by uint8 r/g/b/i.
    _fields_ = [("x", ctypes.c_uint16), ("y", ctypes.c_uint16), ("r", ctypes.c_uint8), ("g", ctypes.c_uint8), ("b", ctypes.c_uint8), ("i", ctypes.c_uint8)]


MIC_SOURCE = "alsa_input.pci-0000_00_1f.3.analog-stereo"


class Microphone:
    """Bounded nonblocking PCM capture; samples never leave process memory."""
    def __init__(self):
        import json
        import subprocess
        sources = json.loads(subprocess.check_output(
            ["pactl", "-f", "json", "list", "sources"], timeout=2))
        source = next((s for s in sources if s["name"] == MIC_SOURCE), None)
        if (not source or source.get("monitor_source") or source.get("mute")
                or source.get("active_port") != "analog-input-internal-mic"
                or source.get("properties", {}).get("device.form_factor") != "internal"):
            raise CliError("confirmed unmuted internal Surface microphone unavailable")
        self.process = subprocess.Popen(
            ["parec", "--device=" + MIC_SOURCE, "--raw", "--format=s16le",
             "--rate=16000", "--channels=1", "--latency-msec=20",
             "--process-time-msec=10"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.last = time.monotonic()
        self.rms = 0.0
        self.pending = b""
        self.samples = ()
        self.sequence = 0

    def read(self):
        # Select the newest complete 20ms window and drop any stale backlog.
        newest = None
        for _ in range(64):
            try:
                data = os.read(self.process.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not data:
                raise CliError("microphone capture EOF")
            data = self.pending + data
            complete = len(data) // 640 * 640
            self.pending = data[complete:]
            if complete:
                newest = data[complete - 640:complete]
        else:
            raise CliError("microphone capture flood budget exceeded")
        if newest is not None:
            values = struct.unpack("<320h", newest)
            self.samples = values
            self.sequence = getattr(self, 'sequence', 0) + 1
            self.rms = math.sqrt(sum(v * v for v in values) / 320) / 32768
            self.last = time.monotonic()
        if self.process.poll() is not None or time.monotonic() - self.last > 0.8:
            raise CliError("microphone capture stopped or stalled")
        return min(1.0, self.rms)

    def close(self):
        import subprocess
        self.process.terminate()
        try:
            self.process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=0.5)
        finally:
            self.process.stdout.close()


class OnsetDetector:
    """RMS transients, not tempo or frequency analysis."""
    def __init__(self):
        self.baseline = None
        self.previous = 0.0
        self.last_onset = -math.inf

    def update(self, level, now, dt):
        if self.baseline is None:
            self.baseline = level
        # Scale the excess threshold to remaining headroom: a hot microphone
        # must not require a level above the normalized maximum of 1.0.
        threshold = self.baseline + max(0.015, 0.06 * (1.0 - self.baseline))
        onset = (level > max(0.025, threshold)
                 and level - self.previous > 0.015
                 and now - self.last_onset >= 0.22)
        self.baseline += (level - self.baseline) * (1 - math.exp(-dt / 0.6))
        self.previous = level
        if onset:
            self.last_onset = now
        return onset


class RippleContour:
    def __init__(self, seed=None, depth=.07, frequency=5):
        if not math.isfinite(depth) or not 0 <= depth <= .10:
            raise CliError('ripple depth must be finite and between 0 and 0.10')
        if not isinstance(frequency, int) or not 1 <= frequency <= 8:
            raise CliError('ripple frequency must be an integer between 1 and 8')
        self.depth, self.frequency = depth, frequency
        self.random = random.Random(seed)
        self.impulses = []
        self.previous = None

    def sample(self, level, now, dt, onset=False):
        self.impulses = [p for p in self.impulses if now - p[0] < 1.8]
        if onset:
            self.impulses = (self.impulses + [(now, self.random.uniform(0, 2 * math.pi),
                                              self.random.choice((-1, 1)))])[-4:]
        rx, ry = 2500 + 13500 * level, 2500 + 10500 * level
        coords = []
        for i in range(500):
            angle = 2 * math.pi * i / 500
            wave = 0.0
            for start, direction, travel in self.impulses:
                age = max(0.0, now - start)
                delta = angle - direction - travel * 2.4 * age
                # Periodic analytic packet: smooth across the contour seam.
                packet = math.exp(3 * (math.cos(delta) - 1))
                wave += (1 - math.exp(-age / .08)) * math.exp(-age / .55) * packet * math.sin(self.frequency * delta - 9 * age)
            # Reserved inward headroom keeps every ripple within the old ellipse.
            scale = 1 - self.depth * len(self.impulses) + self.depth * wave
            coords.append((rx * scale * math.cos(angle), ry * scale * math.sin(angle)))
        if self.previous is not None:
            movement = max(math.hypot(x - px, y - py)
                           for (x, y), (px, py) in zip(coords, self.previous))
            blend = min(1.0, min(560.0, 22400 * dt) / max(1e-9, movement))
            coords = [(px + blend * (x - px), py + blend * (y - py))
                      for (x, y), (px, py) in zip(coords, self.previous)]
        self.previous = coords
        return coords + coords[:1]


LIQUID_PATTERNS = ('liquid', 'liquid2')   # share the Bands -> contour -> 450-point frame path


class AudioFrames:
    def __init__(self, microphone, clock=time.monotonic, gain=1.0, meter=False,
                 pattern='plain', seed=None, ripple_depth=.07, ripple_frequency=5,
                 base_size=0.0, expansion=1.0):
        self.microphone, self.clock = microphone, clock
        self.gain, self.meter = gain, meter
        self.pattern = pattern
        self.base_size, self.expansion = base_size, expansion
        self.detector = OnsetDetector()
        self.ripple = RippleContour(seed, ripple_depth, ripple_frequency)
        from liquid import Bands, LiquidContour
        self.bands = Bands()
        if pattern == 'liquid2':
            from dynamic import DynamicContour     # shape-morphing variant, same frame contract
            self.liquid = DynamicContour()
        else:
            self.liquid = LiquidContour()
        self.pcm_sequence = None
        self.level = 0.0
        self.geometry = 0.0
        self.last = clock()
        self.next_meter = self.last
        self.onset_count = 0
        self.emitters = []  # pulses radiating outward like radio waves between circles
        self.emit_at = 0.0

    def emit_radio(self, now, energy):
        """Every emitter circle periodically broadcasts an expanding ring that the
        other circles catch and ripple in response to."""
        if now >= self.emit_at and energy > 0.18:
            self.emit_at = now + 2.2
            self.emitters.append((now, 0.0))
        self.emitters = [(t, r) for t, r in self.emitters if now - t < 2.4]
        return [(t, r) for t, r in self.emitters]

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        rms = self.microphone.read()
        target = min(1.0, max(0.0, rms * self.gain))
        now = self.clock()
        dt = max(0.0, now - self.last)
        self.last = now
        tau = 0.04 if target > self.level else 0.18
        self.level += (target - self.level) * (1 - math.exp(-dt / tau))
        # Preserve <=40 DAC-coordinate steps even after a scheduling pause.
        step = min(0.10, 4.0 * dt)
        size_target = self.base_size + self.expansion * self.level
        self.geometry += max(-step, min(step, size_target - self.geometry))
        rx = 10000 + 6000 * self.geometry
        ry = 10000 + 3000 * self.geometry
        coords = None
        if self.pattern == 'ripple':
            onset = self.detector.update(target, now, dt)
            self.onset_count += int(onset)
            coords = self.ripple.sample(self.geometry, now, dt, onset)
        if self.pattern in LIQUID_PATTERNS:
            sequence = getattr(self.microphone, 'sequence', None)
            if sequence != self.pcm_sequence:
                bands = self.bands.update(getattr(self.microphone, 'samples', ()), min(dt, .1))
                self.pcm_sequence = sequence
            else:
                bands = self.bands.decay(min(dt, .1))
            bands = [math.sqrt(min(1.0, max(0.0, v))) for v in bands]
            coords, colors, blanks = self.liquid.sample(self.geometry, bands, now, dt,
                                                        self.ripple.depth, self.ripple.frequency,
                                                        flux=self.bands.flux, centroid=self.bands.centroid,
                                                        pitch_angle=self.bands.pitch_angle,
                                                        sub_flux=self.bands.sub_flux,
                                                        beat_phase=self.bands.beat_phase,
                                                        hat_phase=self.bands.hat_phase,
                                                        beat_lock=self.bands.beat_lock,
                                                        hat_lock=self.bands.hat_lock,
                                                        activity=self.bands.activity,
                                                        beat_period=self.bands.beat_period,
                                                        accent=self.bands.accent)
            self.onset_count = self.liquid.onsets
        if self.meter and now >= self.next_meter:
            if self.pattern in LIQUID_PATTERNS:
                print(f"liquid bands={','.join(f'{v:.4f}' for v in self.bands.energy)} "
                      f"capture={self.bands.diagnostic} onsets={self.onset_count}", file=sys.stderr)
            print(f"audio rms={rms:.4f} target={target:.4f} gain={self.gain:g} "
                  f"onsets={self.onset_count} active_waves={len(self.ripple.impulses)}", file=sys.stderr)
            self.next_meter = now + 1.0
        points = []
        if self.pattern in LIQUID_PATTERNS:
            for (x, y), (r, g, b), blank in zip(coords, colors, blanks):
                points.append((round(x), round(y), r, g, b, blank))
            return prepare_frames([Frame(0, points)], 20000)[0]
        for i in range(501):
            angle = 2 * math.pi * i / 500
            blank = i in (0, 500)
            x, y = coords[i] if coords is not None else (rx * math.cos(angle), ry * math.sin(angle))
            r, g, b = (0, 0, 0) if blank else (16, 48, 96)
            points.append((round(x), round(y), r, g, b, blank))
        return prepare_frames([Frame(0, points)], 20000)[0]


def run_audio(args, sdk_factory=HeliosSDK):
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise CliError("audio duration must be finite and > 0 seconds (omit for unbounded)")
    gain = getattr(args, 'gain', 1.0)
    if not math.isfinite(gain) or not 0 < gain <= 32:
        raise CliError("audio gain must be finite, > 0 and <= 32")
    depth = getattr(args, 'ripple_depth', .07)
    frequency = getattr(args, 'ripple_frequency', 5)
    base_size, expansion = getattr(args, 'base_size', 0.0), getattr(args, 'expansion', 1.0)
    if (not all(math.isfinite(v) and 0 <= v <= 1 for v in (base_size, expansion))
            or base_size + expansion > 1):
        raise CliError('base-size and expansion must be 0..1 with sum <=1')
    RippleContour(depth=depth, frequency=frequency)  # Validate before capture or SDK access.
    control = None
    if getattr(args, 'live_control', False):
        from live_control import Control
        control = Control(gain, depth, frequency, base_size, expansion)
    args._control = control
    microphone = None
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt()
    try:
        for s in previous:
            signal.signal(s, interrupt)
        microphone = Microphone()
        # Confirm incoming PCM before any SDK construction/open.
        start = microphone.last
        until = time.monotonic() + 0.5
        while microphone.last == start:
            microphone.read()
            if time.monotonic() >= until:
                raise CliError("microphone startup timeout")
            time.sleep(0.005)
        run_play(args, sdk_factory=sdk_factory, live_frames=AudioFrames(microphone, gain=gain, meter=getattr(args, 'meter', False),
                                                        pattern=getattr(args, 'pattern', 'plain'), seed=getattr(args, 'seed', None),
                                                        ripple_depth=depth, ripple_frequency=frequency,
                                                        base_size=base_size, expansion=expansion))
    except KeyboardInterrupt:
        pass
    finally:
        error = str(sys.exc_info()[1]) if sys.exc_info()[1] else None
        for s in previous:
            signal.signal(s, signal.SIG_IGN)
        try:
            if microphone is not None:
                microphone.close()
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            try:
                if control:
                    control.finish(error)
            finally:
                for s, handler in previous.items():
                    signal.signal(s, handler)


def default_library():
    if "HELIOS_LIBRARY" in os.environ:
        return os.environ["HELIOS_LIBRARY"]
    base = Path(__file__).resolve().parent
    for c in (
        base / "native" / "libHeliosDacAPI.so",
        base / "libHeliosDacAPI.so",
        Path.home() / ".local" / "opt" / "heliosctl" / "libHeliosDacAPI.so",
    ):
        if c.is_file():
            return str(c)
    return "libHeliosDacAPI.so"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heliosctl", description="Offline ILDA inspection and explicitly bounded Helios USB playback")
    sub = parser.add_subparsers(dest="cmd", required=True)
    devices = sub.add_parser("devices", help="list USB Helios devices (opens SDK; not invoked by install)")
    devices.add_argument("--library", default=default_library())
    inspect = sub.add_parser("inspect", help="inspect supported ILDA Format 5 file offline")
    inspect.add_argument("file")
    pv = sub.add_parser("preview", help="render first frame to PNG or all frames to GIF offline")
    pv.add_argument("file")
    pv.add_argument("--out", required=True)
    play = sub.add_parser("play", help="bounded playback; each frame uses SDK single-mode")
    play.add_argument("file")
    play.add_argument("--duration", type=float, default=None, help="playback time in seconds (>0); omit to play until stopped")
    play.add_argument("--pps", type=int, default=20000)
    play.add_argument("--ready-timeout", type=float, default=2.0)
    play.add_argument("--library", default=default_library())
    audio = sub.add_parser("audio", help="bounded internal-microphone reactive ellipse (opens DAC)")
    audio.add_argument("--duration", type=float, default=None, help="seconds (>0); omit to run until stopped (Ctrl-C, SIGTERM or `heliosctl stop`)")
    audio.add_argument("--library", default=default_library())
    audio.add_argument('--gain', type=float, default=1.0, help='RMS geometry gain, >0 and <=32 (default 1; never changes intensity)')
    audio.add_argument('--meter', action='store_true', help='print aggregate RMS/target once per second during playback')
    audio.add_argument('--pattern', choices=('plain', 'ripple', 'liquid', 'liquid2'), default='plain',
                       help='plain, ripple, liquid (60-250/250-2000/2000-6000 Hz; 50Hz resolution) or liquid2 (liquid body that morphs between shapes on the beat, ring style per event); unchanged max extent/power')
    audio.add_argument('--ripple-depth', type=float, default=.07,
                       help='radial deformation fraction per packet, 0..0.10 (default .07); reserves matching inward headroom, never boosts power')
    audio.add_argument('--ripple-frequency', type=int, default=5,
                       help='spatial wave cycles around contour, 1..8 (default 5); not scan rate or beat rate; bounded for 500 samples')
    audio.add_argument('--seed', type=int, help='repeatable ripple directions (optional)')
    audio.add_argument('--base-size', type=float, default=0.0, help='resting size offset within existing size range, 0..1')
    audio.add_argument('--expansion', type=float, default=1.0, help='audio size modulation fraction, 0..1; base-size + expansion <=1')
    audio.set_defaults(pps=20000, ready_timeout=2.0)
    audio.add_argument('--live-control', action='store_true', help='enable private local status/tune/stop/renew socket; no automatic renewal')
    sub.add_parser('status', help='query this user controller or last terminal receipt; never opens SDK')
    sub.add_parser('stop', help='request this controller stop; stopping is not cleanup acknowledgement')
    tune = sub.add_parser('tune', help='atomically tune geometry only; never opens SDK')
    tune.add_argument('--gain', type=float)
    tune.add_argument('--ripple-depth', type=float)
    tune.add_argument('--frequency', type=int)
    tune.add_argument('--base-size', type=float)
    tune.add_argument('--expansion', type=float)
    renew = sub.add_parser('renew', help='explicitly replace deadline with <=300 seconds from now')
    renew.add_argument('seconds', type=float)
    args = parser.parse_args(argv)
    if args.cmd in ('status', 'stop', 'tune', 'renew'):
        import json
        from live_control import request
        message = {'command': args.cmd}
        if args.cmd == 'tune':
            message['params'] = {key: getattr(args, key.replace('-', '_')) for key in ('gain', 'ripple-depth', 'frequency', 'base-size', 'expansion') if getattr(args, key.replace('-', '_')) is not None}
        if args.cmd == 'renew':
            message['seconds'] = args.seconds
        try:
            result = request(message)
        except ValueError as exc:
            raise CliError(str(exc)) from exc
        print(json.dumps(result))
        if 'error' in result and result['error']:
            raise CliError(result['error'])
        return
    if args.cmd == "inspect":
        describe(args.file)
    elif args.cmd == "preview":
        preview(args.file, args.out)
    elif args.cmd == "devices":
        sdk = HeliosSDK(args.library)
        opened_devices = 0
        try:
            opened_devices = sdk.open()
            if opened_devices < 0:
                print(f"USB Helios devices: error ({opened_devices})", file=sys.stderr)
            else:
                print(f"USB Helios devices: {opened_devices}")
        finally:
            if opened_devices > 0:
                for device in range(opened_devices):
                    result = sdk.stop(device)
                    if result != 1:
                        print(f"warning: Stop({device}) failed ({result})", file=sys.stderr)
            result = sdk.close()
            if result != 1:
                print(f"warning: CloseDevices failed ({result})", file=sys.stderr)
    elif args.cmd == "play":
        run_play(args)
    elif args.cmd == "audio":
        run_audio(args)


if __name__ == "__main__":
    try:
        main()
    except (CliError, OSError, ValueError, struct.error) as exc:
        print(f"heliosctl: error: {exc}", file=sys.stderr)
        raise SystemExit(2)
