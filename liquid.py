"""Offline-safe 16 kHz liquid contour analysis using NumPy & SciPy DSP; no device imports."""
import math
import random
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window

_WIN320 = get_window('hann', 320, fftbins=True).astype(np.float64)
_WIN1024 = get_window('hann', 1024, fftbins=True).astype(np.float64)


class Bands:
    """NumPy/SciPy STFT + Harmonic/Percussive Separation + Macro-Activity & 6s Comb PLL."""
    def __init__(self):
        self.baseline = [.015] * 3
        self.envelope = [0.] * 3
        self.energy = [0.] * 3
        self.pcm_ring = np.zeros(1024, dtype=np.float64)
        self.frames_seen = 0
        self.spec_hist = []
        self.prev_log_spec = None
        self.flux = 0.
        self.flux_base = .04
        self.sub_flux = [0., 0., 0.]
        self.sub_flux_base = [.03, .03, .02]
        self.perc_ratio = 0.
        self.activity = 0.
        self.fast_loudness = 0.
        self.slow_peak = .08
        self.noise_floor = .008
        self.loud_hist = []
        self.beat_hist = []
        self.hat_hist = []
        self.beat_period = 0.50
        self.hat_period = 0.125
        self.beat_phase = 0.
        self.hat_phase = 0.
        self.beat_lock = 0.
        self.hat_lock = 0.
        self.centroid = .25
        self.pitch_angle = 0.
        self.diagnostic = 'waiting'

    def update(self, samples, dt):
        if len(samples) != 320:
            self.diagnostic = 'no-pcm'
            return self.decay(dt)
        arr = np.asarray(samples, dtype=np.float64)
        clipped = float(np.mean(np.abs(arr) >= 32700))
        arr = (arr - float(np.mean(arr))) / 32768.0
        spec320 = np.abs(np.fft.rfft(arr * _WIN320))[2:120] ** 2
        spec320 = np.maximum(1e-20, spec320)
        total_power = float(np.sum(spec320))
        flatness = float(np.exp(np.mean(np.log(spec320))) / (total_power / len(spec320)))
        norm = 320.0 * 320.0 * 0.1875
        self.energy = [
            float(np.sqrt(np.sum(spec320[0:3]) / norm)),
            float(np.sqrt(np.sum(spec320[3:38]) / norm)),
            float(np.sqrt(np.sum(spec320[38:118]) / norm)),
        ]
        self.diagnostic = 'clipped' if clipped > .75 else 'broadband-noise' if flatness > .35 else 'ok'
        if self.diagnostic != 'ok':
            self.flux = 0.
            self.sub_flux = [0., 0., 0.]
            self.activity *= math.exp(-dt / 0.25)
            return self.decay(dt)

        self.pcm_ring = np.roll(self.pcm_ring, -320)
        self.pcm_ring[-320:] = arr
        self.frames_seen += 1

        frame_loud = float(np.sqrt(np.mean(arr * arr)))
        self.loud_hist = (self.loud_hist + [frame_loud])[-200:]
        if len(self.loud_hist) >= 20:
            self.noise_floor = max(0.004, min(0.035, float(np.percentile(self.loud_hist, 15))))
        if frame_loud > self.slow_peak:
            self.slow_peak += (frame_loud - self.slow_peak) * (1.0 - math.exp(-dt / 0.25))
        else:
            self.slow_peak = max(0.05, self.slow_peak * math.exp(-dt / 22.0))
        tau_loud = 0.08 if frame_loud > self.fast_loudness else 0.45
        self.fast_loudness += (frame_loud - self.fast_loudness) * (1.0 - math.exp(-dt / tau_loud))
        dyn_span = max(0.025, self.slow_peak - self.noise_floor * 1.25)
        raw_act = min(1.0, max(0.0, (self.fast_loudness - self.noise_floor * 1.25) / dyn_span))
        target_act = (raw_act ** 1.55) * (3.0 - 2.0 * (raw_act ** 1.55)) if raw_act < 1.0 else 1.0
        tau_act = 0.10 if target_act > self.activity else 0.55
        self.activity += (target_act - self.activity) * (1.0 - math.exp(-dt / tau_act))

        log_spec = np.log1p(2500.0 * spec320)
        self.spec_hist.append(log_spec)
        if len(self.spec_hist) > 9:
            self.spec_hist.pop(0)
        if len(self.spec_hist) >= 5:
            mat = np.asarray(self.spec_hist, dtype=np.float64)
            harm_cur = median_filter(mat, size=(5, 1))[-1]
            perc_cur = median_filter(mat, size=(1, 9))[-1]
            p_pow = np.maximum(0.0, log_spec - harm_cur)
            h_pow = np.maximum(0.0, log_spec - perc_cur)
            self.perc_ratio = float(np.sum(p_pow) / max(1e-6, np.sum(p_pow) + np.sum(h_pow)))
        else:
            self.perc_ratio = 0.5

        act_gate = 0.25 + 0.75 * self.activity
        if self.prev_log_spec is not None:
            diffs = np.maximum(0.0, log_spec - self.prev_log_spec)
            raw_flux = float(np.mean(diffs))
            self.flux = act_gate * min(1.0, max(0.0, raw_flux - 0.80 * self.flux_base) / max(0.030, 1.20 * self.flux_base))
            self.flux_base += (raw_flux - self.flux_base) * (1.0 - math.exp(-dt / 1.1))
            for idx, (a, b) in enumerate(((0, 3), (3, 38), (38, 118))):
                rf = float(np.mean(diffs[a:b]))
                base = self.sub_flux_base[idx]
                self.sub_flux[idx] = act_gate * min(1.0, max(0.0, rf - 0.75 * base) / max(0.022, 1.10 * base))
                self.sub_flux_base[idx] += (rf - base) * (1.0 - math.exp(-dt / 1.0))
        self.prev_log_spec = log_spec

        step_dt = max(0.005, min(0.05, dt))
        beat_drive = max(self.sub_flux[0], 0.85 * self.sub_flux[1])
        hat_drive = self.sub_flux[2]
        self.beat_hist = (self.beat_hist + [beat_drive])[-240:]
        self.hat_hist = (self.hat_hist + [hat_drive])[-160:]
        if len(self.beat_hist) >= 64 and len(self.beat_hist) % 4 == 0:
            bh = np.asarray(self.beat_hist, dtype=np.float64)
            n_bh = len(bh)
            weights = 0.992 ** np.arange(n_bh - 1, -1, -1)
            wbh = bh * weights
            energy_norm = float(np.dot(wbh, wbh)) + 1e-6
            best_lag, best_score = 20, 0.0
            for lag in range(13, min(33, n_bh // 2)):
                score = float(np.dot(wbh[lag:], wbh[:-lag]))
                if 2 * lag < n_bh:
                    score += 0.55 * float(np.dot(wbh[2 * lag:], wbh[:-2 * lag]))
                if 3 * lag < n_bh:
                    score += 0.30 * float(np.dot(wbh[3 * lag:], wbh[:-3 * lag]))
                if score > best_score:
                    best_score, best_lag = score, lag
            self.beat_lock = min(1.0, best_score / energy_norm) * self.activity
            if self.beat_lock > 0.16:
                self.beat_period += (best_lag * 0.025 - self.beat_period) * 0.22
        if len(self.hat_hist) >= 32 and len(self.hat_hist) % 3 == 0:
            hh = np.asarray(self.hat_hist, dtype=np.float64)
            n_hh = len(hh)
            hweights = 0.988 ** np.arange(n_hh - 1, -1, -1)
            whh = hh * hweights
            henergy = float(np.dot(whh, whh)) + 1e-6
            best_hlag, best_hscore = 5, 0.0
            for lag in range(3, min(14, n_hh // 2)):
                score = float(np.dot(whh[lag:], whh[:-lag]))
                if 2 * lag < n_hh:
                    score += 0.60 * float(np.dot(whh[2 * lag:], whh[:-2 * lag]))
                if 4 * lag < n_hh:
                    score += 0.35 * float(np.dot(whh[4 * lag:], whh[:-4 * lag]))
                if score > best_hscore:
                    best_hscore, best_hlag = score, lag
            self.hat_lock = min(1.0, best_hscore / henergy) * self.activity
            if self.hat_lock > 0.14:
                self.hat_period += (best_hlag * 0.025 - self.hat_period) * 0.25

        self.beat_phase = (self.beat_phase + 2 * math.pi * step_dt / max(0.28, self.beat_period)) % (2 * math.pi)
        if beat_drive > 0.45 and self.activity > 0.25:
            err = math.atan2(math.sin(-self.beat_phase), math.cos(-self.beat_phase))
            self.beat_phase = (self.beat_phase + 0.35 * beat_drive * err) % (2 * math.pi)
        self.hat_phase = (self.hat_phase + 2 * math.pi * step_dt / max(0.06, self.hat_period)) % (2 * math.pi)
        if hat_drive > 0.35 and self.activity > 0.25:
            herr = math.atan2(math.sin(-self.hat_phase), math.cos(-self.hat_phase))
            self.hat_phase = (self.hat_phase + 0.45 * hat_drive * herr) % (2 * math.pi)

        if max(self.energy) > .004:
            raw_c = float(np.dot(np.arange(2, 120), spec320) / (total_power * 60.0))
            self.centroid += (min(1.0, max(0.0, raw_c)) - self.centroid) * (1.0 - math.exp(-dt / 0.08))
            if self.frames_seen >= 3:
                spec1024 = np.abs(np.fft.rfft(self.pcm_ring * _WIN1024))[4:100] ** 2
                k_peak = int(np.argmax(spec1024)) + 4
                hz = k_peak * (16000.0 / 1024.0)
            else:
                k_peak = int(np.argmax(spec320[:32])) + 2
                hz = k_peak * 50.0
            self.pitch_angle = (12.0 * math.log2(max(55.0, hz) / 110.0) % 12.0) * (2.0 * math.pi / 12.0)

        for i, e in enumerate(self.energy):
            floor = max((.004, .004, .002)[i], .65 * self.baseline[i])
            span = max((.018, .016, .012)[i], .78 * self.baseline[i])
            raw_target = min(1., max(0., e - floor) / span)
            target = raw_target * (0.35 + 0.65 * self.activity)
            tau_base = 1.2 if e > self.baseline[i] else 4.5
            self.baseline[i] += (max((.008, .008, .004)[i], e) - self.baseline[i]) * (1.0 - math.exp(-dt / tau_base))
            tau = .016 if target > self.envelope[i] else (.14, .11, .09)[i]
            self.envelope[i] += (target - self.envelope[i]) * (1.0 - math.exp(-dt / tau))
        return self.envelope[:]

    def decay(self, dt):
        self.envelope = [v * math.exp(-dt / .12) for v in self.envelope]
        return self.envelope[:]


class LiquidContour:
    """Centered liquid sphere with outward expanding concentric radio rings.

    Total points across core sphere + active concentric rings is ~400 points,
    yielding 50 Hz refresh rate at 20 kpps (completely flicker-free).
    """
    def __init__(self, seed=42):
        self.random = random.Random(seed)
        self.rings = []  # (birth_time, color)
        self.last_onset = -10.0
        self.phase = 0.0
        self.color_phase = 0.0
        self.tilt_phase = 0.0
        self.onsets = 0

    def sample(self, level, bands, now, dt, depth=.07, frequency=8, flux=0., centroid=.25, pitch_angle=None,
               sub_flux=None, beat_phase=0., hat_phase=0., beat_lock=0., hat_lock=0., activity=None,
               emitters=None):
        bass, mid, high = bands
        kick_f, snare_f, hat_f = sub_flux if sub_flux is not None else (flux * bass, flux * mid, flux * high)
        act = min(1.0, max(0.0, activity if activity is not None else max(level, max(bands))))
        step_dt = min(dt, 0.05)
        self.phase += (1.4 + 2.4 * bass + 1.2 * act) * step_dt
        self.color_phase += (1.8 + 4.5 * max(bands) + 2.0 * centroid) * step_dt
        self.tilt_phase += (0.85 + 1.6 * bass) * step_dt

        # Spawn new expanding concentric radio ring on beat kicks / snare hits
        if act > 0.18 and (kick_f > 0.24 or bass > 0.20 or snare_f > 0.28) and (now - self.last_onset) > 0.28:
            self.last_onset = now
            self.onsets += 1
            col = (110, 220, 255) if kick_f >= snare_f else (255, 185, 80)
            self.rings.append((now, col))

        MAX_AGE = 1.35
        updated = []
        for birth, col in self.rings:
            age = now - birth
            if age < MAX_AGE:
                updated.append((birth, col))
        self.rings = updated[-2:]

        N_CORE = 160
        amount = depth / 0.10
        r_core = (2400.0 + 1900.0 * level) * (0.82 + 0.18 * act)
        r_max = 8200.0
        coords = []
        colors = []
        blanks = []

        # --- 1. Centered Liquid Core Sphere (160 points) ---
        for i in range(N_CORE):
            a = 2 * math.pi * i / (N_CORE - 1)
            blank = (i in (0, N_CORE - 1))
            w = (0.20 * amount * bass * math.sin(2 * a - self.phase)
                 + 0.10 * amount * mid * math.sin(3 * a + self.phase)
                 + 0.025 * amount * high * math.sin(frequency * a - 2 * self.phase))
            r = r_core * (1.0 + 0.22 * math.tanh(1.5 * w))
            px = r * math.cos(a)
            py = r * math.sin(a)
            coords.append((round(px), round(py)))
            blanks.append(blank)
            s_r = max(0.0, math.sin(a - self.color_phase)) ** 1.6
            s_g = max(0.0, math.sin(a - self.color_phase + 2.094)) ** 1.6
            s_b = max(0.0, math.sin(a - self.color_phase + 4.188)) ** 1.6
            rc = int(min(255, 18 + 220 * s_r * (0.2 + 0.8 * bass)))
            gc = int(min(255, 22 + 200 * s_g * (0.2 + 0.8 * (mid + high))))
            bc = int(min(255, 60 + 190 * s_b))
            colors.append((0, 0, 0) if blank else (rc, gc, bc))

        # --- 2. Outward Expanding Concentric Radio Rings (240 points each) ---
        N_RING = 240
        for birth, col in self.rings:
            age = now - birth
            frac = age / MAX_AGE
            rw = r_core + (r_max - r_core) * frac
            fade = max(0.0, (1.0 - frac) ** 0.85) * act
            if fade < 0.05: continue
            coords.append(coords[-1])
            colors.append((0, 0, 0))
            blanks.append(True)
            coords.append((round(rw), 0))
            colors.append((0, 0, 0))
            blanks.append(True)
            for j in range(N_RING):
                a = 2 * math.pi * j / (N_RING - 1)
                blank = (j in (0, N_RING - 1))
                ax = rw * math.cos(a)
                ay = rw * math.sin(a)
                coords.append((round(ax), round(ay)))
                blanks.append(blank)
                colors.append((0, 0, 0) if blank else (int(col[0] * fade), int(col[1] * fade), int(col[2] * fade)))

        return coords, colors, blanks
