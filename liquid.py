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
        self.beat_t = []         # timestamp of each beat_hist entry (s): frames are ~22.5 ms, not 25
        self.clock = 0.          # Bands-local time, advanced by every update/decay dt
        self.last_kick = -1.
        self.kick_misses = 0
        self.bass_log = math.log(1e-4)
        self.kick_peak = 0.3     # decaying peak of the bass log-energy jump (kick vs snare discriminator)
        self.tempo_votes = 0     # consecutive comb estimates asking for a different tempo octave
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
            self.beat_lock *= math.exp(-dt / 0.6)    # no music: confidence fades, never sticks
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
        self.clock += dt
        self.beat_hist = (self.beat_hist + [beat_drive])[-300:]
        self.beat_t = (self.beat_t + [self.clock])[-300:]
        self.hat_hist = (self.hat_hist + [hat_drive])[-160:]
        if len(self.beat_hist) >= 64 and len(self.beat_hist) % 4 == 0:
            self._estimate_tempo()
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
                frame_s = (self.beat_t[-1] - self.beat_t[0]) / max(1, len(self.beat_t) - 1)
                self.hat_period += (best_hlag * frame_s - self.hat_period) * 0.25

        # Beat PLL (2nd order).  Phase 0 is the kick itself: detection lands BEAT_LATENCY after
        # the audio, so a detected onset should find the phase already that far past 0.
        period = max(0.28, self.beat_period)
        self.beat_phase = (self.beat_phase + 2 * math.pi * step_dt / period) % (2 * math.pi)
        # Only kick-like onsets steer the phase: a bass log-energy jump near its recent peak.
        # Snares and fills feed the tempo comb but would otherwise pull phase 0 onto the backbeat.
        bass_log = math.log(self.energy[0] + 1e-4)
        kick = max(0.0, bass_log - self.bass_log)
        self.bass_log = bass_log
        self.kick_peak = max(kick, 0.3, self.kick_peak * math.exp(-step_dt / 2.5))
        # The bass jump, not the first flux frame, times the kick: a kick starting late in a
        # 20 ms block shows flux there but lands its energy in the next block.
        is_kick = (beat_drive > 0.16 and kick > 0.6 * self.kick_peak
                   and self.clock - self.last_kick > 0.12 and self.activity > 0.05)
        if is_kick:
            self.last_kick = self.clock
            err = _wrap_pi(2 * math.pi * self.BEAT_LATENCY / period - self.beat_phase)
            # off-beat kicks (syncopation) are ignored, but 3 in a row means we locked onto the
            # wrong phase: accept the next one whatever its error (re-acquire)
            in_window = abs(err) < 0.30 * 2 * math.pi
            self.kick_misses = 0 if in_window else self.kick_misses + 1
            if in_window or self.kick_misses > 3:
                self.kick_misses = 0
                w = min(1.0, beat_drive)
                kp = 0.55 - 0.25 * self.beat_lock        # pull hard to acquire, gently once locked
                self.beat_phase = (self.beat_phase + kp * w * err) % (2 * math.pi)
                # integral path: an early onset (err > 0) means our period is too long
                self.beat_period = min(1.25, max(0.28, period * (1.0 - 0.10 * w * err / (2 * math.pi))))
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
            # Soft saturation curve instead of hard min(1., ...): preserves dynamic
            # contrast across loud sections and prevents flat-topping/clipping.
            raw_target = math.tanh(max(0., e - floor) / span)
            target = raw_target * (0.35 + 0.65 * self.activity)
            tau_base = 1.2 if e > self.baseline[i] else 4.5
            self.baseline[i] += (max((.008, .008, .004)[i], e) - self.baseline[i]) * (1.0 - math.exp(-dt / tau_base))
            tau = .016 if target > self.envelope[i] else (.14, .11, .09)[i]
            self.envelope[i] += (target - self.envelope[i]) * (1.0 - math.exp(-dt / tau))
        return self.envelope[:]

    TEMPO_MIN, TEMPO_MAX = 0.30, 1.20   # beat period search, s (50-200 BPM)
    TEMPO_GRID = 0.01                   # onset curve is resampled to 10 ms before the comb
    BEAT_LATENCY = 0.035                # kick audio -> detected onset frame, s (block fill + newest-block wait + frame)

    def _estimate_tempo(self):
        """Comb tempo estimate on a uniform time grid (entries are timestamped, so the frame rate
        does not bias the period), sub-grid peak interpolation, and a lock confidence that is
        a normalised autocorrelation: loudness-independent, ~0 for noise or silence."""
        t = np.asarray(self.beat_t, dtype=np.float64)
        grid = np.arange(t[0], t[-1], self.TEMPO_GRID)
        if len(grid) < 2 * int(self.TEMPO_MAX / self.TEMPO_GRID) + 4:
            return
        x = np.interp(grid, t, np.asarray(self.beat_hist, dtype=np.float64))
        x = (x - float(np.mean(x))) * np.exp(-(grid[-1] - grid) / 3.2)
        n = len(x)
        spec = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] < 1e-7:
            self.beat_lock *= 0.7
            return
        ac /= ac[0]
        lo, hi = int(self.TEMPO_MIN / self.TEMPO_GRID), int(self.TEMPO_MAX / self.TEMPO_GRID)
        lags = np.arange(lo, hi + 1)
        comb = ac[lags] + 0.5 * ac[np.minimum(2 * lags, n - 1)] + 0.25 * ac[np.minimum(3 * lags, n - 1)]
        prior = np.exp(-0.5 * (np.log2(lags * self.TEMPO_GRID / 0.5) / 1.1) ** 2)   # mild 120 BPM prior
        score = comb * (0.7 + 0.3 * prior)
        k = int(np.argmax(score))
        lag = float(lags[k])
        if 0 < k < len(score) - 1:
            a, b, c = score[k - 1], score[k], score[k + 1]
            den = a - 2 * b + c
            if den < 0:
                lag += 0.5 * (a - c) / den
        est = lag * self.TEMPO_GRID
        raw_lock = min(1.0, max(0.0, (float(ac[lags[k]]) - 0.08) / 0.45))
        self.beat_lock += (raw_lock - self.beat_lock) * 0.35
        if raw_lock < 0.2:
            return
        cur = self.beat_period / self.TEMPO_GRID
        if abs(math.log2(est / self.beat_period)) > 0.12 and lo <= cur <= hi:
            # Another tempo (or octave) peak.  Kick+snare music has near-equal comb peaks at
            # P and 1.5P/2P, so jump only when the new peak clearly wins for ~0.5 s, and never
            # slew through the tempos in between; meanwhile keep tracking the local peak.
            sc_cur = float(np.interp(cur, lags, score))
            self.tempo_votes = self.tempo_votes + 1 if score[k] > 1.15 * sc_cur else 0
            if self.tempo_votes >= 6:
                self.beat_period, self.tempo_votes = est, 0
                return
            near = np.abs(np.log2(lags / cur)) < 0.08
            k = int(np.argmax(np.where(near, score, -np.inf)))
            lag = float(lags[k])
            if 0 < k < len(score) - 1 and near[k - 1] and near[k + 1]:
                a, b, c = score[k - 1], score[k], score[k + 1]
                den = a - 2 * b + c
                if den < 0:
                    lag += 0.5 * (a - c) / den
            est = lag * self.TEMPO_GRID
        else:
            self.tempo_votes = 0
        self.beat_period += (est - self.beat_period) * 0.25

    def decay(self, dt):
        # frames without fresh PCM still pass time: keep the clock and beat phase running
        self.clock += dt
        self.beat_phase = (self.beat_phase + 2 * math.pi * max(0., min(0.1, dt)) / max(0.28, self.beat_period)) % (2 * math.pi)
        self.envelope = [v * math.exp(-dt / .12) for v in self.envelope]
        return self.envelope[:]


TWO_PI = 2.0 * math.pi
N_CORE = 220            # twist body points (indices 0..219, seam closes 0 == 219)
N_RING = 220            # max points for one expanding ring (actual count shrinks as transits grow)
N_FRAME = 450           # constant frame size: 22.5 ms @ 20 kpps = 44.4 Hz refresh
DWELL_OFF = 2           # extra blank repeats of the last curve point before a jump (laser-off latency)
DWELL_ON = 3            # blank repeats at a landing point before relighting (galvo settle)
BLANK_STEP = 760.0      # max ILDA step of an eased blank move (smoothstep peak ~1.5x mean)
FOLD_OCTAVES = 3
IMPRINT_MODES = 4       # body outline modes carried out by a ring (higher ones would kink a 200-pt ring)
IMPRINT_BUDGET = 0.6    # max sum(|a_m| m^2) of the carried outline: <= ~4 units of ring curvature at r 6600
CURV_MAX = 14.6         # emitted body second-difference ceiling (heliosctl galvo bound 15)
GRAD_MAX = 0.20
CURL_GAIN = 0.012       # shape contribution of the curl noise (psi itself still drives the 4D fold gradient)
FOLD_GAIN = 1.0         # peak 4D fold angle = FOLD_GAIN*2 rad*grad (tuned against curvature bound)
RING_R_MAX = 6600.0     # with transits taking ~30 pts, keeps ring chord spacing under the 225-unit bound
_A = np.linspace(0.0, TWO_PI, N_CORE)
_COS_A = np.cos(_A)
_SIN_A = np.sin(_A)


def _wrap_pi(x):
    return math.atan2(math.sin(x), math.cos(x))


def _blank_move(a, b):
    """Eased (smoothstep) blank path from a (exclusive) to b (inclusive); every step <= BLANK_STEP."""
    d = math.dist(a, b)
    n = max(1, math.ceil(1.5 * d / BLANK_STEP))
    t = np.arange(1, n + 1) / n
    e = t * t * (3.0 - 2.0 * t)
    return [(round(a[0] + (b[0] - a[0]) * u), round(a[1] + (b[1] - a[1]) * u)) for u in e.tolist()]


class BeatRings:
    """One expanding ring at a time, locked to the Bands beat PLL.

    Locked (beat_lock high, hysteresis LOCK_ON/LOCK_OFF):
      * the PLL beat_phase is unwrapped into a continuous cycle count; a ring spawns on
        the predicted phase wrap (phase 0 is where the PLL puts the kick), every `beats`
        wraps: one ring per beat up to ~130 BPM, one per two beats above (a ring that
        spans 2-4 beats at mid tempos read as 'beat adjacent' on the wall);
      * a ring's expansion fraction advances at a rate that is slewed (tau RATE_TAU)
        toward 1/(beats*beat_period) so it reaches the edge exactly when the next ring
        is born, and a small proportional pull toward the PLL-derived ideal fraction
        keeps it phase-locked without visible speed jumps when the tempo estimate moves;
    Unlocked: rings follow flux/bass onsets with a fixed lifetime, and a slow idle ring
    is born whenever none has been seen for IDLE_GAP seconds while music plays.
    Silence (activity below SIG_ON/SIG_OFF, with hysteresis) births nothing on any path;
    a ring already in flight still finishes, and the first ring after silence comes
    with the music (the spawn timer is reset on the silence->signal edge).
    """
    LOCK_ON = 0.25
    LOCK_OFF = 0.12
    FREE_LIFE = 1.35
    IDLE_LIFE = 2.4
    IDLE_GAP = 1.0
    SIG_ON = 0.06           # activity above which there is music to answer
    SIG_OFF = 0.03          # ...and below which it is silence again (room floor sits at ~0)
    RATE_TAU = 0.45
    RATE_SLEW = 0.7         # max change of the expansion rate, 1/s per s (no visible speed jump on a tempo change)
    PHASE_GAIN = 1.2
    MIN_LIFE = 0.46         # one ring per beat down to 0.46 s (~130 BPM); faster tempos double up
    MAX_LIFE = 0.50         # ...and halve again only if the shorter ring would still live >= MAX_LIFE s (> MIN_LIFE: no flapping)

    def __init__(self):
        self.prev_phase = None
        self.theta = 0.0
        self.max_cycle = 0
        self.last_spawn_cycle = 0
        self.last_spawn_time = -1e9
        self.locked = False
        self.signal = False
        self.beats = 1
        self.rate = 1.0 / self.FREE_LIFE
        self.ring = None   # dict: frac, color, kind, life, beats, theta0
        self.spawned = {'beat': 0, 'onset': 0, 'idle': 0}

    def _spawn(self, kind, now, color, life, theta0=None, frac0=0.0):
        self.ring = {'frac': frac0, 'color': color, 'kind': kind, 'life': life,
                     'beats': self.beats, 'theta0': theta0}
        self.last_spawn_time = now
        self.spawned[kind] += 1

    def update(self, now, dt, beat_phase, beat_period, beat_lock, act, onset, color):
        dt = max(0.0, min(dt, 0.1))
        beat_period = max(0.28, beat_period)
        # --- unwrap the PLL phase into continuous beat cycles ---
        if self.prev_phase is None:
            self.theta = beat_phase
            self.max_cycle = int(math.floor(self.theta / TWO_PI))
            self.last_spawn_cycle = self.max_cycle
        else:
            self.theta += _wrap_pi(beat_phase - self.prev_phase)
        self.prev_phase = beat_phase
        # Spawn on the *predicted* beat: the frame nearest phase 0 (half a frame of lead), not
        # the first frame after it, so the ring is born on the beat rather than up to a frame late.
        lead = math.pi * dt / beat_period
        cycle = int(math.floor((self.theta + lead) / TWO_PI))
        wrapped = cycle > self.max_cycle     # a backward PLL nudge can never re-fire a beat
        if wrapped:
            self.max_cycle = cycle
        # --- lock state and beats-per-ring, both with hysteresis ---
        if self.locked:
            self.locked = beat_lock > self.LOCK_OFF
        else:
            self.locked = beat_lock > self.LOCK_ON
        while self.beats < 4 and self.beats * beat_period < self.MIN_LIFE:
            self.beats *= 2
        while self.beats > 1 and (self.beats // 2) * beat_period >= self.MAX_LIFE:
            self.beats //= 2
        # --- signal gate (hysteresis): no audio, no new rings ---
        was_signal = self.signal
        self.signal = act > (self.SIG_OFF if was_signal else self.SIG_ON)
        if self.signal and not was_signal:
            # first ring arrives with the music: onsets may fire at once, and the idle
            # fallback waits only 0.25 s (activity crosses SIG_ON a frame or two before
            # the kick itself clears the onset threshold, so idle must not pre-empt it)
            self.last_spawn_time = now - (self.IDLE_GAP - 0.25)
        # --- spawn ---
        spawned_on_beat = False
        if not self.signal:
            pass
        elif self.locked and wrapped and cycle - self.last_spawn_cycle >= self.beats:
            ideal0 = (self.theta - TWO_PI * cycle) / (TWO_PI * self.beats)
            self._spawn('beat', now, color, None, theta0=TWO_PI * cycle,
                        frac0=min(0.25, max(0.0, ideal0)))
            self.last_spawn_cycle = cycle
            spawned_on_beat = True
        elif (not self.locked and onset and now - self.last_spawn_time > 0.28
                and (self.ring is None or self.ring['frac'] > 0.5)):
            self._spawn('onset', now, color, self.FREE_LIFE)
            self.last_spawn_cycle = cycle
        elif self.ring is None:
            gap = self.IDLE_GAP if not self.locked else max(self.IDLE_GAP, 1.5 * self.beats * beat_period)
            if now - self.last_spawn_time > gap:
                self._spawn('idle', now, (110, 220, 255), self.IDLE_LIFE)
                self.last_spawn_cycle = cycle
        # --- advance ---
        ring = self.ring
        if ring is not None:
            if self.locked:
                target = 1.0 / (self.beats * beat_period)
            else:
                target = 1.0 / (ring['life'] or self.FREE_LIFE)
            step = (target - self.rate) * (1.0 - math.exp(-dt / self.RATE_TAU))
            self.rate += max(-self.RATE_SLEW * dt, min(self.RATE_SLEW * dt, step))
            speed = self.rate
            if self.locked and ring['kind'] == 'beat':
                ideal = (self.theta - ring['theta0']) / (TWO_PI * ring['beats'])
                speed += self.PHASE_GAIN * (min(1.2, max(0.0, ideal)) - ring['frac'])
                speed = min(1.6 * self.rate, max(0.6 * self.rate, speed))
            if not spawned_on_beat:
                ring['frac'] += speed * dt
            if ring['frac'] >= 1.0:
                self.ring = None
        return spawned_on_beat


class LiquidContour:
    """Twist-sphere body (torsion-spring twist, curl-noise octaves, 4D fold) with one
    BPM-locked expanding ring (see BeatRings).  Every frame is exactly N_FRAME points
    (core 220 + dwell/eased-blank transit + ring (<=220, it yields points to longer
    transits) + dwell/eased walk-back, or core twice when no ring is visible) so the DAC refresh stays ~44 Hz at 20 kpps and brightness stays even.
    """
    MODES = ((2, 3), (3, 4), (2, 4), (1, 3))

    def __init__(self, seed=42):
        self.previous = None
        self.phase = 0.
        self.knot_phase = 0.
        self.morph_clock = 0.4
        self.rot4d = 0.0
        self.color_phase = 0.
        self.tilt_phase = 0.
        self.velocity = 0.35
        self.direction = 1
        self.last_onset = -math.inf
        self.previous_bass = 0.
        self.previous_energy = 0.
        self.onsets = 0
        self.impulses = []
        self.random = random.Random(seed)
        self.mode_cur = 0
        self.mode_nxt = 1
        self.mode_blend = 1.0
        self.twist = 0.0
        self.twist_vel = 0.0
        self.noise_clock = 0.0
        # (k, omega, phase, amplitude). Curl term amplitude is A*k and its second
        # difference scales A*k^3, so A ~ 1/k^2 keeps high octaves from kinking the line.
        octs = [(3, 0.55, 0.00, 1.00), (5, 0.83, 2.10, 0.40), (7, 1.19, 4.40, 0.20),
                (11, 1.61, 1.30, 0.08), (13, 2.13, 5.20, 0.06)]
        self._oct_k = np.array([o[0] for o in octs], dtype=np.float64)
        self._oct_w = np.array([o[1] for o in octs], dtype=np.float64)
        self._oct_p = np.array([o[2] for o in octs], dtype=np.float64)
        self._oct_a = np.array([o[3] for o in octs], dtype=np.float64)
        self.rings = BeatRings()
        self.colors = None
        self.punch = 0.0            # beat-birth body kick, 1 -> 0 over ~0.1 s
        self.ring_obj = None        # identity of the ring being drawn (to catch births)
        self.ring_hue = 0.0         # palette walk: each new ring steps by the golden angle
        self.imprint = np.zeros(IMPRINT_MODES, dtype=np.complex128)   # body outline shed into the ring

    def sample(self, level, bands, now, dt, depth=.07, frequency=8, flux=0., centroid=.25, pitch_angle=None,
               sub_flux=None, beat_phase=0., hat_phase=0., beat_lock=0., hat_lock=0., activity=None,
               emitters=None, beat_period=0.5):
        bass, mid, high = bands
        kick_f, snare_f, hat_f = sub_flux if sub_flux is not None else (flux * bass, flux * mid, flux * high)
        energy = max(bass, .85 * mid, .7 * high, .9 * flux)
        act = min(1.0, max(0.0, activity if activity is not None else max(level, energy)))
        # --- body onsets: splash impulses and mode morphs (not the rings) ---
        if act > 0.22 and ((kick_f > .32) or (snare_f > .36) or (flux > .34)
                or (bass > .24 and bass - self.previous_bass > .09)
                or (energy > .28 and energy - self.previous_energy > .11)) and now - self.last_onset > .18:
            if snare_f > kick_f * 1.1:
                self.direction *= -1
            self.last_onset = now
            self.onsets += 1
            angle0 = ((pitch_angle + self.random.uniform(-.20, .20)) % TWO_PI if pitch_angle is not None
                      else self.random.uniform(0, TWO_PI))
            self.impulses = (self.impulses + [(now, angle0, self.random.choice((-1, 1)))])[-4:]
            if self.mode_blend >= 0.85:
                self.mode_cur = self.mode_nxt
                self.mode_nxt = (self.mode_cur + 1 + self.random.randrange(len(self.MODES) - 1)) % len(self.MODES)
                self.mode_blend = 0.0
        self.previous_bass = bass
        self.previous_energy = energy
        self.impulses = [p for p in self.impulses if now - p[0] < 0.85]
        step_dt = min(dt, .05)
        self.mode_blend = min(1.0, self.mode_blend + step_dt / 0.28)
        target_vel = self.direction * (0.22 + act * (0.95 + 1.5 * mid + 0.9 * centroid + 0.7 * bass))
        self.velocity += (target_vel - self.velocity) * (1 - math.exp(-dt / .24))
        # Phases feed integer multiples only, so wrapping mod 2*pi is seamless and
        # keeps float precision for unbounded sessions.
        self.phase = (self.phase + self.velocity * step_dt) % TWO_PI
        beat_pulse = max(0.0, math.cos(beat_phase)) ** 2 * beat_lock * act
        hat_sync = (0.5 + 0.5 * math.cos(hat_phase)) * max(hat_f, 0.7 * high * hat_lock) * act
        self.morph_clock = (self.morph_clock + (0.04 + act * (0.28 + 0.50 * energy + 0.40 * snare_f + 0.25 * beat_pulse)) * step_dt) % (8 * math.pi)
        self.knot_phase = (self.knot_phase + self.direction * (0.12 + act * (0.65 + 1.1 * bass + 0.7 * mid + 0.5 * beat_pulse)) * step_dt) % TWO_PI
        self.color_phase = (self.color_phase + (0.45 + act * (1.8 + 4.0 * energy + 2.0 * centroid + 1.8 * hat_sync)) * step_dt) % TWO_PI
        self.tilt_phase = (self.tilt_phase + (0.18 + act * (0.85 + 1.6 * bass + 1.2 * kick_f)) * step_dt) % (20 * math.pi)
        self.noise_clock += (0.35 + 1.4 * energy + 0.5 * beat_pulse) * step_dt

        # --- BPM-locked ring (spawn / expansion) ---
        ring_onset = act > 0.18 and (kick_f > 0.24 or bass > 0.20 or snare_f > 0.28)
        ring_color = (110, 220, 255) if kick_f >= snare_f else (255, 185, 80)
        on_beat = self.rings.update(now, dt, beat_phase, beat_period, beat_lock, act, ring_onset, ring_color)
        # Torsion spring: audio torque and each beat-ring birth wind the twist up;
        # spring + damping release it, so the fold breathes in time with the BPM.
        drive = 0.55 * (beat_pulse + 0.45 * kick_f + 0.25 * snare_f) * act
        if on_beat:
            self.twist_vel += 1.2 * act
            self.punch = 1.0
        self.punch *= math.exp(-step_dt / 0.09)
        self.twist_vel += (2.2 * drive - 6.0 * self.twist - 2.2 * self.twist_vel) * step_dt
        self.twist = max(-0.85, min(0.85, self.twist + self.twist_vel * step_dt))
        # Smooth periodic fold 0..2 (the snapshot's sawtooth rot4d popped at its wrap).
        self.rot4d = (self.rot4d + (0.045 + act * (0.22 + 0.40 * energy + 0.18 * beat_pulse)) * step_dt) % TWO_PI
        fold = FOLD_GAIN * (1.0 - math.cos(self.rot4d))

        amount = depth / .10
        r_core = (2400.0 + 1900.0 * level) * (0.82 + 0.18 * act) * (1.0 + 0.07 * self.punch * act)
        c_boost = 0.75 + 0.50 * centroid
        f_boost = 1.0 + 0.40 * max(flux, kick_f)
        low_amp = min(.40 * amount, .34 * amount * (bass + 0.28 * beat_pulse) * f_boost)
        mid_amp = min(.26 * amount, .23 * amount * (mid + 0.30 * snare_f) * c_boost)
        fine_amp = min(.038 * amount, .035 * amount * max(high, hat_sync) * c_boost)
        imp_gain = .26 * amount * f_boost * act
        m0_low, m0_mid = self.MODES[self.mode_cur]
        m1_low, m1_mid = self.MODES[self.mode_nxt]
        b1 = 0.5 - 0.5 * math.cos(math.pi * self.mode_blend)
        b0 = 1.0 - b1
        tilt_strength = 0.34 * max(0.5, amount) * act * (0.25 + 0.50 * bass + 0.25 * mid)
        tx = tilt_strength * math.sin(self.tilt_phase)
        ty = tilt_strength * math.cos(0.7 * self.tilt_phase)
        hat_step_phase = 3 * self.phase + hat_phase

        a = _A
        splash = np.zeros(N_CORE)
        for start, angle0, travel in self.impulses:
            age = max(0., now - start)
            delta = a - angle0 - travel * 3.2 * age
            packet = np.exp(2.0 * (np.cos(delta) - 1))
            env = (1 - math.exp(-age / .02)) * math.exp(-age / .28)
            splash += env * packet * np.cos(4 * delta - 12 * age)
        w_low = b0 * np.sin(m0_low * a - self.phase) + b1 * np.sin(m1_low * a - self.phase)
        w_mid = b0 * np.sin(m0_mid * a + 2 * self.phase) + b1 * np.sin(m1_mid * a + 2 * self.phase)
        w_hat = np.sin(frequency * a - hat_step_phase)
        arg = np.outer(self._oct_k, a) - (self._oct_w * self.noise_clock - self._oct_p)[:, None]
        curl = ((self._oct_a * self._oct_k) @ np.cos(arg)) * (CURL_GAIN * amount * act)
        # Fold gradient follows only the smooth low octaves (k<=7): at 220 pts/loop the k=11,13
        # octaves alias into kinks once the fold rotates the ring through W.
        psi = self._oct_a[:FOLD_OCTAVES] @ np.sin(arg[:FOLD_OCTAVES])
        wave = low_amp * w_low + mid_amp * w_mid + fine_amp * w_hat + imp_gain * splash + curl
        mod = 1.0 + 0.14 * np.tanh(1.5 * wave)
        x0 = _COS_A * mod + 0.045 * wave * _COS_A
        y0 = _SIN_A * mod + 0.045 * wave * _SIN_A
        z0 = 0.09 * wave
        # 4D fold: per-point gradient from the curl-noise potential, scaled by how
        # wound the torsion spring is; only part of the ring sweeps through W.
        g = np.clip(psi * 0.62, 0.0, 1.0)
        grad = GRAD_MAX * (0.25 + 0.75 * g * g) * (0.35 + 0.65 * abs(self.twist))
        ang4 = (fold + self.twist) * grad
        c4, s4 = np.cos(ang4), np.sin(ang4)
        w0 = x0 * s4
        x0 = x0 * c4
        z1 = z0 * c4 - w0 * s4
        w0 = 0.08 * np.tanh((z0 * s4 + w0 * c4) / 0.08)   # smooth limit: a hard clip kinks the line
        persp4 = 1.0 / (1.0 - 0.5 * w0)          # at most about +/-4% magnification
        x0 = x0 * persp4
        y0 = y0 * persp4
        z0 = z1 * persp4
        xr = x0 * math.cos(ty) + z0 * math.sin(ty)
        yr = y0 * math.cos(tx) - z0 * math.sin(tx)
        zr = -x0 * math.sin(ty) + y0 * math.sin(tx) + z0 * math.cos(tx) * math.cos(ty)
        persp = 1.0 / (1.0 - 0.15 * np.clip(zr, -0.8, 0.8))
        pts = np.stack((r_core * xr * persp, r_core * yr * persp), axis=1)
        if self.previous is not None:
            movement = float(np.max(np.hypot(*(pts - self.previous).T)))
            blend = min(1.0, min(560.0, 22400.0 * max(dt, 0.01)) / max(1e-9, movement))
            pts = self.previous + blend * (pts - self.previous)
        pts[-1] = pts[0]                        # exact seam closure
        # Galvo guard on what is actually emitted (integer rounding alone adds up to ~2): the
        # second difference is linear in scale, so shrink the body about its centre the few
        # percent needed whenever beat kicks stack past CURV_MAX.
        core = np.rint(pts).astype(np.int64)
        for _ in range(4):
            curv = float(np.max(np.hypot(*(core[2:] - 2 * core[1:-1] + core[:-2]).T)))
            if curv <= CURV_MAX:
                break
            pts = pts * ((CURV_MAX - 1.0) / curv)
            core = np.rint(pts).astype(np.int64)
        self.previous = pts

        crest = np.minimum(1.0, np.maximum(0.0, splash) * act + 0.65 * hat_sync * np.maximum(0.0, w_hat)
                           + 0.85 * act * np.abs(w0) * grad)
        bass_glow = np.maximum(0.0, w_low) * max(bass, kick_f) * act
        mid_glow = np.maximum(0.0, w_mid) * max(mid, snare_f) * act
        hue = a * 2.0 - self.color_phase + 2.2 * wave
        s_r = np.maximum(0.0, np.sin(hue)) ** 1.6
        s_g = np.maximum(0.0, np.sin(hue + 2.094)) ** 1.6
        s_b = np.maximum(0.0, np.sin(hue + 4.188)) ** 1.6
        cr = np.minimum(255, 14 + 210 * s_r * (0.12 + 0.88 * act * (bass + 0.5 * energy)) + 150 * bass_glow + 140 * crest)
        cg = np.minimum(255, 22 + 200 * s_g * (0.18 + 0.82 * act * (mid + high)) + 170 * crest)
        cb = np.minimum(255, 55 + 195 * s_b * (0.40 + 0.60 * (1.0 - 0.3 * bass)) + 120 * mid_glow + 90 * crest)
        core_col = np.stack((cr, cg, cb), axis=1)
        flash = 0.40 * self.punch * act                    # the body lights up as it sheds a ring
        core_col = np.minimum(255, core_col * (1.0 - flash) + 255.0 * flash).astype(np.int64)

        coords = [tuple(p) for p in core.tolist()]
        colors = [tuple(c) for c in core_col.tolist()]
        blanks = [False] * N_CORE
        blanks[0] = blanks[-1] = True
        colors[0] = colors[-1] = (0, 0, 0)

        # --- ring (one at a time, expansion fraction from BeatRings) ---
        ring = self.rings.ring
        fade = 0.0
        if ring is not None:
            frac = ring['frac']
            fade = max(0.0, 1.0 - frac ** 1.8) * (0.35 + 0.65 * act)   # holds its light, dies at the edge
        if ring is not None and fade >= 0.03:
            r0 = 1.18 * r_core
            rw = r0 + (RING_R_MAX - r0) * frac
            if ring is not self.ring_obj:
                # A new ring: next palette colour, and freeze the body's current low-order
                # outline (modes 1..IMPRINT_MODES) so the ring leaves as a shell of the sphere.
                self.ring_obj = ring
                self.ring_hue = (self.ring_hue + 2.39996) % TWO_PI
                prof = np.hypot(pts[:-1, 0], pts[:-1, 1])
                spec = np.fft.rfft(prof / max(1e-6, float(np.mean(prof))) - 1.0) / len(prof)
                imp = 2.0 * spec[1:IMPRINT_MODES + 1]
                # curvature budget: a mode-m ripple of relative size a adds ~r*a*m^2*dtheta^2
                # to the second difference, so cap sum(|a_m| m^2) instead of clipping (a clip kinks)
                load = float(np.sum(np.abs(imp) * np.arange(1, IMPRINT_MODES + 1) ** 2))
                self.imprint = imp * min(1.0, IMPRINT_BUDGET / max(1e-9, load))
            # Transit out and walk back cover the same radial gap at angle 0: dwell, eased
            # blank move, settle -- the mirrors must stop before the laser relights, or the
            # beam paints the jump (the 'line from the sphere to the ring').
            modes = np.arange(1, IMPRINT_MODES + 1)
            shell = (1.0 - frac) ** 1.3

            def ring_radius(ang):
                w_ring = (0.12 * bass * np.sin(2 * ang - self.phase) + 0.07 * mid * np.sin(3 * ang + self.phase)
                          + 0.06 * self.twist * np.sin(2 * ang + 2 * self.phase))
                carried = np.real(self.imprint @ np.exp(1j * np.outer(modes, ang)))
                return rw * (1.0 + 0.05 * np.tanh(w_ring) * min(1.0, 2.0 * fade) + shell * carried)
            seam = coords[-1]
            land = (int(np.rint(ring_radius(np.zeros(1))[0])), 0)
            out = _blank_move(seam, land)
            back = _blank_move(land, coords[0])     # same gap (core seam == coords[0]): same length
            n_ring = min(N_RING, N_FRAME - N_CORE - 2 * (DWELL_OFF + DWELL_ON) - len(out) - len(back))
            ring_a = np.linspace(0.0, TWO_PI, n_ring)
            r_ring = ring_radius(ring_a)
            rx = np.rint(r_ring * np.cos(ring_a)).astype(np.int64)
            ry = np.rint(r_ring * np.sin(ring_a)).astype(np.int64)
            rpts = list(zip(rx.tolist(), ry.tolist()))
            rpts[0] = rpts[-1] = land
            dark = [seam] * DWELL_OFF + out + [rpts[0]] * DWELL_ON
            # Colour: white-hot at birth, cooling into a hue sweep around the ring that turns
            # with the body; successive rings walk the palette so each beat reads as new.
            hue = self.ring_hue + 1.1 * np.cos(ring_a - 1.5 * self.phase)
            rgb = np.stack([np.maximum(0.0, np.sin(hue + off)) ** 1.2 for off in (0.0, 2.094, 4.188)], axis=1)
            rgb = rgb / np.maximum(1e-6, rgb.max(axis=1, keepdims=True))
            white = math.exp(-frac / 0.06)
            bright = fade * (1.0 + 0.25 * white)
            ring_rgb = np.clip(255.0 * (rgb * (1.0 - white) + white) * bright, 0, 255).astype(np.int64)
            ring_cols = [tuple(c) for c in ring_rgb.tolist()]
            ring_cols[0] = ring_cols[-1] = (0, 0, 0)
            ring_blank = [False] * n_ring
            ring_blank[0] = ring_blank[-1] = True
            tail = [rpts[-1]] * DWELL_OFF + back
            pad = N_FRAME - len(coords) - len(dark) - n_ring - len(tail)
            tail += [coords[0]] * pad          # settle on the core seam (== DWELL_ON points)
            coords += dark + rpts + tail
            colors += [(0, 0, 0)] * len(dark) + ring_cols + [(0, 0, 0)] * len(tail)
            blanks += [True] * len(dark) + ring_blank + [True] * len(tail)
        else:
            # No visible ring: retrace the body so lit duty and refresh stay constant.
            coords.extend(coords[1:N_CORE]); colors.extend(colors[1:N_CORE]); blanks.extend(blanks[1:N_CORE])
            while len(coords) < N_FRAME:
                coords.append(coords[0]); colors.append((0, 0, 0)); blanks.append(True)
        self.colors = colors
        return coords, colors, blanks
