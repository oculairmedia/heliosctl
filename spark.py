"""`--pattern liquid3`: liquid2's body and rings, but the form changes only on musical *events*
and percussive *hits* sparkle.

SparkContour keeps everything DynamicContour does (figures, rigid spin, ring styles, the
liquid body, earned rings, galvo guard) and changes two things:

* Morphs are earned, not cadenced.  liquid2 morphs every beat / second / fourth beat by
  energy: a clock with shapes on it.  Here a shape change needs an event: a sudden accent
  rise (a drop, a section entry: the same measure BeatRings credits at once), an accent
  ring birth, a *surprising* hit (one well above the recent hits), a fill (a run of snare
  hits inside one beat), or the music starting.  Accent events wait for the beat frame
  when locked so the new form lands on the beat; hits commit at once and morph fast.
  Between events the form holds and keeps breathing (twist, curl noise, fold, spin).
  There is no fallback: a long passage without an event keeps its shape.  Silence still
  drifts home to the sphere.

* Hits sparkle.  A hit is a frame where some band's normalised flux fires and the raw band
  energies (Bands.energy, passed as `energy=`) jump by a good fraction of their level.  The
  jump's spectral *shares* (low / mid / high) name it:
      kick   low-dominated, no high content          thump: body flash, a few heavy warm
                                                     sparks that droop (gravity)
      snare  mid (+ high) jump, or kick + crack       crack: full white body flash, body
                                                     shatters into fragments for 3 frames,
                                                     8 fast white sparks scatter all round
      hat    high-dominated                          tick: 3 tiny fast cyan sparks off the
                                                     ring (or body), ring flashes white
      stab   low AND mid jump together, little high  fan of sparks in the pitch direction,
                                                     hue-inverted body flash, longer shatter
  Sparks are short lit streaks (2-3 points) with their own dwell / eased blank move / settle,
  alive 0.09-0.20 s; they take their points from the body retrace (no ring) or from the
  ring, which yields an open arc at its full chord spacing (dynamic.py `budget`).  The frame
  is always N_FRAME points; spark travel per frame is bounded (SPARK_STEP_MAX) and every
  spark block follows the FrameBlankingTests dwell/settle rules, so nothing pops the galvos.
"""
import math
import random
import numpy as np
from dynamic import DynamicContour, MORPH_MIN, MORPH_MAX, SPIN_MAX
from liquid import _blank_move, N_CORE, N_FRAME, DWELL_OFF, DWELL_ON, RING_R_MAX, TWO_PI

# --- hit detection ---
HIT_FLUX = 0.30          # some band's normalised flux must fire...
HIT_RISE = 0.6           # ...and some raw band energy must rise by this fraction of its own level
HIT_MIN_JUMP = 0.003     # ...and by this much in absolute terms (a hat under a decaying kick still counts)
HIT_MIN_ACT = 0.08
REFRACTORY = {'kick': 0.10, 'snare': 0.10, 'hat': 0.07, 'stab': 0.15}   # s, per kind
SURPRISE = 2.0           # a hit this much above its kind's recent median is an event (morph); within a
                         # steady passage a kind's jumps stay within ~1.6x of their median (split blocks go
                         # lower, never higher), section entries are 3.7-10x
HIT_HIST = 8             # jumps per kind the reference median is taken over (needs 4 to judge)
FILL_HITS, FILL_WINDOW = 4, 1.0     # >= 4 snare/stab hits within a beat (<= 1 s) is a fill
# --- morph events ---
MORPH_GAP = 0.30         # least time between two morph starts, s
MORPH_HIT = 0.16         # morph time on a hit event, s (fast: the form commits with the hit)
PENDING_LIFE = 1.5       # an accent event waits at most this many beats for a beat frame
# --- sparks ---
SPARK_STEP_MAX = 340.    # max travel of a spark between frames, ILDA units (bounded steps)
SPARK_SEG = 115.         # spacing of a streak's lit points (well under the 225 segment bound)
SPARK_PTS_MAX = 128      # points sparks may take from the body retrace (no ring)
RING_YIELD = 44          # points sparks may take from a live ring (an open arc of ~20%)
SPARK_R_MAX = 10500.     # a spark past this radius dies (frame extent is +/-16000 x 13000)
SHATTER_FRAGS = 7
# kind: (sparks, life s, speed u/s, streak pts, spread rad, gravity u/s^2, rgb, body flash, flash rgb, shatter s)
KINDS = {
    'kick':  (4, 0.13, 4600., 3, 0.35, 16000., (255, 110, 30), 0.55, (255, 150, 60), 0.0),
    'snare': (8, 0.16, 9800., 3, math.pi, 0., (255, 225, 240), 1.00, (255, 255, 255), 0.07),
    'hat':   (3, 0.09, 11500., 2, 0.5, 0., (120, 235, 255), 0.0, (255, 255, 255), 0.0),
    'stab':  (7, 0.20, 8200., 3, 0.6, 0., (255, 90, 255), 0.85, (255, 255, 255), 0.11),
}


class SparkContour(DynamicContour):
    # liquid3 *holds* a figure for seconds at a time (liquid2 never did), so a saturated
    # passage stacks the beat punch and the body's splash impulses onto the same figure beat
    # after beat; 5% smaller figures keep the held body under CURV_MAX so the guard does not
    # shrink and regrow it by 7% every few beats (that regrow is a body step of ~650 units
    # between frames).  The guard itself is untouched and still the backstop.
    FIGURES = [f * 0.95 for f in DynamicContour.FIGURES]

    def __init__(self, seed=42):
        super().__init__(seed)
        self.prev_energy = None
        self.last_hit = {k: -1e9 for k in KINDS}
        self.hits = []                  # (kind, strength) detected on this frame
        self.hit_count = {k: 0 for k in KINDS}
        self.hit_hist = {k: [] for k in KINDS}   # recent energy jumps per kind: the ordinary hit of this passage
        self.hit_ref_since = None       # when the first hit was seen (surprise needs history)
        self.snare_times = []
        self.last_fill = -1e9
        self.prev_acc = 0.0
        self.pending = None             # (event, time) waiting for the beat frame
        self.events = []                # (time, event, shape) morph events, newest last
        self.morph_time = MORPH_MAX
        self.flash = 0.0                # body flash 0..1
        self.flash_rgb = (255, 255, 255)
        self.flash_invert = False
        self.shatter_until = -1e9
        self.shatter_len = 0.07
        self.shatter_rot = 0.0
        self.ring_flash = 0.0
        self.sparks = []                # dicts: x, y, vx, vy, born, life, kind, rgb, n
        self.spark_births = {k: 0 for k in KINDS}
        self.spark_pts = 0              # points sparks used on the last frame
        self._now = 0.0
        self._dt = 0.0225
        self._pitch = None
        self.spark_random = random.Random(seed + 1)   # sparks never steal the morph picker's draws

    # ----- hits -----
    def _classify(self, energy, sub_flux, act):
        """One hit kind for this frame or None, from the raw band-energy jump's spectral shares."""
        if self.prev_energy is None:
            return None, 0.0
        d = [max(0.0, e - p) for e, p in zip(energy, self.prev_energy)]
        s = sum(d)
        rel = max(v / max(1e-4, p) for v, p in zip(d, self.prev_energy))
        if max(sub_flux) < HIT_FLUX or act < HIT_MIN_ACT or rel < HIT_RISE or s < HIT_MIN_JUMP:
            return None, 0.0
        L, M, H = (v / s for v in d)
        if L >= 0.55 and M < 0.25 and H < 0.035:
            kind = 'kick'
        elif L >= 0.4:
            kind = 'snare'              # kick with a clap/snare on top: the crack is what shows
        elif M >= L >= 0.2 and H < 0.2:
            kind = 'stab'
        elif H >= 0.45:
            kind = 'hat'
        elif M >= 0.4:
            kind = 'snare'
        else:
            return None, 0.0
        return kind, s

    def sample(self, level, bands, now, dt, *args, energy=None, **kw):
        self._now, self._dt = now, max(0.0, min(dt, 0.05))
        self._pitch = kw.get('pitch_angle')
        flux = kw.get('flux', args[2] if len(args) > 2 else 0.0)
        sub_flux = kw.get('sub_flux')
        if sub_flux is None:
            sub_flux = (flux * bands[0], flux * bands[1], flux * bands[2])
        activity = kw.get('activity')
        act = min(1.0, max(0.0, activity if activity is not None else max(level, max(bands))))
        if energy is None:
            energy = bands                  # no raw energies (tests, previews): the envelopes stand in
        self.hits = []
        kind, jump = self._classify(energy, sub_flux, act)
        if kind is not None and now - self.last_hit[kind] >= REFRACTORY[kind]:
            self.last_hit[kind] = now
            self.hit_count[kind] += 1
            # strength = this hit's energy jump over the median of its kind's recent jumps: ~1 for
            # an ordinary hit of the passage whatever the gain, well above 1 for one that stands out
            hist = self.hit_hist[kind]
            strength = jump / float(np.median(hist)) if len(hist) >= 4 else 1.0
            if self.hit_ref_since is None:
                self.hit_ref_since = now
            self.hit_hist[kind] = (hist + [jump])[-HIT_HIST:]
            self.hits.append((kind, strength))
        self.prev_energy = list(energy)
        return super().sample(level, bands, now, dt, *args, **kw)

    # ----- morph on events -----
    def _fire(self, event, now, act, acc, strong):
        if now - self.last_morph < MORPH_GAP:
            return
        self._start_morph(self._pick(act, max(acc, 0.8 if strong else 0.0)), now)
        self.morph_time = MORPH_HIT if event in ('hit', 'fill') else self.morph_time
        self.events = (self.events + [(now, event, self.NAMES[self.shape_nxt])])[-64:]

    def _figure(self, now, step_dt, act, accent, beat_period, on_beat):
        acc = 0.0 if accent is None else accent
        rings = self.rings
        self.morph_time = min(MORPH_MAX, max(MORPH_MIN, 0.55 * beat_period))
        if not rings.signal:
            self.pending = None
            if self.shape_nxt != 0 and self.morph >= 1.0 and now - self.last_morph > 1.0:
                self._start_morph(0, now)
                self.events = (self.events + [(now, 'silence', 'sphere')])[-64:]
        else:
            beat = on_beat or rings.beat_hit
            event, strong = None, False
            if rings.last_spawn_time == now and rings.spawned['beat'] + rings.spawned['onset'] + rings.spawned['idle'] == 1 \
                    and self.morphs == 0:
                event = 'start'                                   # music (re)starting is an event
            sudden = acc - self.prev_acc - 0.8 * self.prev_acc - 0.4
            if sudden > 0.0:
                event, strong = 'accent', True                    # a drop, a section entry
            if on_beat and rings.ring is not None and rings.ring.get('weight', 0.0) >= 0.85:
                event, strong = 'accent', acc >= 0.8              # a ring earned by an accent, not the floor
            for kind, s in self.hits:
                if kind in ('snare', 'stab'):
                    self.snare_times = [t for t in self.snare_times if now - t <= min(FILL_WINDOW, beat_period)] + [now]
                    if len(self.snare_times) >= FILL_HITS and now - self.last_fill > 1.0:
                        self.last_fill = now
                        event, strong = 'fill', True
                if now - self.hit_ref_since > 2.0 and s > SURPRISE:
                    event, strong = 'hit', kind == 'stab' or s > 2.0
            if event in ('accent', 'start') and rings.locked and not beat:
                self.pending = (event, now, strong)             # land it on the beat
            elif event is not None:
                self._fire(event, now, act, acc, strong)
                self.pending = None
            if self.pending is not None:
                if beat:
                    self._fire(self.pending[0], now, act, acc, self.pending[2])
                    self.pending = None
                elif now - self.pending[1] > PENDING_LIFE * beat_period:
                    self.pending = None
        self.prev_acc = acc
        self.morph_rate = 1.0 / self.morph_time
        self.morph = min(1.0, self.morph + step_dt * self.morph_rate)
        self.spin = (self.spin + self.direction * SPIN_MAX * (0.15 + 0.85 * act) * step_dt) % TWO_PI
        f = self._current_figure()
        c, s = math.cos(self.spin), math.sin(self.spin)
        return f[:, 0] * c - f[:, 1] * s, f[:, 0] * s + f[:, 1] * c, f[:, 2]

    # ----- sparkle -----
    def _spawn_sparks(self, pts, r_core, act):
        ring = self.rings.ring
        rw = None
        if ring is not None:
            r0 = 1.18 * r_core
            rw = r0 + (RING_R_MAX - r0) * ring['frac']
        for kind, strength in self.hits:
            n, life, speed, streak, spread, grav, rgb, bflash, frgb, shatter = KINDS[kind]
            gain = min(1.0, max(0.55, 0.75 * strength)) * (0.6 + 0.4 * act)
            if bflash > 0.0:
                self.flash = max(self.flash, bflash * gain)
                self.flash_rgb = frgb
                self.flash_invert = kind == 'stab'
            if shatter > 0.0:
                self.shatter_until = self._now + shatter
                self.shatter_len = shatter
                self.shatter_rot = self.spark_random.random()
            if kind == 'hat' and ring is not None:
                self.ring_flash = 1.0
            if kind == 'stab' and self._pitch is not None:
                centre = self._pitch
            else:
                centre = self.spark_random.uniform(0.0, TWO_PI)
            for i in range(n):
                th = centre + (self.spark_random.uniform(-spread, spread) if kind != 'snare'
                               else TWO_PI * (i + self.spark_random.uniform(0.0, 0.6)) / n)
                if kind == 'hat' and rw is not None:
                    x, y = rw * math.cos(th), rw * math.sin(th)
                else:
                    k = int((th % TWO_PI) / TWO_PI * (N_CORE - 1))
                    x, y = float(pts[k, 0]), float(pts[k, 1])
                    th = math.atan2(y, x)
                v = speed * self.spark_random.uniform(0.7, 1.0) * gain
                tang = self.spark_random.uniform(-0.6, 0.6) if kind == 'snare' else self.spark_random.uniform(-0.2, 0.2)
                self.sparks.append({'x': x, 'y': y, 'vx': v * (math.cos(th) - tang * math.sin(th)),
                                    'vy': v * (math.sin(th) + tang * math.cos(th)), 'born': self._now,
                                    'life': life * self.spark_random.uniform(0.75, 1.0), 'kind': kind, 'rgb': rgb,
                                    'n': streak, 'grav': grav})
                self.spark_births[kind] += 1

    def _advance_sparks(self):
        dt = self._dt
        alive = []
        for s in self.sparks:
            age = self._now - s['born']
            if age > s['life']:
                continue
            s['vy'] -= s['grav'] * dt
            v = math.hypot(s['vx'], s['vy'])
            cap = SPARK_STEP_MAX / max(1e-6, dt)
            if v > cap:
                s['vx'] *= cap / v; s['vy'] *= cap / v
            s['x'] += s['vx'] * dt; s['y'] += s['vy'] * dt
            if math.hypot(s['x'], s['y']) > SPARK_R_MAX:
                continue
            alive.append(s)
        self.sparks = alive

    def _spark_block(self, s):
        """Lit streak points (head first) and colours of one spark on this frame, plus the
        blank settle point in front of the head and the blank dwell point behind the tail.
        Both lie on the streak's own line one step out, so the lit run with its blank
        neighbours is straight and evenly spaced: zero second difference by construction
        (a dwell *on* the end point would kink the run by a whole step)."""
        age = max(0.0, self._now - s['born'])
        env = max(0.0, 1.0 - age / s['life']) ** 0.7
        white = math.exp(-age / 0.03)
        v = math.hypot(s['vx'], s['vy'])
        ux, uy = (s['vx'] / v, s['vy'] / v) if v > 1e-6 else (1.0, 0.0)
        step = min(SPARK_SEG, max(40.0, 0.012 * v))
        line = [(int(round(s['x'] - k * step * ux)), int(round(s['y'] - k * step * uy))) for k in range(-1, s['n'] + 1)]
        a, pts, b = line[0], line[1:-1], line[-1]
        cols = []
        for k in range(s['n']):
            tail = 1.0 - 0.35 * k / max(1, s['n'] - 1)
            lum = max(0.35, env) * tail
            cols.append(tuple(int(min(255.0, (c * (1.0 - white) + 255.0 * white) * lum)) for c in s['rgb']))
        return pts, cols, a, b

    def _append_sparks(self, coords, colors, blanks, start, budget):
        """Append spark blocks and return to coords[0] with a settle, appending exactly
        `budget` points (the remainder settles on the seam).  `start` is the blank dwell
        point the walk leaves from: it must continue the curve the frame was drawing
        (the seam, or the body point after a partial retrace), never duplicate its last lit
        point."""
        home = coords[0]
        order = sorted(self.sparks, key=lambda s: math.atan2(s['y'], s['x']))
        blocks = []
        cur = start
        used = 0
        for s in order:
            pts, cols, a, b = self._spark_block(s)
            move = _blank_move(cur, a)
            back = _blank_move(b, home)
            need = (DWELL_OFF + 1) + len(move) + (DWELL_ON + 1) + len(pts)
            if used + need + (DWELL_OFF + 1) + len(back) + (DWELL_ON + 1) > budget:
                break
            blocks.append((cur, move, a, pts, cols))
            used += need
            cur = b
        for prev, move, a, pts, cols in blocks:
            coords += [prev] * (DWELL_OFF + 1) + move + [a] * (DWELL_ON + 1)
            colors += [(0, 0, 0)] * (DWELL_OFF + 1 + len(move) + DWELL_ON + 1)
            blanks += [True] * (DWELL_OFF + 1 + len(move) + DWELL_ON + 1)
            coords += pts; colors += cols; blanks += [False] * len(pts)
        back = _blank_move(cur, home)
        tail = [cur] * (DWELL_OFF + 1) + back
        pad = budget - used - len(tail)
        tail += [home] * pad
        coords += tail; colors += [(0, 0, 0)] * len(tail); blanks += [True] * len(tail)
        return len(blocks)

    def _append_ring(self, coords, colors, blanks, pts, r_core, bass, mid, act):
        now, dt = self._now, self._dt
        # --- body flash / shatter (colours and in-place blanks of the body just built) ---
        self._spawn_sparks(pts, r_core, act)
        self._advance_sparks()
        if self.flash > 0.01:
            f = self.flash
            body = np.asarray(colors[1:N_CORE - 1], dtype=np.float64)
            if self.flash_invert:
                body = 255.0 - body * 0.6
            fr = np.asarray(self.flash_rgb, dtype=np.float64)
            body = np.minimum(255.0, body * (1.0 - f) + fr * f)
            colors[1:N_CORE - 1] = [tuple(c) for c in body.astype(np.int64).tolist()]
        self.flash *= math.exp(-dt / 0.045)
        if now < self.shatter_until:
            prog = 1.0 - (self.shatter_until - now) / self.shatter_len           # 0 -> 1 over the shatter
            duty = 0.55 + 0.45 * prog * prog
            idx = np.arange(4, N_CORE - 5)
            u = (idx / (N_CORE - 1) * SHATTER_FRAGS + self.shatter_rot) % 1.0
            for i in idx[u >= duty].tolist():
                blanks[i] = True; colors[i] = (0, 0, 0)
        # --- how many points do the sparks want ---
        want = 0
        for s in self.sparks:
            want += (DWELL_OFF + 1) + 5 + (DWELL_ON + 1) + s['n']
        ring = self.rings.ring
        fade = 0.0
        if ring is not None:
            frac = ring['frac']
            fade = max(0.0, 1.0 - frac ** 1.8) * (0.35 + 0.65 * act)
            fade *= 0.6 + 0.4 * ring.get('weight', 1.0)
        ring_visible = ring is not None and fade >= 0.03
        if want == 0:
            self.spark_pts = 0
            super()._append_ring(coords, colors, blanks, pts, r_core, bass, mid, act)
            if self.ring_flash > 0.01 and ring_visible:
                self._flash_ring(colors, blanks)
            self.ring_flash *= math.exp(-dt / 0.03)
            return
        if ring_visible:
            k = min(want + 24, RING_YIELD)
            super()._append_ring(coords, colors, blanks, pts, r_core, bass, mid, act, budget=N_FRAME - k)
            if self.ring_flash > 0.01:
                self._flash_ring(colors, blanks)
            # the ring's tail settled on the seam: the spark walk starts there
            start = coords[-1]
        else:
            k = min(want + 24, SPARK_PTS_MAX)
            m = N_FRAME - k - N_CORE                 # partial body retrace coords[1..m], lit
            while m > 1 and blanks[m]:               # (never end it on a shattered gap)
                m -= 1
            k = N_FRAME - N_CORE - m
            coords.extend(coords[1:1 + m]); colors.extend(colors[1:1 + m]); blanks.extend(blanks[1:1 + m])
            start = coords[m + 1]                    # dwell on the next body point: the run stays smooth
        self.ring_flash *= math.exp(-dt / 0.03)
        self.spark_pts = k
        self._append_sparks(coords, colors, blanks, start, k)

    def _flash_ring(self, colors, blanks):
        f = self.ring_flash
        for i in range(N_CORE, len(colors)):
            if not blanks[i]:
                c = colors[i]
                colors[i] = tuple(int(min(255.0, v * (1.0 - f) + 255.0 * f)) for v in c)
