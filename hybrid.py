"""`--pattern liquid4`: liquid3 (event morphs, hit sparkle) plus nine things Emmanuel asked to see.

HybridContour keeps SparkContour's hit classifier, morph events, flash and shatter, and adds:

1. Sparks inherit body motion.  A spark leaves with the local body velocity at its detach
   point (this frame's body point minus last frame's, over dt) plus a reduced outward push,
   so a hit during a twist throws them in a spiral instead of a radial fan.
2. Persistent ring trails.  Every TRAIL_EVERY seconds the live ring leaves a dim arc at its
   current radius; the arc stays put while the ring grows on, fading over TRAIL_LIFE.  Up to
   TRAIL_MAX arcs: the ring's history inside it.  They take their points from the ring, which
   yields them as an open arc (dynamic.py `budget`) -- a full extra circle costs 110+ points.
3. Spark colour by pitch.  Bands.centroid colours every spark: low spectrum red, high cyan
   (snare keeps half its white so the crack still reads).
4. Impact ring.  A hard hit (well above its kind's recent median, or the biggest whole-spectrum
   energy step of the last seconds) spawns a tiny white circle at the detach point that
   expands IMPACT_R0 -> IMPACT_R1 and is dead in IMPACT_LIFE (~90 ms, 4 frames).  Drawn as a
   pseudo-spark block (IMPACT_N lit points: the second-difference bound sets the radius a
   40-point circle may have, see below).
5. Beam-width illusion.  On accents (Bands.accent high, or a body flash) the body retrace is
   drawn shifted by BEAM_OFF units and BEAM_GAIN brighter: a thicker, hotter line.  The shift
   is a translation, so segment and curvature are those of the body itself.
6. Arcing sparks.  Kick and stab sparks fall under gravity; snare and hat sparks curl (their
   velocity turns at a few rad/s), so stab/kick droop and the rest arc.
7. Silhouette flash.  On a big hit the body is drawn full white, twice, and nothing else is
   lit, for SIL_FRAMES frames.
8. Section-aware palette.  `heat` follows activity over seconds (rises in HEAT_UP s, falls in
   HEAT_DOWN s) and jumps on a drop (a large accent); every lit colour is tinted cold -> hot
   by it.  `section` names it: intro / build / drop.
9. Bands improvements (liquid.py): the tempo comb weights kick frames up (eighth-note hats and
   snares no longer outvote the kicks, the 70 BPM-read-as-140 case) and exposes whole-spectrum
   `hit_flux` / `hit_rms` (positive linear-magnitude flux and energy step, normalised to their
   recent peak).  Here they size a hit: a 'kick' or 'snare' whose energy step is tiny for the
   passage is a hat, and the big-hit effects (impact, silhouette) key on them.

Frame contract is unchanged: N_FRAME points, body first, every lit run bounded by the galvo
limits (segment 225, second difference 15) with the FrameBlankingTests dwell/settle rules.
"""
import math
import numpy as np
from dynamic import DynamicContour
from spark import SparkContour, KINDS, SPARK_STEP_MAX, SPARK_R_MAX, SHATTER_FRAGS
from liquid import N_CORE, N_FRAME, DWELL_OFF, DWELL_ON, RING_R_MAX, TWO_PI

# 1 / 6 sparks
INHERIT = 1.0            # share of the local body velocity a spark leaves with
OUTWARD = 0.45           # share of the kind's own speed still pushing it off the body
CURL = {'kick': 0.0, 'snare': 3.0, 'hat': 4.5, 'stab': 1.2}        # rad/s turn of the path (+/-)
GRAVITY = {'kick': 16000., 'snare': 0., 'hat': 0., 'stab': 9000.}  # ILDA units / s^2
BODY_V_MAX = 9000.       # cap on the inherited body velocity, u/s
# 3 pitch colour
PITCH_LOW, PITCH_HIGH = (255, 60, 30), (80, 230, 255)
# 2 trails
TRAIL_EVERY, TRAIL_LIFE, TRAIL_MAX = 0.11, 0.45, 2
TRAIL_SPAN = 0.8         # rad of arc per trail
TRAIL_CHORD = 200.       # lit spacing on the arc (chord^2 / r <= 15 keeps the curvature; see _trail_n)
# 4 impact ring: a circle of n lit points at radius r has curvature r (2 pi / n)^2 and segment
# 2 pi r / n; 56 points allow r 900 under the 15-unit bound with rounding slack (40 points: r 550)
IMPACT_N, IMPACT_R0, IMPACT_R1, IMPACT_LIFE = 56, 160., 900., 0.09
IMPACT_STRENGTH, IMPACT_SIZE, IMPACT_GAP = 1.5, 0.9, 0.12
# 5 beam double
BEAM_OFF, BEAM_GAIN, BEAM_ACCENT = 55., 1.18, 0.8
# 7 silhouette
SIL_FRAMES, SIL_STRENGTH, SIL_SIZE = 2, 2.6, 0.6
# 8 palette
HEAT_UP, HEAT_DOWN = 4.0, 9.0
COLD, HOT = (0.70, 0.95, 1.25), (1.30, 0.90, 0.70)
# 9 hit size gate
HAT_SIZE = 0.22          # a kick/snare/stab with a smaller whole-spectrum step than this is a hat
# point caps: sparks, trails and impacts take from the body retrace or the ring (open arc)
SPARK_PTS_MAX = 200
RING_YIELD = 120


class HybridContour(SparkContour):
    def __init__(self, seed=42):
        super().__init__(seed)
        self._prev_pts = None
        self._hit_size = None
        self._centroid = 0.25
        self._accent = 0.0
        self.last_trail = -1e9
        self.trail_angle = 0.0
        self.last_impact = -1e9
        self.impacts = 0
        self.sil_frames = 0
        self.silhouettes = 0
        self.beam_frames = 0
        self.act_slow = 0.0
        self.heat = 0.1
        self.section = 'intro'
        self.hat_demotions = 0

    # ----- frame entry -----
    def sample(self, level, bands, now, dt, *args, hit_flux=None, hit_rms=None, **kw):
        if hit_flux is None and hit_rms is None:
            self._hit_size = None
        else:
            self._hit_size = 0.6 * (hit_rms or 0.0) + 0.4 * (hit_flux or 0.0)
        self._centroid = kw.get('centroid', args[3] if len(args) > 3 else 0.25)
        self._accent = kw.get('accent') or 0.0
        self._prev_pts = None if self.previous is None else self.previous.copy()
        act = kw.get('activity')
        if act is None:
            act = max(level, max(bands))
        act = min(1.0, max(0.0, act))
        sdt = max(0.0, min(dt, 0.1))
        self.act_slow += (act - self.act_slow) * (1.0 - math.exp(-sdt / 5.0))
        target = 0.1 + 0.9 * self.act_slow
        if self._accent >= 1.0:
            self.heat = max(self.heat, min(1.0, 0.55 + 0.25 * self._accent))     # a drop: hot at once
        tau = HEAT_UP if target > self.heat else HEAT_DOWN
        self.heat += (target - self.heat) * (1.0 - math.exp(-sdt / tau))
        self.section = 'intro' if self.heat < 0.35 else 'build' if self.heat < 0.7 else 'drop'
        return super().sample(level, bands, now, dt, *args, **kw)

    # ----- 9: hit size gates the kind -----
    def _classify(self, energy, sub_flux, act):
        kind, jump = super()._classify(energy, sub_flux, act)
        if kind in ('kick', 'snare', 'stab') and self._hit_size is not None and self._hit_size < HAT_SIZE:
            kind = 'hat'
            self.hat_demotions += 1
        return kind, jump

    # ----- 3 -----
    def _pitch_rgb(self, kind):
        t = min(1.0, max(0.0, (self._centroid - 0.10) / 0.45))
        rgb = tuple(PITCH_LOW[i] * (1.0 - t) + PITCH_HIGH[i] * t for i in range(3))
        if kind == 'snare':
            rgb = tuple(0.5 * v + 127.5 for v in rgb)
        return tuple(int(min(255.0, v)) for v in rgb)

    # ----- 1, 3, 4, 6, 7: spawn -----
    def _spawn_sparks(self, pts, r_core, act):
        ring = self.rings.ring
        rw = None
        if ring is not None:
            r0 = 1.18 * r_core
            rw = r0 + (RING_R_MAX - r0) * ring['frac']
        dt = max(1e-3, self._dt)
        pv = self._prev_pts if self._prev_pts is not None and len(self._prev_pts) == len(pts) else None
        rnd = self.spark_random
        for kind, strength in self.hits:
            n, life, speed, streak, spread, grav, _rgb, bflash, frgb, shatter = KINDS[kind]
            size = 1.0 if self._hit_size is None else self._hit_size
            gain = min(1.0, max(0.55, 0.75 * strength)) * (0.6 + 0.4 * act)
            if bflash > 0.0:
                self.flash = max(self.flash, bflash * gain)
                self.flash_rgb = frgb
                self.flash_invert = kind == 'stab'
            if shatter > 0.0:
                self.shatter_until = self._now + shatter
                self.shatter_len = shatter
                self.shatter_rot = rnd.random()
            if kind == 'hat' and ring is not None:
                self.ring_flash = 1.0
            if kind == 'stab' and self._pitch is not None:
                centre = self._pitch
            else:
                centre = rnd.uniform(0.0, TWO_PI)
            rgb = self._pitch_rgb(kind)
            kc = int((centre % TWO_PI) / TWO_PI * (N_CORE - 1))
            for i in range(n):
                th = centre + (rnd.uniform(-spread, spread) if kind != 'snare'
                               else TWO_PI * (i + rnd.uniform(0.0, 0.6)) / n)
                bvx = bvy = 0.0
                if kind == 'hat' and rw is not None:
                    x, y = rw * math.cos(th), rw * math.sin(th)
                else:
                    k = int((th % TWO_PI) / TWO_PI * (N_CORE - 1))
                    x, y = float(pts[k, 0]), float(pts[k, 1])
                    th = math.atan2(y, x)
                    if pv is not None:
                        bvx = (x - float(pv[k, 0])) / dt
                        bvy = (y - float(pv[k, 1])) / dt
                        bv = math.hypot(bvx, bvy)
                        if bv > BODY_V_MAX:
                            bvx *= BODY_V_MAX / bv; bvy *= BODY_V_MAX / bv
                v = speed * rnd.uniform(0.7, 1.0) * gain
                tang = rnd.uniform(-0.6, 0.6) if kind == 'snare' else rnd.uniform(-0.2, 0.2)
                ox, oy = math.cos(th) - tang * math.sin(th), math.sin(th) + tang * math.cos(th)
                self.sparks.append({'x': x, 'y': y,
                                    'vx': INHERIT * bvx + OUTWARD * v * ox,
                                    'vy': INHERIT * bvy + OUTWARD * v * oy,
                                    'born': self._now, 'life': life * rnd.uniform(0.75, 1.0), 'kind': kind,
                                    'rgb': rgb, 'n': streak, 'grav': GRAVITY[kind],
                                    'curl': CURL[kind] * rnd.choice((-1.0, 1.0)) * rnd.uniform(0.6, 1.0)})
                self.spark_births[kind] += 1
            # 4: impact ring on a hard hit
            hard = strength >= IMPACT_STRENGTH or size >= IMPACT_SIZE
            if kind != 'hat' and hard and self._now - self.last_impact >= IMPACT_GAP:
                self.last_impact = self._now
                cx, cy = (0.0, 0.0) if kind == 'snare' else (float(pts[kc, 0]), float(pts[kc, 1]))
                self.sparks.append({'x': cx, 'y': cy, 'vx': 0.0, 'vy': 0.0, 'born': self._now,
                                    'life': IMPACT_LIFE, 'kind': 'impact', 'rgb': (255, 255, 255),
                                    'n': IMPACT_N, 'grav': 0.0, 'rot': rnd.uniform(0.0, TWO_PI)})
                self.impacts += 1
            # 7: silhouette flash on a big hit
            if kind != 'hat' and strength >= SIL_STRENGTH and size >= SIL_SIZE:
                self.sil_frames = SIL_FRAMES
                self.silhouettes += 1

    # ----- 6: curved paths -----
    def _advance_sparks(self):
        dt = self._dt
        alive = []
        cap = SPARK_STEP_MAX / max(1e-6, dt)
        for s in self.sparks:
            age = self._now - s['born']
            if age > s['life']:
                continue
            if s['kind'] in ('impact', 'trail'):
                alive.append(s)
                continue
            c = s.get('curl', 0.0) * dt
            if c:
                cc, sc = math.cos(c), math.sin(c)
                s['vx'], s['vy'] = s['vx'] * cc - s['vy'] * sc, s['vx'] * sc + s['vy'] * cc
            s['vy'] -= s['grav'] * dt
            v = math.hypot(s['vx'], s['vy'])
            if v > cap:
                s['vx'] *= cap / v; s['vy'] *= cap / v
            s['x'] += s['vx'] * dt; s['y'] += s['vy'] * dt
            if math.hypot(s['x'], s['y']) > SPARK_R_MAX:
                continue
            alive.append(s)
        self.sparks = alive

    # ----- 2: trails -----
    @staticmethod
    def _trail_n(r):
        chord = min(TRAIL_CHORD, math.sqrt(11.0 * r))     # chord^2 / r <= 11 (+rounding) under the 15 bound
        return max(6, int(math.ceil(TRAIL_SPAN * r / chord)) + 1)

    def _update_trails(self, coords, r_core, ring, fade):
        now = self._now
        if ring is None or fade < 0.03 or now - self.last_trail < TRAIL_EVERY:
            return
        self.last_trail = now
        r0 = 1.18 * r_core
        rw = r0 + (RING_R_MAX - r0) * ring['frac']
        seam = coords[N_CORE - 1]
        th0 = math.atan2(seam[1], seam[0]) + math.pi - 0.5 * TRAIL_SPAN   # opposite the transit
        hue = self.ring_hue
        rgb = tuple(int(255 * max(0.0, math.sin(hue + off)) ** 1.2) for off in (0.0, 2.094, 4.188))
        m = max(rgb) or 1
        rgb = tuple(int(255 * v / m) for v in rgb)
        trails = [s for s in self.sparks if s['kind'] == 'trail']
        if len(trails) >= TRAIL_MAX:
            self.sparks.remove(min(trails, key=lambda s: s['born']))
        n = self._trail_n(rw)
        mid = th0 + 0.5 * TRAIL_SPAN
        self.sparks.append({'x': rw * math.cos(mid), 'y': rw * math.sin(mid), 'vx': 0.0, 'vy': 0.0,
                            'born': now, 'life': TRAIL_LIFE, 'kind': 'trail', 'rgb': rgb, 'n': n,
                            'grav': 0.0, 'r': rw, 'th0': th0, 'span': TRAIL_SPAN})

    # ----- block rendering for the three block kinds -----
    def _spark_block(self, s):
        kind = s['kind']
        if kind not in ('impact', 'trail'):
            return super()._spark_block(s)
        age = max(0.0, self._now - s['born'])
        prog = min(1.0, age / s['life'])
        n = s['n']
        if kind == 'impact':
            r = IMPACT_R0 + (IMPACT_R1 - IMPACT_R0) * prog
            dth = TWO_PI / n
            ang = s['rot'] + np.arange(-1, n + 1) * dth
            lum = max(0.55, (1.0 - prog) ** 0.5)
            col = tuple(int(255 * lum) for _ in range(3))
        else:
            r = s['r']
            dth = s['span'] / (n - 1)
            ang = s['th0'] + np.arange(-1, n + 1) * dth
            lum = 0.10 + 0.32 * (1.0 - prog)
            col = tuple(int(c * lum) for c in s['rgb'])
        xs = np.rint(s['x'] + r * np.cos(ang)).astype(np.int64) if kind == 'impact' else np.rint(r * np.cos(ang)).astype(np.int64)
        ys = np.rint(s['y'] + r * np.sin(ang)).astype(np.int64) if kind == 'impact' else np.rint(r * np.sin(ang)).astype(np.int64)
        line = list(zip(xs.tolist(), ys.tolist()))
        return line[1:-1], [col] * n, line[0], line[-1]

    def _append_sparks(self, coords, colors, blanks, start, budget):
        # impacts first (they are the point of the frame), streaks, then trails: the parent walks
        # them by angle and drops what no longer fits, so the cheap history goes before the hit
        prio = {'impact': 0, 'trail': 2}
        saved = self.sparks
        self.sparks = sorted(saved, key=lambda s: (prio.get(s['kind'], 1), math.atan2(s['y'], s['x'])))
        try:
            return self._walk_sparks(coords, colors, blanks, start, budget)
        finally:
            self.sparks = saved

    def _walk_sparks(self, coords, colors, blanks, start, budget):
        from liquid import _blank_move
        home = coords[0]
        blocks, cur, used = [], start, 0
        for s in self.sparks:                    # already in priority order
            pts, cols, a, b = self._spark_block(s)
            move = _blank_move(cur, a)
            back = _blank_move(b, home)
            need = (DWELL_OFF + 1) + len(move) + (DWELL_ON + 1) + len(pts)
            if used + need + (DWELL_OFF + 1) + len(back) + (DWELL_ON + 1) > budget:
                continue
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
        tail += [home] * (budget - used - len(tail))
        coords += tail; colors += [(0, 0, 0)] * len(tail); blanks += [True] * len(tail)
        return len(blocks)

    # ----- 5: body retrace, plain or beam-doubled -----
    def _retrace(self, coords, colors, blanks, m, beam, full):
        """Retrace body points 1..m (lit).  Plain: identical to the body, as the base pattern.
        Beam: translated by BEAM_OFF and BEAM_GAIN brighter, with the seam copy settled first so
        the lit run's blank neighbours lie on the shifted curve.  Returns the point the frame
        should dwell on next (the next body point, shifted or not)."""
        if not beam:
            coords.extend(coords[1:1 + m]); colors.extend(colors[1:1 + m]); blanks.extend(blanks[1:1 + m])
            if full:
                coords.append(coords[N_CORE - 1]); colors.append((0, 0, 0)); blanks.append(True)
            return coords[m + 1]
        ph = self.spin + 0.5 * math.pi
        dx, dy = int(round(BEAM_OFF * math.cos(ph))), int(round(BEAM_OFF * math.sin(ph)))
        seam = (coords[0][0] + dx, coords[0][1] + dy)
        coords.extend([seam] * (DWELL_ON + 1)); colors.extend([(0, 0, 0)] * (DWELL_ON + 1)); blanks.extend([True] * (DWELL_ON + 1))
        body = np.asarray(coords[1:1 + m], dtype=np.int64) + np.array((dx, dy))
        col = np.minimum(255, (np.asarray(colors[1:1 + m], dtype=np.float64) * BEAM_GAIN)).astype(np.int64)
        coords.extend(tuple(p) for p in body.tolist())
        colors.extend(tuple(c) for c in col.tolist())
        blanks.extend(blanks[1:1 + m])
        if full:
            coords.append(seam); colors.append((0, 0, 0)); blanks.append(True)
        self.beam_frames += 1
        nxt = coords[m + 1]
        return (nxt[0] + dx, nxt[1] + dy)

    # ----- frame assembly -----
    def _append_ring(self, coords, colors, blanks, pts, r_core, bass, mid, act):
        now, dt = self._now, self._dt
        self._spawn_sparks(pts, r_core, act)
        self._advance_sparks()
        ring = self.rings.ring
        fade = 0.0
        if ring is not None:
            frac = ring['frac']
            fade = max(0.0, 1.0 - frac ** 1.8) * (0.35 + 0.65 * act)
            fade *= 0.6 + 0.4 * ring.get('weight', 1.0)
        ring_visible = ring is not None and fade >= 0.03
        self._update_trails(coords, r_core, ring, fade)
        # 7: silhouette -- the outline, white, twice, nothing else
        if self.sil_frames > 0:
            self.sil_frames -= 1
            for i in range(1, N_CORE - 1):
                if not blanks[i]:
                    colors[i] = (255, 255, 255)
            coords.extend(coords[1:N_CORE]); colors.extend(colors[1:N_CORE]); blanks.extend(blanks[1:N_CORE])
            while len(coords) < N_FRAME:
                coords.append(coords[0]); colors.append((0, 0, 0)); blanks.append(True)
            self.flash *= math.exp(-dt / 0.045)
            self.ring_flash *= math.exp(-dt / 0.03)
            self.spark_pts = 0
            return
        # body flash / shatter (as liquid3)
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
            prog = 1.0 - (self.shatter_until - now) / self.shatter_len
            duty = 0.55 + 0.45 * prog * prog
            idx = np.arange(4, N_CORE - 5)
            u = (idx / (N_CORE - 1) * SHATTER_FRAGS + self.shatter_rot) % 1.0
            for i in idx[u >= duty].tolist():
                blanks[i] = True; colors[i] = (0, 0, 0)
        beam = self._accent >= BEAM_ACCENT or self.flash > 0.3
        want = 0
        for s in self.sparks:
            want += (DWELL_OFF + 1) + 5 + (DWELL_ON + 1) + s['n']
        if want == 0:
            self.spark_pts = 0
            if ring_visible:
                DynamicContour._append_ring(self, coords, colors, blanks, pts, r_core, bass, mid, act)
                if self.ring_flash > 0.01:
                    self._flash_ring(colors, blanks)
            else:
                m = N_CORE - 2
                self._retrace(coords, colors, blanks, m, beam, True)
                while len(coords) < N_FRAME:
                    coords.append(coords[0]); colors.append((0, 0, 0)); blanks.append(True)
        else:
            if ring_visible:
                k = min(want + 24, RING_YIELD)
                DynamicContour._append_ring(self, coords, colors, blanks, pts, r_core, bass, mid, act, budget=N_FRAME - k)
                if self.ring_flash > 0.01:
                    self._flash_ring(colors, blanks)
                start = coords[-1]
            else:
                k = min(want + 24, SPARK_PTS_MAX)
                extra = (DWELL_ON + 1) if beam else 0
                m = N_FRAME - k - N_CORE - extra
                while m > 1 and blanks[m]:
                    m -= 1
                k = N_FRAME - N_CORE - m - extra
                start = self._retrace(coords, colors, blanks, m, beam, False)
            self.spark_pts = k
            self._append_sparks(coords, colors, blanks, start, k)
        self.ring_flash *= math.exp(-dt / 0.03)
        # 8: section palette tint on everything lit
        h = self.heat
        tint = np.array([COLD[i] * (1.0 - h) + HOT[i] * h for i in range(3)])
        arr = np.asarray(colors, dtype=np.float64)
        arr = np.minimum(255.0, arr * tint)
        arr[np.asarray(blanks, dtype=bool)] = 0.0
        colors[:] = [tuple(c) for c in arr.astype(np.int64).tolist()]
