"""`--pattern liquid2`: the liquid body changes *form* on the beat.

DynamicContour keeps everything LiquidContour does (bands, twist spring, curl noise, 4D
fold, BPM-locked earned rings, galvo guard) and replaces two hooks:

* `_figure`: the body is a closed unit-scale figure that morphs between shapes on the
  beat -- sphere, torus band, figure-eight, clover (rose) and trefoil.  The morph is a
  smoothstep convex blend of two fixed point sets over the same parameter, so every
  intermediate frame's segment and second difference are bounded by the two endpoints
  (both are linear in the points): the morph itself can never pop the galvos.  Each
  figure is pre-scaled so its own curvature stays inside the body budget; the c8fd019
  guard in `sample` is still the last line of defence after rounding.
  Cadence follows the song: every beat when the music is energetic or accented, every
  second beat in a quieter passage, every fourth when soft, never in silence (the body
  drifts home to the sphere).  Accents (a drop, a section entry) pick from the wild set.
* `_append_ring`: the ring style depends on the event that earned it: an accent beat
  ring bursts as a rotating dashed ring, a floor beat ring is a soft scalloped ring,
  an unlocked kick onset is a spiral, a snare onset a counter-rotating dash ring, and
  an idle ring a plain dim ghost.  Frame size stays N_FRAME; the transits are the same
  dwell / eased blank move / settle as the base pattern.
"""
import math
import numpy as np
from liquid import (LiquidContour, _A, _COS_A, _SIN_A, _blank_move, N_CORE, N_RING, N_FRAME,
                    DWELL_OFF, DWELL_ON, RING_R_MAX, RING_R0, IMPRINT_MODES, IMPRINT_BUDGET, TWO_PI)

SPIN_MAX = 1.3          # rad/s rigid spin of the figure (rotation preserves segment and curvature)
MORPH_MIN, MORPH_MAX = 0.18, 0.34   # morph duration bounds, s (0.55 x beat period in between)
GOLDEN = 2.39996
SCALLOP_M, SCALLOP_A = 5, 0.025     # floor-ring ripple: mode 5 at 2.5% adds ~3.9 units of ring curvature at r 6600


def _unit_figures():
    """Closed figures over _A, normalised to max radius 1, then sized by their curvature
    load L (max second difference relative to the unit circle's): size = min(1, 4/(2.5+L)).
    The twist body at saturated input spends the whole CURV_MAX budget on the sphere
    (~3 parts wave to 1 part circle), so a figure with load L is drawn at that size to
    roughly keep r*(3*size + L*size) within the same budget.  The constant was 3.0; it was
    loosened to 2.5 (torus/eight/clover/limacon 0.79/0.78/0.77/0.73 -> 0.87/0.87/0.86/0.80)
    when the body grew, because `sample` now pre-limits the drawn radius to the curvature
    budget continuously, so the figures no longer have to carry the whole safety margin.
    A (2,3) torus-knot trefoil was tried and dropped: two windings on 220 points give it
    load 7 (40% size) and double the chord length.  Order: sphere, torus, eight, clover, limacon."""
    a = _A
    figs = {
        'sphere': (_COS_A, _SIN_A, np.zeros(N_CORE)),
        'torus': ((0.88 + 0.12 * np.cos(3 * a)) * _COS_A, (0.88 + 0.12 * np.cos(3 * a)) * _SIN_A, 0.16 * np.sin(3 * a)),
        'eight': (_COS_A, 0.5 * np.sin(2 * a), 0.30 * _SIN_A),
        'clover': ((0.70 + 0.30 * np.cos(2 * a)) * _COS_A, (0.70 + 0.30 * np.cos(2 * a)) * _SIN_A, 0.25 * np.sin(2 * a)),
        'limacon': ((0.5 + np.cos(a)) * _COS_A - 0.5, (0.5 + np.cos(a)) * _SIN_A, 0.25 * np.sin(2 * a)),
    }
    kappa_circle = float(np.max(np.hypot(_COS_A[2:] - 2 * _COS_A[1:-1] + _COS_A[:-2], _SIN_A[2:] - 2 * _SIN_A[1:-1] + _SIN_A[:-2])))
    out, names, sizes = [], [], []
    for name, (x, y, z) in figs.items():
        p = np.stack((x, y, z), axis=1).copy()
        p[-1] = p[0]
        p /= float(np.max(np.hypot(p[:, 0], p[:, 1])))
        load = float(np.max(np.hypot(*(p[2:, :2] - 2 * p[1:-1, :2] + p[:-2, :2]).T))) / kappa_circle
        size = min(1.0, 4.0 / (2.5 + load))
        out.append(p * size); names.append(name); sizes.append(size)
    return names, out, sizes


class DynamicContour(LiquidContour):
    NAMES, FIGURES, SIZES = _unit_figures()
    CALM = ('sphere', 'torus')              # quiet passages
    WILD = ('eight', 'clover', 'limacon')   # accents: a drop, a section entry

    def __init__(self, seed=42):
        super().__init__(seed)
        self.shape_cur = 0                  # index into FIGURES (sphere)
        self.shape_nxt = 0
        self.fig_from = self.FIGURES[0].copy()   # frozen start of the current morph
        self.morph = 1.0                    # 0 -> 1 blend from fig_from to FIGURES[shape_nxt]
        self.morph_rate = 1.0 / MORPH_MAX
        self.spin = 0.0
        self.beat_count = 0
        self.morphs = 0                     # morphs started (tests / diagnostics)
        self.last_morph = -1e9
        self.ring_style = None
        self.shape_hist = []

    # ----- body -----
    def _current_figure(self):
        s = self.morph * self.morph * (3.0 - 2.0 * self.morph)
        return (1.0 - s) * self.fig_from + s * self.FIGURES[self.shape_nxt]

    def _start_morph(self, target, now):
        self.fig_from = self._current_figure()
        self.shape_cur = self.shape_nxt
        self.shape_nxt = target
        self.morph = 0.0
        self.morphs += 1
        self.last_morph = now
        self.color_phase = (self.color_phase + GOLDEN) % TWO_PI    # each form gets its own palette
        self.shape_hist = (self.shape_hist + [self.NAMES[target]])[-8:]

    def _pick(self, act, accent):
        if accent >= 0.8:
            pool = self.WILD
        elif act < 0.25:
            pool = self.CALM
        else:
            pool = self.NAMES
        cands = [self.NAMES.index(n) for n in pool if self.NAMES.index(n) != self.shape_nxt]
        return self.random.choice(cands)

    def _figure(self, now, step_dt, act, accent, beat_period, on_beat):
        acc = 0.0 if accent is None else accent
        rings = self.rings
        if not rings.signal:
            # silence: drift home to the sphere, nothing else changes form
            if self.shape_nxt != 0 and self.morph >= 1.0 and now - self.last_morph > 1.0:
                self._start_morph(0, now)
        else:
            if act >= 0.4 or acc >= 0.5:
                every = 1
            elif act >= 0.18:
                every = 2
            else:
                every = 4
            beat = on_beat or rings.beat_hit
            if beat:
                self.beat_count += 1
            trigger = (beat and self.beat_count % every == 0) if rings.locked else \
                      (self.last_onset == now and now - self.last_morph > 0.35)
            if trigger and (self.morph >= 0.6 or acc >= 0.8):
                self._start_morph(self._pick(act, acc), now)
        self.morph_rate = 1.0 / min(MORPH_MAX, max(MORPH_MIN, 0.55 * beat_period))
        self.morph = min(1.0, self.morph + step_dt * self.morph_rate)
        self.spin = (self.spin + self.direction * SPIN_MAX * (0.15 + 0.85 * act) * step_dt) % TWO_PI
        f = self._current_figure()
        c, s = math.cos(self.spin), math.sin(self.spin)
        return f[:, 0] * c - f[:, 1] * s, f[:, 0] * s + f[:, 1] * c, f[:, 2]

    # ----- ring -----
    @staticmethod
    def _style(ring):
        kind, weight = ring['kind'], ring.get('weight', 1.0)
        if kind == 'beat':
            return 'burst' if weight >= 0.85 else 'scallop'
        if kind == 'onset':
            return 'spiral' if ring['color'] == (110, 220, 255) else 'dash'
        return 'ghost'

    def _append_ring(self, coords, colors, blanks, pts, r_core, bass, mid, act, budget=N_FRAME):
        """Ring or body retrace so the frame is exactly `budget` points (N_FRAME by default).
        A caller that needs points for something else (liquid3's sparks) passes a smaller
        budget: the ring then yields them as an *open arc* drawn at the same chord spacing a
        full ring would have, so the missing sector costs no segment length or curvature,
        and the walk-back starts from the arc's end."""
        ring = self.rings.ring
        fade = 0.0
        if ring is not None:
            frac = ring['frac']
            fade = max(0.0, 1.0 - frac ** 1.8) * (0.35 + 0.65 * act)
            fade *= 0.6 + 0.4 * ring.get('weight', 1.0)
        if ring is None or fade < 0.03:
            coords.extend(coords[1:N_CORE]); colors.extend(colors[1:N_CORE]); blanks.extend(blanks[1:N_CORE])
            while len(coords) < budget:
                coords.append(coords[0]); colors.append((0, 0, 0)); blanks.append(True)
            return
        r0 = RING_R0 * r_core
        rw = r0 + (RING_R_MAX - r0) * frac
        if ring is not self.ring_obj:
            self.ring_obj = ring
            self.ring_style = self._style(ring)
            self.ring_hue = (self.ring_hue + GOLDEN) % TWO_PI
            if self.ring_style == 'burst':
                prof = np.hypot(pts[:-1, 0], pts[:-1, 1])
                spec = np.fft.rfft(prof / max(1e-6, float(np.mean(prof))) - 1.0) / len(prof)
                imp = 2.0 * spec[1:IMPRINT_MODES + 1]
                load = float(np.sum(np.abs(imp) * np.arange(1, IMPRINT_MODES + 1) ** 2))
                self.imprint = imp * min(1.0, IMPRINT_BUDGET / max(1e-9, load))
            else:
                self.imprint = np.zeros(IMPRINT_MODES, dtype=np.complex128)
        style = self.ring_style
        modes = np.arange(1, IMPRINT_MODES + 1)
        shell = (1.0 - frac) ** 1.3
        ghost = style == 'ghost'

        def ring_radius(ang):
            w_ring = (0.12 * bass * np.sin(2 * ang - self.phase) + 0.07 * mid * np.sin(3 * ang + self.phase)
                      + 0.06 * self.twist * np.sin(2 * ang + 2 * self.phase))
            r = np.ones_like(ang) + (0.0 if ghost else 0.05 * np.tanh(w_ring) * min(1.0, 2.0 * fade))
            if style == 'burst':
                r = r + shell * np.real(self.imprint @ np.exp(1j * np.outer(modes, ang)))
            elif style == 'scallop':
                r = r + SCALLOP_A * shell * np.cos(SCALLOP_M * ang)   # static scallop
            elif style == 'spiral':
                r = r * (1.0 - 0.16 * (1.0 - ang / TWO_PI))      # free start inside, lands on the full radius
            return rw * r
        # The ring starts at the body seam's own angle (the figure spins, the seam with it),
        # so the transit out and the walk-back stay short and radial as in the base pattern.
        seam = coords[-1]
        th0 = math.atan2(seam[1], seam[0])
        c0, s0 = math.cos(th0), math.sin(th0)
        land = tuple(int(v) for v in np.rint(ring_radius(np.zeros(1))[0] * np.array((c0, s0))))
        last = tuple(int(v) for v in np.rint(ring_radius(np.full(1, TWO_PI))[0] * np.array((c0, s0))))
        out = _blank_move(seam, land)
        back = _blank_move(last, coords[0])
        n_ring = min(N_RING, N_FRAME - N_CORE - 2 * (DWELL_OFF + DWELL_ON) - len(out) - len(back))
        arc = TWO_PI
        if budget < N_FRAME:
            # open arc: keep the full ring's chord (same angular step), drop the sector at the
            # end; the longer walk-back from the arc's end is paid out of the same budget
            n_full = n_ring
            n_ring = min(N_RING, budget - N_CORE - 2 * (DWELL_OFF + DWELL_ON) - len(out) - len(back))
            for _ in range(8):      # a shorter arc ends farther from the seam: shrink until it fits
                arc = TWO_PI * (n_ring - 1) / (n_full - 1)
                last = tuple(int(v) for v in np.rint(ring_radius(np.full(1, arc))[0]
                                                     * np.array((math.cos(th0 + arc), math.sin(th0 + arc)))))
                back = _blank_move(last, coords[0])
                n_ok = min(N_RING, budget - N_CORE - 2 * (DWELL_OFF + DWELL_ON) - len(out) - len(back))
                if n_ok >= n_ring:
                    break
                n_ring = n_ok
        ring_a = np.linspace(0.0, arc, n_ring)
        r_ring = ring_radius(ring_a)
        rx = np.rint(r_ring * np.cos(ring_a + th0)).astype(np.int64)
        ry = np.rint(r_ring * np.sin(ring_a + th0)).astype(np.int64)
        rpts = list(zip(rx.tolist(), ry.tolist()))
        rpts[0], rpts[-1] = land, last
        dark = [seam] * DWELL_OFF + out + [rpts[0]] * DWELL_ON
        # colour per style
        white = math.exp(-frac / 0.06)
        if style == 'burst':
            hue = self.ring_hue + 1.1 * np.cos(ring_a - 1.5 * self.phase)
            white = math.exp(-frac / 0.09)
        elif style == 'scallop':
            hue = self.ring_hue + 0.5 * np.cos(SCALLOP_M * ring_a)   # static hue
        elif style == 'spiral':
            hue = self.ring_hue + 2.0 * ring_a / TWO_PI               # hue travels along the spiral
        elif style == 'dash':
            hue = self.ring_hue + 0.8 * np.sin(2 * ring_a + self.phase)
        else:
            hue = np.full(n_ring, self.ring_hue)
        rgb = np.stack([np.maximum(0.0, np.sin(hue + off)) ** 1.2 for off in (0.0, 2.094, 4.188)], axis=1)
        rgb = rgb / np.maximum(1e-6, rgb.max(axis=1, keepdims=True))
        bright = fade * (1.0 + 0.25 * white) * (0.55 if ghost else 1.0)
        if style == 'spiral':
            bright = bright * (0.45 + 0.55 * ring_a / TWO_PI)[:, None]   # comet: dim tail, bright head
        ring_rgb = np.clip(255.0 * (rgb * (1.0 - white) + white) * bright, 0, 255).astype(np.int64)
        ring_blank = np.zeros(n_ring, dtype=bool)
        if style in ('burst', 'dash'):
            # dashes: blank points in place (no coordinate jump, so no dwell is needed); the
            # pattern rotates with the body phase, against it for a snare dash ring
            n_dash, duty = (8, 0.78) if style == 'burst' else (6, 0.66)
            rot = 0.0      # static: the ring must not turn with the body
            u = (ring_a * n_dash / TWO_PI + rot) % 1.0
            ring_blank[1:-1] = (u[1:-1] >= duty) & (white < 0.5)   # born solid, breaks into dashes
            # the ring's first and last few points stay lit, so the dashes never merge with the
            # transit's blank run (whose dwell and settle points must stay identical)
            ring_blank[1:1 + DWELL_ON + 1] = False
            ring_blank[-(DWELL_OFF + 3):-1] = False
        ring_blank[0] = ring_blank[-1] = True
        ring_rgb[ring_blank] = 0
        ring_cols = [tuple(c) for c in ring_rgb.tolist()]
        tail = [rpts[-1]] * DWELL_OFF + back
        pad = budget - len(coords) - len(dark) - n_ring - len(tail)
        tail += [coords[0]] * pad
        coords += dark + rpts + tail
        colors += [(0, 0, 0)] * len(dark) + ring_cols + [(0, 0, 0)] * len(tail)
        blanks += [True] * len(dark) + ring_blank.tolist() + [True] * len(tail)
