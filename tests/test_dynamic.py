import itertools
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heliosctl as h
from liquid import Bands, LiquidContour
from dynamic import DynamicContour

DT = .0225   # heliosctl frame interval: 450 pts @ 20 kpps


def lit_runs_metrics(coords, blanks):
    """Max lit segment and second difference over every lit run plus its blank seam
    neighbours (the extremes-test rule), max blank step, lit count."""
    n = len(coords); seg = curv = blank = 0.
    j = 0
    while j < n:
        if blanks[j]:
            j += 1
            continue
        k = j
        while k < n and not blanks[k]:
            k += 1
        run = coords[max(0, j - 1):min(n, k + 1)]
        for p, q in zip(run, run[1:]):
            seg = max(seg, math.dist(p, q))
        for r, p, q in zip(run, run[1:], run[2:]):
            curv = max(curv, math.hypot(q[0] - 2 * p[0] + r[0], q[1] - 2 * p[1] + r[1]))
        j = k
    for i in range(n):
        if blanks[i] and blanks[i - 1]:
            blank = max(blank, math.dist(coords[i - 1], coords[i]))
    return seg, curv, blank, sum(1 for b in blanks if not b)


def assert_blank_transits_settled(tc, coords, blanks, colors, lit_step=225, blank_step=800, off=3, on=4):
    """FrameBlankingTests rules: no lit point ends or starts a jump, blank steps eased,
    every blank run containing a jump dwells before and settles after it."""
    n = len(coords)
    steps = [math.dist(coords[i - 1], coords[i]) for i in range(n)]
    for i in range(n):
        if not blanks[i]:
            tc.assertLessEqual(steps[i], lit_step)
            tc.assertLessEqual(steps[(i + 1) % n], lit_step)
        else:
            tc.assertEqual(tuple(colors[i]), (0, 0, 0))
        tc.assertLessEqual(steps[i], blank_step)
    i = 0
    while i < n:
        if blanks[i] and not blanks[i - 1]:
            j = i
            while blanks[j % n]:
                j += 1
            run = [coords[k % n] for k in range(i, j)]
            jump = max(math.dist(p, q) for p, q in zip(run, run[1:])) if len(run) > 1 else 0.
            if jump > lit_step:
                tc.assertGreaterEqual(len(run), off + on)
                tc.assertEqual(len(set(run[:off])), 1)
                tc.assertEqual(len(set(run[-on:])), 1)
        i += 1


def locked_frames(c, period, bands, act, accent, seconds, level=None, lock=.9, flux=0., start=0.):
    ph, now = 0., start
    for _ in range(int(seconds / DT)):
        ph = (ph + 2 * math.pi * DT / period) % (2 * math.pi)
        yield now, c.sample(level if level is not None else (1. if act >= 1 else .3), bands, now, DT,
                            .1 if act >= 1 else .07, 8, beat_phase=ph, beat_lock=lock, activity=act,
                            beat_period=period, flux=flux, accent=accent)
        now += DT


class PatternChoiceTests(unittest.TestCase):
    def test_liquid2_is_a_choice_and_liquid_is_unchanged(self):
        import io, contextlib
        # the real CLI parser: audio --pattern liquid2 must parse (duration 0 then fails as a CliError, not exit 2)
        with self.assertRaises(h.CliError), contextlib.redirect_stderr(io.StringIO()):
            h.main(['audio', '--pattern', 'liquid2', '--duration', '0'])   # parses, then duration 0 is a CliError
        src = type('S', (), {'read': lambda self: .1, 'samples': [0] * 320, 'sequence': 0})()
        self.assertIsInstance(h.AudioFrames(src, pattern='liquid2').liquid, DynamicContour)
        self.assertIs(type(h.AudioFrames(src, pattern='liquid').liquid), LiquidContour)


class ShapeMorphTests(unittest.TestCase):
    def test_every_shape_and_every_morph_stays_inside_galvo_bounds(self):
        """All 20 ordered shape pairs under saturated bands, locked at 0.5 s with a burst
        ring on every beat; every frame (intermediate morph frames included) must be 450
        points, lit segment < 225 and second difference < 15 on every lit run, blank steps
        eased, and the body must not jump between frames."""
        c = DynamicContour()
        pairs = list(itertools.permutations(range(len(c.NAMES)), 2))
        seen_shapes, mid_frames, worst_curv, worst_seg = set(), 0, 0., 0.
        prev = None; ph = now = 0.; k = 0
        for i in range(int(0.5 * len(pairs) / DT) + 40):
            ph = (ph + 2 * math.pi * DT / .5) % (2 * math.pi)
            if i % 22 == 0 and i > 0 and k < len(pairs):
                a, b = pairs[k]; k += 1
                c.shape_nxt = a; c.fig_from = c.FIGURES[a].copy(); c.morph = 1.0
                c._start_morph(b, now)
            coords, colors, blanks = c.sample(1., [1., 1., 1.], now, DT, .1, 8, beat_phase=ph, beat_lock=.9,
                                              activity=1., beat_period=.5, accent=1.5)
            now += DT
            self.assertEqual(len(coords), 450)
            for x, y in coords:
                self.assertLessEqual(abs(x), 16000); self.assertLessEqual(abs(y), 13000)
            seg, curv, blank, lit = lit_runs_metrics(coords, blanks)
            worst_seg, worst_curv = max(worst_seg, seg), max(worst_curv, curv)
            self.assertLess(seg, 225, f'{c.NAMES[c.shape_cur]}->{c.NAMES[c.shape_nxt]} morph {c.morph:.2f}')
            self.assertLess(curv, 15, f'{c.NAMES[c.shape_cur]}->{c.NAMES[c.shape_nxt]} morph {c.morph:.2f}')
            self.assertLessEqual(blank, 800)
            self.assertGreaterEqual(lit, 360)
            if prev is not None:
                move = max(math.dist(p, q) for p, q in zip(coords[:220], prev[:220]))
                self.assertLessEqual(move, 560, 'body popped between frames')
            prev = coords
            seen_shapes.add(c.shape_nxt)
            mid_frames += 0.05 < c.morph < 0.95
        self.assertEqual(k, len(pairs))
        self.assertEqual(seen_shapes, set(range(len(c.NAMES))))
        self.assertGreater(mid_frames, 100)       # the sweep really measured intermediate frames
        self.assertLessEqual(c.guard_hits, 5)      # the figures are sized for the budget; the guard is a backstop
        print('liquid2 forced morph sweep max segment/curvature ILDA:', worst_seg, worst_curv)

    def test_morphs_follow_the_beat_and_the_structure(self):
        # energetic, locked: one morph per beat; each morph starts on a beat frame
        c = DynamicContour(); starts = []; beats = []
        for now, _ in locked_frames(c, .5, [1., 1., 1.], 1., 0., 10):
            if c.rings.beat_hit: beats.append(now)
            if c.last_morph == now: starts.append(now)
        self.assertGreaterEqual(len(beats), 18)
        self.assertGreaterEqual(len(starts), 0.9 * len(beats))
        for s in starts[1:]:
            self.assertLessEqual(min(abs(s - b) for b in beats), DT / 2)
        # quieter: every second beat
        c2 = DynamicContour(); n2 = 0; b2 = 0
        for now, _ in locked_frames(c2, .5, [.3, .2, .2], .22, 0., 10):
            b2 += c2.rings.beat_hit; n2 += c2.last_morph == now
        self.assertLess(n2, 0.65 * b2); self.assertGreater(n2, 0.35 * b2)
        self.assertTrue(set(c2.shape_hist) <= {'sphere', 'torus'}, c2.shape_hist)   # calm set
        # accent: wild set only
        c3 = DynamicContour()
        for _ in locked_frames(c3, .5, [1., 1., 1.], 1., 1.5, 6): pass
        self.assertTrue(set(c3.shape_hist) <= {'eight', 'clover', 'limacon'}, c3.shape_hist)
        # silence: no morphs, and the body goes home to the sphere
        for _ in locked_frames(c3, .5, [0., 0., 0.], 0., 0., 3, level=0., start=6.): pass
        m = c3.morphs
        for _ in locked_frames(c3, .5, [0., 0., 0.], 0., 0., 5, level=0., start=9.): pass
        self.assertEqual(c3.morphs, m)
        self.assertEqual(c3.NAMES[c3.shape_nxt], 'sphere'); self.assertEqual(c3.morph, 1.0)

    def test_sphere_at_rest_matches_the_liquid_body_within_rounding(self):
        """Before the first morph the variant IS the liquid body (same twist sphere)."""
        a, b = LiquidContour(), DynamicContour()
        for i in range(40):   # quiet enough for no body onset (those morph when unlocked)
            fa = a.sample(.5, [.2, .15, .1], i * DT, DT, .07, 8, activity=.2)
            fb = b.sample(.5, [.2, .15, .1], i * DT, DT, .07, 8, activity=.2)
        self.assertEqual(b.morphs, 0)
        # the variant spins its figure rigidly; undo the spin and the two bodies agree to a few
        # units (integer rounding, and the tilt is applied about world axes, not body axes)
        cs, sn = math.cos(-b.spin), math.sin(-b.spin)
        d = max(math.dist(p, (q[0] * cs - q[1] * sn, q[0] * sn + q[1] * cs)) for p, q in zip(fa[0][:220], fb[0][:220]))
        self.assertLessEqual(d, 8.0)


class RingStyleTests(unittest.TestCase):
    def styles_seen(self, frames, c):
        seen = {}
        for now, (coords, colors, blanks) in frames:
            assert_blank_transits_settled(self, coords, blanks, colors)
            self.assertEqual(len(coords), 450)
            seg, curv, blank, lit = lit_runs_metrics(coords, blanks)
            self.assertLess(seg, 225); self.assertLess(curv, 15)
            if c.rings.ring is not None and c.ring_style is not None and c.ring_obj is c.rings.ring:
                ring_lit = sum(1 for b in blanks[220:] if not b)
                st = seen.setdefault(c.ring_style, {'frames': 0, 'min_lit': 999, 'frac': []})
                st['frames'] += 1; st['min_lit'] = min(st['min_lit'], ring_lit); st['frac'].append(c.rings.ring['frac'])
        return seen

    def test_accent_beat_ring_bursts_into_dashes_floor_ring_scallops(self):
        c = DynamicContour()
        seen = self.styles_seen(locked_frames(c, .5, [1., 1., 1.], 1., 1.5, 8), c)
        self.assertIn('burst', seen); self.assertNotIn('scallop', seen)
        self.assertLess(seen['burst']['min_lit'], 170)          # dashes: ~22% of the ring is dark
        c = DynamicContour()
        seen = self.styles_seen(locked_frames(c, .5, [1., 1., 1.], 1., 0., 12), c)
        self.assertIn('scallop', seen); self.assertNotIn('burst', seen)
        self.assertGreaterEqual(seen['scallop']['min_lit'], 195)  # solid ring

    def test_unlocked_kick_spirals_and_snare_dashes(self):
        c = DynamicContour(); spiral = dash = 0
        for now, (coords, colors, blanks) in locked_frames(c, .6, [.9, .5, .2], .8, 0., 8, lock=0., flux=.5):
            assert_blank_transits_settled(self, coords, blanks, colors)
            if c.rings.ring is not None and c.ring_obj is c.rings.ring and 0.3 < c.rings.ring['frac'] < 0.9:
                lit = [i for i in range(220, 450) if not blanks[i]]
                if c.ring_style == 'spiral':
                    spiral += 1
                    r_head, r_tail = math.hypot(*coords[lit[-1]]), math.hypot(*coords[lit[0]])
                    self.assertGreater(r_head, 1.1 * r_tail)        # open spiral: the head is outside the tail
                if c.ring_style == 'dash':
                    dash += 1
        self.assertGreater(spiral, 10)
        # snare-coloured onset rings dash; drive the ring colour path directly
        c = DynamicContour(); c.rings.update(0., DT, 0., .6, 0., .8, True, (255, 185, 80), accent=None)
        self.assertEqual(c._style(c.rings.ring), 'dash')
        self.assertEqual(c._style({'kind': 'idle', 'color': (110, 220, 255), 'weight': 1.}), 'ghost')

    def test_idle_ghost_rings_render_and_are_blank_safe(self):
        # unlocked, quiet, no onsets: BeatRings' idle path births ghost rings; every frame drawn
        c = DynamicContour(); ghost_frames = 0
        for now, (coords, colors, blanks) in locked_frames(c, .6, [.15, .1, .05], .3, 0., 8, lock=0.):
            self.assertEqual(len(coords), 450)
            assert_blank_transits_settled(self, coords, blanks, colors)
            seg, curv, blank, lit = lit_runs_metrics(coords, blanks)
            self.assertLess(seg, 225); self.assertLess(curv, 15)
            if c.rings.ring is not None and c.ring_obj is c.rings.ring and c.ring_style == 'ghost':
                ghost_frames += 1
        self.assertGreater(c.rings.spawned['idle'], 2)
        self.assertEqual(c.rings.spawned['beat'] + c.rings.spawned['onset'], 0)
        self.assertGreater(ghost_frames, 40)

    def test_beat_lock_and_ring_births_identical_to_liquid_on_the_real_chain(self):
        """The variant changes only the drawing: Bands and BeatRings are shared, so the
        beat lock and every ring birth are identical to --pattern liquid."""
        from test_liquid import drum_track
        pcm, beats, period = drum_track(128, .4, 10)
        births = []
        for cls in (LiquidContour, DynamicContour):
            b, c, seq_prev, out = Bands(), cls(), -1, []
            for i in range(int(10 / DT)):
                now = (i + 1) * DT; seq = int(now / .02) - 1
                if seq != seq_prev:
                    lv, seq_prev = b.update(pcm[seq * 320:(seq + 1) * 320].tolist(), DT), seq
                else:
                    lv = b.decay(DT)
                c.sample(.45, [math.sqrt(min(1, max(0, v))) for v in lv], now, DT, .07, 8, flux=b.flux,
                         sub_flux=b.sub_flux, beat_phase=b.beat_phase, beat_lock=b.beat_lock, activity=b.activity,
                         beat_period=b.beat_period, centroid=b.centroid, pitch_angle=b.pitch_angle, accent=b.accent)
                if c.rings.ring is not None and c.rings.last_spawn_time == now:
                    out.append((round(now, 4), c.rings.ring['kind']))
            births.append((out, dict(c.rings.spawned), round(b.beat_lock, 6), round(b.beat_period, 6)))
        self.assertEqual(births[0], births[1])
        self.assertGreaterEqual(len(births[0][0]), 4)


if __name__ == '__main__':
    unittest.main()
