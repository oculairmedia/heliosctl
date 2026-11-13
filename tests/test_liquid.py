import math
import random
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heliosctl as h
from liquid import Bands, BeatRings, LiquidContour


def tone(hz, amplitude=.12):
    return [round(32767 * amplitude * math.sin(2*math.pi*hz*n/16000)) for n in range(320)]


class Source:
    def __init__(self, hz):
        self.samples = tone(hz)
        self.sequence = 0
    def read(self):
        self.sequence += 1
        return .085


class LiquidTests(unittest.TestCase):
    def test_liquid_capture_failure_and_deadline_shell(self):
        from test_audio import AudioTests
        from unittest.mock import patch
        original = h.AudioFrames
        def liquid_frames(*args, **kwargs):
            kwargs['pattern'] = 'liquid'
            return original(*args, **kwargs)
        with patch.object(h, 'AudioFrames', side_effect=liquid_frames):
            case = AudioTests()
            case.test_capture_failure_stops_dac()
            case.test_deadline_crossing_during_capture_never_writes()

    def test_bands_and_final_native_contours(self):
        final = []
        for index, hz in enumerate((150, 800, 4000)):
            b = Bands()
            levels = b.update(tone(hz), .025)
            self.assertEqual(max(range(3), key=lambda i: b.energy[i]), index)
            self.assertGreater(levels[index], .1)
            now = [0.]
            f = h.AudioFrames(Source(hz), clock=lambda: now[0], pattern='liquid', base_size=.35, expansion=.25)
            previous = None
            for _ in range(80):
                now[0] += .025
                native = f[0]
                self.assertGreaterEqual(len(native), 220)  # Core (220) + optional concentric rings (220)
                # Core seam closes
                self.assertEqual((native[0].x, native[0].y), (native[219].x, native[219].y))
                self.assertEqual((native[0].r, native[0].g, native[0].b, native[0].i), (0, 0, 0, 0))
                self.assertEqual((native[219].r, native[219].g, native[219].b, native[219].i), (0, 0, 0, 0))
                for p in native:
                    self.assertLessEqual(abs(p.x*16-32768), 16016)
                    self.assertLessEqual(abs(p.y*16-32768), 13016)
                    if p.i:  # blank seam points have i=0
                        self.assertEqual(p.i, 255)
                        self.assertTrue(0 <= p.r <= 255 and 0 <= p.g <= 255 and 0 <= p.b <= 255)
                self.assertGreater(len({(p.r, p.g, p.b) for p in native if p.i}), 20)
                final.append([(p.x, p.y) for p in native])
            r_base = (2400.0 + 1900.0 * f.geometry)
            radii = [math.hypot(p.x*16-32768, p.y*16-32768) / r_base
                     for p in native[1:219]]
            self.assertGreater(max(radii) - min(radii), (.03, .015, .003)[index])
        self.assertNotEqual(final[0],final[1])
        self.assertNotEqual(final[1],final[2])

    def test_silence_clipping_noise(self):
        b = Bands()
        for _ in range(200):
            self.assertEqual(b.update([0]*320,.025),[0.,0.,0.])
        self.assertEqual(b.update([32767,-32768]*160,.025),[0.,0.,0.])
        self.assertEqual(b.diagnostic,'clipped')
        rng = random.Random(17)
        b.update([rng.randrange(-8000,8000) for _ in range(320)],.025)
        self.assertEqual(b.diagnostic,'broadband-noise')

    def test_simultaneous_extrema_sampling_curvature(self):
        c = LiquidContour()
        max_segment = max_curve = 0
        for i in range(250):
            dt = .025 if i%17 else .5
            coords, colors, blanks = c.sample(1., [1.,1.,1.], i*.025, dt, .1, 8)
            for x, y in coords:
                self.assertLessEqual(abs(x), 16000)
                self.assertLessEqual(abs(y), 13000)
            # Check core segments (0..219)
            for j in range(219):
                p, q = coords[j], coords[j + 1]
                max_segment = max(max_segment, math.dist(p, q))
            for j in range(1, 219):
                p, q, r = coords[j], coords[j + 1], coords[j - 1]
                max_curve = max(max_curve, math.hypot(q[0]-2*p[0]+r[0], q[1]-2*p[1]+r[1]))
            # Check every other drawn curve (ring, or body retrace) wherever it sits in the
            # frame: each lit run plus its blank seam neighbours, exactly as the core above.
            # (The ring no longer has a fixed 222..441 slot: it yields points to the eased
            # blank transits, which FrameBlankingTests bounds separately.)
            j = 220
            while j < len(coords):
                if blanks[j]:
                    j += 1
                    continue
                k = j
                while k < len(coords) and not blanks[k]:
                    k += 1
                run = coords[j - 1:k + 1]          # lit run with its seam neighbours
                for p, q in zip(run, run[1:]):
                    max_segment = max(max_segment, math.dist(p, q))
                for r, p, q in zip(run, run[1:], run[2:]):
                    max_curve = max(max_curve, math.hypot(q[0]-2*p[0]+r[0], q[1]-2*p[1]+r[1]))
                j = k
        self.assertLess(max_segment, 225)
        self.assertLess(max_curve, 15)
        print('concentric liquid max segment/curvature ILDA:', max_segment, max_curve)


class BeatRingTests(unittest.TestCase):
    """Rings are locked to Bands.beat_phase / beat_period, not to onset thresholds."""
    def drive(self, period, seconds, lock=.8, ring_onsets=False, bpm_change=None, dt=.025):
        c = LiquidContour()
        phase, now, births, sizes, rates = 0., 0., [], set(), []
        for i in range(int(seconds / dt)):
            if bpm_change and now >= bpm_change[0]:
                period = bpm_change[1]
            phase = (phase + 2 * math.pi * dt / period) % (2 * math.pi)
            before = c.rings.spawned['beat']
            # No flux / kick / bass onsets at all: only the beat PLL can create beat rings.
            coords, colors, blanks = c.sample(.3, [0.05, 0.05, 0.05] if not ring_onsets else [.9, .5, .2], now, dt, .07, 5,
                                              beat_phase=phase, beat_lock=lock, activity=.8, beat_period=period)
            sizes.add(len(coords))
            if c.rings.spawned['beat'] > before:
                births.append(now)
            rates.append(c.rings.rate)
            now += dt
        return c, births, sizes, rates

    def test_spawn_on_phase_wrap_every_beat_count_without_onsets(self):
        # 0.9 s period -> 1 ring per beat; births are exactly one beat apart even though
        # audio bands never cross any onset threshold.
        c, births, sizes, _ = self.drive(.9, 20)
        self.assertEqual(c.rings.beats, 1)
        self.assertEqual(c.rings.spawned['onset'], 0)
        gaps = [b - a for a, b in zip(births, births[1:])]
        self.assertGreaterEqual(len(births), 20)
        for g in gaps:
            self.assertAlmostEqual(g, .9, delta=.03)
        # Faster tempo (0.45 s): ring spans 2 beats so it stays on screen ~0.9 s.
        c, births, sizes, _ = self.drive(.45, 20)
        self.assertEqual(c.rings.beats, 2)
        for a, b in zip(births, births[1:]):
            self.assertAlmostEqual(b - a, .9, delta=.03)

    def test_ring_reaches_edge_when_next_ring_is_born(self):
        c = LiquidContour()
        period, dt, phase, now = .9, .025, 0., 0.
        peaks, last_ring, last_frac = [], None, 0.
        for _ in range(int(14 / dt)):
            phase = (phase + 2 * math.pi * dt / period) % (2 * math.pi)
            c.sample(.3, [.1, .1, .1], now, dt, .07, 5, beat_phase=phase, beat_lock=.9, activity=.8, beat_period=period)
            ring = c.rings.ring
            if last_ring is not None and ring is not last_ring:
                peaks.append(last_frac)   # furthest extent this ring reached before it was retired/replaced
            last_ring, last_frac = ring, (ring['frac'] if ring else 0.)
            now += dt
        self.assertGreater(len(peaks), 10)
        for f in peaks[3:]:
            self.assertGreater(f, .93)    # edge reached on the beat, every beat

    def test_tempo_change_slews_expansion_rate(self):
        c, births, sizes, rates = self.drive(.9, 40, bpm_change=(20., .6))
        settled = rates[100:]
        steps = [abs(b - a) for a, b in zip(settled, settled[1:])]
        self.assertLess(max(steps), .02)          # no visible jump per 25 ms frame
        self.assertAlmostEqual(rates[790], 1 / .9, delta=.02)
        self.assertAlmostEqual(rates[-1], 1 / .6 / 1.0 if c.rings.beats == 1 else 1 / (2 * .6), delta=.03)

    def test_unlocked_falls_back_to_onsets_and_idle_and_never_stops(self):
        c, births, sizes, _ = self.drive(.5, 12, lock=0., ring_onsets=True)
        self.assertEqual(c.rings.spawned['beat'], 0)
        self.assertGreater(c.rings.spawned['onset'], 3)
        c, births, sizes, _ = self.drive(.5, 12, lock=0., ring_onsets=False)
        self.assertEqual(c.rings.spawned['beat'] + c.rings.spawned['onset'], 0)
        self.assertGreaterEqual(c.rings.spawned['idle'], 3)   # quiet material still breathes

    def test_constant_point_budget_and_refresh(self):
        for lock in (0., .9):
            for ring_onsets in (False, True):
                c, births, sizes, _ = self.drive(.5, 10, lock=lock, ring_onsets=ring_onsets)
                self.assertEqual(sizes, {450})
        # 450 pts @ 20 kpps = 22.5 ms -> 44.4 Hz
        self.assertGreaterEqual(20000 / 450, 44)

    def test_phase_wrap_backwards_nudge_never_double_spawns(self):
        r = BeatRings()
        spawned = 0
        phase = 6.0
        for i, ph in enumerate((6.0, 0.1, 6.2, 0.05, 0.2, 0.3)):  # PLL jitters across the wrap
            before = r.spawned['beat']
            r.update(i * .025, .025, ph, .9, .9, .8, False, (1, 2, 3))
            spawned += r.spawned['beat'] - before
        self.assertEqual(spawned, 1)

    def test_long_session_state_stays_bounded(self):
        c = LiquidContour()
        now = 0.
        for i in range(4000):
            now += .025
            c.sample(.5, [.6, .4, .3], now, .025, .07, 5, beat_phase=(now * 7.0) % (2 * math.pi), beat_lock=.8,
                     activity=.9, beat_period=.9)
        for v in (c.phase, c.knot_phase, c.color_phase, c.rot4d):
            self.assertLess(v, 2 * math.pi + 1e-9)
        self.assertLess(c.tilt_phase, 20 * math.pi + 1e-9)
        self.assertLess(c.morph_clock, 8 * math.pi + 1e-9)


class FrameBlankingTests(unittest.TestCase):
    """The core->ring transit and the ring->core walk-back must be dark *and* settled.

    Galvos lag the colour channel: a blanked jump of thousands of ILDA units in one
    point leaves the mirrors still travelling when the next point is lit, which
    paints the 'line from the sphere to the ring' seen on the wall.
    """
    LIT_STEP = 225       # galvo bound for any segment drawn lit (heliosctl limit)
    BLANK_STEP = 800     # blank moves are eased over several points, never one jump
    SETTLE_OFF = 3       # identical blank points before a jump leaves a lit curve
    SETTLE_ON = 4        # identical blank points at the landing before relighting

    def frames(self, seconds=8, dt=.025, period=.9):
        c = LiquidContour()
        phase, now = 0., 0.
        for _ in range(int(seconds / dt)):
            phase = (phase + 2 * math.pi * dt / period) % (2 * math.pi)
            coords, colors, blanks = c.sample(.3, [.6, .4, .3], now, dt, .07, 5, beat_phase=phase,
                                              beat_lock=.9, activity=.8, beat_period=period)
            yield coords, colors, blanks, c.rings.ring
            now += dt

    def test_ring_transit_is_blank_bounded_and_settled(self):
        n_big = 0
        for coords, colors, blanks, ring in self.frames():
            n = len(coords)
            self.assertEqual(n, 450)
            steps = [math.dist(coords[i - 1], coords[i]) for i in range(n)]   # cyclic: frame repeats
            for i in range(n):
                if not blanks[i]:
                    # a lit point never ends a long move, and never starts one
                    self.assertLessEqual(steps[i], self.LIT_STEP, f'lit point {i} reached by a jump')
                    self.assertLessEqual(steps[(i + 1) % n], self.LIT_STEP, f'lit point {i} starts a jump')
                else:
                    self.assertEqual(colors[i], (0, 0, 0))
                self.assertLessEqual(steps[i], self.BLANK_STEP, f'blank step {i} is a single galvo jump')
            # every maximal blank run that contains a jump must settle before and after it
            i = 0
            while i < n:
                if blanks[i] and not blanks[i - 1]:
                    j = i
                    while blanks[j % n]:
                        j += 1
                    run = [coords[k % n] for k in range(i, j)]
                    jump = max(math.dist(p, q) for p, q in zip(run, run[1:])) if len(run) > 1 else 0.
                    if jump > self.LIT_STEP:
                        n_big += 1
                        self.assertGreaterEqual(len(run), self.SETTLE_OFF + self.SETTLE_ON)
                        self.assertEqual(len(set(run[:self.SETTLE_OFF])), 1, f'no laser-off dwell at {i}')
                        self.assertEqual(len(set(run[-self.SETTLE_ON:])), 1, f'no galvo settle before relight at {j}')
                i += 1
        self.assertGreater(n_big, 50)   # the scenario really exercised ring transits


if __name__ == '__main__':
    unittest.main()
