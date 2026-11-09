import math
import random
import sys
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heliosctl as h
from liquid import Bands, LiquidContour


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
                self.assertGreaterEqual(len(native), 160)  # Core (160) + optional concentric rings (240)
                # Core seam closes
                self.assertEqual((native[0].x, native[0].y), (native[159].x, native[159].y))
                self.assertEqual((native[0].r, native[0].g, native[0].b, native[0].i), (0, 0, 0, 0))
                self.assertEqual((native[159].r, native[159].g, native[159].b, native[159].i), (0, 0, 0, 0))
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
                     for p in native[1:159]]
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
            # Check core segments (0..159)
            for j in range(159):
                p, q = coords[j], coords[j + 1]
                max_segment = max(max_segment, math.dist(p, q))
            for j in range(1, 159):
                p, q, r = coords[j], coords[j + 1], coords[j - 1]
                max_curve = max(max_curve, math.hypot(q[0]-2*p[0]+r[0], q[1]-2*p[1]+r[1]))
            # Check concentric ring segments if present
            if len(coords) >= 402:
                for j in range(239):
                    p, q = coords[162 + j], coords[162 + j + 1]
                    max_segment = max(max_segment, math.dist(p, q))
                for j in range(1, 239):
                    p, q, r = coords[162 + j], coords[162 + j + 1], coords[162 + j - 1]
                    max_curve = max(max_curve, math.hypot(q[0]-2*p[0]+r[0], q[1]-2*p[1]+r[1]))
        self.assertLess(max_segment, 225)
        self.assertLess(max_curve, 15)
        print('concentric liquid max segment/curvature ILDA:', max_segment, max_curve)


if __name__ == '__main__':
    unittest.main()
