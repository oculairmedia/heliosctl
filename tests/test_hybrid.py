import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heliosctl as h
from liquid import Bands, N_CORE
from hybrid import HybridContour
from test_dynamic import lit_runs_metrics, assert_blank_transits_settled
from test_spark import drive

DT = .0225


class HybridTests(unittest.TestCase):
    def test_liquid4_is_a_choice(self):
        import io, contextlib
        with self.assertRaises(h.CliError), contextlib.redirect_stderr(io.StringIO()):
            h.main(['audio', '--pattern', 'liquid4', '--duration', '0'])
        src = type('S', (), {'read': lambda self: .1, 'samples': [0] * 320, 'sequence': 0})()
        self.assertIs(type(h.AudioFrames(src, pattern='liquid4').liquid), HybridContour)
        self.assertIn('liquid4', h.LIQUID_PATTERNS)

    def test_every_frame_stays_inside_the_galvo_bounds(self):
        """Scripted hits of all kinds (strengths vary via energy gain so impacts and silhouettes
        fire), accent high so beam doubling and trails share frames with open-arc rings: 450
        points, segment < 225, second difference < 15, dwell/settle rules, body never pops."""
        kinds = ('kick', 'snare', 'hat', 'stab')
        worst, prev, seen = (0., 0.), None, set()
        for accent in (0., 1.5):
            c = HybridContour()
            gain = [1.]
            def hits(i, now):
                if i % 9 == 0:
                    gain[0] = 3.0 if (i // 9) % 7 == 6 else 1.0
                    return kinds[(i // 9) % 4]
                return None
            frames = []
            ph, now = 0., 0.
            from test_spark import HIT_E, FLOOR_E
            for i in range(int(16. / DT)):
                ph = (ph + 2 * math.pi * DT / .5) % (2 * math.pi)
                kind = hits(i, now)
                e = [v * gain[0] for v in HIT_E[kind]] if kind else FLOOR_E
                sf = [1., 1., 1.] if kind else [0., 0., 0.]
                coords, colors, blanks = c.sample(1., [1., 1., 1.], now, DT, .1, 8, beat_phase=ph, beat_lock=.9,
                                                  activity=1., beat_period=.5, flux=sf[0], sub_flux=sf, accent=accent,
                                                  energy=e, centroid=.4 + .3 * math.sin(now), hit_flux=1., hit_rms=1.)
                now += DT
                self.assertEqual(len(coords), 450)
                for x, y in coords:
                    self.assertLessEqual(abs(x), 16000); self.assertLessEqual(abs(y), 13000)
                assert_blank_transits_settled(self, coords, blanks, colors)
                seg, curv, blank, lit = lit_runs_metrics(coords, blanks)
                self.assertLess(seg, 225); self.assertLess(curv, 15); self.assertLessEqual(blank, 800)
                self.assertGreaterEqual(lit, 180)
                if prev is not None:
                    self.assertLessEqual(max(math.dist(p, q) for p, q in zip(coords[:N_CORE], prev[:N_CORE])), 560)
                prev = coords
                worst = (max(worst[0], seg), max(worst[1], curv))
                for s in c.sparks:
                    seen.add(s['kind'])
            seen.add('beam' if c.beam_frames else 'nobeam')
            seen.add('impact' if c.impacts else 'noimpact')
            seen.add('sil' if c.silhouettes else 'nosil')
            prev = None
        print(f'liquid4: max segment/curvature ILDA: {worst[0]:.1f} {worst[1]:.2f}; drew {sorted(seen)}')
        self.assertIn('trail', seen); self.assertIn('impact', seen); self.assertIn('sil', seen); self.assertIn('beam', seen)

    def test_real_chain_runs_and_bands_expose_hit_size(self):
        from test_liquid import drum_track
        pcm, beats, period = drum_track(128, .4, 8)
        b, c, seq_prev = Bands(), HybridContour(), -1
        for i in range(int(8 / DT)):
            now = (i + 1) * DT; seq = int(now / .02) - 1
            if seq != seq_prev:
                lv, seq_prev = b.update(pcm[seq * 320:(seq + 1) * 320].tolist(), DT), seq
            else:
                lv = b.decay(DT)
            coords, colors, blanks = c.sample(.45, [math.sqrt(min(1, max(0, v))) for v in lv], now, DT, .07, 8, flux=b.flux,
                                              sub_flux=b.sub_flux, beat_phase=b.beat_phase, beat_lock=b.beat_lock,
                                              activity=b.activity, beat_period=b.beat_period, centroid=b.centroid,
                                              pitch_angle=b.pitch_angle, accent=b.accent, energy=b.energy,
                                              hit_flux=b.hit_flux, hit_rms=b.hit_rms)
            self.assertEqual(len(coords), 450)
        self.assertAlmostEqual(b.beat_period, period, delta=.02 * period)
        self.assertGreater(sum(c.spark_births.values()), 30)
        self.assertIn(c.section, ('intro', 'build', 'drop'))


if __name__ == '__main__':
    unittest.main()
