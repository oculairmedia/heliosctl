import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heliosctl as h
from liquid import Bands, LiquidContour, N_CORE
from dynamic import DynamicContour
from spark import SparkContour, KINDS, REFRACTORY
from test_dynamic import lit_runs_metrics, assert_blank_transits_settled

DT = .0225   # heliosctl frame interval: 450 pts @ 20 kpps

# raw Bands.energy vectors of the four hit kinds (from the synthetic-hit probe) and a quiet floor
HIT_E = {'kick': [.25, .03, .001], 'snare': [.02, .13, .04], 'hat': [.012, .012, .04], 'stab': [.07, .10, .02]}
FLOOR_E = [.01, .01, .005]


def drive(c, seconds, hits, period=.5, lock=.9, act=1., accent=0., bands=(1., 1., 1.), level=1., start=0., energy_gain=1.):
    """Locked frames with scripted hits: `hits(i, now)` -> kind or None.  Yields (now, frame)."""
    ph, now = 0., start
    for i in range(int(seconds / DT)):
        ph = (ph + 2 * math.pi * DT / period) % (2 * math.pi)
        kind = hits(i, now)
        e = [v * energy_gain for v in HIT_E[kind]] if kind else FLOOR_E
        sf = [1., 1., 1.] if kind else [0., 0., 0.]
        extra = {'energy': e} if isinstance(c, SparkContour) else {}
        yield now, c.sample(level, list(bands), now, DT, .1, 8, beat_phase=ph, beat_lock=lock, activity=act,
                            beat_period=period, flux=sf[0], sub_flux=sf, accent=accent, **extra)
        now += DT


def every_beat(kind, frames=22):
    return lambda i, now: kind if i % frames == 0 else None


class PatternChoiceTests(unittest.TestCase):
    def test_liquid3_is_a_choice_and_the_others_are_unchanged(self):
        import io, contextlib
        with self.assertRaises(h.CliError), contextlib.redirect_stderr(io.StringIO()):
            h.main(['audio', '--pattern', 'liquid3', '--duration', '0'])   # parses, then duration 0 is a CliError
        src = type('S', (), {'read': lambda self: .1, 'samples': [0] * 320, 'sequence': 0})()
        self.assertIs(type(h.AudioFrames(src, pattern='liquid3').liquid), SparkContour)
        self.assertIs(type(h.AudioFrames(src, pattern='liquid2').liquid), DynamicContour)
        self.assertIs(type(h.AudioFrames(src, pattern='liquid').liquid), LiquidContour)
        self.assertIn('liquid3', h.LIQUID_PATTERNS)


class HitClassifierTests(unittest.TestCase):
    def test_hit_kinds_from_energy_jumps_with_refractory(self):
        c = SparkContour()
        for kind in ('kick', 'snare', 'hat', 'stab'):
            seen = []
            for i, (now, _) in enumerate(drive(c, 2., every_beat(kind, 11), start=len(seen) * 10.)):
                seen += [k for k, s in c.hits]
            self.assertEqual(set(seen), {kind}, f'{kind}: {seen}')
            self.assertGreaterEqual(len(seen), 7)
        # the same hit held over consecutive frames (a block repeated by decay()) counts once
        c = SparkContour(); n = 0
        for now, _ in drive(c, 1., lambda i, now: 'snare' if 10 <= i < 13 else None):
            n += len(c.hits)
        self.assertEqual(n, 1)
        # no flux, no hit; a slow swell (energy rising a little each frame) is not a hit either
        c = SparkContour(); n = 0; ph = 0.
        for i in range(60):
            e = [.01 + .002 * i, .01 + .001 * i, .005]
            c.sample(1., [1., 1., 1.], i * DT, DT, .1, 8, beat_lock=.9, activity=1., sub_flux=[0., 0., 0.], energy=e)
            n += len(c.hits)
        self.assertEqual(n, 0)
        for k in KINDS:
            self.assertGreaterEqual(REFRACTORY[k], 0.06)


class EventMorphTests(unittest.TestCase):
    def test_steady_groove_holds_its_shape_where_liquid2_morphs_every_beat(self):
        """Saturated, locked, accent 0, the same kick on every beat for 20 s: liquid2 morphs on
        every beat; liquid3 morphs at most once (the music starting) and then holds."""
        c2, c3 = DynamicContour(), SparkContour()
        for now, _ in drive(c2, 20., every_beat('kick')): pass
        for now, _ in drive(c3, 20., every_beat('kick')): pass
        self.assertGreaterEqual(c2.morphs, 36)
        self.assertLessEqual(c3.morphs, 1)
        self.assertEqual(c3.morph, 1.0)                     # settled, breathing, not drifting
        self.assertTrue(all(e[1] == 'start' for e in c3.events), c3.events)
        self.assertGreaterEqual(c3.hit_count['kick'], 38)   # the hits were seen: they just were not events

    def test_accent_step_morphs_on_the_next_beat_and_a_surprising_hit_morphs_at_once(self):
        c = SparkContour(); beats = []
        for now, _ in drive(c, 8., every_beat('kick')):
            if c.rings.beat_hit: beats.append(now)
        m0, t0 = c.morphs, 8.
        # a sudden accent (a drop): morph within one beat, landing on a beat frame
        for now, _ in drive(c, 2., every_beat('kick'), accent=1.6, start=t0):
            if c.rings.beat_hit: beats.append(now)
        self.assertGreaterEqual(c.morphs, m0 + 1)
        t_morph, ev, shape = next(e for e in c.events if e[0] >= t0)
        self.assertLessEqual(t_morph - t0, .5 + DT)
        self.assertLessEqual(min(abs(t_morph - b) for b in beats), DT / 2, 'accent morph lands on a beat frame')
        self.assertEqual(ev, 'accent')
        self.assertIn(shape, ('eight', 'clover', 'limacon'))        # a strong event picks from the wild set
        # a hit three times louder than the ordinary ones: the morph starts on that very frame
        c = SparkContour()
        for now, _ in drive(c, 8., every_beat('snare')): pass
        m0 = c.morphs
        hit_t = None
        for now, _ in drive(c, 2., lambda i, now: 'snare' if i % 22 == 0 else None, start=8.,
                            energy_gain=3.):
            if c.hits and hit_t is None: hit_t = now
            if hit_t is not None and c.morphs > m0:
                self.assertEqual(c.last_morph, hit_t)
                break
        else:
            self.fail('a surprising hit did not morph')
        self.assertEqual(c.events[-1][1], 'hit')
        self.assertLess(1.0 / c.morph_rate, .2)                      # a hit morph commits fast

    def test_no_events_no_morphs_however_long_and_silence_goes_home(self):
        c = SparkContour()
        for now, _ in drive(c, 30., every_beat('hat', 11), act=.5, bands=(.4, .3, .3), level=.4): pass
        self.assertLessEqual(c.morphs, 1)
        for now, _ in drive(c, 6., lambda i, now: None, act=0., bands=(0., 0., 0.), level=0., start=30.): pass
        self.assertEqual(c.NAMES[c.shape_nxt], 'sphere'); self.assertEqual(c.morph, 1.0)


class SparkleTests(unittest.TestCase):
    def check_frame(self, coords, colors, blanks, prev):
        self.assertEqual(len(coords), 450)
        for x, y in coords:
            self.assertLessEqual(abs(x), 16000); self.assertLessEqual(abs(y), 13000)
        assert_blank_transits_settled(self, coords, blanks, colors)
        seg, curv, blank, lit = lit_runs_metrics(coords, blanks)
        self.assertLess(seg, 225); self.assertLess(curv, 15); self.assertLessEqual(blank, 800)
        self.assertGreaterEqual(lit, 180)
        if prev is not None:
            self.assertLessEqual(max(math.dist(p, q) for p, q in zip(coords[:N_CORE], prev[:N_CORE])), 560)
        return seg, curv, lit

    def test_every_frame_with_sparks_rings_and_shatter_stays_inside_the_galvo_bounds(self):
        """All four hit kinds, saturated bands, locked with earned rings (accent rings too, so
        sparks share frames with burst rings), hits every 11 frames: every frame is 450 points,
        obeys the lit segment / second difference / blank step bounds and the dwell/settle
        rules, the body never pops, and sparks were really drawn."""
        kinds = ('kick', 'snare', 'hat', 'stab')
        for accent in (0., 1.5):
            c = SparkContour(); prev = None; spark_frames = 0; ring_spark_frames = 0; worst = (0., 0.)
            for now, (coords, colors, blanks) in drive(c, 16., lambda i, now: kinds[(i // 11) % 4] if i % 11 == 0 else None,
                                                        accent=accent):
                seg, curv, lit = self.check_frame(coords, colors, blanks, prev)
                worst = (max(worst[0], seg), max(worst[1], curv))
                prev = coords
                if c.spark_pts:
                    spark_frames += 1
                    if c.rings.ring is not None and c.ring_obj is c.rings.ring:
                        ring_spark_frames += 1
            self.assertGreater(spark_frames, 200)
            self.assertGreater(ring_spark_frames, 20)       # sparks drawn while a ring is alive (open arc)
            self.assertEqual(sum(c.spark_births.values()), sum(c.hit_count[k] * KINDS[k][0] for k in kinds))
            print(f'liquid3 sparks accent={accent}: max segment/curvature ILDA: {worst[0]:.1f} {worst[1]:.2f}')

    def test_sparks_are_short_lived_and_differ_by_hit_kind(self):
        """One hit of each kind alone: sparks vanish within the kind's life, and the kinds are
        distinguishable: snares scatter the most, hats the fewest and fastest-dying, kicks are
        warm, snares shatter the body for a few frames and the body comes back whole."""
        results = {}
        for kind in ('kick', 'snare', 'hat', 'stab'):
            c = SparkContour()
            for now, _ in drive(c, 3., lambda i, now: None): pass          # settle, no hits
            frames_with_sparks, max_sparks, shatter_frames, hues = 0, 0, 0, []
            t_hit = None
            for now, (coords, colors, blanks) in drive(c, 1.0, lambda i, now: kind if i == 0 else None, start=3.):
                if c.hits: t_hit = now
                if c.spark_pts:
                    frames_with_sparks += 1
                    last_spark = now
                max_sparks = max(max_sparks, len(c.sparks))
                body_blank = sum(1 for b in blanks[4:N_CORE - 5] if b)
                if body_blank > 0: shatter_frames += 1
                for s in c.sparks: hues.append(s['rgb'])
            self.assertIsNotNone(t_hit)
            self.assertLessEqual(last_spark - t_hit, KINDS[kind][1] + 2 * DT, f'{kind} sparks linger')
            self.assertEqual(len(c.sparks), 0)
            self.assertFalse(any(blanks[4:N_CORE - 5]))                     # body whole again
            results[kind] = (frames_with_sparks, max_sparks, shatter_frames, set(hues))
        self.assertGreater(results['snare'][1], results['kick'][1])
        self.assertGreater(results['kick'][1], results['hat'][1])
        self.assertLess(results['hat'][0], results['snare'][0])            # hats die first
        self.assertGreaterEqual(results['snare'][2], 2); self.assertLessEqual(results['snare'][2], 5)
        self.assertEqual(results['kick'][2], 0); self.assertEqual(results['hat'][2], 0)
        self.assertGreater(results['stab'][2], results['snare'][2])
        r, g, b = next(iter(results['kick'][3]))
        self.assertGreater(r, b + 100)                                     # kick sparks are warm
        r, g, b = next(iter(results['hat'][3]))
        self.assertGreater(b, r + 80)                                      # hat sparks are cold
        for kind in results:
            self.assertLessEqual(results[kind][0], 10)                     # never texture: <= 0.23 s of sparks

    def test_snare_flashes_the_body_white_and_the_flash_is_gone_in_a_tenth(self):
        c = SparkContour(); base = []; t_hit = None; whites = []
        for now, (coords, colors, blanks) in drive(c, 2., lambda i, now: 'snare' if i == 40 else None, accent=0.):
            body = [col for col, bl in zip(colors[1:N_CORE - 1], blanks[1:N_CORE - 1]) if not bl]
            mean = sum(sum(col) for col in body) / (3 * len(body))
            if c.hits and c.hits[0][0] == 'snare':
                t_hit = now
            if t_hit is None: base.append(mean)                             # includes the beat punch's own flash
            else: whites.append((now - t_hit, mean))
        self.assertIsNotNone(t_hit)
        self.assertGreater(whites[0][1], max(base[10:]) + 50)               # hit frame: well above anything before
        self.assertLess(max(m for t, m in whites if t > .15), max(base[10:]) + 5)   # and gone 0.15 s later

    def test_beat_lock_ring_births_and_hits_on_the_real_chain(self):
        """Bands and BeatRings are shared: lock and ring births are identical to --pattern liquid
        with the same PCM; on a four-on-the-floor track the classifier sees kicks, the snare
        (clap on 2 and 4, with the kick) and offbeat hats, and sparks on them."""
        from test_liquid import drum_track
        pcm, beats, period = drum_track(128, .4, 10)
        births = []
        for cls in (LiquidContour, SparkContour):
            b, c, seq_prev, out = Bands(), cls(), -1, []
            for i in range(int(10 / DT)):
                now = (i + 1) * DT; seq = int(now / .02) - 1
                if seq != seq_prev:
                    lv, seq_prev = b.update(pcm[seq * 320:(seq + 1) * 320].tolist(), DT), seq
                else:
                    lv = b.decay(DT)
                extra = {'energy': b.energy} if cls is SparkContour else {}
                coords, colors, blanks = c.sample(.45, [math.sqrt(min(1, max(0, v))) for v in lv], now, DT, .07, 8, flux=b.flux,
                                                  sub_flux=b.sub_flux, beat_phase=b.beat_phase, beat_lock=b.beat_lock,
                                                  activity=b.activity, beat_period=b.beat_period, centroid=b.centroid,
                                                  pitch_angle=b.pitch_angle, accent=b.accent, **extra)
                self.assertEqual(len(coords), 450)
                if c.rings.ring is not None and c.rings.last_spawn_time == now:
                    out.append((round(now, 4), c.rings.ring['kind']))
            births.append((out, dict(c.rings.spawned), round(b.beat_lock, 6), round(b.beat_period, 6)))
        self.assertEqual(births[0], births[1])
        self.assertGreaterEqual(c.hit_count['kick'], 8)
        self.assertGreaterEqual(c.hit_count['snare'], 4)
        self.assertGreaterEqual(c.hit_count['hat'], 8)
        self.assertGreater(sum(c.spark_births.values()), 50)
        self.assertLessEqual(c.morphs, 4)      # a steady track: the start and perhaps the groove settling, not a beat clock


if __name__ == '__main__':
    unittest.main()
