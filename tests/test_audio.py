import os
import signal
import struct
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from test_heliosctl import h, FakeSDK


class AudioTests(unittest.TestCase):
    def args(self, duration=.02):
        return SimpleNamespace(duration=duration, pps=20000, ready_timeout=.1, library='fake')

    def test_silence_loud_bounds_slew_and_constant_colour(self):
        now = [0.0]
        mic = SimpleNamespace(read=lambda: 0.0)
        frames = h.AudioFrames(mic, clock=lambda: now[0])
        silent = frames[0]
        self.assertEqual(len(silent), 501)
        self.assertEqual(silent[0].x, (10000 + 32768) >> 4)
        mic.read = lambda: 1.0
        prior = silent[0].x
        for _ in range(100):
            now[0] += .025
            frame = frames[0]
            self.assertLessEqual(abs(frame[0].x - prior), 40)
            prior = frame[0].x
            for p in frame:
                self.assertTrue(1048 <= p.x <= 3048 and 1235 <= p.y <= 2860)
                self.assertIn((p.r, p.g, p.b, p.i), [(0,0,0,0),(16,48,96,255)])
            self.assertEqual(frame[0].i, 0)
            self.assertEqual(frame[-1].i, 0)
        mic.read = lambda: 0.0
        for _ in range(200):
            now[0] += .025
            frames[0]
        self.assertLess(frames.level, .001)

    def test_pcm_silence_and_clipping(self):
        for sample, expected in [(0, 0.0), (32767, 32767 / 32768), (-32768, 1.0)]:
            read_fd, write_fd = os.pipe()
            os.set_blocking(read_fd, False)
            os.write(write_fd, struct.pack('<h', sample) * 400)
            mic = h.Microphone.__new__(h.Microphone)
            mic.process = SimpleNamespace(stdout=os.fdopen(read_fd,'rb'), poll=lambda: None)
            mic.last, mic.rms, mic.pending = time.monotonic(), 0.0, b''
            self.assertEqual(mic.read(), expected)
            os.close(write_fd)
            with self.assertRaisesRegex(h.CliError, 'EOF'):
                mic.read()
            mic.process.stdout.close()

    def test_observed_hot_microphone_not_flattened(self):
        radii = []
        for rms in (.4743, .7363, .8607):
            now = [0.0]
            frames = h.AudioFrames(SimpleNamespace(read=lambda: rms), clock=lambda: now[0])
            for _ in range(100):
                now[0] += .025
                frame = frames[0]
            radii.append(frame[0].x)
        self.assertGreater(radii[1], radii[0])
        self.assertGreater(radii[2], radii[1])

    def test_gain_clamps_geometry_and_rejects_invalid_before_sdk(self):
        frames = h.AudioFrames(SimpleNamespace(read=lambda: 1.0), gain=32)
        for _ in range(10):
            frames[0]
            self.assertTrue(0 <= frames.level <= 1)
        for gain in (0, -1, 33, float('inf'), float('nan')):
            args = self.args()
            args.gain = gain
            with patch.object(h, 'Microphone') as mic:
                with self.assertRaises(h.CliError):
                    h.run_audio(args)
                mic.assert_not_called()

    def test_capture_failure_stops_dac(self):
        frames = h.AudioFrames(SimpleNamespace(read=lambda: (_ for _ in ()).throw(h.CliError('capture EOF'))))
        with self.assertRaisesRegex(h.CliError, 'capture EOF'):
            h.run_play(self.args(), sdk_factory=FakeSDK, live_frames=frames)
        sdk = FakeSDK.instances[-1]
        self.assertEqual((sdk.stops, sdk.closed, sdk.writes), ([0], 1, []))

    def test_deadline_crossing_during_capture_never_writes(self):
        now = [0.0]
        def read():
            now[0] = 2.0
            return 1.0
        frames = h.AudioFrames(SimpleNamespace(read=read), clock=lambda: now[0])
        h.run_play(self.args(1), sdk_factory=FakeSDK, clock=lambda: now[0], live_frames=frames)
        self.assertEqual(FakeSDK.instances[-1].writes, [])
        self.assertEqual(FakeSDK.instances[-1].closed, 1)

    def test_child_cleanup_even_if_sigterm_ignored(self):
        mic = h.Microphone.__new__(h.Microphone)
        mic.process = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(10)'], stdout=subprocess.PIPE)
        self.assertEqual(mic.process.stdout.readline(), b'ready\n')
        start = time.monotonic()
        mic.close()
        self.assertIsNotNone(mic.process.poll())
        self.assertLess(time.monotonic() - start, 1.2)

    def test_stalled_capture_rejected_without_blocking(self):
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        mic = h.Microphone.__new__(h.Microphone)
        mic.process = SimpleNamespace(stdout=os.fdopen(read_fd, 'rb'), poll=lambda: None)
        mic.last, mic.rms, mic.pending = time.monotonic() - 1, 0.0, b''
        try:
            with self.assertRaisesRegex(h.CliError, 'stalled'):
                mic.read()
        finally:
            mic.process.stdout.close()
            os.close(write_fd)

    def test_monitor_or_wrong_port_rejected_before_capture(self):
        import json
        source = {'name': h.MIC_SOURCE, 'monitor_source': '', 'mute': False,
                  'active_port': 'analog-input-internal-mic',
                  'properties': {'device.form_factor': 'internal'}}
        for changes in ({'monitor_source': 'monitor'}, {'active_port': 'analog-input-mic'}, {'mute': True}):
            with patch('subprocess.check_output', return_value=json.dumps([dict(source, **changes)])), patch('subprocess.Popen') as popen:
                with self.assertRaises(h.CliError):
                    h.Microphone()
                popen.assert_not_called()

    def test_audio_cleanup_on_sdk_error_and_sigterm(self):
        class Mic:
            instances = []
            def __init__(self):
                self.last = time.monotonic()
                self.closed = False
                self.instances.append(self)
            def read(self):
                self.last += .001
                return 0.0
            def close(self):
                self.closed = True
        class Bad(FakeSDK):
            def status(self, _dev):
                raise h.CliError('status failed')
        class Term(FakeSDK):
            def status(self, _dev):
                os.kill(os.getpid(), signal.SIGTERM)
        with patch.object(h, 'Microphone', Mic):
            with self.assertRaisesRegex(h.CliError, 'status failed'):
                h.run_audio(self.args(), sdk_factory=Bad)
            self.assertTrue(Mic.instances[-1].closed)
            h.run_audio(self.args(), sdk_factory=Term)
            self.assertTrue(Mic.instances[-1].closed)
            self.assertEqual((FakeSDK.instances[-1].stops, FakeSDK.instances[-1].closed), ([0],1))
        for duration in (0, 301, float('nan')):
            with self.assertRaises(h.CliError):
                h.run_audio(self.args(duration))


class LatencyTests(unittest.TestCase):
    def mic(self):
        mic = h.Microphone.__new__(h.Microphone)
        mic.last, mic.rms, mic.pending = time.monotonic(), 0.0, b''
        mic.process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 99), poll=lambda: None)
        return mic

    def test_newest_block_and_split_sample_remainder(self):
        mic = self.mic()
        pcm = struct.pack('<320h', *([1000] * 320)) + struct.pack('<320h', *([2000] * 320))
        with patch.object(h.os, 'read', side_effect=[pcm[:701], pcm[701:] + b'\x00', BlockingIOError()]):
            self.assertAlmostEqual(mic.read(), 2000 / 32768)
        self.assertEqual(mic.pending, b'\x00')
        with patch.object(h.os, 'read', side_effect=[b'\x40' + struct.pack('<319h', *([16384] * 319)), BlockingIOError()]):
            self.assertEqual(mic.read(), .5)
        self.assertEqual(mic.pending, b'')

    def test_flood_budget_fails_closed(self):
        mic = self.mic()
        with patch.object(h.os, 'read', return_value=b'\x00' * 4096) as read:
            with self.assertRaisesRegex(h.CliError, 'budget'):
                mic.read()
            self.assertLessEqual(read.call_count, 64)

    def test_attack_release_wall_clock_invariance(self):
        for steps in (1, 4, 20):
            now = [0.0]
            value = [1.0]
            frames = h.AudioFrames(SimpleNamespace(read=lambda: value[0]), clock=lambda: now[0])
            for i in range(steps):
                now[0] = .08 * (i + 1) / steps
                frames[0]
            self.assertAlmostEqual(frames.level, 1 - h.math.exp(-2))
            value[0] = 0.0
            initial = frames.level
            for i in range(steps):
                now[0] = .08 + .18 * (i + 1) / steps
                frames[0]
            self.assertAlmostEqual(frames.level, initial / h.math.e)


class RippleTests(unittest.TestCase):
    def test_ripple_native_frames_keep_blank_colour_cadence_and_slew(self):
        now, value = [0.0], [0.0]
        frames = h.AudioFrames(SimpleNamespace(read=lambda: value[0]), clock=lambda: now[0], pattern='ripple', seed=11)
        previous = frames[0]
        self.assertEqual(previous[0].x, (32768+2500) >> 4)
        for i in range(250):
            now[0] += .025
            value[0] = 1.0 if i%20 < 4 else 0.0
            native = frames[0]
            self.assertEqual(len(native), 501)
            self.assertGreater(len(native)/20000, .010)
            self.assertEqual(native[0].i, 0)
            self.assertEqual(native[-1].i, 0)
            for p, q in zip(native, previous):
                self.assertLessEqual(h.math.hypot(p.x-q.x, p.y-q.y), 37)
                self.assertIn((p.r,p.g,p.b,p.i), [(0,0,0,0),(16,48,96,255)])
            previous = native

    def test_base_size_independent_of_audio_and_onsets(self):
        now = [0.0]
        level = [0.0]
        mic = SimpleNamespace(read=lambda: level[0])
        old = h.AudioFrames(mic, clock=lambda: now[0], pattern='ripple', seed=42, gain=.6)
        new = h.AudioFrames(mic, clock=lambda: now[0], pattern='ripple', seed=42,
                            gain=.6, base_size=.35, expansion=.25)
        geometries = []
        for volume in (0.0, .5, 1.0):
            level[0] = volume
            for _ in range(150):
                now[0] += .025
                old[0]
                frame = new[0]
                self.assertEqual(old.onset_count, new.onset_count)
                self.assertTrue(all(0 <= p.x <= 4095 and 0 <= p.y <= 4095 for p in frame))
            geometries.append((old.geometry, new.geometry))
        self.assertAlmostEqual(geometries[0][1], .35, places=4)
        self.assertAlmostEqual(geometries[-1][1]-geometries[0][1],
                               .25*(geometries[-1][0]-geometries[0][0]), places=4)

    def test_detector_silence_sustained_and_pulses(self):
        detector = h.OnsetDetector()
        self.assertFalse(any(detector.update(0, i*.025, .025) for i in range(40)))
        hits = [detector.update(.8, 1+i*.025, .025) for i in range(100)]
        self.assertEqual(sum(hits), 1)
        detector = h.OnsetDetector()
        hits = []
        for i in range(200):
            hits.append(detector.update(.8 if i % 24 == 12 else .01, i*.025, .025))
        self.assertGreaterEqual(sum(hits), 7)

    def test_hot_baseline_beats_reach_native_frames(self):
        now = [0.0]
        level = [0.90]
        frames = h.AudioFrames(SimpleNamespace(read=lambda: level[0]),
                               clock=lambda: now[0], pattern='ripple', seed=42)
        deviations = []
        for i in range(200):
            level[0] = .98 if i >= 80 and i % 24 in (8, 9, 10) else .90
            now[0] += .025
            native = frames[0]
            if frames.onset_count:
                rx = 2500 + 13500 * frames.geometry
                ry = 2500 + 10500 * frames.geometry
                radii = [h.math.hypot((p.x * 16 - 32768) / rx,
                                     (p.y * 16 - 32768) / ry)
                         for p in native[1:-1]]
                deviations.append(max(radii) - min(radii))
        self.assertGreaterEqual(frames.onset_count, 4)
        self.assertGreater(max(deviations), .02)

    def test_extreme_overlapping_geometry_and_time_steps(self):
        contour = h.RippleContour(7)
        previous = None
        now = 0
        for i in range(600):
            dt = (0, .001, .025, .1, 2)[i % 5]
            now += dt
            coords = contour.sample(float(i % 2), now, dt, True)
            self.assertLessEqual(len(contour.impulses), 4)
            self.assertEqual(coords[0], coords[-1])
            self.assertTrue(all((x/16000)**2 + (y/13000)**2 <= 1.000001 for x,y in coords))
            gaps = [h.math.hypot(x-px, y-py) for (x,y),(px,py) in zip(coords[1:],coords)]
            self.assertGreater(min(gaps), 10)
            self.assertLess(max(gaps), 260)
            if previous:
                self.assertLessEqual(max(h.math.hypot(x-px,y-py) for (x,y),(px,py) in zip(coords,previous)), min(560,22400*dt)+1e-8)
            previous = coords

    def test_knob_validation_before_capture(self):
        for key, values in [('ripple_depth', [-.01, .101, float('nan'), float('inf')]),
                            ('ripple_frequency', [0, 9, 2.5, float('nan')])]:
            for value in values:
                args = AudioTests().args()
                setattr(args, key, value)
                with patch.object(h, 'Microphone') as mic:
                    with self.assertRaises(h.CliError):
                        h.run_audio(args)
                    mic.assert_not_called()

    def test_depth_and_frequency_survive_frame_interpolation(self):
        def profile(depth, frequency):
            c = h.RippleContour(11, depth, frequency)
            for i in range(13):
                coords = c.sample(1, i*.025, .025, i == 0)
            # Remove reserved headroom to measure wave, not ellipse shrinkage.
            return [h.math.hypot(x/16000, y/13000) - (1-depth) for x,y in coords[:-1]]
        old = profile(.035, 3)
        deeper = profile(.07, 3)
        new = profile(.07, 5)
        self.assertGreater(max(deeper)-min(deeper), 1.9*(max(old)-min(old)))
        self.assertGreater(max(new)-min(new), 1.7*(max(old)-min(old)))
        crossings = lambda p: sum(a*b < 0 for a,b in zip(p, p[1:]+p[:1]))
        self.assertGreater(crossings(new), crossings(old))

    def test_extreme_knobs_sampling_curvature_and_transitions(self):
        for depth in (0, .10):
            for frequency in (1, 8):
                c = h.RippleContour(7, depth, frequency)
                previous = None
                for i in range(160):
                    coords = c.sample(1 if i%20 < 10 else 0, i*.025, .025, i%9 == 0)
                    self.assertTrue(all((x/16000)**2+(y/13000)**2 <= 1.000001 for x,y in coords))
                    p = coords[:-1]
                    gaps = [h.math.hypot(p[(j+1)%500][0]-x,p[(j+1)%500][1]-y) for j,(x,y) in enumerate(p)]
                    self.assertLess(max(gaps), 260)
                    self.assertGreater(min(gaps), 5)
                    curvature = [h.math.hypot(p[(j+1)%500][0]-2*x+p[j-1][0], p[(j+1)%500][1]-2*y+p[j-1][1]) for j,(x,y) in enumerate(p)]
                    self.assertLess(max(curvature), 15)
                    if previous:
                        self.assertLessEqual(max(h.math.hypot(x-a,y-b) for (x,y),(a,b) in zip(coords,previous)),560.000001)
                    previous = coords

    def test_minimum_maximum_and_seed(self):
        a, b = h.RippleContour(42), h.RippleContour(42)
        rest = a.sample(0, 0, .025)
        self.assertAlmostEqual(rest[0][0], 2500)
        self.assertTrue(all(abs(h.math.hypot(x,y)-2500)<1e-8 for x,y in rest))
        b.sample(0, 0, .025)
        for i in range(100):
            self.assertEqual(a.sample(.7, i*.025, .025, i%12==0), b.sample(.7, i*.025, .025, i%12==0))
        c = h.RippleContour(1)
        maximum = c.sample(1, 0, .025)
        self.assertEqual(max(x for x,y in maximum), 16000)
        self.assertAlmostEqual(max(y for x,y in maximum), 13000)


if __name__ == '__main__':
    unittest.main()
