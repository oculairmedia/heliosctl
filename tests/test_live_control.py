import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_control import Control
from test_heliosctl import h, FakeSDK


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, XDG_RUNTIME_DIR=self.tmp.name)
        self.env.start()
        self.c = Control(1, .07, 5)
        self.c.deadline = time.monotonic() + 3

    def tearDown(self):
        if self.c.server.fileno() != -1:
            self.c.finish(None)
        self.env.stop()
        self.tmp.cleanup()

    def test_validation_atomic_and_renewal(self):
        baseline = self.c.params.copy()
        for changes in ({'gain': 2, 'frequency': 9}, {'gain': float('nan')}, {'gain': float('inf')}, {'frequency': 1.5}, {'ripple-depth': -.1}, {'unknown': 1}, {'gain': True}):
            with self.assertRaises(ValueError):
                self.c.command({'command': 'tune', 'params': changes}, 1)
            self.assertEqual(self.c.params, baseline)
        self.c.command({'command': 'renew', 'seconds': 300}, 10)
        self.assertEqual(self.c.deadline, 310)
        self.c.command({'command': 'renew', 'seconds': 5}, 11)
        self.assertEqual(self.c.deadline, 16)
        for seconds in (301, 0, float('nan')):
            with self.assertRaises(ValueError):
                self.c.command({'command': 'renew', 'seconds': seconds}, 12)
        with self.assertRaises(ValueError):
            self.c.command({'command': 'renew', 'seconds': 5}, 17)

    def test_socket_malformed_slow_eof_flood(self):
        clients = []
        for payload in (b'{}\n', b'NaN\n', b'{', b'x' * 2049, b'') * 3:
            client = socket.socket(socket.AF_UNIX)
            client.connect(str(self.c.path))
            clients.append(client)
            if payload:
                client.sendall(payload)
            else:
                client.shutdown(socket.SHUT_WR)
            self.c.poll(time.monotonic())
        start = time.monotonic()
        for _ in range(100):
            self.c.poll(time.monotonic())
        self.assertLess(time.monotonic() - start, .5)
        self.assertEqual(self.c.params, {'gain': 1, 'ripple-depth': .07, 'frequency': 5,
                                        'base-size': 0.0, 'expansion': 1.0})
        for client in clients:
            client.close()

    def test_other_controller_and_stale_endpoint_refused(self):
        with self.assertRaises((OSError, ValueError)):
            Control(1, .07, 5)
        self.c.finish(None)
        self.c.path.touch(mode=0o600)
        with self.assertRaises(ValueError):
            Control(1, .07, 5)
        self.c.path.unlink()

    def test_tune_changes_actual_native_frames_and_stop_cleans(self):
        mic = SimpleNamespace(read=lambda: .6)
        frames = h.AudioFrames(mic, pattern='ripple', seed=1)
        args = SimpleNamespace(duration=.1, pps=20000, ready_timeout=.1, library='fake', _control=self.c)
        controller = self.c
        snapshots = []
        class SDK(FakeSDK):
            def write(self, device, pps, flags, points, count):
                snapshots.append([(p.x, p.y, p.r, p.g, p.b, p.i) for p in points])
                if len(snapshots) == 1:
                    controller.command({'command': 'tune', 'params': {'ripple-depth': 0, 'gain': .1, 'frequency': 8}}, time.monotonic())
                if len(snapshots) == 3:
                    client = socket.socket(socket.AF_UNIX)
                    client.connect(str(controller.path))
                    client.sendall(b'{"command":"stop"}\n')
                    self.stop_client = client
                return 1
        sdk = SDK('fake')
        h.run_play(args, sdk_factory=lambda _: sdk, live_frames=frames)
        self.assertEqual(frames.ripple.depth, 0)
        self.assertEqual(frames.ripple.frequency, 8)
        self.assertEqual(frames.gain, .1)
        self.assertNotEqual(snapshots[0], snapshots[1])
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(json.loads(sdk.stop_client.recv(4096))['state'], 'stopping')
        sdk.stop_client.close()
        self.assertEqual((sdk.stops, sdk.closed), ([0], 1))
        self.c.finish(None)
        self.assertEqual(json.loads(self.c.receipt.read_text())['state'], 'stopped')

    def test_audio_receipt_after_mic_reap_and_sdk_cleanup(self):
        self.c.finish(None)
        self.c.receipt.unlink()
        receipt = self.c.receipt
        events = []
        class Mic:
            def __init__(self):
                self.last = time.monotonic()
            def read(self):
                self.last += .001
                return .2
            def close(self):
                self_test.assertFalse(receipt.exists())
                events.append('reaped')
        self_test = self
        args = SimpleNamespace(duration=.003, pps=20000, ready_timeout=.1, library='fake', live_control=True)
        with patch.object(h, 'Microphone', Mic):
            h.run_audio(args, sdk_factory=FakeSDK)
        self.assertEqual(events, ['reaped'])
        self.assertEqual(json.loads(receipt.read_text())['state'], 'stopped')
        self.assertEqual((FakeSDK.instances[-1].stops, FakeSDK.instances[-1].closed), ([0], 1))

    def test_deadline_without_renewal(self):
        args = SimpleNamespace(duration=.002, pps=20000, ready_timeout=.1, library='fake', _control=self.c)
        frames = h.AudioFrames(SimpleNamespace(read=lambda: .1))
        h.run_play(args, sdk_factory=lambda _: FakeSDK('fake'), live_frames=frames)
        self.assertLessEqual(self.c.deadline, time.monotonic())


if __name__ == '__main__':
    unittest.main()
