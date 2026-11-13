import ctypes
import importlib.util
import math
import signal
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location("heliosctl", Path(__file__).parents[1] / "heliosctl.py")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)


def make_ilda(frames=None):
    frames = frames or [[(0, 0, 0, 0, 0, True), (100, 200, 5, 10, 20, False)]]
    out = bytearray()
    for fi, points in enumerate(frames):
        hdr = b"ILDA" + bytes(3) + bytes([5]) + b"TEST    " + b"UNIT    " + struct.pack(">HHHBB", len(points), fi, len(frames), 0, 0)
        out += hdr
        for i, (x, y, r, g, b, blank) in enumerate(points):
            st = (0x40 if blank else 0) | (0x80 if i == len(points) - 1 else 0)
            out += struct.pack(">hhBBBB", x, y, st, b, g, r)
    out += b"ILDA" + bytes(3) + bytes([5]) + b"END     " + b"        " + struct.pack(">HHHBB", 0, 0, 0, 0, 0)
    return bytes(out)


class FakeSDK:
    instances = []
    def __init__(self, lib):
        self.statuses = []
        self.writes = []
        self.stops = []
        self.closed = 0
        self.opened = False
        FakeSDK.instances.append(self)
    def open(self):
        self.opened = True
        return 1
    def status(self, dev):
        return self.statuses.pop(0) if self.statuses else 1
    def write(self, dev, pps, flags, array, count):
        self.writes.append((dev, pps, flags, count, [(p.x, p.y, p.r, p.g, p.b, p.i) for p in array[:count]]))
        return 1
    def stop(self, dev):
        self.stops.append(dev)
        return 1
    def close(self):
        self.closed += 1
        return 1


class TestHeliosCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ok.ild"
        self.path.write_bytes(make_ilda([[(-32768, 32767, 0, 0, 0, True), (1600, 3200, 5, 10, 20, False)] * 200]))
        FakeSDK.instances.clear()
    def tearDown(self): self.tmp.cleanup()

    def test_valid_parse_and_single_mode_write(self):
        frames, points = h.parse_ilda5(self.path)
        self.assertEqual((len(frames), points), (1, 400))
        args = SimpleNamespace(file=str(self.path), duration=0.002, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=FakeSDK)
        sdk = FakeSDK.instances[-1]
        self.assertEqual(sdk.writes[0][2], h.HeliosSDK.SINGLE_MODE)
        self.assertEqual(ctypes.sizeof(h.HeliosPoint), 8)
        self.assertEqual({name: getattr(h.HeliosPoint, name).offset for name, *_ in h.HeliosPoint._fields_}, {"x": 0, "y": 2, "r": 4, "g": 5, "b": 6, "i": 7})
        self.assertEqual(sdk.writes[0][4][:2], [(0, 4095, 0, 0, 0, 0), (2148, 2248, 5, 10, 20, 255)])
        self.assertEqual((sdk.stops, sdk.closed), ([0], 1))

    def test_coordinate_boundaries_map_to_12_bit(self):
        self.path.write_bytes(make_ilda([[(-32768, 32767, 1, 2, 3, False)] * 400]))
        args = SimpleNamespace(file=str(self.path), duration=0.002, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=FakeSDK)
        self.assertEqual(FakeSDK.instances[-1].writes[0][4], [(0, 4095, 1, 2, 3, 255)] * 400)

    def test_blank_point_suppresses_rgb_and_intensity(self):
        self.path.write_bytes(make_ilda([[(20, -20, 0, 0, 0, True)] * 400]))
        args = SimpleNamespace(file=str(self.path), duration=0.002, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=FakeSDK)
        self.assertEqual(FakeSDK.instances[-1].writes[0][4], [(2049, 2046, 0, 0, 0, 0)] * 400)

    def test_readiness_timeout_stops_and_closes(self):
        class NotReady(FakeSDK):
            def status(self, dev): return 0
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=0.001, library="fake")
        with self.assertRaisesRegex(h.CliError, "readiness timeout"):
            h.run_play(args, sdk_factory=NotReady)
        sdk = NotReady.instances[-1]
        self.assertEqual((sdk.stops, sdk.closed), ([0], 1))

    def test_status_error_stops_and_closes(self):
        class BadStatus(FakeSDK):
            def status(self, dev): return -8
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaisesRegex(h.CliError, "GetStatus failed \\(-8\\)"):
            h.run_play(args, sdk_factory=BadStatus)
        sdk = BadStatus.instances[-1]
        self.assertEqual((sdk.writes, sdk.stops, sdk.closed), ([], [0], 1))

    def test_write_error_stops_and_closes(self):
        class BadWrite(FakeSDK):
            def write(self, *args): return -17
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaisesRegex(h.CliError, "WriteFrame failed"):
            h.run_play(args, sdk_factory=BadWrite)
        sdk = BadWrite.instances[-1]
        self.assertEqual((sdk.stops, sdk.closed), ([0], 1))

    def test_stop_and_close_failure_are_reported(self):
        class BadCleanup(FakeSDK):
            def stop(self, dev):
                self.stops.append(dev)
                return -21
            def close(self):
                self.closed += 1
                return -22
        args = SimpleNamespace(file=str(self.path), duration=0.002, pps=20000, ready_timeout=1, library="fake")
        with tempfile.TemporaryFile(mode="w+t") as err:
            import contextlib
            with contextlib.redirect_stderr(err), self.assertRaisesRegex(h.CliError, "cleanup failed"):
                h.run_play(args, sdk_factory=BadCleanup)
            err.seek(0)
            output = err.read()
        self.assertIn("Stop(0) failed (-21)", output)
        self.assertIn("CloseDevices failed (-22)", output)

    def test_sigint_cleanup(self):
        class Interrupt(FakeSDK):
            def status(self, dev): raise KeyboardInterrupt()
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=Interrupt)
        sdk = Interrupt.instances[-1]
        self.assertEqual((sdk.stops, sdk.closed), ([0], 1))

    def test_actual_orbit_fixture_and_group_cadence(self):
        import hashlib
        path = Path(__file__).with_name("rotating-orbit.ild")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "1dab392d80f20ad63a7e7209bdf24302d482cec90e4480d1359a069a41906b80")
        frames, total = h.parse_ilda5(path)
        groups = h.group_file_frames(frames, 20000)
        self.assertEqual((len(groups), total), (60, 26280))
        self.assertTrue(all(len(g) == 438 for g in groups))
        self.assertEqual((groups[0][0].x, groups[0][0].i), ((17694 + 32768) >> 4, 0))
        original = [p for f in frames for p in f.points]
        actual = [p for g in groups for p in g]
        self.assertEqual([(p.x, p.y, p.i) for p in actual],
                         [((p[0]+32768)>>4, (p[1]+32768)>>4, 0 if p[5] else 255) for p in original])

    def test_invalid_native_frame_rejected_before_open(self):
        # Only the 4096-point overflow is invalid; 1-point frames are valid ILDA
        # and are padded with blanks up to the SDK's 10ms minimum.
        self.path.write_bytes(make_ilda([[(0, 0, 1, 2, 3, False)] * 4096]))
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaises(h.CliError):
            h.run_play(args, sdk_factory=FakeSDK)
        self.assertFalse(FakeSDK.instances)

    def test_exactly_one_dac_guard(self):
        class Multiple(FakeSDK):
            def open(self): return 2
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaisesRegex(h.CliError, "exactly one"):
            h.run_play(args, sdk_factory=Multiple)
        self.assertEqual(Multiple.instances[-1].writes, [])
        self.assertEqual(Multiple.instances[-1].stops, [0])
        self.assertEqual(Multiple.instances[-1].closed, 1)

    def test_sigterm_cleanup_and_handler_restore(self):
        previous = signal.getsignal(signal.SIGTERM)
        class Terminate(FakeSDK):
            def status(self, dev):
                signal.raise_signal(signal.SIGTERM)
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=Terminate)
        self.assertEqual((Terminate.instances[-1].stops, Terminate.instances[-1].closed), ([0], 1))
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_deadline_crossing_in_status_does_not_write(self):
        now = [0.0]
        class SlowStatus(FakeSDK):
            def status(self, dev):
                now[0] = 2.0
                return 1
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        h.run_play(args, sdk_factory=SlowStatus, clock=lambda: now[0])
        self.assertEqual(SlowStatus.instances[-1].writes, [])
        self.assertEqual(SlowStatus.instances[-1].closed, 1)

    def test_open_error_does_not_write(self):
        class BadOpen(FakeSDK):
            def open(self): return -9
        args = SimpleNamespace(file=str(self.path), duration=1, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaisesRegex(h.CliError, "OpenDevicesOnlyUsb failed"):
            h.run_play(args, sdk_factory=BadOpen)
        self.assertEqual(BadOpen.instances[-1].writes, [])

    def test_rejects_nonfinite_duration(self):
        args = SimpleNamespace(file=str(self.path), duration=math.inf, pps=20000, ready_timeout=1, library="fake")
        with self.assertRaisesRegex(h.CliError, "duration"):
            h.run_play(args, sdk_factory=FakeSDK)
        self.assertFalse(FakeSDK.instances)

    def test_duration_cap_removed_and_none_is_unbounded(self):
        # The old <=300 s cap was removed on request; None runs until stopped.
        import signal as sig
        class Stopper(FakeSDK):
            def status(self, dev):
                sig.raise_signal(sig.SIGTERM)
                return 1
        for duration in (301, 3600.0, None):
            args = SimpleNamespace(file=str(self.path), duration=duration, pps=20000, ready_timeout=1, library="fake")
            h.run_play(args, sdk_factory=Stopper)
            self.assertEqual(FakeSDK.instances[-1].stops, [0])
        for duration in (0, -5, math.nan):
            args = SimpleNamespace(file=str(self.path), duration=duration, pps=20000, ready_timeout=1, library="fake")
            with self.assertRaisesRegex(h.CliError, "duration"):
                h.run_play(args, sdk_factory=FakeSDK)

    def test_cli_devices_checks_open_and_cleanup_results(self):
        class CliSDK(FakeSDK):
            def __init__(self, lib):
                super().__init__(lib)
                self.open_result = -7
                self.stop_result = 1
                self.close_result = 1
                if hasattr(CliSDK, "default_open"):
                    self.open_result = CliSDK.default_open
                    self.stop_result = -4
                    self.close_result = -5
            def open(self): return self.open_result
            def stop(self, dev):
                self.stops.append(dev)
                return self.stop_result
            def close(self):
                self.closed += 1
                return self.close_result
        original = h.HeliosSDK
        h.HeliosSDK = CliSDK
        try:
            import contextlib
            with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    h.main(["devices", "--library", "fake"])
                err.seek(0)
                self.assertIn("USB Helios devices: error (-7)", err.read())
            sdk = CliSDK.instances[-1]
            self.assertEqual((sdk.stops, sdk.closed), ([], 1))
            CliSDK.instances.clear()
            CliSDK.default_open = 1
            with tempfile.TemporaryFile(mode="w+t") as out, tempfile.TemporaryFile(mode="w+t") as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    h.main(["devices", "--library", "fake"])
                err.seek(0)
                output = err.read()
                out.seek(0)
                output += out.read()
            self.assertIn("Stop(0) failed (-4)", output)
            self.assertIn("CloseDevices failed (-5)", output)
        finally:
            h.HeliosSDK = original

    def test_rejects_truncation_bad_end_marker_and_other_format(self):
        for payload in (make_ilda()[:-1], make_ilda()[:-32], make_ilda()[:7] + bytes([4]) + make_ilda()[8:]):
            self.path.write_bytes(payload)
            with self.assertRaises(h.CliError): h.parse_ilda5(self.path)

    def test_rejects_missing_last_point_and_invalid_blank_color(self):
        payload = bytearray(make_ilda())
        payload[32 + 8 + 4] = 0
        self.path.write_bytes(payload)
        with self.assertRaisesRegex(h.CliError, "last-point"):
            h.parse_ilda5(self.path)
        payload = bytearray(make_ilda())
        payload[32 + 5] = 99
        self.path.write_bytes(payload)
        with self.assertRaisesRegex(h.CliError, "blanked point"):
            h.parse_ilda5(self.path)


if __name__ == "__main__": unittest.main()
