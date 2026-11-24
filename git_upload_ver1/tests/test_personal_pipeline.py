from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import can

from can_live import CAN_LIST_MAP_ID, SessionClock, can_list_decoder, receive
from steering_profile import Assessment, COMPANY_SPEED_MAX_AGE_US, NormalRecorder, build_profile, load_profile


class EndOfFrames(Exception):
    pass


class ReplayClock(SessionClock):
    def __init__(self):
        super().__init__(1_700_000_000)
        self.latest = 0

    def frame_us(self, timestamp):
        self.latest = super().frame_us(timestamp)
        return self.latest

    def now_us(self):
        return self.latest


def stream_for_test(duration_ms, angle_at, observer, expiry=None, speed_at=lambda _: 100):
    decoder = can_list_decoder()
    clock = ReplayClock()
    with can.Bus(interface="virtual", channel="personal-pipeline", can_filters=decoder.filters(),
                 ignore_config=True) as receiver, can.Bus(interface="virtual", channel="personal-pipeline",
                 preserve_timestamps=True, ignore_config=True) as sender:
        for elapsed in range(0, duration_ms, 10):
            timestamp = 1_700_000_000 + elapsed / 1000
            speed = speed_at(elapsed)
            if elapsed % 100 == 0 and speed is not None:
                data = bytes(7) + bytes([speed])
                sender.send(can.Message(timestamp=timestamp, arbitration_id=0x102, is_extended_id=False, data=data))
            angle = angle_at(elapsed)
            if angle is not None:
                raw = round(angle * 10).to_bytes(2, "little", signed=True)
                sender.send(can.Message(timestamp=timestamp + 0.005, arbitration_id=0x125, is_extended_id=False,
                                        is_fd=True, data=bytes(3) + raw + bytes(3)))

        class FiniteBus:
            def recv(self, timeout):
                frame = receiver.recv(0)
                if frame is None:
                    raise EndOfFrames
                return frame

        try:
            receive(FiniteBus(), decoder, clock, lambda event: None, session_id="test", origin="simulated",
                    signal_map_id=CAN_LIST_MAP_ID, speed_max_age_us=COMPANY_SPEED_MAX_AGE_US,
                    on_steering=observer, on_snapshot=expiry)
        except EndOfFrames:
            pass


class PersonalPipelineTest(unittest.TestCase):
    def test_same_live_input_differs_for_two_generated_profiles(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for participant, normal_hold_ms in [("A", 200), ("B", 800)]:
                for session in ("n1", "n2"):
                    with NormalRecorder(root, participant, session) as recorder:
                        stream_for_test(32_000, lambda ms: 5.2 if 100 <= ms % 1500 < 100 + normal_hold_ms else 0,
                                        recorder.observe)
                build_profile(root, participant, ("n1", "n2"))
            a, b = [load_profile(root, participant, "holdout") for participant in ("A", "B")]
            self.assertEqual(a.limits[5], 0.30)
            self.assertEqual(b.limits[5], 0.90)
            with self.assertRaisesRegex(ValueError, "distinct"):
                load_profile(root, "A", "n1")

            for participant, hits in [(a, 2), (b, 1)]:
                results = []
                assessment = Assessment(participant, "holdout", results.append)

                def angle(ms):
                    if 900 <= ms < 1100:
                        return 3276.7  # CAN raw 32767 is the company's steering error value.
                    if 2500 <= ms < 2800:
                        return None
                    if 100 <= ms < 750 or 1100 <= ms < 2300:
                        return 5.2
                    return 0

                stream_for_test(3700, angle, assessment.observe, assessment.expire,
                                speed_at=lambda ms: None if 2900 <= ms < 3150 else 100)
                deviations = [event for event in results if event["deviation_status"] == "deviation"]
                self.assertEqual(len(deviations), hits, results)
                self.assertTrue(all(event["detail"]["angle_band_deg"] == 5 for event in deviations))
                missing = [event["t_rel_s"] for event in results if event["deviation_status"] == "unavailable"]
                self.assertTrue(any(0.9 <= stamp < 1.1 for stamp in missing))
                self.assertTrue(any(2.9 <= stamp < 3.2 for stamp in missing))

    def test_incomplete_recording_cannot_create_profile(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError), NormalRecorder(root, "A", "interrupted"):
                raise RuntimeError("simulate failed capture")
            with self.assertRaisesRegex(ValueError, "complete recording"):
                build_profile(root, "A", ("interrupted", "normal-02"))


if __name__ == "__main__":
    unittest.main()
