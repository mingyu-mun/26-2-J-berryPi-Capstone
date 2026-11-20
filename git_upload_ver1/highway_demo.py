"""Run a synthetic 100 km/h highway scenario through the live CAN receiver."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep, time

import can

from can_live import CAN_LIST_MAP_ID, SessionClock, can_list_decoder, json_line, receive
from steering_profile import COMPANY_SPEED_MAX_AGE_US


def frame_schedule(speed, steering):
    """Vehicle CAN list cadence with a steering error burst and a dropout."""
    for stamp in range(0, 5200, 10):
        if stamp % 100 == 0:
            speed_kmh = 99 if stamp % 1000 < 200 else 100
            yield stamp, speed, speed.encode({"Speed": speed_kmh})
        if not 4000 <= stamp < 4500:
            # Small steering corrections, a sustained turn, then a return to center.
            steer_deg = 5.0 if 1800 <= stamp < 3000 else (0, 0.4, 0, -0.6)[(stamp // 100) % 4]
            data = bytes.fromhex("00 00 00 FF 7F") if 3300 <= stamp < 3600 else steering.encode({"SAS_angleVal": steer_deg})
            yield stamp + 5, steering, data.ljust(8, b"\x00")


def run_highway(emit, blf_path=None):
    decoder = can_list_decoder()
    speed, _ = decoder.messages[(0x102, False)]
    steering, _ = decoder.messages[(0x125, False)]

    with ExitStack() as stack:
        listener = stack.enter_context(can.Bus(interface="virtual", channel="highway-demo",
                                               can_filters=decoder.filters()))
        sender = stack.enter_context(can.Bus(interface="virtual", channel="highway-demo",
                                             preserve_timestamps=True))
        log = stack.enter_context(can.BLFWriter(blf_path)) if blf_path is not None else None
        producer = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        clock = SessionClock()

        def send_frames():
            start = monotonic()
            for stamp, signal, data in frame_schedule(speed, steering):
                wait = start + stamp / 1000 - monotonic()
                if wait > 0:
                    sleep(wait)
                frame = can.Message(timestamp=time(), arbitration_id=signal.frame_id, is_extended_id=False,
                                    channel=0, data=data, is_fd=signal is steering)
                sender.send(frame)
                if log is not None:
                    log.on_message_received(frame)

        scheduled = producer.submit(send_frames)
        receive(listener, decoder, clock, emit, session_id="highway-demo-01", origin="simulated",
                signal_map_id=CAN_LIST_MAP_ID, duration_s=5.8, speed_max_age_us=COMPANY_SPEED_MAX_AGE_US)
        scheduled.result()
        if decoder.bad_frames:
            raise RuntimeError(f"Unexpected malformed demo frames: {decoder.bad_frames}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blf", type=Path, help="optional destination for the generated raw CAN recording")
    args = parser.parse_args()
    if args.blf is not None and args.blf.exists():
        parser.error("--blf requires a new output path")
    run_highway(lambda event: print(json_line(event), flush=True), args.blf)
