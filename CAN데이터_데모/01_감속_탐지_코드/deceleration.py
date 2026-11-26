"""Build and replay the personal deceleration rule from synthetic CAN."""

import argparse
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

MAX_SPEED_GAP = 0.150001
SPEED_BANDS = (35, 50, 65)
BAND_RADIUS = 5
MIN_EVENTS_PER_BAND = 4
MARGIN_M_S2 = 0.45
SIGNALS = {
    "Speed": (0x102, 56, 8, False, 1.0),
    "SAS_angleVal": (0x125, 24, 16, True, 0.1),
}


def read_can(path, names=("Speed",)):
    import can

    path = Path(path)
    reader = {".blf": can.BLFReader, ".asc": can.ASCReader}.get(path.suffix.lower())
    if reader is None:
        raise ValueError("Expected a BLF or ASC file")
    records = {name: [] for name in names}
    channels = set()
    origin = None
    with reader(path) as frames:
        for frame in frames:
            if origin is None:
                origin = frame.timestamp
            if frame.is_extended_id:
                continue
            for name in names:
                identifier, start, length, signed, factor = SIGNALS[name]
                if frame.arbitration_id != identifier:
                    continue
                if len(frame.data) * 8 < start + length:
                    raise ValueError(f"Incomplete CAN frame for {name}")
                raw = int.from_bytes(frame.data, "little") >> start & ((1 << length) - 1)
                if signed and raw & (1 << (length - 1)):
                    raw -= 1 << length
                records[name].append((round(frame.timestamp - origin, 6), raw * factor))
                channels.add(frame.channel)
    if len(channels) != 1 or not all(records.values()):
        raise ValueError(f"Missing signals or ambiguous CAN channels in {path}")
    for stream in records.values():
        if any(a[0] >= b[0] for a, b in zip(stream, stream[1:])):
            raise ValueError("CAN signal timestamps must be strictly increasing")
    return records


def interpolate(times, values, time):
    i = bisect_left(times, time)
    if i < len(times) and times[i] == time:
        return values[i]
    if not 0 < i < len(times):
        raise ValueError("Interpolation requires surrounding samples")
    fraction = (time - times[i - 1]) / (times[i] - times[i - 1])
    return values[i - 1] + fraction * (values[i] - values[i - 1])


def valid_events(samples, intervals, max_gap, deceleration=False):
    times, values = map(list, zip(*samples))
    events = []
    previous_end = 0.0
    for start, end in intervals:
        if not 0 <= previous_end <= start < end < float("inf"):
            raise ValueError("Expected ordered, non-overlapping event windows")
        previous_end = end
        left, right = bisect_left(times, start), bisect_right(times, end)
        if (right - left < 2 or times[left] - start > max_gap or end - times[right - 1] > max_gap
                or any(times[i] - times[i - 1] > max_gap for i in range(left + 1, right))):
            continue
        if deceleration:
            if round(times[right - 1] - times[left], 6) < 1:
                continue
            deltas = ((values[i] - interpolate(times, values, round(times[i] + 1, 6))) / 3.6
                      for i in range(left, right) if round(times[i] + 1, 6) <= times[right - 1])
            feature = max(0.0, *deltas)
        else:
            feature = max(values[left:right]) - min(values[left:right])
        events.append({"end_s": end, "entry_kmh": values[left], "value": feature})
    return events


def nearest_band(bands, speed):
    # Choose the lower band on an exact tie, as in the fixed prototype.
    key = min(bands, key=lambda band: (abs(band - speed), band), default=None)
    return key if key is not None and abs(key - speed) <= BAND_RADIUS else None


def build_baseline(samples, windows):
    valid = valid_events(samples, windows, MAX_SPEED_GAP, deceleration=True)
    groups = {band: [] for band in SPEED_BANDS}
    for event in valid:
        band = nearest_band(SPEED_BANDS, event["entry_kmh"])
        if band is not None:
            groups[band].append(event["value"])
    return {str(band): {
        "observations": len(peaks),
        "threshold_m_s2": max(peaks) + MARGIN_M_S2 if len(peaks) >= MIN_EVENTS_PER_BAND else None,
    } for band, peaks in groups.items()}


def detect_deceleration(samples, baseline):
    times, speeds = map(list, zip(*samples))
    bands = {int(key): row["threshold_m_s2"] for key, row in baseline.items()}
    latest_gap = 0
    active = False
    band = None
    alerts, unmatched = [], 0
    for i, time in enumerate(times):
        if i and time - times[i - 1] > MAX_SPEED_GAP:
            latest_gap = i
            active = False
            band = None
        target = round(time - 1, 6)
        if target < times[latest_gap]:
            continue
        earlier_speed = interpolate(times, speeds, target)
        decel = (earlier_speed - speeds[i]) / 3.6
        threshold = None
        if decel <= 0:
            band = None
        else:
            if band is None:
                band = nearest_band(bands, earlier_speed)
            threshold = bands.get(band)
        if decel > 0 and threshold is None:
            unmatched += 1
        strong = threshold is not None and decel > threshold
        if strong and not active:
            alerts.append({"time_s": time, "entry_band_kmh": band,
                           "decel_m_s2": round(decel, 4), "threshold_m_s2": round(threshold, 4)})
        active = strong
    return {"alerts": alerts, "decelerating_samples_without_baseline": unmatched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    fit = modes.add_parser("fit", help="Build a personal baseline from known brake windows")
    fit.add_argument("can_log", type=Path)
    fit.add_argument("windows_json", type=Path)
    fit.add_argument("output_json", type=Path)
    replay = modes.add_parser("replay", help="Detect strong deceleration using a fixed baseline")
    replay.add_argument("can_log", type=Path)
    replay.add_argument("baseline_json", type=Path)
    args = parser.parse_args()
    speed = read_can(args.can_log)["Speed"]
    if args.mode == "fit":
        windows = json.loads(args.windows_json.read_text(encoding="utf-8"))
        baseline = build_baseline(speed, windows["windows_s"]["brake"])
        args.output_json.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(baseline, indent=2))
    else:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        print(json.dumps(detect_deceleration(speed, baseline), indent=2))


if __name__ == "__main__":
    main()
