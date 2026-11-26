"""Replay the Seoul collection demo and summarize the original 300-run results."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01_감속_탐지_코드"))

from deceleration import MAX_SPEED_GAP, build_baseline, detect_deceleration, read_can, valid_events

DATA = HERE / "samples"
REQUIRED = 12
DEMO_CASES = ("downtown_1900000", "teheran_1910005", "dmc_1920000")


def replay(name):
    metadata = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    signals = read_can(DATA / f"{name}.blf", ("Speed", "SAS_angleVal"))
    windows = metadata["windows_s"]
    straight = valid_events(signals["SAS_angleVal"], windows["straight"], 0.015001)
    brake = valid_events(signals["Speed"], windows["brake"], MAX_SPEED_GAP, deceleration=True)
    twelfth = [events[REQUIRED - 1]["end_s"] if len(events) >= REQUIRED else None
               for events in (straight, brake)]
    ready = max(twelfth) if None not in twelfth else None
    counts = [sum(event["end_s"] <= 900 for event in events) for events in (straight, brake)]
    if ready != metadata["ready_at_s"] or counts != [metadata["straight_15min"], metadata["brake_15min"]]:
        raise ValueError(f"Stored and recomputed results differ for {name}")
    baseline = None
    if ready is not None:
        baseline = {
            "steering_range_median_deg": round(median(e["value"] for e in straight if e["end_s"] <= ready), 4),
            "max_1s_decel_median_m_s2": round(median(e["value"] for e in brake if e["end_s"] <= ready), 4),
        }
    return {"case": name, "course": metadata["course"], "ready_at_s": ready,
            "checkpoint_15min": {"straight": counts[0], "decel": counts[1],
                                "missing_straight": max(0, REQUIRED - counts[0]),
                                "missing_decel": max(0, REQUIRED - counts[1]),
                                "progress_percent": round(100 * sum(min(n, REQUIRED) for n in counts)
                                                          / (2 * REQUIRED), 2)},
            "collection_demo_features": baseline}


def summarize():
    with (DATA / "run_results.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 300 or len({(r["course"], r["seed"]) for r in rows}) != 300:
        raise ValueError("Expected 300 distinct original runs")
    courses = {}
    for course in sorted({row["course"] for row in rows}):
        selected = [row for row in rows if row["course"] == course]
        completed = sorted(float(r["ready_at_s"]) for r in selected if r["ready_at_s"])
        target = math.ceil(0.95 * len(selected))
        courses[course] = {"runs": len(selected), "ready_by_15min": sum(t <= 900 for t in completed),
                           "first_at_95_percent_s": completed[target - 1] if len(completed) >= target else None}
    reach = [row["first_at_95_percent_s"] for row in courses.values()]
    return {"source": "original SUMO result table", "courses": courses,
            "all_courses_at_95_percent_s": max(reach) if None not in reach else None,
            "total_ready_by_15min": sum(row["ready_by_15min"] for row in courses.values())}


def replay_deceleration(name):
    metadata = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    speed = read_can(DATA / f"{name}.blf")["Speed"]
    baseline_windows = [(start, end) for start, end in metadata["windows_s"]["brake"] if end <= 900]
    baseline = build_baseline(speed, baseline_windows)
    evaluation = [sample for sample in speed if sample[0] >= 900]
    return {"case": name, "baseline_end_s": 900, "evaluation_end_s": evaluation[-1][0],
            "baseline": baseline, **detect_deceleration(evaluation, baseline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "summary", "decel"))
    parser.add_argument("--case", choices=DEMO_CASES)
    args = parser.parse_args()
    if args.mode == "summary" and args.case:
        parser.error("--case is only available in replay or decel mode")
    if args.mode == "summary":
        result = summarize()
    else:
        function = replay if args.mode == "replay" else replay_deceleration
        result = [function(name) for name in ((args.case,) if args.case else DEMO_CASES)]
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
