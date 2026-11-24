"""Compare the browser demo with the copied Python CAN and steering pipeline. Requires Node.js."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steering_profile import Assessment, NormalRecorder, build_profile, load_profile
from test_personal_pipeline import stream_for_test


def main():
    hold_times = [200, 520, 530, 650, 900, 910, 920, 1200, 1400]
    with TemporaryDirectory() as directory:
        data_dir = Path(directory)
        profiles = []
        for participant, normal_hold_ms in [("A", 200), ("B", 800)]:
            for session in ("normal-1", "normal-2"):
                with NormalRecorder(data_dir, participant, session) as recorder:
                    stream_for_test(32_000, lambda ms: 5.2 if 100 <= ms % 1500 < 100 + normal_hold_ms else 0,
                                    recorder.observe)
            build_profile(data_dir, participant, ("normal-1", "normal-2"))
            profiles.append(load_profile(data_dir, participant, "evaluation"))

        expected = []
        for hold_ms in hold_times:
            transitions = []
            for profile in profiles:
                source = []
                assessment = Assessment(profile, "evaluation", source.append)
                stream_for_test(1700, lambda ms: 5.2 if 100 <= ms < 100 + hold_ms else 0,
                                assessment.observe, assessment.expire)
                transitions.append([round(e["t_rel_s"] * 1000) for e in source if e["deviation_status"] == "deviation"])
            expected.append(transitions)

    script = """const s = require('./web/can-simulator.js');
    const results = JSON.parse(process.argv[1]).map(duration => s.run(duration));
    console.log(JSON.stringify({
      limits: results[0].profiles.map(p => p.limits[5]),
      transitions: results.map(r => r.transitions.map(events => events.filter(e => e.status === 'deviation').map(e => e.time)))
    }));"""
    process = subprocess.run(["node", "-e", script, json.dumps(hold_times)], cwd=ROOT,
                             check=True, capture_output=True, text=True)
    browser = json.loads(process.stdout)
    assert browser["limits"] == [round(profile.limits[5] * 1000) for profile in profiles], browser
    assert browser["transitions"] == expected, (browser["transitions"], expected)
    print(f"Python and browser results matched for {len(hold_times)} hold durations and two generated profiles.")


if __name__ == "__main__":
    main()
