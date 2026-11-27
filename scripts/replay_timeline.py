#!/usr/bin/env python3
"""웹보드 판정 재생(LocalReplay, Pi 와 같은 경로)을 화면 없이 끝까지 돌려 판정 단계 시각표를 뽑는다.

기준 기록으로 기준을 계산·저장한 뒤, 판정 기록을 가장 빠른 속도로 재생해 ai.decision 을 모은다.
시각은 판정 기록의 원본 시각(recording.json 의 source_start_s 를 더함)으로 바꿔 적는다.

예)
  python scripts/replay_timeline.py --recordings-dir outputs/local-recordings/demo_1118 \
      --baseline 1118_baseline --replay 1118_full --rpeak-threshold rolling --out out/timeline_full.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from driver_state.local_replay import LocalReplay  # noqa: E402
from driver_state.web_dashboard import Dashboard  # noqa: E402
from driver_state.webboard_library import BaselineLibrary  # noqa: E402


class Recorder(Dashboard):
    def __init__(self):
        super().__init__()
        self.decisions = []

    def ingest(self, data):
        for event in data if isinstance(data, list) else [data]:
            if event.get("type") == "ai.decision":
                self.decisions.append(event)
        return super().ingest(data)


def find(local, folder_name):
    for item in local.catalog.refresh()["recordings"]:
        if Path(item["path"]).name == folder_name:
            return item["id"]
    raise SystemExit(f"기록 폴더 {folder_name} 이 목록에 없습니다")


def wait(local, stages, timeout=3600):
    began = time.monotonic()
    while time.monotonic() - began < timeout:
        status = local.status()
        if status["stage"] in stages:
            return status
        if status["stage"] == "error":
            raise SystemExit(f"재생 오류: {status['error']}")
        time.sleep(.2)
    raise SystemExit("시간 초과")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recordings-dir", type=Path, required=True)
    p.add_argument("--baseline", required=True, help="기준 계산에 쓸 기록 폴더 이름")
    p.add_argument("--replay", required=True, nargs="+", help="판정 재생할 기록 폴더 이름(여러 개 가능)")
    p.add_argument("--rpeak-threshold", choices=("global", "rolling"), default="rolling")
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--can-sim-scenario", help="판정 재생에 붙일 시뮬레이터 CAN 기록 (선택)")
    p.add_argument("--out", type=Path, required=True, help="CSV 출력 폴더")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    options = {}
    if a.can_sim_scenario:
        from driver_state.can_assist import build_can_assist
        options["can_assist_factory"] = lambda: build_can_assist(a.can_sim_scenario)
    with tempfile.TemporaryDirectory() as tmp:
        library = BaselineLibrary(Path(tmp) / "library")
        engine = Recorder()
        local = LocalReplay(engine, library, a.recordings_dir, model_dir=a.models, rpeak_threshold=a.rpeak_threshold,
                            **options)
        try:
            user = library.add_user("timeline")
            local.select(user["id"], find(local, a.baseline))
            local.start("calibrate", 0)
            status = wait(local, ("baseline_ready",))
            print(f"기준 완료: {a.baseline} 유효 창 {status['clean_windows']} (재생 위치 {status['position_s']:.0f}s)")
            saved = local.save(user["id"], "timeline", status["session_id"])
            for name in a.replay:
                local.select(user["id"], find(local, name))
                local.apply(user["id"], saved["id"])
                manifest = json.loads((a.recordings_dir / name / "recording.json").read_text(encoding="utf-8"))
                offset = float(manifest.get("source_start_s", 0.))
                engine.decisions.clear()
                local.start("replay", 0)
                wait(local, ("completed",))
                path = a.out / f"timeline_{name}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    w = csv.writer(handle)
                    w.writerow(["clip_t_s", "source_t_s", "level", "level_name", "candidate", "score", "p_used",
                                "channel", "frozen", "can_drowsy", "confirmation_source"])
                    for e in engine.decisions:
                        pl = e["payload"]
                        out = pl.get("output")
                        if out is None:
                            continue
                        w.writerow([out["t"], round(out["t"] + offset, 1), out["level"], out["level_name"],
                                    int(out["candidate"]), out["score"], out["p_used"], out["channel"],
                                    int(out["frozen"]), pl["input"].get("can_drowsy"), pl.get("confirmation_source")])
                print(f"판정 재생 완료: {name} → {path}")
        finally:
            local.close()
            library.close()


if __name__ == "__main__":
    main()
