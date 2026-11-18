"""AI 쪽 결과를 웹보드 규약 v1.2 이벤트로 바꿔 대시보드 서버(/api/events)로 보낸다.

web/dashboard/contract/WEBBOARD_DATA_CONTRACT.md 의 규칙을 따른다.
- 측정값을 만들지 않는다. 없는 값은 None (JSON null). NaN/Inf 도 None.
- ai.decision 의 output 은 실제 StateMachineV2.step() 이 돌려준 StepOut 을 그대로 직렬화한다.
- 기준이 준비되기 전에는 output = None.
- confirmation_source 는 증명할 수 있을 때만 채운다 (CAN 입력이 전혀 없을 때 졸음 확정 = "persistence").

사용 예 (실시간 루프 안에서):

    pub = WebboardPublisher("http://127.0.0.1:8000")
    pub.send(bio_wave(sid, t, ecg=wave_channel(t0, 100.0, ecg_chunk, "a.u.", "common_preprocessed", ok)))
    step = sm.step(win_in)
    pub.send(ai_decision(sid, win_in, step, baseline_status="ready", bio_origin="live"))
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable
from urllib.request import Request, urlopen

SCHEMA_VERSION = "1.2"
LEVEL_NAMES = {-1: "측정 불가", 0: "정상", 1: "주의", 2: "졸음"}


def clean(x: Any) -> Any:
    """비유한 수는 None, numpy 스칼라·배열은 파이썬 값으로."""
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def _opt(x, lo=None, hi=None):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


def envelope(kind: str, session_id: str, origin: str, t_rel_s: float, payload: dict) -> dict:
    return clean({"schema_version": SCHEMA_VERSION, "type": kind, "session_id": str(session_id),
                  "origin": origin, "t_rel_s": max(0.0, float(t_rel_s)), "payload": payload})


# ------------------------------------------------------------------ bio.wave
def wave_channel(t_start_rel_s: float, fs_hz: float, samples, unit: str, stage: str, quality_ok: bool | None) -> dict | None:
    """unit: mV / adc_count / a.u. / z_score, stage: common_preprocessed / student_model_input.
    비유한 표본이 있으면 채널 전체를 None 으로 둔다 (임의로 메우지 않음)."""
    vals = [float(v) for v in samples]
    if not vals or not all(math.isfinite(v) for v in vals):
        return None
    return {"t_start_rel_s": max(0.0, float(t_start_rel_s)), "fs_hz": float(fs_hz), "samples": vals,
            "unit": unit, "stage": stage, "quality_ok": quality_ok}


def bio_wave(session_id: str, t_rel_s: float, ecg: dict | None = None, ppg: dict | None = None,
             origin: str = "live") -> dict:
    return envelope("bio.wave", session_id, origin, t_rel_s, {"ecg": ecg, "ppg": ppg})


# ------------------------------------------------------------------ ai.decision
def window_in_dict(w) -> dict:
    """WindowIn (또는 같은 키의 dict) → 규약 input. 누락은 None 그대로."""
    d = asdict(w) if is_dataclass(w) else dict(w)
    return {"t": _opt(d.get("t"), 0), "p_ep": _opt(d.get("p_ep"), 0, 1), "q_ecg": _opt(d.get("q_ecg"), 0, 1),
            "p_ppg": _opt(d.get("p_ppg"), 0, 1), "q_ppg": _opt(d.get("q_ppg"), 0, 1),
            "hr_ecg": _opt(d.get("hr_ecg"), 0), "hr_ppg": _opt(d.get("hr_ppg"), 0),
            "ok_ecg": d.get("ok_ecg"), "ok_ppg": d.get("ok_ppg"), "can_drowsy": d.get("can_drowsy")}


def step_out_dict(s) -> dict | None:
    if s is None:
        return None
    d = asdict(s) if is_dataclass(s) else dict(s)
    return {"t": float(d["t"]), "level": int(d["level"]), "level_name": LEVEL_NAMES[int(d["level"])],
            "candidate": bool(d["candidate"]), "score": _opt(d.get("score"), 0, 1), "p_used": _opt(d.get("p_used"), 0, 1),
            "confidence": float(d["confidence"]), "channel": d["channel"], "frozen": bool(d["frozen"])}


def ai_decision(session_id: str, win_in, step, *, baseline_status: str, bio_origin: str = "live",
                can_origin: str = "unavailable", window_sec: float = 30.0,
                confirmation_source: str | None = None, origin: str | None = None) -> dict:
    inp = window_in_dict(win_in)
    out = step_out_dict(step) if baseline_status == "ready" else None
    t_end = out["t"] if out else inp["t"]
    if out is not None and out["level"] == 2 and confirmation_source is None and inp["can_drowsy"] is None \
            and can_origin == "unavailable":
        confirmation_source = "persistence"          # CAN 입력이 없으면 확정 경로는 지속 조건뿐
    if out is None or out["level"] != 2:
        confirmation_source = None
    payload = {"model_id": "S1-EP", "state_machine_id": "v2",
               "window_start_s": None if t_end is None else max(0.0, t_end - window_sec),
               "window_end_s": t_end, "baseline_status": baseline_status, "bio_origin": bio_origin,
               "can_origin": can_origin, "input": inp, "output": out, "confirmation_source": confirmation_source}
    return envelope("ai.decision", session_id, origin or bio_origin, t_end if t_end is not None else 0.0, payload)


# ------------------------------------------------------------------ system.status
def system_status(session_id: str, t_rel_s: float, *, bio_status: str, clean_windows: int | None,
                  required_clean_windows: int = 20, bio_elapsed_s: float | None = None,
                  can_status: str = "unavailable", can_coverage_pct: float | None = None,
                  can_elapsed_s: float | None = None, connections: dict | None = None,
                  origin: str = "live") -> dict:
    conn = {"ecg": "unknown", "ppg": "unknown", "can": "unknown", "model": "unknown"} | (connections or {})
    payload = {"bio_baseline": {"status": bio_status, "clean_windows": clean_windows,
                                "required_clean_windows": int(required_clean_windows), "elapsed_s": _opt(bio_elapsed_s, 0)},
               "can_baseline": {"status": can_status, "coverage_pct": _opt(can_coverage_pct, 0, 100),
                                "elapsed_s": _opt(can_elapsed_s, 0)},
               "connections": conn}
    return envelope("system.status", session_id, origin, t_rel_s, payload)


# ------------------------------------------------------------------ 전송
class WebboardPublisher:
    """POST /api/events. 한 번에 1,000개·2 MB 이하로 나눠 보낸다."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 3.0):
        self.url = base_url.rstrip("/") + "/api/events"
        self.timeout = timeout

    def send(self, events: dict | Iterable[dict]) -> list[dict]:
        batch = [events] if isinstance(events, dict) else list(events)
        results, chunk, size = [], [], 2
        for ev in batch:
            n = len(json.dumps(ev, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 1
            if chunk and (len(chunk) >= 1000 or size + n > 1_900_000):
                results.append(self._post(chunk)); chunk, size = [], 2
            chunk.append(ev); size += n
        if chunk:
            results.append(self._post(chunk))
        return results

    def _post(self, chunk: list[dict]) -> dict:
        body = json.dumps(chunk if len(chunk) > 1 else chunk[0], ensure_ascii=False, allow_nan=False).encode("utf-8")
        req = Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=self.timeout) as r:
            return json.load(r)
