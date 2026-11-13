"""학생 입력 파형 캐시 (docs/student-f2c-stage1.md 2.2). GPU 서버에서 실행.

    python scripts/student_cache.py --labels out/f2c_label/f2c_labels.parquet --data data/converted \
        --cache data/teacher_cache --student data/cache_student --out data/student_wave

결과:
  wave.npy      (N, 2, 1920) float16  — 라벨 표 순서, ch0 ECG (0.5–40 Hz), ch1 PPG (0.5–8 Hz), 64 Hz, 창마다 z-score
  ref_wave.npy  (K, 2, 1920) float16  — 기준 구간 창
  index.parquet (라벨 표 행 순서: dataset_id, row_id, profile_key, has_ecg, has_ppg)
  ref_index.parquet (profile_key, has_ecg, has_ppg)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from driver_state.data_io import load_session  # noqa: E402
from driver_state.teacher.features import baseline_plan  # noqa: E402
from driver_state.teacher.rr import load_rr_cache  # noqa: E402
from driver_state.teacher.spec import TeacherSpec  # noqa: E402

FS, WIN = 64, 30.0
L = int(FS * WIN)
BANDS = {"ecg": (0.5, 25.0), "ppg": (0.5, 8.0)}     # 64 Hz 나이퀴스트 32 Hz 아래로
MAX_REF = 30


def prep(x: np.ndarray, fs: float, ch: str) -> np.ndarray:
    lo, hi = BANDS[ch]
    sos = butter(3, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, np.float64))
    fr = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(y, fr.numerator, fr.denominator).astype(np.float32)


def cut(sig: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(starts), L), np.float16)
    ok = np.zeros(len(starts), bool)
    for i, t in enumerate(starts):
        a = int(round(t * FS))
        w = sig[a:a + L]
        if a < 0 or len(w) < L:
            continue
        sd = w.std()
        if sd < 1e-6:
            continue
        out[i] = ((w - w.mean()) / sd).clip(-10, 10)
        ok[i] = True
    return out, ok


def session_signals(d: Path):
    try:
        rec = load_session(d, channels=("ecg", "ppg"))
    except Exception:
        rec = load_session(d, channels=("ecg",))
    return {ch: prep(rec.signals[ch], rec.fs[ch], ch) for ch in ("ecg", "ppg") if ch in rec.signals}


def main():
    ap = argparse.ArgumentParser()
    for k in ("labels", "data", "cache", "student", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = TeacherSpec()
    lab = pd.read_parquet(a.labels)
    wins = pd.concat([pd.read_parquet(Path(a.cache) / "windows.parquet"), pd.read_parquet(Path(a.student) / "windows.parquet")])
    wins = wins[["dataset_id", "row_id", "subject_id", "profile_key"]]
    idx = lab[["dataset_id", "row_id", "session_id", "t_rel"]].merge(wins, on=["dataset_id", "row_id"], how="left")
    assert len(idx) == len(lab) and idx["profile_key"].notna().all()
    N = len(idx)
    wave = np.lib.format.open_memmap(out / "wave.npy", mode="w+", dtype=np.float16, shape=(N, 2, L))
    has = np.zeros((N, 2), bool)
    t0 = time.time()
    pos = np.arange(N)
    for n_done, ((ds, subj, ses), g) in enumerate(idx.groupby(["dataset_id", "subject_id", "session_id"], dropna=False)):
        sig = session_signals(Path(a.data) / ds / (subj if isinstance(subj, str) and subj else "anon") / ses)
        rows = pos[g.index.to_numpy()]
        for c, ch in enumerate(("ecg", "ppg")):
            if ch in sig:
                w, ok = cut(sig[ch], g["t_rel"].to_numpy(float))
                wave[rows, c] = w
                has[rows, c] = ok
        if n_done % 50 == 0:
            print(f"  세션 {n_done}, {time.time() - t0:.0f}s", flush=True)
    wave.flush()
    idx["has_ecg"], idx["has_ppg"] = has[:, 0], has[:, 1]
    idx.drop(columns=["session_id", "t_rel"]).to_parquet(out / "index.parquet")
    print(f"창 {N}: ECG {has[:, 0].mean():.3f}, PPG {has[:, 1].mean():.3f}, {time.time() - t0:.0f}s", flush=True)

    # 기준 구간 창
    rrs = load_rr_cache(Path(a.cache) / "rr") + load_rr_cache(Path(a.student) / "rr")
    groups: dict[str, list] = {}
    for r in rrs:
        groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
    need = set(idx["profile_key"])
    rw, rk, rh = [], [], []
    for gname, lst in groups.items():
        for info in baseline_plan(lst, spec).values():
            if info is None:
                continue
            key = f"{gname}#{info.session_id}"
            if key not in need or key in rk:
                continue
            r = next(x for x in lst if x.session_id == info.session_id)
            sig = session_signals(Path(a.data) / r.dataset_id / (r.subject_id or "anon") / r.session_id)
            starts = np.arange(info.start, info.end - WIN + 1e-6, spec.stride_sec)[:MAX_REF]
            if len(starts) == 0:
                starts = np.array([max(0.0, info.start)])
            W = np.zeros((len(starts), 2, L), np.float16); H = np.zeros((len(starts), 2), bool)
            for c, ch in enumerate(("ecg", "ppg")):
                if ch in sig:
                    W[:, c], H[:, c] = cut(sig[ch], starts)
            rw.append(W); rh.append(H); rk += [key] * len(starts)
    np.save(out / "ref_wave.npy", np.concatenate(rw))
    H = np.concatenate(rh)
    pd.DataFrame({"profile_key": rk, "has_ecg": H[:, 0], "has_ppg": H[:, 1]}).to_parquet(out / "ref_index.parquet")
    print(f"기준 창 {len(rk)}, 프로필 {len(set(rk))} / 필요 {len(need)}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
