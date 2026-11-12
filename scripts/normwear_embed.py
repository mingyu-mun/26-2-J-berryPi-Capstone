"""NormWear 표현 추출 (docs/teacher-normwear-stage1.md 1장). GPU 서버에서 실행.

    python scripts/normwear_embed.py --normwear-dir ~/jberry --ckpt ~/jberry/normwear_ckpt/model.safetensors \
        --data ~/jberry/data/converted --cache ~/jberry/data/teacher_cache --student ~/jberry/data/cache_student \
        --out ~/jberry/out/normwear

결과: <out>/emb_<dataset>.npz (row_id, emb[N,768] float16), <out>/ref_<dataset>.npz (profile_key, emb[K,768] 평균 전 각 창)
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
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

FS_NW, SEG_SEC, WIN_SEC, N_SEG = 65, 6.0, 30.0, 5


def prep_ecg(x: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(3, [0.5, min(40.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64))
    from fractions import Fraction
    fr = Fraction(FS_NW / fs).limit_denominator(1000)
    return resample_poly(y, fr.numerator, fr.denominator).astype(np.float32)


def segments(sig65: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """창 시작 [s] → (n, N_SEG, L) 6초 구간, 창마다 z-score. 범위를 벗어나면 0."""
    L = int(round(SEG_SEC * FS_NW))
    out = np.zeros((len(starts), N_SEG, L), np.float32)
    for i, t in enumerate(starts):
        a = int(round(t * FS_NW))
        w = sig65[a:a + N_SEG * L]
        if len(w) < N_SEG * L:
            continue
        w = (w - w.mean()) / (w.std() + 1e-6)
        out[i] = w.reshape(N_SEG, L)
    return out


class Embedder:
    def __init__(self, nw_dir: str, ckpt: str, device: str = "cuda"):
        import torch
        from safetensors.torch import load_file
        sys.path.insert(0, nw_dir)
        from NormWear.main_model import NormWearModel
        self.torch = torch
        self.dev = torch.device(device)
        m = NormWearModel(weight_path="", optimized_cwt=True)
        sd = load_file(ckpt)
        bk = m.backbone.state_dict()
        missing, unexpected = m.backbone.load_state_dict({k: v for k, v in sd.items() if k in bk}, strict=False)
        assert not missing, missing
        self.m = m.to(self.dev).eval()

    def __call__(self, seg: np.ndarray, bs: int = 256) -> np.ndarray:
        """seg: (n, N_SEG, L) → (n, 768): 구간마다 패치 평균, 구간 평균."""
        n = len(seg)
        flat = seg.reshape(n * N_SEG, 1, -1)
        outs = []
        with self.torch.no_grad():
            for i in range(0, len(flat), bs):
                e = self.m.get_embedding(flat[i:i + bs], sampling_rate=FS_NW, device=self.dev)   # b,1,P,768
                outs.append(e.mean(2)[:, 0].float().cpu().numpy())
        return np.concatenate(outs).reshape(n, N_SEG, -1).mean(1).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normwear-dir", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--student", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = TeacherSpec()
    emb = Embedder(a.normwear_dir, a.ckpt)
    tables = {"teacher": (pd.read_parquet(Path(a.cache) / "windows.parquet"), load_rr_cache(Path(a.cache) / "rr")),
              "student": (pd.read_parquet(Path(a.student) / "windows.parquet"), load_rr_cache(Path(a.student) / "rr"))}
    for src, (win, rrs) in tables.items():
        for ds in sorted(win["dataset_id"].unique()):
            if (a.datasets and ds not in a.datasets) or ds == "mand":
                continue
            if (out / f"emb_{ds}.npz").exists():
                print(f"{ds}: 있음, 건너뜀", flush=True)
                continue
            t0 = time.time()
            wd = win[win["dataset_id"] == ds]
            row_ids, embs, ref_keys, ref_embs = [], [], [], []
            groups: dict[str, list] = {}
            for r in rrs:
                if r.dataset_id == ds:
                    groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
            plans = {g: baseline_plan(s, spec) for g, s in groups.items()}
            for (subj, ses), g in wd.groupby(["subject_id", "session_id"]):
                d = Path(a.data) / ds / (subj or "anon") / ses
                rec = load_session(d, channels=("ecg",))
                sig = prep_ecg(rec.signals["ecg"], rec.fs["ecg"])
                E = emb(segments(sig, g["t_rel"].to_numpy(float)))
                row_ids.append(g["row_id"].to_numpy()); embs.append(E)
            # 기준 구간 창
            done = set()
            for gname, plan in plans.items():
                for info in plan.values():
                    if info is None:
                        continue
                    key = f"{gname}#{info.session_id}"
                    if key in done:
                        continue
                    done.add(key)
                    r = next(x for x in groups[gname] if x.session_id == info.session_id)
                    d = Path(a.data) / ds / (r.subject_id or "anon") / r.session_id
                    rec = load_session(d, channels=("ecg",))
                    sig = prep_ecg(rec.signals["ecg"], rec.fs["ecg"])
                    starts = np.arange(info.start, info.end - WIN_SEC + 1e-6, spec.stride_sec)
                    if len(starts) == 0:
                        starts = np.array([max(0.0, info.start)])
                    E = emb(segments(sig, starts))
                    ref_keys += [key] * len(E); ref_embs.append(E)
            np.savez_compressed(out / f"emb_{ds}.npz", row_id=np.concatenate(row_ids), emb=np.concatenate(embs))
            if ref_embs:
                np.savez_compressed(out / f"ref_{ds}.npz", profile_key=np.array(ref_keys), emb=np.concatenate(ref_embs))
            print(f"{ds}: 창 {sum(len(x) for x in row_ids)}개, 기준 창 {len(ref_keys)}개, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
