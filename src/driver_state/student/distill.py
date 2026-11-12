"""학생 증류 (docs/student-f2c-stage1.md 2~3장).

조건
  S1-E  : ECG 파형 인코더 g → ẑ_nw (32) + ECG RR z (36) → 교사 W 로 ‖W z‖
  S1-EP : ECG·PPG 2채널 인코더 (채널 드롭아웃) + ECG RR z
  S1-P  : PPG 인코더 + PRV z
  S2-E  : ECG 인코더 → 기준 차감 → MLP → logit (점수만 증류)

fold (3장, 4 fold 로 묶음): fold fi 는 교사 데이터셋 TEACHER_DATASETS[fi] 와 학생 데이터(CLAS·MAUS·WESAD)
참가자 묶음 fi 를 학습에서 뺀다. 학생 데이터 목표는 fold 교사 (lodo_<test>) 로 다시 매긴 logit.
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..teacher.pairs import TEACHER_DATASETS

CONDS = {
    "S1-E": dict(ch=("ecg",), rr="ecg", head="dist"),
    "S1-EP": dict(ch=("ecg", "ppg"), rr="ecg", head="dist"),
    "S1-P": dict(ch=("ppg",), rr="prv", head="dist"),
    "S2-E": dict(ch=("ecg",), rr=None, head="mlp"),
}
HP = dict(steps=3000, batch=256, n_ref=4, lr=1e-3, lr_w=1e-4, wd=1e-4, lam=0.5, noise=0.05, ch_drop=0.3)
CH = {"ecg": 0, "ppg": 1}
STUDENT_SETS = ("clas", "maus", "wesad")


def n_folds() -> int:
    return len(TEACHER_DATASETS)


def group_fold(groups: pd.Series, n: int) -> np.ndarray:
    """참가자 → 묶음 (이름 해시, 데이터셋마다 고르게)."""
    return groups.map(lambda g: zlib.crc32(str(g).encode()) % n).to_numpy()


class Encoder(nn.Module):
    def __init__(self, c_in: int, d_out: int = 32):
        super().__init__()
        chs = [c_in, 16, 32, 64, 64]
        layers = []
        for a, b in zip(chs[:-1], chs[1:]):
            layers += [nn.Conv1d(a, b, 7, stride=2, padding=3), nn.BatchNorm1d(b), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(64, d_out)

    def forward(self, x):
        return self.fc(self.conv(x).mean(-1))


class Student(nn.Module):
    def __init__(self, c_in: int, head: str, W: np.ndarray | None, alpha: float, b: float, d_rr: int):
        super().__init__()
        self.g = Encoder(c_in)
        self.head, self.d_rr = head, d_rr
        if head == "dist":
            self.W = nn.Parameter(torch.as_tensor(W, dtype=torch.float32))
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
            self.b = nn.Parameter(torch.tensor(float(b)))
        else:
            self.mlp = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))

    def logit(self, zr, dz):
        if self.head == "dist":
            z = torch.cat([zr, dz], 1) if self.d_rr else dz
            if not self.d_rr:                                 # W 의 NW 열만 (RR 없음)
                v = dz @ self.W[:, -32:].T
            else:
                v = z @ self.W.T
            s = torch.sqrt((v ** 2).sum(-1) + 1e-8) - 1e-4
            return nn.functional.softplus(self.alpha) * s + self.b
        return self.mlp(dz).squeeze(-1)


def fold_logits(params: dict, tag: str, Z: np.ndarray) -> np.ndarray:
    """저장된 교사 파라미터 (시드 평균 logit)."""
    out = []
    i = 0
    while f"{tag}/s{i}/W" in params:
        W, a, b = params[f"{tag}/s{i}/W"], float(params[f"{tag}/s{i}/alpha"]), float(params[f"{tag}/s{i}/b"])
        s = np.sqrt(((Z @ W.T) ** 2).sum(1) + 1e-8) - 1e-4
        out.append(np.log1p(np.exp(a)) * s + b)
        i += 1
    return np.mean(out, 0)


class Data:
    def __init__(self, lab, index, wave, ref_index, ref_wave, prv_z: np.ndarray | None, rr_names, params):
        self.lab, self.wave, self.ref_wave = lab, wave, ref_wave
        self.pk = index["profile_key"].to_numpy()
        self.has = index[["has_ecg", "has_ppg"]].to_numpy(bool)
        self.zr = lab[[f"zr_{n}" for n in rr_names]].to_numpy(np.float32)
        self.zn = lab[[c for c in lab.columns if c.startswith("zn_")]].to_numpy(np.float32)
        self.prv = prv_z
        self.params = params
        self.ds = lab["dataset_id"].to_numpy()
        self.is_student = np.isin(self.ds, STUDENT_SETS)
        self.gfold = group_fold(lab["group"], n_folds())
        rh = ref_index[["has_ecg", "has_ppg"]].to_numpy(bool)
        self.refs: dict[tuple, np.ndarray] = {}
        for need in ((True, False), (False, True), (True, True)):
            ok = np.ones(len(ref_index), bool)
            for c in range(2):
                if need[c]:
                    ok &= rh[:, c]
            d = pd.Series(np.flatnonzero(ok)).groupby(ref_index["profile_key"].to_numpy()[ok]).apply(np.array).to_dict()
            self.refs[need] = d

    def target(self, fold: int) -> np.ndarray:
        """교사 데이터: LODO oof logit, 학생 데이터: fold 교사로 다시 매김."""
        y = self.lab["logit"].to_numpy(np.float32).copy()
        Z = np.concatenate([self.zr, self.zn], 1)
        tag = f"lodo_{TEACHER_DATASETS[fold]}"
        y[self.is_student] = fold_logits(self.params, tag, Z[self.is_student])
        return y


def weights(ds: np.ndarray, grp: np.ndarray) -> np.ndarray:
    n_ds = len(np.unique(ds))
    per = pd.Series(grp).map(pd.Series(grp).value_counts()).to_numpy(float)
    subj = pd.Series(ds).map(pd.Series(grp).groupby(ds).nunique()).to_numpy(float)
    return 1.0 / (n_ds * subj * per)


def _need(cfg, use_ch):
    return tuple(c in use_ch for c in ("ecg", "ppg"))


def train_eval(D: Data, cond: str, fold: int, seed: int, device: str, log=print, hp=None):
    hp = {**HP, **(hp or {})}
    cfg = CONDS[cond]
    chs = [CH[c] for c in cfg["ch"]]
    final = fold < 0                                          # 최종 학생: 전체 데이터, 교사 "final"
    test = None if final else TEACHER_DATASETS[fold]
    target = D.lab["logit"].to_numpy(np.float32).copy() if final else D.target(fold)
    need = _need(cfg, cfg["ch"])
    has_any = np.zeros(len(D.lab), bool)
    for c in cfg["ch"]:
        has_any |= D.has[:, CH[c]]
    if cond == "S1-EP":
        need_ref = (True, False)                              # 기준 창은 ECG 필수, PPG 는 있으면
    else:
        need_ref = need
    has_ref = pd.Series(D.pk).isin(set(D.refs[need_ref])).to_numpy()
    rr = None
    if cfg["rr"] == "ecg":
        rr = D.zr
    elif cfg["rr"] == "prv":
        rr = D.prv
    usable = has_any & has_ref & np.isfinite(target)
    if final:
        tr, ev = usable.copy(), np.zeros(len(usable), bool)
    else:
        tr = usable & (D.ds != test) & ~(D.is_student & (D.gfold == fold))
        ev = usable & ((D.ds == test) | (D.is_student & (D.gfold == fold)))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 101)
    pr = D.params
    tag = "final/s0" if final else f"lodo_{test}/s0"
    d_rr = 0 if rr is None else rr.shape[1]
    W = pr[f"{tag}/W"]
    m = Student(len(chs), cfg["head"], W, float(pr[f"{tag}/alpha"]), float(pr[f"{tag}/b"]), d_rr).to(device)
    enc = [p for n, p in m.named_parameters() if n.startswith(("g.", "mlp."))]
    oth = [p for n, p in m.named_parameters() if not n.startswith(("g.", "mlp."))]
    opt = torch.optim.AdamW([{"params": enc, "lr": hp["lr"]}, {"params": oth, "lr": hp["lr_w"]}], weight_decay=hp["wd"])
    idx = np.flatnonzero(tr)
    w = weights(D.ds[idx], D.lab["group"].to_numpy()[idx])
    p = w / w.sum()
    p_t = 1 / (1 + np.exp(-target))
    refs = D.refs[need_ref]

    def batch(rows, train: bool):
        x = torch.as_tensor(D.wave[rows][:, chs].astype(np.float32))
        R = []
        for r in rows:
            cand = refs[D.pk[r]]
            R.append(D.ref_wave[rng.choice(cand, hp["n_ref"])][:, chs])
        R = torch.as_tensor(np.stack(R).astype(np.float32))           # (B, n_ref, C, L)
        if train:
            x = x + hp["noise"] * torch.randn_like(x)
            R = R + hp["noise"] * torch.randn_like(R)
        if len(chs) == 2:
            M = torch.ones(len(rows), 2)
            hp_ = torch.as_tensor(D.has[rows, 1])
            M[:, 1] = hp_.float()
            if train:
                u = torch.rand(len(rows))
                both = hp_
                M[(u < hp["ch_drop"]) & both, 1] = 0.0                  # PPG 끔
                M[(u >= hp["ch_drop"]) & (u < 2 * hp["ch_drop"]) & both, 0] = 0.0   # ECG 끔
            x = x * M[:, :, None]
            R = R * M[:, None, :, None]
        return x, R

    def forward(rows, train):
        x, R = batch(rows, train)
        B, K = R.shape[:2]
        x, R = x.to(device), R.to(device)
        gx = m.g(x)
        gr = m.g(R.reshape(B * K, *R.shape[2:])).reshape(B, K, -1).mean(1)
        dz = gx - gr
        zr_ = torch.as_tensor(rr[rows]).to(device) if rr is not None else None
        return dz, m.logit(zr_, dz)

    t0 = time.perf_counter()
    m.train()
    zn_t = torch.as_tensor(D.zn)
    for step in range(hp["steps"]):
        rows = rng.choice(idx, hp["batch"], p=p)
        dz, lg = forward(rows, True)
        yt = torch.as_tensor(p_t[rows], dtype=torch.float32, device=device)
        l_score = nn.functional.binary_cross_entropy_with_logits(lg, yt)
        if cfg["head"] == "dist":
            l_feat = ((dz - zn_t[rows].to(device)) ** 2).mean()
            loss = l_feat + hp["lam"] * l_score
        else:
            loss = l_score
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            log(f"      {cond} f{fold} s{seed} step {step} loss {float(loss):.4f} score {float(l_score):.4f}")
    m.eval()

    if final:
        log(f"    {cond} 최종 seed {seed}: 학습 {len(idx)} 창 ({time.perf_counter() - t0:.0f}s)")
        return {}, m
    # 평가: 기준 = 기준 창 g 의 중앙값 (전부)
    out = {}
    modes = {"": None}
    if cond == "S1-EP":
        modes = {"": None, "_ecg": 0, "_ppg": 1}               # 한 채널만 넣은 평가
    with torch.no_grad():
        for suf, only in modes.items():
            rows_all = np.flatnonzero(ev)
            if only == 1:
                rows_all = rows_all[D.has[rows_all, 1]]
            gref = {}
            keys = pd.unique(D.pk[rows_all])
            for k in keys:
                cand = refs.get(k)
                Rw = torch.as_tensor(D.ref_wave[cand][:, chs].astype(np.float32))
                if len(chs) == 2:
                    M = torch.tensor([1.0 if only in (None, 0) else 0.0, 1.0 if only in (None, 1) else 0.0])
                    Rw = Rw * M[None, :, None]
                gref[k] = m.g(Rw.to(device)).median(0).values
            res = np.full(len(D.lab), np.nan, np.float32)
            for i in range(0, len(rows_all), 1024):
                rows = rows_all[i:i + 1024]
                x = torch.as_tensor(D.wave[rows][:, chs].astype(np.float32))
                if len(chs) == 2:
                    M = torch.ones(len(rows), 2)
                    M[:, 1] = torch.as_tensor(D.has[rows, 1]).float()
                    if only == 0:
                        M[:, 1] = 0
                    if only == 1:
                        M[:, 0] = 0
                    x = x * M[:, :, None]
                gx = m.g(x.to(device))
                gr = torch.stack([gref[k] for k in D.pk[rows]])
                zr_ = torch.as_tensor(rr[rows]).to(device) if rr is not None else None
                res[rows] = m.logit(zr_, gx - gr).cpu().numpy()
            out[f"s_{cond}{suf}"] = res
    out["target"] = np.where(ev, target, np.nan).astype(np.float32)
    log(f"    {cond} fold {test} seed {seed}: 학습 {len(idx)} 창, 평가 {int(ev.sum())} 창 ({time.perf_counter() - t0:.0f}s)")
    return out, m


def n_params(cond: str) -> int:
    c = CONDS[cond]
    m = Student(len(c["ch"]), c["head"], np.zeros((8, 68 if c["rr"] else 32)), 0.0, 0.0, 36 if c["rr"] else 0)
    return sum(p.numel() for p in m.parameters())
