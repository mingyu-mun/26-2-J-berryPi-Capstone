"""rrnet 인코더 + 방향 없는 거리 (docs/teacher-anomaly-stage1.md 초안 2, 조건 R1·R2·R3).

d_t = ‖W (g(x_t) − g_ref)‖₂ , g = rrnet CNN+BiGRU 인코더(64), g_ref = 그 참가자 기준 타코그램 임베딩 평균.
logit = softplus(α)·d + b.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import baseline_plan, tachogram
from .models.rrnet import make_input
from .rr import RRSeries, load_rr_cache
from .spec import TeacherSpec

HP = dict(lr=1e-3, wd=1e-4, batch=256, epochs=30, patience=5, shift_sec=10.0, k_ref_train=4, k_ref_eval=16,
          n_rpairs=64, lam_r=0.5, max_missing=0.4, val_frac=0.2)


# ---------------------------------------------------------------- 기준 타코그램 (2.2)
def reference_tachograms(rrs: list[RRSeries], spec: TeacherSpec, max_missing: float = 0.4):
    """profile_key → (k, 2, T) float16. 기준 구간이 300초보다 짧으면 중심 1개 (short=True)."""
    groups: dict[str, list[RRSeries]] = {}
    for r in rrs:
        groups.setdefault(f"{r.dataset_id}/{r.group}", []).append(r)
    out, short = {}, {}
    half = spec.tacho_sec / 2
    for g, ses in groups.items():
        plan = baseline_plan(ses, spec)
        by_id = {s.session_id: s for s in ses}
        for info in {(v.session_id, v.start, v.end) for v in plan.values() if v is not None}:
            sid, a, b = info
            key = f"{g}#{sid}"
            if key in out:
                continue
            rr = by_id[sid]
            if b - a >= spec.tacho_sec:
                centers = np.arange(a + half, b - half + 1e-6, spec.stride_sec)
                short[key] = False
            else:
                centers = np.array([(a + b) / 2])
                short[key] = True
            T = [tachogram(rr, c, spec) for c in centers]
            T = [t for t in T if t[1].mean() <= max_missing]
            if T:
                out[key] = np.stack(T).astype(np.float16)
    return out, short


def base_rr_of(prof: pd.DataFrame) -> pd.Series:
    b = prof.get("m_rr_mean_300")
    fb = prof.get("m_rr_mean_30")
    v = b.where(np.isfinite(b), fb) if b is not None else fb
    v = v.where(prof["baseline_valid"] > 0.5)
    return pd.Series(v.to_numpy(), index=prof["profile_key"])


# ---------------------------------------------------------------- 모델
class Encoder(nn.Module):
    """rrnet 의 CNN + BiGRU 와 같은 구조 → 시간 평균 64차원."""

    def __init__(self):
        super().__init__()
        layers, c_in = [], 3
        for c, k, p in ((16, 7, 4), (32, 5, 4), (32, 5, 2)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(p)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)

    def forward(self, x):
        h, _ = self.gru(self.conv(x).transpose(1, 2))
        return h.mean(1)


class RRDistance(nn.Module):
    def __init__(self, k: int = 16):
        super().__init__()
        self.enc = Encoder()
        self.W = nn.Linear(64, k, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(()))

    def distance(self, g, g_ref):
        v = self.W(g - g_ref)
        return torch.sqrt((v ** 2).sum(-1) + 1e-8)

    def logit(self, d):
        return nn.functional.softplus(self.alpha) * d + self.b


def _inp(tacho: np.ndarray, base: np.ndarray) -> torch.Tensor:
    x, _ = make_input(tacho.astype(np.float32), base)
    return x


class Trainer:
    def __init__(self, cond: str, seed: int, device: str = "cpu", hp: dict | None = None):
        self.cond, self.seed, self.dev = cond, seed, device
        self.hp = {**HP, **(hp or {})}
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed + 101)
        self.m = RRDistance().to(device)

    def _shift(self, x):
        s = int(round(self.hp["shift_sec"] * 4))
        return torch.roll(x, int(self.rng.integers(-s, s + 1)), dims=2) if s else x

    def _gref(self, keys, refs, base, k, train):
        """배치의 고유 profile 마다 기준 타코그램 k 개를 인코딩해 평균."""
        uniq = list(dict.fromkeys(keys))
        xs, owner = [], []
        for i, key in enumerate(uniq):
            R = refs[key]
            sel = self.rng.choice(len(R), min(k, len(R)), replace=False) if train else np.arange(min(k, len(R)))
            xs.append(_inp(R[sel], np.full(len(sel), base[key], np.float32)))
            owner += [i] * len(sel)
        x = torch.cat(xs).to(self.dev)
        if train:
            x = self._shift(x)
        e = self.m.enc(x)
        own = torch.as_tensor(owner, device=self.dev)
        g = torch.zeros(len(uniq), e.shape[1], device=self.dev).index_add_(0, own, e)
        g = g / torch.bincount(own, minlength=len(uniq)).unsqueeze(1).float()
        pos = {key: i for i, key in enumerate(uniq)}
        return g[torch.as_tensor([pos[k_] for k_ in keys], device=self.dev)]

    def _batch_logit(self, T, keys, refs, base, idx, train):
        x = _inp(T[idx], np.array([base[keys[i]] for i in idx], np.float32)).to(self.dev)
        if train:
            x = self._shift(x)
        g = self.m.enc(x)
        gr = self._gref([keys[i] for i in idx], refs, base, self.hp["k_ref_train"] if train else self.hp["k_ref_eval"], train)
        d = self.m.distance(g, gr)
        return self.m.logit(d), d

    def fit(self, T, keys, y, w, refs, base, val_idx=None, rpairs=None, log=None):
        hp = self.hp
        opt = torch.optim.AdamW(self.m.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        tr = np.setdiff1d(np.arange(len(y)), val_idx if val_idx is not None else [])
        p = w[tr] / w[tr].sum()
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.dev)
        wt = torch.as_tensor(w, dtype=torch.float32, device=self.dev)
        steps = max(1, len(tr) // hp["batch"])
        best, best_state, bad = np.inf, None, 0
        for ep in range(hp["epochs"]):
            self.m.train()
            for _ in range(steps):
                b = tr[self.rng.choice(len(tr), hp["batch"], p=p)]
                logit, d = self._batch_logit(T, keys, refs, base, b, True)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, yt[b])
                if self.cond == "R3" and rpairs is not None:
                    ra, rb = rpairs(self.rng, hp["n_rpairs"])
                    if len(ra):
                        _, da = self._batch_logit(T, keys, refs, base, ra, True)
                        _, db = self._batch_logit(T, keys, refs, base, rb, True)
                        loss = loss + hp["lam_r"] * nn.functional.softplus(-(da - db)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if val_idx is None or len(val_idx) == 0:
                continue
            lv = self.predict(T, keys, refs, base, val_idx)
            vl = float((nn.functional.binary_cross_entropy_with_logits(
                torch.as_tensor(lv, dtype=torch.float32), yt[val_idx].cpu(), reduction="none") * wt[val_idx].cpu()).sum()
                / wt[val_idx].sum().cpu())
            if log:
                log(f"      ep {ep} val {vl:.4f}")
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.m.state_dict().items()}
            else:
                bad += 1
                if bad >= hp["patience"]:
                    break
        if best_state is not None:
            self.m.load_state_dict(best_state)
        self.m.eval()
        return self

    @torch.no_grad()
    def predict(self, T, keys, refs, base, idx, bs: int = 1024) -> np.ndarray:
        self.m.eval()
        out = []
        for i in range(0, len(idx), bs):
            lg, _ = self._batch_logit(T, keys, refs, base, idx[i:i + bs], False)
            out.append(lg.cpu().numpy())
        return np.concatenate(out).astype(np.float64) if out else np.zeros(0)
