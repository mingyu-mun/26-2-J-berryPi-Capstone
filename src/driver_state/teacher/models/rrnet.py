"""B. RR 타코그램 1D CNN + BiGRU (PyTorch, CPU).

입력: 창 중심 앞뒤 2.5분(5분) 정상 RR 을 4 Hz 로 보간한 1200 점 × 3 채널
  0: (RR − 기준 평균 RR) / 100 ms   (기준 무효면 0)
  1: 창 내 z-score RR                (개인 기준 없이도 쓰는 모양 정보)
  2: 결측 마스크
+ baseline_valid (0/1) 를 풀링 뒤에 붙인다.

Conv1d(3→16,k7)→BN→ReLU→MaxPool4 → Conv1d(16→32,k5)→…→MaxPool4 → Conv1d(32→32,k5)→…→MaxPool2
→ BiGRU(32) → 시간축 평균 (64) ‖ baseline_valid → Dropout 0.3 → Linear(65→1) = 로짓
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import TeacherInput, TeacherModel

DEFAULTS = dict(epochs=30, patience=5, batch_size=256, lr=1e-3, weight_decay=1e-4,
                shift_sec=10.0, invalid_rate=0.2, threads=0)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c_in = [], 3
        for c, k, p in ((16, 7, 4), (32, 5, 4), (32, 5, 2)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(p)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(65, 1))

    def forward(self, x, bv):
        h = self.conv(x).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(torch.cat([h.mean(1), bv[:, None]], 1)).squeeze(1)


def make_input(tacho: np.ndarray, base_rr: np.ndarray | None,
               force_invalid: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """배치 단위로 만든다 (전체를 float32 로 펼치면 창 10만 개에 1.4 GB)."""
    t = np.asarray(tacho, dtype=np.float32)
    rr, mask = t[:, 0], t[:, 1]
    base = base_rr.astype(np.float32) if base_rr is not None else np.full(len(t), np.nan, np.float32)
    bv = np.isfinite(base)
    if force_invalid is not None:
        bv &= ~force_invalid
    ch0 = np.where(bv[:, None], (rr - np.nan_to_num(base)[:, None]) / 100.0, 0.0)
    valid = mask < 0.5
    cnt = np.maximum(valid.sum(1, keepdims=True), 1)
    mu = (rr * valid).sum(1, keepdims=True) / cnt
    sd = np.sqrt((((rr - mu) * valid) ** 2).sum(1, keepdims=True) / cnt) + 1.0
    ch1 = np.where(valid, (rr - mu) / sd, 0.0)
    ch0 = np.where(valid, ch0, 0.0)
    x = np.clip(np.stack([ch0, ch1, mask], 1).astype(np.float32), -10, 10)
    return torch.as_tensor(x), torch.as_tensor(bv.astype(np.float32))


class RRNetTeacher(TeacherModel):
    name = "rrnet"
    uses_tacho = True

    def __init__(self, seed: int = 0, **params):
        super().__init__(seed, **{**DEFAULTS, **params})
        self.net: _Net | None = None

    def _threads(self):
        if self.params["threads"]:
            torch.set_num_threads(int(self.params["threads"]))

    def fit(self, X, y, w=None, X_val=None, y_val=None, w_val=None):
        p = self.params
        self._threads()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.net = _Net()
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        w = np.ones(len(y)) if w is None else w
        yt = torch.as_tensor(y, dtype=torch.float32)
        wt = torch.as_tensor(w, dtype=torch.float32)
        have_val = X_val is not None and len(X_val) and len(np.unique(y_val)) == 2
        if have_val:
            yv = torch.as_tensor(y_val, dtype=torch.float32)
            wv = torch.as_tensor(np.ones(len(y_val)) if w_val is None else w_val, dtype=torch.float32)
        shift = int(round(p["shift_sec"] * 4))
        best, best_state, bad = np.inf, None, 0
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        for ep in range(int(p["epochs"])):
            self.net.train()
            perm = rng.permutation(len(y))
            for i in range(0, len(y), p["batch_size"]):
                b = np.sort(perm[i:i + p["batch_size"]])
                force = rng.random(len(b)) < p["invalid_rate"]
                xb, bvb = make_input(X.tacho[b], None if X.base_rr is None else X.base_rr[b], force)
                if shift:
                    xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), dims=2)
                bt = torch.as_tensor(b)
                loss = (loss_fn(self.net(xb, bvb), yt[bt]) * wt[bt]).sum() / wt[bt].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if not have_val:
                continue
            self.net.eval()
            with torch.no_grad():
                vl = float((loss_fn(self._batched(X_val), yv) * wv).sum() / wv.sum())
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _batched(self, X: TeacherInput, bs: int = 2048) -> torch.Tensor:
        outs = []
        for i in range(0, len(X), bs):
            sl = slice(i, i + bs)
            x, bv = make_input(X.tacho[sl], None if X.base_rr is None else X.base_rr[sl])
            outs.append(self.net(x, bv))
        return torch.cat(outs) if outs else torch.zeros(0)

    def predict_logit(self, X):
        self._threads()
        self.net.eval()
        with torch.no_grad():
            return self._batched(X).numpy().astype(np.float64)

    def _save_state(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path / "rrnet.pt")

    def _load_state(self, path: Path) -> None:
        self.net = _Net()
        self.net.load_state_dict(torch.load(path / "rrnet.pt", map_location="cpu"))
        self.net.eval()
