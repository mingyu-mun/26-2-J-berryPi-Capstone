"""쌍 비교 교사 모델 P0·P1·P2 (docs/teacher-pair-stage1.md 5장). 기존 TeacherModel 인터페이스는 쓰지 않는다.

몸통: x(36) → Linear(36,32) → GELU → Dropout(0.3) → Linear(32,32) → GELU → Dropout(0.3) → h(32)
P0·P1: s = Linear(32,1)(h)                       P2: s = logaddexp(s_U, s_D) − ln 2
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

HP = dict(m=0.5, lam_h=0.5, lam_t=1.0, lam_s=0.5, lr=1e-3, wd=1e-2, steps=2000,
          noise=0.1, drop120=0.1, drop300=0.1, batch_windows=256)


class _Body(nn.Module):
    def __init__(self, d_in: int = 36, heads: int = 1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 32), nn.GELU(), nn.Dropout(0.3),
                                  nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.3))
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(heads)])

    def forward(self, x):
        h = self.body(x)
        outs = [hd(h).squeeze(-1) for hd in self.heads]
        if len(outs) == 1:
            return outs[0], outs
        return torch.logaddexp(outs[0], outs[1]) - math.log(2.0), outs


def n_params(heads: int) -> int:
    return sum(p.numel() for p in _Body(36, heads).parameters())


def _l_r(sa, sb):
    return nn.functional.softplus(-(sa - sb))


def _l_t(sa, sb, m):
    return torch.clamp((sa - sb).abs() - m, min=0) ** 2


def _l_s(sb, sref, m):
    return nn.functional.softplus(sb - sref - m)


def _mean(x):
    return x.mean() if x.numel() else torch.zeros((), dtype=torch.float32)


class PairNet:
    """cond: 'P0' (창 BCE), 'P1' (쌍, 헤드 1), 'P2' (쌍, 헤드 2)."""

    def __init__(self, cond: str, seed: int, groups: dict[str, np.ndarray], hp: dict | None = None):
        self.cond, self.seed = cond, seed
        self.hp = {**HP, **(hp or {})}
        self.groups = groups                       # {"120": bool[36], "300": bool[36]}
        torch.manual_seed(seed)
        self.net = _Body(36, 2 if cond == "P2" else 1)
        self.rng = np.random.default_rng(seed + 7919)

    # -------------------------------------------------------------- 입력 가공
    def _pair_inputs(self, A: np.ndarray, B: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """쌍 결측 규칙 + 스케일 묶음 드롭 + 서로 다른 잡음."""
        M = np.isfinite(A) & np.isfinite(B)
        n = len(A)
        for key, p in (("120", self.hp["drop120"]), ("300", self.hp["drop300"])):
            drop = self.rng.random(n) < p
            M[np.ix_(drop, self.groups[key])] = False
        A = np.where(M, A, 0.0) + self.rng.normal(0, self.hp["noise"], A.shape) * M
        B = np.where(M, B, 0.0) + self.rng.normal(0, self.hp["noise"], B.shape) * M
        return torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(B, dtype=torch.float32)

    def _win_inputs(self, X: np.ndarray) -> torch.Tensor:
        M = np.isfinite(X)
        n = len(X)
        for key, p in (("120", self.hp["drop120"]), ("300", self.hp["drop300"])):
            drop = self.rng.random(n) < p
            M[np.ix_(drop, self.groups[key])] = False
        X = np.where(M, X, 0.0) + self.rng.normal(0, self.hp["noise"], X.shape) * M
        return torch.as_tensor(X, dtype=torch.float32)

    # -------------------------------------------------------------- 학습
    def fit_pairs(self, sampler, Z: np.ndarray, ref_of: np.ndarray, Zref: np.ndarray,
                  ds_of_row: np.ndarray, log=None) -> "PairNet":
        """Z: 학습 창 표준화 특징 (NaN 결측). ref_of[row] = Zref 의 행 번호 (기준 벡터)."""
        hp = self.hp
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        m = hp["m"]
        self.net.train()
        for step in range(hp["steps"]):
            ra, rb, rds = sampler.sample_r()
            ta, tb = sampler.sample_t()
            sb = sampler.sample_s()
            loss = torch.zeros(())
            if len(ra):
                xa, xb = self._pair_inputs(Z[ra], Z[rb])
                (sa, ha), (sb_, hb) = self.net(xa), self.net(xb)
                loss = loss + _mean(_l_r(sa, sb_))
                if self.cond == "P2":
                    kss = torch.as_tensor(np.isin(rds, ["advitam", "drozy"]))
                    eeg = torch.as_tensor(np.isin(rds, ["mpddf"]))
                    loss = loss + hp["lam_h"] * (_mean(_l_r(ha[0], hb[0])[kss]) + _mean(_l_r(ha[1], hb[1])[eeg]))
            if len(ta):
                xa, xb = self._pair_inputs(Z[ta], Z[tb])
                (sa, ha), (sb_, hb) = self.net(xa), self.net(xb)
                if self.cond == "P2":
                    loss = loss + hp["lam_t"] * _mean(_l_t(ha[0], hb[0], m) + _l_t(ha[1], hb[1], m))
                else:
                    loss = loss + hp["lam_t"] * _mean(_l_t(sa, sb_, m))
            if len(sb):
                xb, xr = self._pair_inputs(Z[sb], Zref[ref_of[sb]])
                (s_b, hb), (s_r, hr) = self.net(xb), self.net(xr)
                if self.cond == "P2":
                    loss = loss + hp["lam_s"] * _mean(_l_s(hb[0], hr[0], m) + _l_s(hb[1], hr[1], m))
                else:
                    loss = loss + hp["lam_s"] * _mean(_l_s(s_b, s_r, m))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log and step % 500 == 0:
                log(f"      step {step} loss {float(loss):.4f}")
        self.net.eval()
        return self

    def fit_windows(self, Z: np.ndarray, y: np.ndarray, w: np.ndarray, log=None) -> "PairNet":
        """P0: 가중치에 비례해 창 256개씩 뽑아 BCE (가중 BCE 와 기댓값이 같다)."""
        hp = self.hp
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        p = w / w.sum()
        yt = torch.as_tensor(y, dtype=torch.float32)
        self.net.train()
        for step in range(hp["steps"]):
            b = self.rng.choice(len(y), hp["batch_windows"], p=p)
            s, _ = self.net(self._win_inputs(Z[b]))
            loss = nn.functional.binary_cross_entropy_with_logits(s, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if log and step % 500 == 0:
                log(f"      step {step} loss {float(loss):.4f}")
        self.net.eval()
        return self

    # -------------------------------------------------------------- 점수
    @torch.no_grad()
    def score(self, X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """(n,) 결합 점수와 (n, heads) 헤드 점수. mask 가 0 인 특징은 0 으로 (평가 땐 잡음·드롭 없음)."""
        self.net.eval()                            # 드롭아웃 끔 (평가는 항상 결정적)
        M = np.isfinite(X) if mask is None else (mask & np.isfinite(X))
        Xt =torch.as_tensor(np.where(M, X, 0.0), dtype=torch.float32)
        s, hs = self.net(Xt)
        return s.numpy().astype(np.float64), np.stack([h.numpy() for h in hs], 1).astype(np.float64)

    def score_diff(self, X: np.ndarray, Xref: np.ndarray) -> np.ndarray:
        """d_t = f(x_t ⊙ M) − f(x_ref ⊙ M), M = 둘 다 유한 (3.2)."""
        M = np.isfinite(X) & np.isfinite(Xref)
        st, _ = self.score(X, M)
        sr, _ = self.score(Xref, M)
        return st - sr
