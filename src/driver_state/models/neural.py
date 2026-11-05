"""신경망 후보: MLP (features 입력), CNN+BiLSTM (sequence + personal 입력)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import CandidateModel, class_weights


class MLPNet(nn.Module):
    """F → h1 → h2 → K. 각 은닉층: Linear → ReLU → Dropout."""

    def __init__(self, in_dim: int, n_classes: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNNBiLSTMNet(nn.Module):
    """파형 (B, C, T) [+ 개인화 특징 (B, P)] → (B, K).

    파형 경로: 1D-CNN 블록 3개 → BiLSTM → 시간축 평균 → 64차원
    개인화 경로 (P > 0): Linear(P→16) → ReLU → 16차원
    두 벡터를 이어 붙여(concat) Dropout → Linear 로 분류한다 (분류층 직전 결합).

    첫 커널 15 = 100 Hz 에서 150 ms (QRS 폭 정도, 참고논문 2 설계 근거).
    MaxPool 로 시간축을 4·4·2 = 32 배 줄여 BiLSTM 계산량을 낮춘다.
    """

    def __init__(self, in_ch: int, n_classes: int, personal_dim: int = 0, conv=(16, 32, 32),
                 kernels=(15, 7, 5), pools=(4, 4, 2), lstm_hidden: int = 32, personal_hidden: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        blocks, c = [], in_ch
        for co, k, p in zip(conv, kernels, pools):
            blocks += [nn.Conv1d(c, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(p)]
            c = co
        self.cnn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(c, lstm_hidden, batch_first=True, bidirectional=True)
        self.personal_dim = personal_dim
        d = 2 * lstm_hidden
        if personal_dim > 0:
            self.personal = nn.Sequential(nn.Linear(personal_dim, personal_hidden), nn.ReLU())
            d += personal_hidden
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))

    def forward(self, x, p=None):
        h = self.cnn(x)                        # (B, C', T')
        h, _ = self.lstm(h.transpose(1, 2))    # (B, T', 2H)
        h = h.mean(dim=1)                      # (B, 2H)
        if self.personal_dim > 0:
            h = torch.cat([h, self.personal(p)], dim=1)
        return self.head(h)


def _as_list(X) -> list[np.ndarray]:
    return list(X) if isinstance(X, (tuple, list)) else [X]


class _TorchModel(CandidateModel):
    """공통 학습 루프: Adam + 가중 CrossEntropy + 검증 손실 기준 early stopping.

    입력 X 는 배열 하나 또는 배열 튜플 (CNN+BiLSTM 의 (sequence, personal)).
    """

    defaults = dict(epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=10)

    def __init__(self, n_classes: int, seed: int = 0, **params):
        super().__init__(n_classes, seed, **{**self.defaults, **params})
        self.net: nn.Module | None = None
        self.in_shapes: list[tuple[int, ...]] | None = None

    def _build(self, in_shapes: list[tuple[int, ...]]) -> nn.Module:
        raise NotImplementedError

    def _net_kwargs(self) -> dict:
        skip = set(_TorchModel.defaults)
        return {k: tuple(v) if isinstance(v, list) else v for k, v in self.params.items() if k not in skip}

    @staticmethod
    def _tensors(X) -> list[torch.Tensor]:
        return [torch.from_numpy(np.ascontiguousarray(a)).float() for a in _as_list(X)]

    def _forward(self, ts: list[torch.Tensor]):
        return self.net(*ts)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        p = self.params
        Xs = self._tensors(X)
        self.in_shapes = [tuple(t.shape[1:]) for t in Xs]
        self.net = self._build(self.in_shapes)
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        w = torch.tensor(class_weights(y, self.n_classes), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        yt = torch.from_numpy(y).long()
        has_val = X_val is not None and len(_as_list(X_val)[0]) > 0
        if has_val:
            Xv, yv = self._tensors(X_val), torch.from_numpy(y_val).long()
        best, best_state, wait = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        n = len(yt)
        for _ in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, p["batch_size"]):
                b = perm[i:i + p["batch_size"]]
                if len(b) < 2:
                    continue   # BatchNorm 은 배치 1개로 학습 불가
                opt.zero_grad()
                loss_fn(self._forward([t[b] for t in Xs]), yt[b]).backward()
                opt.step()
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = nn.functional.cross_entropy(self._forward(Xv), yv).item()
                if vl < best - 1e-4:
                    best, best_state, wait = vl, copy.deepcopy(self.net.state_dict()), 0
                else:
                    wait += 1
                    if wait >= p["patience"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self._forward(self._tensors(X)), dim=1).numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "model.pt")
        (path / "meta.json").write_text(json.dumps({
            "name": self.name, "n_classes": self.n_classes, "seed": self.seed,
            "params": self.params, "in_shapes": [list(s) for s in self.in_shapes]}, indent=2))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        m = cls(meta["n_classes"], meta["seed"], **meta["params"])
        m.in_shapes = [tuple(s) for s in meta["in_shapes"]]
        m.net = m._build(m.in_shapes)
        m.net.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        m.net.eval()
        return m

    def export_onnx(self, path: str | Path) -> None:
        """보드 이식용 ONNX 내보내기 (batch 축 가변). 입력 이름: x (파형/특징), p (개인화, 있을 때)."""
        names = ["x", "p"][:len(self.in_shapes)]
        dummies = tuple(torch.zeros((1, *s)) for s in self.in_shapes)
        axes = {n: {0: "batch"} for n in names} | {"logits": {0: "batch"}}
        torch.onnx.export(self.net, dummies, str(path), input_names=names, output_names=["logits"],
                          dynamic_axes=axes, dynamo=False)


class MLPModel(_TorchModel):
    name = "mlp"
    input_kind = "features"

    def _build(self, in_shapes):
        return MLPNet(in_shapes[0][0], self.n_classes, **self._net_kwargs())


class CNNBiLSTMModel(_TorchModel):
    """input_kind 'sequence+personal': (sequence, personal) 튜플. personal 이 (N, 0) 이면 파형만."""

    name = "cnn_bilstm"
    input_kind = "sequence+personal"
    defaults = dict(_TorchModel.defaults, epochs=40, batch_size=32, patience=8)

    def _tensors(self, X):
        ts = super()._tensors(X)
        return ts[:1] if len(ts) > 1 and ts[1].shape[1] == 0 else ts

    def _build(self, in_shapes):
        pdim = in_shapes[1][0] if len(in_shapes) > 1 else 0
        return CNNBiLSTMNet(in_shapes[0][0], self.n_classes, personal_dim=pdim, **self._net_kwargs())
