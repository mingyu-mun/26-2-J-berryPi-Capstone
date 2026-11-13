"""쌍 비교 교사 지표 M1·M3·M4·M5 와 참가자 단위 부트스트랩 (docs/teacher-pair-stage1.md 8장)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _wauc(y, s, w=None) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def m1_dataset(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """한 데이터셋의 시간 층화 AUROC 와 비교에 쓰인 창 비율.

    층 b 마다 w_i = 1/(그 참가자가 층 b 에서 가진 같은 라벨 창 수), A_b = 가중 AUROC,
    W_b = (졸음 창 있는 참가자 수) × (정상 창 있는 참가자 수). 두 라벨 다 있는 층만.
    """
    num = den = 0.0
    used = 0
    for _, g in df.groupby("tbin"):
        if g["y"].nunique() < 2:
            continue
        cnt = g.groupby(["group", "y"])["y"].transform("size").to_numpy(float)
        w = 1.0 / cnt
        a = _wauc(g["y"].to_numpy(), g[col].to_numpy(), w)
        W = g.loc[g["y"] == 1, "group"].nunique() * g.loc[g["y"] == 0, "group"].nunique()
        num += W * a
        den += W
        used += len(g)
    return (num / den if den else float("nan")), (used / len(df) if len(df) else float("nan"))


def m3_dataset(df: pd.DataFrame, col: str) -> float:
    """세션마다 점수를 t_rel 에 선형 회귀한 잔차로 같은 세션 안 AUROC (가중: 졸음×정상 수)."""
    num = den = 0.0
    for _, s in df.groupby(["group", "session_id"]):
        if s["y"].nunique() < 2:
            continue
        x = s[col].to_numpy(float)
        t = s["t_rel"].to_numpy(float)
        if np.ptp(t) > 0:
            x = x - np.polyval(np.polyfit(t, x, 1), t)
        n = (s["y"] == 1).sum() * (s["y"] == 0).sum()
        num += _wauc(s["y"].to_numpy(), x) * n
        den += n
    return num / den if den else float("nan")


def m4_dataset(df: pd.DataFrame, col: str) -> float:
    """두 라벨 다 있는 참가자 안의 AUROC 평균."""
    vals = [_wauc(g["y"].to_numpy(), g[col].to_numpy()) for _, g in df.groupby("group") if g["y"].nunique() == 2]
    return float(np.mean(vals)) if vals else float("nan")


def m2_dataset(df: pd.DataFrame, col: str) -> float:
    return _wauc(df["y"].to_numpy(), df[col].to_numpy())


METRICS = {"M1": lambda d, c: m1_dataset(d, c)[0], "M2": m2_dataset, "M3": m3_dataset, "M4": m4_dataset}


def per_dataset(df: pd.DataFrame, col: str, metric: str) -> dict[str, float]:
    return {ds: METRICS[metric](g, col) for ds, g in df.groupby("dataset_id")}


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def _layer_cache(df: pd.DataFrame, cols: list[str]):
    """M1 부트스트랩용: 데이터셋별 층마다 정렬 순서·동점 묶음·행 가중치를 미리 계산."""
    out = {}
    for ds, d in df.groupby("dataset_id"):
        groups = {g: i for i, g in enumerate(sorted(d["group"].unique()))}
        layers = []
        for _, g in d.groupby("tbin"):
            if g["y"].nunique() < 2:
                continue
            gi = g["group"].map(groups).to_numpy()
            y = g["y"].to_numpy()
            base = 1.0 / g.groupby(["group", "y"])["y"].transform("size").to_numpy(float)
            per_col = {}
            for c in cols:
                s = g[c].to_numpy(float)
                o = np.argsort(s, kind="mergesort")
                ss = s[o]
                starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
                per_col[c] = (o, starts)
            pos_g = np.unique(gi[y == 1]); neg_g = np.unique(gi[y == 0])
            layers.append((gi, y, base, per_col, pos_g, neg_g))
        out[ds] = (len(groups), layers)
    return out


def _wauc_sorted(y_o, w_o, starts) -> float:
    pos = np.add.reduceat(np.where(y_o == 1, w_o, 0.0), starts)
    neg = np.add.reduceat(np.where(y_o == 0, w_o, 0.0), starts)
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def bootstrap_m1(df: pd.DataFrame, cols: list[str], n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """참가자 단위 부트스트랩 M1 (데이터셋 평균), (n_boot, len(cols)).

    같은 참가자가 k 번 뽑히면 다른 사람 k 명으로 본다: 행 가중치 ×k, W_b 의 참가자 수에도 k 번 센다.
    m1_dataset 과 같은 정의를 정렬 한 번으로 빠르게 계산한다.
    """
    rng = np.random.default_rng(seed)
    cache = _layer_cache(df, cols)
    res = np.full((n_boot, len(cols)), np.nan)
    for it in range(n_boot):
        per_ds = np.full((len(cache), len(cols)), np.nan)
        for di, (ds, (n_g, layers)) in enumerate(sorted(cache.items())):
            mult = np.bincount(rng.integers(n_g, size=n_g), minlength=n_g).astype(float)
            num = np.zeros(len(cols)); den = 0.0
            for gi, y, base, per_col, pos_g, neg_g in layers:
                W = mult[pos_g].sum() * mult[neg_g].sum()
                if W <= 0:
                    continue
                w = base * mult[gi]
                for k, c in enumerate(cols):
                    o, starts = per_col[c]
                    a = _wauc_sorted(y[o], w[o], starts)
                    if np.isfinite(a):
                        num[k] += W * a
                den += W
            if den > 0:
                per_ds[di] = num / den
        res[it] = np.nanmean(per_ds, 0)
    return res


def bootstrap_diff(df: pd.DataFrame, col_a: str, col_b: str, metric: str = "M1",
                   n_boot: int = 2000, seed: int = 0, datasets=None) -> dict:
    """참가자 단위 쌍체 부트스트랩: 데이터셋마다 group 복원 추출, 데이터셋 평균 지표의 차이 (a − b)."""
    rng = np.random.default_rng(seed)
    datasets = datasets or sorted(df["dataset_id"].unique())
    parts = {ds: {g: d for g, d in df[df["dataset_id"] == ds].groupby("group")} for ds in datasets}
    f = METRICS[metric]
    point = np.nanmean([f(df[df["dataset_id"] == ds], col_a) - f(df[df["dataset_id"] == ds], col_b) for ds in datasets])
    diffs = []
    for _ in range(n_boot):
        per = []
        for ds in datasets:
            keys = list(parts[ds])
            pick = rng.integers(len(keys), size=len(keys))
            chunks = []
            for k, i in enumerate(pick):
                d = parts[ds][keys[i]].copy()
                d["group"] = f"{keys[i]}#{k}"          # 같은 참가자가 두 번 뽑혀도 다른 사람으로
                chunks.append(d)
            b = pd.concat(chunks, ignore_index=True)
            per.append(f(b, col_a) - f(b, col_b))
        diffs.append(np.nanmean(per))
    diffs = np.asarray(diffs)
    return {"diff": float(point), "lo": float(np.nanpercentile(diffs, 2.5)),
            "hi": float(np.nanpercentile(diffs, 97.5)), "n_boot": n_boot}
