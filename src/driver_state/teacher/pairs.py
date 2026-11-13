"""쌍 비교 교사(teacher-pair) 1단계: 창 주석, RR 특징 변환, 쌍 표집 (docs/teacher-pair-stage1.md).

이 모듈의 열(group, session_id, t_rel, …)은 표집에만 쓰고 모델 입력에는 넣지 않는다 (9.1).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .spec import TeacherSpec

TEACHER_DATASETS = ("advitam", "dddb", "drozy", "mpddf")
STUDENT_TRAIN = ("clas", "maus")          # S 쌍 (모든 fold 학습에 포함)
EVAL_ONLY = ("wesad",)                    # 어느 fold 에도 학습에 안 들어감
LOG_PREFIX = ("hr", "rr_mean", "sdnn", "rmssd", "sd1", "sd2")
GAP_SEC = 300.0
KSS_DATASETS = ("advitam", "drozy")       # P2 KSS 헤드 (s_U)
EEG_DATASETS = ("mpddf",)                 # P2 EEG 헤드 (s_D)


def feature_names(spec: TeacherSpec | None = None) -> list[str]:
    return list((spec or TeacherSpec()).baseline_feature_names)


# ---------------------------------------------------------------- 1.1 주석
def read_session_info(rr_roots: list[str | Path]) -> pd.DataFrame:
    """RR 캐시 npz 의 info 만 읽는다 (group, session_id, is_rest, t0, meta)."""
    rows = []
    for root in rr_roots:
        for p in sorted(Path(root).rglob("*.npz")):
            with np.load(p, allow_pickle=False) as z:
                info = json.loads(str(z["info"]))
            meta = info.get("meta") or {}
            rows.append({"group": f"{info['dataset_id']}/{info['group']}", "session_id": info["session_id"],
                         "is_rest": bool(info.get("is_rest", False)), "t0": float(info["t0"]),
                         "has_range": meta.get("baseline_range") is not None,
                         "per_session": bool(meta.get("baseline_per_session", False))})
    return pd.DataFrame(rows)


def baseline_kinds(info: pd.DataFrame, prof: pd.DataFrame) -> pd.Series:
    """profile_key → rest_session / range / default (3.1)."""
    look = info.set_index(["group", "session_id"])
    out = {}
    for key in prof["profile_key"]:
        g, _, ses = key.partition("#")
        if (g, ses) in look.index:
            r = look.loc[(g, ses)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            out[key] = "rest_session" if r["is_rest"] else ("range" if r["has_range"] else "default")
        else:
            out[key] = "default"
    return pd.Series(out, name="base_kind")


# CLAS: 자극 이름 → 구간 종류 (10장 대응표). 과제는 휴식 화면(begin / n#.jpg) 다음 자극부터
# 그 과제의 정답 화면(…testanswer.png) 직전까지. 정답 화면 자체는 'Response' 블록이라 과제가 아니다.
CLAS_ANSWER = {"mathtestanswer.png": "Math Test", "strooptestanswer.png": "Stroop Test",
               "iqtestanswer.png": "Logic(IQ) Test"}


def _stim_name(s: str) -> str:
    return str(s).replace("\\", "/").split("/")[-1].strip().lower()


def clas_task_intervals(orig: pd.DataFrame) -> list[tuple[float, float, str]]:
    ev = orig.sort_values("t").reset_index(drop=True)
    names = [_stim_name(s) for s in ev["stimulus"]]
    out, start = [], None
    for i, n in enumerate(names):
        if n == "begin.png" or (n.startswith("n") and n.endswith(".jpg") and n[1:-4].isdigit()):
            start = float(ev["t"].iloc[i + 1]) if i + 1 < len(ev) else None
        elif n in CLAS_ANSWER and start is not None:
            out.append((start, float(ev["t"].iloc[i]), CLAS_ANSWER[n]))
            start = None
    return out


def _orig_path(data_root: Path, dataset: str, subject: str, session: str) -> Path:
    return data_root / dataset / subject / session / "labels_orig.csv"


def tag_conditions(win: pd.DataFrame, data_root: str | Path | None, W: float = 30.0) -> np.ndarray:
    """학생용 데이터의 cond: task / stress / other / mixed (4.3). 교사 데이터는 ''."""
    cond = np.full(len(win), "", dtype=object)
    if data_root is None:
        return cond
    data_root = Path(data_root)
    for (ds, subj, ses), g in win.groupby(["dataset_id", "subject_id", "session_id"]):
        idx = g.index.to_numpy()
        t = g["t_rel"].to_numpy(float)
        if ds == "maus":
            cond[idx] = "task" if str(ses).startswith("trial") else "other"
            continue
        if ds not in ("clas", "wesad"):
            continue
        p = _orig_path(data_root, ds, subj, ses)
        if not p.exists():
            cond[idx] = "other"
            continue
        orig = pd.read_csv(p)
        if ds == "clas":
            iv = [(a, b) for a, b, _ in clas_task_intervals(orig)]
            target = "task"
        else:
            tsst = orig[orig["condition"].astype(str).str.upper().str.startswith("TSST")]
            iv = [(float(r["t"]), float(r["t_end"])) for _, r in tsst.iterrows()]
            target = "stress"
        inside = np.zeros(len(t), bool)
        touch = np.zeros(len(t), bool)
        for a, b in iv:
            inside |= (t >= a) & (t + W <= b)
            touch |= (t < b) & (t + W > a)
        cond[idx] = np.where(inside, target, np.where(touch, "mixed", "other"))
    return cond


def annotate(win: pd.DataFrame, prof: pd.DataFrame, info: pd.DataFrame,
             data_root: str | Path | None = None, tbin_sec: float = 300.0) -> pd.DataFrame:
    """tbin_sec: AdVitam 밖의 시간 층 폭 (기본 300초, 8.3 지표 유효 조건이 깨지면 150초로 한 번만 줄임)."""
    w = win.copy()
    t = w["t_rel"].to_numpy(float)
    is_adv = (w["dataset_id"] == "advitam").to_numpy()
    w["tbin"] = np.where(is_adv, (t >= 300).astype(int), np.floor(t / tbin_sec).astype(int))
    t0 = (w["t_start"] - w["t_rel"]).to_numpy(float)
    w["_t0"] = t0
    ses = w.groupby(["group", "session_id"])["_t0"].first().reset_index()
    ses["ses_rank"] = ses.groupby("group")["_t0"].rank(method="first").astype(int) - 1
    w = w.merge(ses[["group", "session_id", "ses_rank"]], on=["group", "session_id"], how="left")
    w["same_as_base"] = (w["profile_key"].str.split("#").str[-1] == w["session_id"]).astype(int)
    kinds = baseline_kinds(info, prof)
    w["base_kind"] = w["profile_key"].map(kinds).fillna("default")
    bv = prof.set_index("profile_key")["baseline_valid"]
    w["baseline_valid"] = w["profile_key"].map(bv).fillna(0.0).astype(float)
    w["cond"] = tag_conditions(w, data_root)
    return w.drop(columns=["_t0"])


# ---------------------------------------------------------------- 2장 특징 변환
@dataclass
class FeatureScaler:
    names: list[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64).copy()
        for j, n in enumerate(self.names):
            if n.rsplit("_", 1)[0] in LOG_PREFIX:
                with np.errstate(divide="ignore", invalid="ignore"):
                    X[:, j] = np.where(X[:, j] > 0, np.log(X[:, j]), np.nan)
        return X

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """데이터셋 균등 × 참가자 균등 가중 평균·표준편차 (유효 창)."""
        X = self._log(df[self.names].to_numpy())
        n_ds = df["dataset_id"].nunique()
        subj = df.groupby("dataset_id")["group"].transform("nunique").to_numpy()
        per = df.groupby("group")["group"].transform("size").to_numpy()
        w = 1.0 / (n_ds * subj * per)
        self.mean = np.zeros(len(self.names))
        self.std = np.ones(len(self.names))
        for j in range(len(self.names)):
            m = np.isfinite(X[:, j])
            if m.sum() > 1:
                ww = w[m] / w[m].sum()
                mu = float((ww * X[m, j]).sum())
                sd = float(np.sqrt((ww * (X[m, j] - mu) ** 2).sum()))
                self.mean[j], self.std[j] = mu, (sd if sd > 1e-9 else 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """로그 → 표준화 → ±5 자르기. 결측은 NaN 으로 남긴다 (쌍 결측 규칙을 적용해야 해서)."""
        Z = (self._log(X) - self.mean) / self.std
        return np.clip(Z, -5, 5).astype(np.float32)


def scale_groups(names: list[str]) -> dict[str, np.ndarray]:
    tag = np.array([n.rsplit("_", 1)[1] for n in names])
    return {"120": tag == "120", "300": tag == "300"}


# ---------------------------------------------------------------- 4장 쌍 색인과 표집
@dataclass
class _Group:
    dataset: str
    idx: np.ndarray           # 창 행 번호 (Z 기준)
    prof: np.ndarray          # profile_key 번호
    ses: np.ndarray           # session 번호
    t: np.ndarray
    y: np.ndarray
    tbin: np.ndarray
    base_ses: np.ndarray      # same_as_base
    cond: np.ndarray
    r_ok: np.ndarray          # R: 후보가 있는 졸음 창 (로컬 위치)
    t_sessions: list          # T: [(ses, label, a 후보 로컬 위치)]
    s_ok: np.ndarray          # S: 과제 창 (로컬 위치)
    has_tm: bool = False


def _r_candidates(g: _Group, i: int) -> tuple[np.ndarray, np.ndarray]:
    """졸음 창 i (로컬) 의 정상 후보와 tm 여부 (4.1)."""
    m = (g.y == 0) & (g.prof == g.prof[i])
    diff_ses = g.ses != g.ses[i]
    m &= diff_ses | (np.abs(g.t - g.t[i]) >= GAP_SEC)
    cand = np.flatnonzero(m)
    tm = diff_ses[cand] & (g.tbin[cand] == g.tbin[i])
    return cand, tm


class PairIndex:
    """학습 창(주석 완료, valid=True)으로 만든 참가자별 배열. 쌍 목록은 미리 만들지 않는다 (4.4)."""

    def __init__(self, ann: pd.DataFrame, s_datasets=STUDENT_TRAIN):
        ann = ann.reset_index(drop=True)
        self.ann = ann
        pk = {k: i for i, k in enumerate(ann["profile_key"].unique())}
        self.groups: dict[str, list[_Group]] = {"R": [], "T": [], "S": []}
        self.by_ds: dict[str, dict[str, list[int]]] = {"R": {}, "T": {}, "S": {}}
        self.all: list[_Group] = []
        for gname, d in ann.groupby("group", sort=True):
            if not d["valid"].all():
                d = d[d["valid"]]
            if len(d) == 0:
                continue
            ses_codes = pd.factorize(d["session_id"])[0]
            g = _Group(dataset=d["dataset_id"].iloc[0], idx=d.index.to_numpy(),
                       prof=d["profile_key"].map(pk).to_numpy(), ses=ses_codes,
                       t=d["t_rel"].to_numpy(float), y=d["y"].to_numpy(int), tbin=d["tbin"].to_numpy(int),
                       base_ses=d["same_as_base"].to_numpy(int), cond=d["cond"].to_numpy(object),
                       r_ok=np.array([], int), t_sessions=[], s_ok=np.array([], int))
            # R
            ok, tm_any = [], False
            for i in np.flatnonzero(g.y == 1):
                cand, tm = _r_candidates(g, i)
                if len(cand):
                    ok.append(i)
                    tm_any |= bool(tm.any())
            g.r_ok, g.has_tm = np.array(ok, int), tm_any
            # T
            for s in np.unique(g.ses):
                if g.dataset == "drozy" and g.base_ses[g.ses == s].any():
                    continue
                for lab in (0, 1):
                    loc = np.flatnonzero((g.ses == s) & (g.y == lab))
                    if len(loc) < 2:
                        continue
                    tt = g.t[loc]
                    has = np.array([(np.abs(tt - x) >= GAP_SEC).any() for x in tt])
                    if has.any():
                        g.t_sessions.append((s, lab, loc[has], loc))
            # S
            if g.dataset in s_datasets:
                bk = d["base_kind"].to_numpy()
                bv = d["baseline_valid"].to_numpy(float)
                g.s_ok = np.flatnonzero((g.cond == "task") & np.isin(bk, ["rest_session", "range"]) & (bv > 0.5))
            gi = len(self.all)
            self.all.append(g)
            for kind, cond in (("R", len(g.r_ok) > 0), ("T", len(g.t_sessions) > 0), ("S", len(g.s_ok) > 0)):
                if cond:
                    self.by_ds[kind].setdefault(g.dataset, []).append(gi)

    def stats(self) -> pd.DataFrame:
        rows = {}
        for g in self.all:
            r = rows.setdefault(g.dataset, {"R_subjects": 0, "R_tm_subjects": 0, "T_subjects": 0,
                                            "S_subjects": 0, "valid_labeled_windows": 0})
            r["R_subjects"] += int(len(g.r_ok) > 0)
            r["R_tm_subjects"] += int(len(g.r_ok) > 0 and g.has_tm)
            r["T_subjects"] += int(len(g.t_sessions) > 0)
            r["S_subjects"] += int(len(g.s_ok) > 0)
            r["valid_labeled_windows"] += int((g.y >= 0).sum())
        return pd.DataFrame(rows).T


class PairSampler:
    """배치 쌍 256개 = R 128, T 77, S 51 (4.4). 반환 값은 Z 행 번호와 S 의 profile_key."""

    def __init__(self, index: PairIndex, seed: int, fold_index: int, n_r=128, n_t=77, n_s=51):
        self.ix = index
        self.rng = np.random.default_rng(1000 * seed + fold_index)
        self.n = {"R": n_r, "T": n_t, "S": n_s}

    def _pick_group(self, kind: str):
        dss = sorted(self.ix.by_ds[kind])
        if not dss:
            return None
        ds = dss[self.rng.integers(len(dss))]
        gl = self.ix.by_ds[kind][ds]
        return self.ix.all[gl[self.rng.integers(len(gl))]]

    def sample_r(self):
        a_out, b_out, ds_out = [], [], []
        for _ in range(self.n["R"]):
            g = self._pick_group("R")
            if g is None:
                break
            for _try in range(20):
                i = g.r_ok[self.rng.integers(len(g.r_ok))]
                cand, tm = _r_candidates(g, i)
                if len(cand):
                    break
            else:
                continue
            if tm.any() and self.rng.random() < 0.5:
                cand = cand[tm]
            j = cand[self.rng.integers(len(cand))]
            a_out.append(g.idx[i]); b_out.append(g.idx[j]); ds_out.append(g.dataset)
        return np.array(a_out, int), np.array(b_out, int), np.array(ds_out, object)

    def sample_t(self):
        a_out, b_out = [], []
        for _ in range(self.n["T"]):
            g = self._pick_group("T")
            if g is None:
                break
            sess = sorted({s for s, _, _, _ in g.t_sessions})
            s = sess[self.rng.integers(len(sess))]
            labs = [x for x in g.t_sessions if x[0] == s]
            _, lab, a_loc, all_loc = labs[self.rng.integers(len(labs))]
            i = a_loc[self.rng.integers(len(a_loc))]
            partners = all_loc[np.abs(g.t[all_loc] - g.t[i]) >= GAP_SEC]
            j = partners[self.rng.integers(len(partners))]
            a_out.append(g.idx[i]); b_out.append(g.idx[j])
        return np.array(a_out, int), np.array(b_out, int)

    def sample_s(self):
        b_out = []
        for _ in range(self.n["S"]):
            g = self._pick_group("S")
            if g is None:
                break
            b_out.append(g.idx[g.s_ok[self.rng.integers(len(g.s_ok))]])
        return np.array(b_out, int)


def pair_stats(ann_teacher: pd.DataFrame, ann_student: pd.DataFrame) -> pd.DataFrame:
    tr = ann_teacher[ann_teacher["valid"]]
    st = ann_student[ann_student["valid"] & ann_student["dataset_id"].isin(STUDENT_TRAIN)]
    ix = PairIndex(pd.concat([tr, st], ignore_index=True))
    return ix.stats()
