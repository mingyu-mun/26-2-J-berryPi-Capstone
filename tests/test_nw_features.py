"""NormWear 표현 열 붙이기 (nw_features) 합성 데이터 검사."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driver_state.teacher.nw_features import NW_COLS, attach, fit_pca, z_of  # noqa: E402


def _fake(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for ds in ("clas", "maus", "drozy"):
        for s in range(3):
            for i in range(20):
                rows.append(dict(row_id=len(rows), dataset_id=ds, subject_id=f"s{s}", session_id="a",
                                 group=f"{ds}/s{s}", profile_key=f"{ds}/s{s}#a"))
    win = pd.DataFrame(rows)
    E = rng.normal(size=(len(win), 768)).astype(np.float16)
    for ds in ("clas", "maus", "drozy"):
        m = (win.dataset_id == ds).to_numpy()
        np.savez(tmp_path / f"emb_{ds}.npz", row_id=win.row_id[m].to_numpy(), emb=E[m])
        keys = np.repeat([f"{ds}/s{s}#a" for s in range(3)], 4)
        np.savez(tmp_path / f"ref_{ds}.npz", profile_key=keys, emb=rng.normal(size=(len(keys), 768)).astype(np.float16))
    prof = win[["profile_key"]].drop_duplicates().reset_index(drop=True)
    return win, prof


def test_attach_and_z(tmp_path):
    win, prof = _fake(tmp_path)
    comps, var = fit_pca(win, tmp_path)
    assert comps.shape == (32, 768) and 0 < var.sum() <= 1
    w2, p2 = attach(win, prof, tmp_path, comps)
    assert np.isfinite(w2[NW_COLS].to_numpy()).all()
    assert np.isfinite(p2[[f"m_{c}" for c in NW_COLS]].to_numpy()).all()
    Z = z_of(w2, w2, p2)
    assert Z.shape == (len(w2), 32) and np.abs(Z).max() <= 5


def test_missing_dataset_is_nan(tmp_path):
    win, prof = _fake(tmp_path)
    comps, _ = fit_pca(win, tmp_path)
    (tmp_path / "emb_drozy.npz").unlink(); (tmp_path / "ref_drozy.npz").unlink()
    w2, p2 = attach(win, prof, tmp_path, comps)
    d = (w2.dataset_id == "drozy").to_numpy()
    assert np.isnan(w2.loc[d, NW_COLS].to_numpy()).all() and np.isfinite(w2.loc[~d, NW_COLS].to_numpy()).all()
