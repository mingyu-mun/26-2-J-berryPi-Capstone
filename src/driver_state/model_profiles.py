"""Versioned numeric baseline profiles. No pickles or measurement synthesis."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import os
import tempfile
import uuid
import zipfile

import numpy as np

from .webboard_library import LibraryError, text_field

MAX_PROFILE_BYTES = 256 * 1024


def model_identity(model_dir):
    root = Path(model_dir)
    paths = [root / 'bundle.json', *sorted((root / 'student').glob('S1-EP_s*.safetensors'))]
    if len(paths) != 4:
        raise LibraryError('S1-EP 가중치 3개와 bundle.json이 필요합니다.')
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        content = (json.dumps(json.loads(path.read_text(encoding='utf-8-sig')),sort_keys=True,
                              separators=(',',':'),allow_nan=False).encode()
                   if path.suffix == '.json' else path.read_bytes())
        digest.update(content)
    code = hashlib.sha256()
    here = Path(__file__).parent
    for name in ('preprocessing.py', 'features.py', 'spec.py', 'teacher/rr.py', 'teacher/hrv.py',
                 'teacher/features.py', 'teacher/spec.py', 'streaming_model.py', 'pi_runtime.py',
                 'local_recordings.py', 'student/distill.py'):
        code.update((here / name).read_bytes().replace(b'\r\n',b'\n'))
    return dict(model_sha256=digest.hexdigest(), pipeline_sha256=code.hexdigest(),
                feature_mode='causal_trailing_v1', model_id='S1-EP')


def validate_profile(snapshot, identity=None, processing=None):
    from .teacher.spec import TeacherSpec
    try:
        meta = snapshot['metadata']
        names = meta['feature_names']
        stats = np.asarray(snapshot['statistics'])
        refs = np.asarray(snapshot['references'])
        if (meta['format_version'] != 1 or names != list(TeacherSpec().baseline_feature_names)
                or not all(isinstance(n, str) for n in names)
                or stats.shape != (2, 2, 36) or refs.shape != (3, 2, 32)
                or stats.dtype != np.float64 or refs.dtype != np.float32
                or np.isinf(stats).any() or not np.isfinite(refs).all()
                or np.any(stats[:, 1][np.isfinite(stats[:, 1])] < 0)
                or meta['clean_windows'] < 20 or meta['end_s'] - meta['start_s'] < 300
                or not isinstance(meta['source_session_id'], str) or not meta['source_session_id']):
            raise ValueError()
        for key in ('hr_30', 'hr_120', 'hr_300'):
            if not np.isfinite(stats[:, 0, names.index(key)]).all():
                raise ValueError()
        ref = meta['ecg_reference']
        if not (np.isfinite(ref['mean_bpm']) and ref['mean_bpm'] > 0 and ref['clean_windows'] > 0):
            raise ValueError()
        json.dumps(meta, allow_nan=False)
        if identity is not None and meta['identity'] != identity:
            raise LibraryError('모델 가중치·특징 처리 버전이 다른 기준입니다. 기준을 다시 측정하세요.', 409)
        if processing is not None and meta['processing'] != processing:
            raise LibraryError('전압·공통 주파수·PPG 보정 설정이 다른 기준입니다.', 409)
    except LibraryError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        raise LibraryError('불완전하거나 손상된 모델 기준입니다. 필요한 통계·참조값을 확인하세요.') from None
    return snapshot


class ModelProfileStore:
    def __init__(self, library):
        self.library = library
        self.directory = library.directory / 'profiles'
        self.directory.mkdir(exist_ok=True)
        with library.lock, library.db:
            library.db.execute('''CREATE TABLE IF NOT EXISTS model_profiles (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), title TEXT NOT NULL,
                created_at TEXT NOT NULL, metadata_json TEXT NOT NULL, sha256 TEXT NOT NULL)''')

    def list(self, user_id):
        self.library.user(user_id)
        with self.library.lock:
            rows = self.library.db.execute('''SELECT id,title,created_at,metadata_json FROM model_profiles
                WHERE user_id=? ORDER BY created_at DESC,id DESC''', (user_id,)).fetchall()
        return [dict(id=r['id'], title=r['title'], created_at=r['created_at'],
                     metadata=json.loads(r['metadata_json'])) for r in rows]

    def save(self, user_id, title, snapshot):
        self.library.user(user_id)
        validate_profile(snapshot)
        title = text_field(title, '기준 이름', 80)
        identifier = uuid.uuid4().hex
        created = datetime.now(timezone.utc).isoformat(timespec='seconds')
        meta = dict(snapshot['metadata'], profile_id=identifier, user_id=user_id,
                    title=title, created_at=created)
        body = io.BytesIO()
        np.savez_compressed(body, metadata=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)),
                            statistics=snapshot['statistics'], references=snapshot['references'])
        data = body.getvalue()
        if len(data) > MAX_PROFILE_BYTES:
            raise LibraryError('모델 기준 파일이 보관 한도를 초과했습니다.')
        path = self.directory / f'{identifier}.npz'
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, path)
            with self.library.lock, self.library.db:
                self.library.db.execute('INSERT INTO model_profiles VALUES(?,?,?,?,?,?)',
                    (identifier, user_id, title, created, json.dumps(meta, ensure_ascii=False),
                     hashlib.sha256(data).hexdigest()))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        return next(item for item in self.list(user_id) if item['id'] == identifier)

    def load(self, user_id, identifier, identity=None, processing=None):
        self.library.user(user_id)
        with self.library.lock:
            row = self.library.db.execute('SELECT * FROM model_profiles WHERE id=? AND user_id=?',
                                          (identifier, user_id)).fetchone()
        if row is None:
            raise LibraryError('이 사용자의 모델 기준을 찾을 수 없습니다.', 404)
        # ID comes from SQLite, never from a user supplied filesystem path.
        path = self.directory / f'{row["id"]}.npz'
        try:
            if path.stat().st_size > MAX_PROFILE_BYTES:
                raise ValueError()
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if len(archive.infolist()) != 3 or sum(x.file_size for x in archive.infolist()) > MAX_PROFILE_BYTES:
                    raise ValueError()
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                snapshot = dict(metadata=json.loads(str(archive['metadata'])),
                                statistics=archive['statistics'].copy(), references=archive['references'].copy())
            if snapshot['metadata'] != json.loads(row['metadata_json']):
                raise ValueError()
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            raise LibraryError('기준 파일이 없거나 손상됐습니다. 기준을 다시 저장하세요.') from None
        return validate_profile(snapshot, identity, processing)
