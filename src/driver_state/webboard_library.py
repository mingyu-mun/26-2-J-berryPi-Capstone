"""Nickname-owned baseline recordings. Playback does not modify model profiles."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import sqlite3
import threading
import unicodedata
import uuid

MAX_RECORDING_BYTES = 4 * 1024 * 1024
MAX_RECORDING_EVENTS = 10000
DEFAULT_LIBRARY_DIR = Path(__file__).resolve().parents[2] / 'outputs' / 'webboard-users'


class LibraryError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def text_field(value, name, limit):
    if not isinstance(value, str):
        raise LibraryError(f'{name}을 입력하세요.')
    value = unicodedata.normalize('NFC', value).strip()
    if not value or len(value) > limit or any(unicodedata.category(c).startswith('C') for c in value):
        raise LibraryError(f'{name}은 제어문자 없이 1~{limit}자로 입력하세요.')
    return value


def is_ready(event):
    return event['type'] == 'system.status' and event['payload']['bio_baseline']['status'] == 'ready'


def prepare_recording(events, validate):
    if not isinstance(events, list) or not 0 < len(events) <= MAX_RECORDING_EVENTS:
        raise LibraryError('1~10,000개의 규약 이벤트가 필요합니다.')
    for event in events:
        validate(event)
    if len({e['session_id'] for e in events}) != 1:
        raise LibraryError('한 사람의 한 세션만 등록할 수 있습니다.')
    ordered = sorted(events, key=lambda e: e['t_rel_s'])
    ready = next((e for e in ordered if is_ready(e)), None)
    if ready is None:
        raise LibraryError('생체 기준 준비 완료(system.status: ready)가 포함된 기록이 필요합니다.')
    end = ready['t_rel_s']
    recording = [e for e in ordered if e['t_rel_s'] <= end]
    if not any(e['type'] == 'system.status' and e['payload']['bio_baseline']['status'] == 'collecting'
               for e in recording):
        raise LibraryError('기준 수집 중(collecting)부터 준비 완료까지의 기록이 필요합니다.')
    if not any(e['type'] == 'bio.wave' and e['payload']['ecg'] is not None and e['payload']['ppg'] is not None
               for e in recording):
        raise LibraryError('ECG와 PPG 파형이 함께 포함된 기준 기록이 필요합니다.')
    encoded = json.dumps(recording, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(encoded.encode('utf-8')) > MAX_RECORDING_BYTES:
        raise LibraryError('기준 기록은 4MB 이하여야 합니다.', 413)
    return recording, encoded, ready['payload']['bio_baseline']


class BaselineCapture:
    """Bounded, unassigned copy of the current session until baseline is ready."""
    def __init__(self):
        self.session = None
        self.events = []
        self.bytes = 0
        self.ready_t = None
        self.overflow = False

    def observe(self, event, size):
        if event['session_id'] != self.session:
            self.__init__()
            self.session = event['session_id']
        if self.overflow or self.ready_t is not None and event['t_rel_s'] > self.ready_t:
            return
        if len(self.events) >= MAX_RECORDING_EVENTS or self.bytes + size > MAX_RECORDING_BYTES:
            self.overflow = True
            self.events.clear()
            return
        self.events.append(event)
        self.bytes += size
        if self.ready_t is None and is_ready(event):
            self.ready_t = event['t_rel_s']

    def status(self):
        events = [e for e in self.events if self.ready_t is None or e['t_rel_s'] <= self.ready_t]
        latest_status = max((e for e in events if e['type'] == 'system.status'),
                            key=lambda e: e['t_rel_s'], default=None)
        collecting = any(e['type'] == 'system.status' and e['payload']['bio_baseline']['status'] == 'collecting'
                         for e in events)
        waves = any(e['type'] == 'bio.wave' and e['payload']['ecg'] is not None and e['payload']['ppg'] is not None
                    for e in events)
        return dict(session_id=self.session, event_count=len(self.events),
                    state=latest_status['payload']['bio_baseline']['status'] if latest_status else 'waiting',
                    ready=self.ready_t is not None and collecting and waves and not self.overflow,
                    overflow=self.overflow)

    def snapshot(self, expected_session):
        if not expected_session or expected_session != self.session:
            raise LibraryError('수신 세션이 바뀌었습니다. 현재 기준 정보를 새로 확인하세요.', 409)
        if not self.status()['ready']:
            raise LibraryError('기준 수집과 ECG·PPG 파형 수신이 완료된 뒤 저장할 수 있습니다.', 409)
        return copy.deepcopy(self.events)


class BaselineLibrary:
    def __init__(self, directory=DEFAULT_LIBRARY_DIR):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory.resolve()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(directory / 'library.sqlite3', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, nickname TEXT NOT NULL, nickname_key TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS baselines (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), title TEXT NOT NULL,
                session_id TEXT NOT NULL, event_count INTEGER NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL,
                clean_windows INTEGER, origins TEXT NOT NULL, created_at TEXT NOT NULL, events_json TEXT NOT NULL
            );
        ''')

    def close(self):
        with self.lock:
            self.db.close()

    def users(self):
        with self.lock:
            return [dict(r) for r in self.db.execute('''
                SELECT u.id,u.nickname,u.created_at,COUNT(b.id) AS baseline_count
                FROM users u LEFT JOIN baselines b ON b.user_id=u.id
                GROUP BY u.id ORDER BY u.nickname_key,u.id
            ''')]

    def user(self, user_id):
        with self.lock:
            row = self.db.execute('SELECT id,nickname,created_at FROM users WHERE id=?', (user_id,)).fetchone()
            if row is None:
                raise LibraryError('사용자를 찾을 수 없습니다.', 404)
            return dict(row)

    def add_user(self, nickname):
        nickname = text_field(nickname, '닉네임', 32)
        user_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self.lock:
            try:
                with self.db:
                    self.db.execute('INSERT INTO users VALUES(?,?,?,?)', (user_id, nickname, nickname.casefold(), now))
            except sqlite3.IntegrityError:
                raise LibraryError('이미 등록된 닉네임입니다. 목록에서 선택하세요.', 409) from None
        return self.user(user_id)

    @staticmethod
    def metadata(row):
        result = {key: row[key] for key in row.keys() if key != 'events_json'}
        result['origins'] = json.loads(result['origins'])
        result['duration_s'] = result['end_s'] - result['start_s']
        return result

    def list_baselines(self, user_id):
        self.user(user_id)
        with self.lock:
            rows = self.db.execute('''SELECT id,user_id,title,session_id,event_count,start_s,end_s,
                clean_windows,origins,created_at FROM baselines WHERE user_id=? ORDER BY created_at DESC,id DESC''', (user_id,))
            return [self.metadata(row) for row in rows]

    def save(self, user_id, title, events, validate):
        self.user(user_id)
        title = text_field(title, '기록 이름', 80)
        recording, encoded, baseline = prepare_recording(events, validate)
        recording_id = uuid.uuid4().hex
        values = (recording_id, user_id, title, recording[0]['session_id'], len(recording),
                  min(e['t_rel_s'] for e in recording), max(e['t_rel_s'] for e in recording), baseline['clean_windows'],
                  json.dumps(sorted({e['origin'] for e in recording})),
                  datetime.now(timezone.utc).isoformat(timespec='seconds'), encoded)
        with self.lock, self.db:
            self.db.execute('INSERT INTO baselines VALUES(?,?,?,?,?,?,?,?,?,?,?)', values)
        return self.get(user_id, recording_id)['baseline']

    def get(self, user_id, recording_id):
        self.user(user_id)
        with self.lock:
            row = self.db.execute('SELECT * FROM baselines WHERE user_id=? AND id=?', (user_id, recording_id)).fetchone()
            if row is None:
                raise LibraryError('이 사용자의 기준 기록을 찾을 수 없습니다.', 404)
            return dict(baseline=self.metadata(row), events=json.loads(row['events_json']))
