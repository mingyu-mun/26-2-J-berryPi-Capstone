"""Server-side catalog and raw ADC/IR reader restricted to one recordings root."""
from contextlib import ExitStack
from itertools import islice
from pathlib import Path
import csv
import hashlib
import json
import math

from .streaming_io import Sample
from .webboard_library import LibraryError


class RecordingCatalog:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}

    def inside(self, path):
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise LibraryError('기록 파일은 지정한 로컬 폴더 안에 있어야 합니다.')
        return path

    def refresh(self):
        entries, errors, folders = {}, [], {self.root}
        folders.update(p for p in self.root.iterdir() if p.is_dir() and not p.is_symlink())
        manifests = list(islice(self.root.rglob('recording.json'), 101))
        if len(manifests) > 100:
            raise LibraryError('기록은 폴더당 최대 100개까지 조회합니다.')
        candidates = []
        for path in manifests:
            folders.discard(path.parent)
            try:
                self.inside(path)
                if path.stat().st_size > 16384:
                    raise ValueError('recording.json은 16KB 이하여야 합니다.')
                meta = json.loads(path.read_text(encoding='utf-8-sig'))
                candidates.append((path.parent, meta['ecg'], meta['ppg'], meta['meta'], meta.get('title', path.parent.name)))
                folders.discard(path.parent)
            except (OSError, ValueError, KeyError, TypeError, LibraryError) as exc:
                errors.append(f'{path.name}: {exc}')
        for folder in sorted(folders):
            ecgs, ppgs = list(folder.glob('원본_ecg_*.csv')), list(folder.glob('원본_ppg_*.csv'))
            if not ecgs and not ppgs:
                continue
            metas = list(islice(folder.rglob('meta.json'), 101))
            if len(ecgs)==len(ppgs)==1 and len(metas)>1:
                # Some supplied archives contain metadata for later recordings.
                # Accept only an explicit source filename declaration, never a
                # nearest timestamp or an inferred sampling rate.
                stem=ecgs[0].stem.removeprefix('원본_ecg_')
                if ppgs[0].stem.removeprefix('원본_ppg_')==stem:
                    declared=[]
                    for path in metas[:100]:
                        try:
                            self.inside(path)
                            if path.stat().st_size<=65536 and json.loads(path.read_text(encoding='utf-8-sig')).get('notes','').startswith(f'원본_*_{stem}.csv '):
                                declared.append(path)
                        except (OSError,ValueError,TypeError,LibraryError):
                            pass
                    metas=declared
            if len(ecgs) == len(ppgs) == len(metas) == 1:
                candidates.append((folder, str(ecgs[0].relative_to(folder)), str(ppgs[0].relative_to(folder)),
                                   str(metas[0].relative_to(folder)), folder.name))
            else:
                errors.append(f'{folder.name}: 원본 쌍·meta.json이 여러 개이거나 빠졌습니다. recording.json으로 지정하세요.')
        for folder, ecg, ppg, meta, title in candidates[:100]:
            try:
                paths = {key: self.inside(folder / value) for key, value in dict(ecg=ecg, ppg=ppg, meta=meta).items()}
                if paths['meta'].stat().st_size > 65536:
                    raise ValueError('meta.json은 64KB 이하여야 합니다.')
                info = json.loads(paths['meta'].read_text(encoding='utf-8-sig'))
                rates = {c: float(info['fs'][c]) for c in ('ecg', 'ppg')}
                if any(not math.isfinite(fs) or not 100 <= fs <= 2000 for fs in rates.values()):
                    raise ValueError('meta.json에 실제 ECG·PPG fs(100~2000Hz)가 필요합니다.')
                signature = {k: [str(v.relative_to(self.root)), v.stat().st_size, v.stat().st_mtime_ns] for k, v in paths.items()}
                identifier = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:32]
                entries[identifier] = dict(id=identifier, title=str(title)[:120], paths=paths, rates=rates,
                    source_session_id=str(info.get('session_id', folder.name)),
                    path=str(folder.relative_to(self.root)) or '.', signature=signature)
            except (OSError, ValueError, KeyError, TypeError, LibraryError) as exc:
                errors.append(f'{folder.name}: {exc}')
        self.entries = entries
        return dict(recordings=[self.public(v) for v in entries.values()], errors=errors,
                    directory=str(self.root))

    @staticmethod
    def public(entry):
        return {k: entry[k] for k in ('id', 'title', 'rates', 'source_session_id', 'path')}

    def get(self, identifier):
        if not isinstance(identifier,str):
            raise LibraryError('기록 ID가 필요합니다.')
        self.refresh()
        if identifier not in self.entries:
            raise LibraryError('기록이 없거나 파일이 변경됐습니다. 목록을 새로고침하세요.', 404)
        return self.entries[identifier]


class RawRecording:
    def __init__(self, entry):
        self.entry = entry
        self.rates = entry['rates']
        self.positions = dict(ecg=0, ppg=0)

    def __enter__(self):
        self.stack = ExitStack()
        self.readers = {}
        durations = []
        try:
            for channel in ('ecg', 'ppg'):
                path = self.entry['paths'][channel]
                if path.stat().st_size > 128 * 1024 * 1024:
                    raise LibraryError('원본 CSV는 채널당 128MB 이하여야 합니다.')
                handle = self.stack.enter_context(path.open(encoding='utf-8-sig', newline=''))
                reader = csv.reader(handle)
                expected = 2 if channel == 'ecg' else 3
                header = next(reader, [])
                if len(header) != expected:
                    raise LibraryError('ECG(time,ADC)·PPG(time,red,IR) 원본이 필요합니다. 가공된 value CSV는 입력할 수 없습니다.')
                value_header=header[1 if channel=='ecg' else 2].lower()
                if not any(word in value_header for word in (('adc','raw','원본') if channel=='ecg' else ('ir','적외선'))):
                    raise LibraryError('원본 ADC·IR 열 이름을 확인하세요. 전처리 파형을 원본으로 해석할 수 없습니다.')
                self.readers[channel] = reader
                with path.open(encoding='utf-8-sig') as count_handle:
                    count = sum(1 for _ in count_handle) - 1
                durations.append(count / self.rates[channel])
            self.duration = float(math.floor(min(durations)))
            if self.duration < 31:
                raise LibraryError('ECG·PPG 공통 원본 구간이 31초 이상 필요합니다.')
            return self
        except Exception:
            self.stack.close()
            raise

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def emit_until(self, runtime, end):
        blocks = {}
        for channel, reader in self.readers.items():
            count = math.ceil(end * self.rates[channel] - 1e-9) - self.positions[channel]
            rows = list(islice(reader, count))
            if len(rows) != count:
                return False
            if any(len(row) != (2 if channel == 'ecg' else 3) for row in rows):
                raise LibraryError('원본 CSV의 열 개수가 맞지 않습니다.')
            values = [float(row[1 if channel == 'ecg' else 2]) for row in rows]
            upper = runtime.config['serial']['adc_max'] if channel == 'ecg' else 262143
            if any(not math.isfinite(v) or v > upper or v < (-1 if channel == 'ecg' else 0) for v in values):
                raise LibraryError('원본 ADC·IR에 범위 밖 값 또는 NaN·무한대가 있습니다.')
            blocks[channel] = values
        for channel, values in blocks.items():
            for index, value in enumerate(values, self.positions[channel]):
                good = value >= 0 if channel == 'ecg' else value >= runtime.config['ppg']['finger_threshold']
                runtime.emit(channel, Sample(runtime.epoch + index / self.rates[channel], value, good))
            self.positions[channel] += len(values)
        return True
