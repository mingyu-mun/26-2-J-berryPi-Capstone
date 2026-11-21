#!/usr/bin/env bash
# Raspberry Pi OS 64-bit: install, check software, or start live sensors.
set -euo pipefail

PI_TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI_PROJECT_ROOT="$(cd -- "$PI_TEST_DIR/.." && pwd)"
cd "$PI_PROJECT_ROOT"
PI_ACTION="${1:-help}"
if (($#)); then shift; fi
PI_VENV_PYTHON="$PI_TEST_DIR/.venv/bin/python"
PI_LOCAL_CONFIG="$PI_TEST_DIR/sensors.local.json"
PI_PREVIOUS_DIR="$PI_PROJECT_ROOT/pi-integration-test"

usage() {
    cat <<'EOF'
사용법: 이 폴더에서 bash pi_test.sh setup|check|run
  setup  이 폴더의 .venv에 패키지 설치, 로컬 설정이 없을 때만 예시 복사
  check  설정·패키지·가중치 로딩 확인 (USB·I²C는 열지 않음)
  run    실제 센서 수집 → 전처리 → CPU 모델 → 웹보드 시작
run 뒤에 --port 8002 등 센서 실행 옵션을 추가할 수 있습니다.
안내: web/dashboard/README.md
EOF
}

case "$PI_ACTION" in
    help|-h|--help) usage; exit 0 ;;
    setup|check|run) ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "$(uname -s)" != Linux ]]; then
    echo '실물 Pi의 Linux 터미널에서 실행하세요.' >&2
    exit 1
fi

if [[ "$PI_ACTION" == setup ]]; then
    python3 - <<'PY'
import platform
import struct
import sys
if struct.calcsize('P') != 8 or not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise SystemExit('64비트 Python 3.10~3.12가 필요합니다. 권장: Raspberry Pi OS Bookworm 64비트 / Python 3.11.')
if platform.machine() not in ('aarch64', 'arm64'):
    raise SystemExit('setup은 실제 Pi의 ARM64 환경에서 실행하세요.')
PY
    python3 -m venv "$PI_TEST_DIR/.venv"
    "$PI_VENV_PYTHON" -X utf8 -m pip install --only-binary=:all: -r "$PI_TEST_DIR/requirements.txt"
    if [[ ! -e "$PI_LOCAL_CONFIG" ]]; then
        if [[ -f "$PI_PREVIOUS_DIR/sensors.local.json" ]]; then
            cp "$PI_PREVIOUS_DIR/sensors.local.json" "$PI_LOCAL_CONFIG"
        elif [[ -f "$PI_PROJECT_ROOT/configs/pi-sensors.local.json" ]]; then
            cp "$PI_PROJECT_ROOT/configs/pi-sensors.local.json" "$PI_LOCAL_CONFIG"
        else
            cp "$PI_TEST_DIR/sensors.example.json" "$PI_LOCAL_CONFIG"
        fi
    fi
    echo '설치 완료. tests/sensors.local.json의 USB 형식·Hz·전압을 확인하세요.'
    echo '다음: 이 폴더에서 bash pi_test.sh check'
    exit 0
fi

# Reuse a previous installation for check/run until setup creates the new one.
if [[ ! -x "$PI_VENV_PYTHON" && -x "$PI_PREVIOUS_DIR/.venv/bin/python" ]]; then
    PI_VENV_PYTHON="$PI_PREVIOUS_DIR/.venv/bin/python"
fi
if [[ ! -f "$PI_LOCAL_CONFIG" && -f "$PI_PREVIOUS_DIR/sensors.local.json" ]]; then
    PI_LOCAL_CONFIG="$PI_PREVIOUS_DIR/sensors.local.json"
fi
if [[ ! -x "$PI_VENV_PYTHON" || ! -f "$PI_LOCAL_CONFIG" ]]; then
    echo '먼저 tests 폴더에서 bash pi_test.sh setup을 실행하세요.' >&2
    exit 1
fi

export PYTHONUTF8=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "$PI_ACTION" == check ]]; then
    exec "$PI_VENV_PYTHON" "$PI_TEST_DIR/run.py" --config "$PI_LOCAL_CONFIG" --check "$@"
fi

exec "$PI_VENV_PYTHON" "$PI_TEST_DIR/run.py" --config "$PI_LOCAL_CONFIG" "$@"
