"""
setup_env.py — 이 PC 에 J_Berry PI 추론 환경(.venv)을 만든다. Unity 메뉴 [Capstone > AI 추론 환경 설치] 가 실행.
  1) Python 3.10~3.13 찾기 (지금 실행한 Python, 안 맞으면 py -3.12/-3.11/-3.13/-3.10)
  2) Tools/jberry_ai/.venv 생성
  3) requirements.txt + torch(CPU) 설치
  4) selftest.py: 저장소 모델 로드 + 실측 기록 앞부분으로 실제 추론 1회
--no-venv: 가상환경을 만들지 않고 지금 실행한 Python 에 바로 설치 (setup_ai.ps1 의 휴대용 Python 용)
결과: SimLogs/ai/setup_log.txt (전체 출력), SimLogs/ai/setup_status.txt (ok / fail: 이유)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOGD = ROOT / "SimLogs" / "ai"
LOGD.mkdir(parents=True, exist_ok=True)
LOG = open(LOGD / "setup_log.txt", "w", encoding="utf-8")
STATUS = LOGD / "setup_status.txt"
NO_VENV = "--no-venv" in sys.argv
VENV = HERE / ".venv"
VPY = Path(sys.executable) if NO_VENV else VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(time.strftime("[%H:%M:%S] ") + s + "\n")
    LOG.flush()


def status(s):
    STATUS.write_text(s + "\n", encoding="utf-8")
    log("STATUS:", s)


def run(cmd, timeout=3600):
    log("$", " ".join(str(c) for c in cmd))
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", cwd=str(HERE))
    for line in p.stdout:
        LOG.write("    " + line)
        LOG.flush()
    p.wait(timeout=timeout)
    log("  → exit", p.returncode)
    return p.returncode


def version_of(cmd):
    try:
        out = subprocess.run(cmd + ["-c", "import sys;print('%d.%d'%sys.version_info[:2])"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        return tuple(int(x) for x in out.split("."))
    except Exception:
        return None


def find_base():
    cands = [[sys.executable]] + [["py", f"-3.{m}"] for m in (12, 11, 13, 10)]
    for c in cands:
        v = version_of(c)
        log("후보", " ".join(c), "→", v)
        if v and (3, 10) <= v <= (3, 13):
            return c
    return None


def main():
    status("running: 준비")
    log("setup 시작", sys.version.replace("\n", " "), sys.executable)
    if NO_VENV:
        log("가상환경 없이 이 Python 에 설치:", VPY)
    elif not (VPY.exists() and version_of([str(VPY)])):
        base = find_base()
        if base is None:
            status("fail: Python 3.10~3.13 이 없음 — python.org 에서 3.12 설치 후 다시 실행")
            return 1
        if run(base + ["-m", "venv", str(VENV)]) != 0:
            status("fail: venv 생성 실패")
            return 1
    else:
        log("기존 .venv 사용:", VPY)
    if run([VPY, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        log("pip 업그레이드 실패 (계속 진행)")
    status("running: 패키지 설치 (numpy·scipy·pandas …)")
    if run([VPY, "-m", "pip", "install", "--no-warn-script-location", "-r", HERE / "requirements.txt"]) != 0:
        status("fail: 패키지 설치 실패 (setup_log.txt 참고)")
        return 1
    status("running: torch(CPU) 내려받기 · 약 200 MB")
    if run([VPY, "-m", "pip", "install", "--no-warn-script-location", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"]) != 0:
        status("fail: torch(CPU) 설치 실패 (setup_log.txt 참고)")
        return 1
    status("running: 모델 시험 추론")
    if run([VPY, "-u", HERE / "selftest.py"]) != 0:
        status("fail: selftest 실패 (setup_log.txt 참고)")
        return 1
    status("ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        status(f"fail: {e!r}")
        raise
