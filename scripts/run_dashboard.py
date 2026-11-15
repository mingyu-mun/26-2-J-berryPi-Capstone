#!/usr/bin/env python3
"""Serve the J_Berry PI dashboard on a Raspberry Pi or development PC."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="기본: 같은 네트워크에서 접속 허용")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--interval", type=float, default=.55, help="자동 재생 1단계의 실제 초 (0.1~10)")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or not .1 <= args.interval <= 10:
        parser.error("port는 0~65535, interval은 0.1~10 범위여야 합니다.")
    try:
        from driver_state.web_dashboard import Dashboard, create_server
    except ModuleNotFoundError as exc:
        parser.exit(1, f"필수 패키지가 없습니다: {exc.name}. web/dashboard/requirements.txt를 설치하세요.\n")
    engine = Dashboard(args.mode, args.interval)
    try:
        server = create_server(args.host, args.port, dashboard=engine)
    except OSError as exc:
        parser.exit(1, f"서버 시작 실패: {exc}\n")
    print(f"J_Berry PI: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"같은 네트워크: http://<Pi IP>:{server.server_port} / 종료: Ctrl+C", flush=True)
    print("모델 구현 미연결. demo는 합성 입력, live는 외부 추론 출력 수신 모드입니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
