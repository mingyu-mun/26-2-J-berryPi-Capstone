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
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port는 0~65535 범위여야 합니다.")
    try:
        from driver_state.web_dashboard import Dashboard, create_server
    except ModuleNotFoundError as exc:
        parser.exit(1, f"필수 패키지가 없습니다: {exc.name}. web/dashboard/requirements.txt를 설치하세요.\n")
    engine = Dashboard()
    try:
        server = create_server(args.host, args.port, dashboard=engine)
    except OSError as exc:
        parser.exit(1, f"서버 시작 실패: {exc}\n")
    print(f"J_Berry PI: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"같은 네트워크: http://<Pi IP>:{server.server_port} / 종료: Ctrl+C", flush=True)
    print("데이터 수신 대기. 실제 JSON 이벤트가 들어오기 전에는 측정값을 표시하지 않습니다.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
