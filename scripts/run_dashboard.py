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
    parser.add_argument("--data-dir", help="닉네임·기준 기록 저장 폴더 (기본: outputs/webboard-users)")
    parser.add_argument('--recordings-dir', default=str(Path(__file__).resolve().parents[1]/'outputs/local-recordings'),
                        help='서버에서 원본 ECG·PPG 기록을 조회할 폴더')
    parser.add_argument('--models', default=str(Path(__file__).resolve().parents[1]/'models'))
    parser.add_argument('--external-events', action='store_true',
                        help='팀 송신 프로그램의 /api/events 입력만 표시; 로컬 기록 제어 비활성화')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("port는 0~65535 범위여야 합니다.")
    try:
        from driver_state.web_dashboard import Dashboard, create_server
    except ModuleNotFoundError as exc:
        parser.exit(1, f"필수 패키지가 없습니다: {exc.name}. web/dashboard/requirements.txt를 설치하세요.\n")
    engine = Dashboard()
    try:
        options = {'library_dir': args.data_dir} if args.data_dir else {}
        if not args.external_events:
            options.update(recordings_dir=args.recordings_dir,model_dir=args.models)
        server = create_server(args.host, args.port, dashboard=engine, **options)
    except OSError as exc:
        parser.exit(1, f"서버 시작 실패: {exc}\n")
    print(f"J_Berry PI: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"같은 네트워크: http://<Pi IP>:{server.server_port} / 종료: Ctrl+C", flush=True)
    print('외부 이벤트 수신 모드: 기존 송신 프로그램을 /api/events에 연결하세요.' if args.external_events else
          '로컬 기록 모드: 사용자 설정·시작에서 원본과 기준을 선택하세요.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버 종료", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
