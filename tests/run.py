#!/usr/bin/env python3
"""Physical Pi sensors -> preprocessing -> CPU model -> webboard."""
from datetime import datetime
from pathlib import Path
from threading import Thread
import argparse
import logging
import platform
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))


def default_config():
    current = HERE / 'sensors.local.json'
    previous = ROOT / 'pi-integration-test' / 'sensors.local.json'
    return current if current.exists() or not previous.exists() else previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(default_config()))
    parser.add_argument('--check', action='store_true', help='load configuration and weights without opening hardware')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--out', help='diagnostic log directory; default: results/<unique run>')
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    output = None
    if not args.check:
        output = Path(args.out).resolve() if args.out else HERE / 'results' / (
            datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output / 'run.log', encoding='utf-8'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=handlers, force=True)
    web = runtime = None
    try:
        if not sys.platform.startswith('linux'):
            raise ValueError('실물 Pi의 Linux 터미널에서 실행하세요.')
        import serial  # Verify packages before starting acquisition threads.
        import smbus2
        from driver_state.pi_runtime import Runtime, load_config
        from driver_state.streaming_io import Publisher
        from driver_state.streaming_model import CausalStudent
        from driver_state.web_dashboard import create_server

        config = load_config(args.config)
        backend = CausalStudent(ROOT / 'models', feature_mode=config['inference']['feature_mode'],
                                experimental=config['inference']['experimental'],
                                threads=config['inference'].get('threads', 1),
                                rpeak_threshold=config['inference'].get('rpeak_threshold', 'global'))
        logging.info('실행 환경: %s / %s / Python %s', platform.system(), platform.machine(), platform.python_version())
        logging.info('S1-EP 가중치 %d개 CPU 로딩 완료', len(backend.models))
        if args.check:
            logging.info('설정·패키지·가중치 확인 완료. USB·I²C는 열지 않았습니다.')
            return 0

        logging.warning('EXPERIMENTAL causal_trailing_v1: 과거 구간만 사용하는 시간 특징이며 판정 정확도는 별도 검증 대상입니다.')
        logging.warning('ECG fs_hz=%s는 설정값입니다. 실제 Arduino 주기와 확인하세요.', config['serial']['fs_hz'])
        web = create_server(args.host, args.port)
        Thread(target=web.serve_forever, daemon=True).start()
        address = f'http://127.0.0.1:{web.server_port}'
        publisher = Publisher(address + '/api/events')
        runtime = Runtime(config, backend, publisher)
        logging.info('Pi 웹보드: %s (같은 네트워크에서는 Pi IP로 접속)', address)
        logging.info('실행 로그: %s', output / 'run.log')
        logging.info('센서 설정: %s', Path(args.config).resolve())
        logging.info('Arduino USB + MAX30102 I²C 수집 시작. 웹보드 입력: 이 Pi의 수신 데이터. 종료: Ctrl+C')
        runtime.run()
        return 0
    except KeyboardInterrupt:
        logging.info('종료 요청을 받았습니다.')
        return 0
    except Exception:
        logging.exception('Pi 통합테스트 실패')
        return 1
    finally:
        if runtime and not runtime.stop.is_set():
            runtime.close()
        if web:
            web.shutdown()
            web.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
