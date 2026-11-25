// LiveBioSource.cs  —  실측 모드 (데모 단계, 틀)
// 라즈베리파이(센서 허브)에서 오는 ECG/PPG 를 실시간으로 받아 BioReplay 와 같은 방식으로 AiLink 에 넘긴다.
//   전송 방식은 아직 미정 → Transport 로 고를 수 있게 틀만 두고, 지금은 UDP 만 동작한다.
//     UDP       : 구현 (Wi-Fi/유선 LAN). 패킷 = docs/06_실측모드.md 의 LivePacket JSON
//     Serial    : 틀 (USB-UART, 라즈베리파이를 USB 가젯으로 쓰는 경우)
//     MQTT      : 틀 (브로커 경유, 웹보드 등 여러 구독자가 같은 데이터를 받을 때 유리)
//     WebSocket : 틀 (웹보드 규약이 제안하는 단일 채널)
//   시험: Tools/rpi_bridge/rpi_sender.py 가 기록 파일을 실시간으로 UDP 전송 → 라즈베리파이 없이 경로 점검
// 수신 품질(패킷 속도, 손실, 도착 간격 흔들림, 마지막 수신 후 경과)을 계산해 메뉴·화면에 보여 준다.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace CapstoneSim
{
    /// <summary>라즈베리파이 → 시뮬레이터 생체 패킷 (UDP JSON 한 개 = 묶음 하나)</summary>
    [Serializable]
    public class LivePacket
    {
        public string type;          // "bio"
        public string device;        // 예: "rpi-01"
        public long seq;             // 1씩 증가 (손실 확인)
        public double t_dev;         // 장치 시계 기준 이 묶음 첫 표본 시각 [s]
        public double fs_ecg, fs_ppg;
        public float[] ecg, ppg;     // 원신호 (각자 원래 주기)
    }

    public class LiveBioSource : BioSource
    {
        public enum Transport { Udp, Serial, Mqtt, WebSocket }
        public static readonly string[] TransportNames = { "UDP · Wi-Fi/LAN", "Serial · USB", "MQTT", "WebSocket" };
        public static bool Implemented(Transport t) => t == Transport.Udp;

        [Header("연결")]
        public Transport transport = Transport.Udp;
        public int udpPort = 5010;
        public string serialPort = "COM3";
        public int serialBaud = 921600;
        public string mqttHost = "raspberrypi.local";
        public int mqttPort = 1883;
        public string mqttTopic = "jberry/bio";
        public string wsUrl = "ws://raspberrypi.local:8765/bio";

        public override string Origin => "live";
        public override float ElapsedSec => FsEcg > 0 ? (float)(_samplesEcg / FsEcg) : 0f;
        public string Device { get; private set; } = "";
        public bool Listening { get; private set; }
        /// <summary>true 면 받은 표본을 AiLink 로 넘기려고 쌓아 둔다. false(메뉴의 수신 시험)면 화면 미리보기만</summary>
        public bool Streaming { get; private set; }

        // ---- 수신 품질 ----
        public long Packets { get; private set; }
        public long Lost { get; private set; }
        public float PacketRate { get; private set; }
        public float JitterMs { get; private set; }
        public float LastAgeSec => _lastRx < 0 ? 999f : (float)(_clock.Elapsed.TotalSeconds - _lastRx);
        public bool Receiving => LastAgeSec < 2f;

        readonly Stopwatch _clock = Stopwatch.StartNew();
        UdpClient _udp;
        Thread _thread;
        volatile bool _run;
        readonly object _lock = new object();
        readonly Queue<(LivePacket p, double t)> _inbox = new Queue<(LivePacket, double)>();
        readonly List<float> _outE = new List<float>(8192), _outP = new List<float>(2048);
        readonly float[] _prevE = new float[1200], _prevP = new float[400];
        int _prevEi, _prevPi;
        long _samplesEcg, _lastSeq = -1;
        double _lastRx = -1, _prevArrival = -1;
        readonly Queue<double> _arrivals = new Queue<double>();
        double _jSum, _jSum2; int _jN;

        public override void Begin()
        {
            StartListening();
            StartStreaming();
        }

        public void StartListening()
        {
            if (Listening) return;
            if (!Implemented(transport))
            {
                Status = $"{TransportNames[(int)transport]}: 아직 구현 전 (틀) — docs/06_실측모드.md";
                return;
            }
            try
            {
                _udp = new UdpClient(new IPEndPoint(IPAddress.Any, udpPort));
                _udp.Client.ReceiveBufferSize = 1 << 20;
                _run = true;
                _thread = new Thread(RxLoop) { IsBackground = true, Name = "LiveBio-UDP" };
                _thread.Start();
                Listening = true;
                Status = $"UDP {udpPort} 수신 대기 중";
            }
            catch (Exception e) { Status = $"UDP {udpPort} 열기 실패: {e.Message}"; }
        }

        /// <summary>세션 시작: 지금부터 받은 표본을 세션 시각 0 으로 보고 쌓는다</summary>
        public void StartStreaming()
        {
            lock (_lock) { _outE.Clear(); _outP.Clear(); }
            _samplesEcg = 0;
            Streaming = true;
        }

        public void StopListening()
        {
            _run = false;
            try { _udp?.Close(); } catch { }
            _udp = null; Listening = false; Streaming = false;
        }

        void RxLoop()
        {
            var any = new IPEndPoint(IPAddress.Any, 0);
            while (_run)
            {
                try
                {
                    var data = _udp.Receive(ref any);
                    var p = JsonUtility.FromJson<LivePacket>(Encoding.UTF8.GetString(data));
                    if (p == null || p.type != "bio") continue;
                    double t = _clock.Elapsed.TotalSeconds;
                    lock (_lock) { _inbox.Enqueue((p, t)); while (_inbox.Count > 500) _inbox.Dequeue(); }
                }
                catch (SocketException) { if (!_run) break; }
                catch (ObjectDisposedException) { break; }
                catch (Exception) { /* 형식이 틀린 패킷은 무시 */ }
            }
        }

        void Update()
        {
            while (true)
            {
                (LivePacket p, double t) item;
                lock (_lock) { if (_inbox.Count == 0) break; item = _inbox.Dequeue(); }
                Accept(item.p, item.t);
            }
            while (_arrivals.Count > 0 && _clock.Elapsed.TotalSeconds - _arrivals.Peek() > 2.0) _arrivals.Dequeue();
            PacketRate = _arrivals.Count / 2f;
            if (Listening)
                Status = Receiving ? $"수신 중 · {Device} · {PacketRate:F0} 패킷/s · 손실 {Lost}"
                                   : $"UDP {udpPort} 수신 대기 중" + (Packets > 0 ? $" (마지막 수신 {LastAgeSec:F0}s 전)" : "");
        }

        void Accept(LivePacket p, double t)
        {
            Packets++;
            Device = string.IsNullOrEmpty(p.device) ? "장치" : p.device;
            if (_lastSeq >= 0 && p.seq > _lastSeq + 1) Lost += p.seq - _lastSeq - 1;
            _lastSeq = p.seq;
            if (_prevArrival >= 0)
            {   // 도착 간격의 표준편차 = 흔들림(jitter)
                double dt = (t - _prevArrival) * 1000.0;
                _jSum += dt; _jSum2 += dt * dt; _jN++;
                if (_jN > 200) { _jSum *= 0.5; _jSum2 *= 0.5; _jN /= 2; }
                double m = _jSum / _jN;
                JitterMs = (float)Math.Sqrt(Math.Max(0, _jSum2 / _jN - m * m));
            }
            _prevArrival = t; _lastRx = t; _arrivals.Enqueue(t);
            if (!(FsEcg > 0) && p.fs_ecg > 0) { FsEcg = p.fs_ecg; FsPpg = p.fs_ppg; SessionName = $"live:{Device}"; Loaded = true; }
            if (p.ecg != null) foreach (var v in p.ecg) { _prevE[_prevEi] = v; _prevEi = (_prevEi + 1) % _prevE.Length; }
            if (p.ppg != null) foreach (var v in p.ppg) { _prevP[_prevPi] = v; _prevPi = (_prevPi + 1) % _prevP.Length; }
            if (!Streaming) return;
            lock (_lock)
            {
                if (p.ecg != null) { _outE.AddRange(p.ecg); _samplesEcg += p.ecg.Length; }
                if (p.ppg != null) _outP.AddRange(p.ppg);
            }
        }

        public override bool Drain(out float[] ecg, out float[] ppg, int maxEcg = 2200)
        {
            lock (_lock)
            {
                ecg = ppg = null;
                if (_outE.Count == 0 && _outP.Count == 0) return false;
                int ne = Math.Min(_outE.Count, maxEcg);
                int np = FsEcg > 0 ? Math.Min(_outP.Count, (int)Math.Ceiling(ne * FsPpg / FsEcg) + 1) : _outP.Count;
                ecg = _outE.GetRange(0, ne).ToArray(); _outE.RemoveRange(0, ne);
                ppg = _outP.GetRange(0, np).ToArray(); _outP.RemoveRange(0, np);
                return true;
            }
        }

        /// <summary>메뉴 미리보기용 최근 원신호 (시간 순서)</summary>
        public float[] Preview(bool ecg)
        {
            var src = ecg ? _prevE : _prevP; int head = ecg ? _prevEi : _prevPi;
            var o = new float[src.Length];
            for (int i = 0; i < src.Length; i++) o[i] = src[(head + i) % src.Length];
            return o;
        }

        void OnDestroy() => StopListening();
    }
}
