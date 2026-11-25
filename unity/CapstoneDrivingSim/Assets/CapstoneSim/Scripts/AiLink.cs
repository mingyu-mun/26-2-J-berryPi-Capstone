// AiLink.cs
// 시뮬레이터 ↔ J_Berry PI 실시간 추론 서버(Tools/jberry_ai/server.py) 연결.
//   이 PC 에서 저장소 모델(S1-EP 학생 3개 평균 + 상태 머신 v2)을 CPU 로 돌린다.
//   송신 (UDP → 5005): hello(1초마다), 실측 ECG/PPG 원신호(각자 원래 주기, 10 Hz 묶음), 차량 CAN 유사 신호(50 Hz → 10 Hz 묶음)
//   수신 (UDP ← 5006): 화면 패킷 'ui' (약 10 Hz) — 상태 머신 출력, 기준 수집 상태, 전처리 파형
// 규약(웹보드 v1.2)에 맞춘 이벤트 기록은 서버가 SimLogs/ai/events_<세션>.jsonl 로 남긴다.
// 'Auto Launch Server' 가 켜져 있으면 Play 때 Tools/jberry_ai/.venv 의 Python 으로 서버를 띄우고 Play 종료 때 끈다.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace CapstoneSim
{
    /// <summary>server.py 가 보내는 화면 패킷. 값 없음(null) 은 -1, 판정 전에는 has_out=false</summary>
    [Serializable]
    public class AiUi
    {
        public string type, session, mode, model_id, sm_id, bio_origin, can_origin, error;
        public float t_rel, preroll, decision_delay = -1;
        public string base_status = "", base_source = "";
        public float feat_ready = -1;
        public bool base_forced;
        public int clean_windows, required_clean = 20, n_windows;
        public float base_wait = 60, base_max = 600, base_a = -1, base_b = -1;
        public bool ecg_rx, can_rx;
        public bool has_out;
        public int level = -1;
        public string level_name = "", channel = "", confirmation_source = "";
        public bool candidate, frozen;
        public float score = -1, p_used = -1, confidence = -1, p_ep = -1, p_ppg = -1, q_ecg = -1, q_ppg = -1,
                     hr_ecg = -1, hr_ppg = -1, logit_ep = -1, win_end = -1, proc_ms = -1, out_age = -1,
                     th_on = 0.55f, th_off = 0.40f;
        public string[] z_names;
        public float[] z_vals;
        public float[] hist_t, hist_p;
        public int[] hist_lv;
        public float wave_fs;
        public float[] ecg_w, ppg_w;
        // 기준 파형 (현재 파형과 같은 길이·같은 배율, 첫 박동 위상 맞춤) / 기준 심박 [bpm]
        public float[] ecg_b, ppg_b;
        public float ecg_hrb = -1, ppg_hrb = -1;
    }

    public class AiLink : MonoBehaviour
    {
        [Header("연결")]
        public string host = "127.0.0.1";
        public int sendPort = 5005;
        public int receivePort = 5006;

        [Header("Python 서버 자동 실행")]
        public bool autoLaunchServer = true;
        [Tooltip("비우면 Tools/jberry_ai/python(휴대용) → .venv 순서")] public string pythonExe = "";
        [Tooltip("causal: 실시간(과거 RR 만, 지연 없음) / exact: 학습·batch 와 같은 계산(판정이 약 140초 늦게 나옴)")]
        public string mode = "causal";
        [Tooltip("시작 위치 앞 기록을 서버에 미리 빠르게 넣어 기준 수집이 끝난 상태로 시작 (시연용, 화면에 표시)")]
        public bool prerollBeforeStart = true;
        [Tooltip("웹보드(대시보드) 주소. 비우면 전송 안 함. 예: http://127.0.0.1:8000")] public string webboardUrl = "";
        [Tooltip("개인 기준(baseline) 측정 파일 세션 폴더. 비우면 입력 앞부분에서 자동 수집")] public string baselineDir = "";

        [Header("참조")]
        public BioSource bio;
        public CarController car;
        public TrackBuilder track;
        public string participantId = "P01";

        public AiUi Latest { get; private set; }
        public float LatestAgeSec => _lastRxRealtime < 0 ? 999f : Time.realtimeSinceStartup - _lastRxRealtime;
        public bool Connected => LatestAgeSec < 3f;
        public string ServerStatus { get; private set; } = "대기";
        public int ResultCount { get; private set; }
        public string SessionId { get; private set; }
        public bool SessionAccepted { get; private set; }

        UdpClient _tx, _rx;
        IPEndPoint _dst;
        Thread _rxThread;
        volatile bool _running;
        readonly object _lock = new object();
        readonly Queue<string> _rxQueue = new Queue<string>();
        float _lastRxRealtime = -1f;
        Process _proc;
        StreamWriter _procLog;
        float _sendAcc, _helloAcc;
        float _preroll;

        // CAN 유사 신호 묶음 (FixedUpdate 에서 50 Hz 샘플)
        readonly List<float> _cT = new List<float>(), _cSpeed = new List<float>(), _cSteer = new List<float>(),
                             _cAcc = new List<float>(), _cBrk = new List<float>();
        readonly List<int> _cGear = new List<int>(), _cTurn = new List<int>();
        float _canAcc;

        [Serializable] class Hello
        {
            public string type = "hello", session, participant, replay, mode, preroll_dir, origin, baseline_dir;
            public double fs_ecg, fs_ppg; public float replay_offset, preroll_sec;
        }
        [Serializable] class BioPacket { public string type = "bio"; public float[] ecg, ppg; }
        [Serializable] class CanPacket
        {
            public string type = "can";
            public float[] t, speed, steer, accel, brake; public int[] gear, turn;
        }

        public string ProjectRoot => SimPaths.Root;
        public string LogDir => SimPaths.AiLogDir;
        public string ServerDir => SimPaths.AiDir;

        IEnumerator Start()
        {
            SessionId = "sim_" + DateTime.Now.ToString("yyyyMMdd_HHmmss");
            Directory.CreateDirectory(LogDir);
            try
            {
                _tx = new UdpClient();
                _dst = new IPEndPoint(IPAddress.Parse(host), sendPort);
                _rx = new UdpClient(new IPEndPoint(IPAddress.Parse(host), receivePort));
                _rx.Client.ReceiveBufferSize = 1 << 20;
                _running = true;
                _rxThread = new Thread(RxLoop) { IsBackground = true, Name = "AiLink-RX" };
                _rxThread.Start();
            }
            catch (Exception e)
            {
                ServerStatus = "UDP 포트 열기 실패: " + e.Message;
                UnityEngine.Debug.LogError("[AiLink] " + ServerStatus);
                yield break;
            }
            while (bio && !bio.Loaded && string.IsNullOrEmpty(bio.Status)) yield return null;

            // 이미 떠 있는 서버가 있는지 먼저 확인
            SendHello();
            ServerStatus = "서버 확인 중…";
            float t0 = Time.realtimeSinceStartup;
            while (Time.realtimeSinceStartup - t0 < 1.5f && _lastRxRealtime < 0) { PumpReceived(); yield return null; }
            if (_lastRxRealtime >= 0) { ServerStatus = "외부 서버 연결됨"; yield break; }

            if (autoLaunchServer) yield return LaunchServer();
            else ServerStatus = "서버 없음 (Tools/jberry_ai/run_server.bat 실행)";
        }

        IEnumerator LaunchServer()
        {
            string script = Path.Combine(ServerDir, "server.py");
            if (!File.Exists(script)) { ServerStatus = "server.py 없음: " + script; UnityEngine.Debug.LogError("[AiLink] " + ServerStatus); yield break; }

            var cand = new List<string>();
            if (!string.IsNullOrWhiteSpace(pythonExe)) cand.Add(pythonExe);
            else
            {
                string py = SimPaths.AiPython;
                if (py != null) cand.Add(py);
                else
                {
                    ServerStatus = "AI 엔진이 없습니다 — 메인 메뉴에서 [AI 엔진 설치]";
                    UnityEngine.Debug.LogError("[AiLink] " + ServerStatus + $" ({SimPaths.PortablePython}, {SimPaths.VenvPython} 없음)");
                    yield break;
                }
            }
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            string args = $"-u \"{script}\" --host {host} --in-port {sendPort} --out-port {receivePort} --mode {mode} " +
                          $"--log-dir \"{LogDir}\" --idle-exit 20" + (string.IsNullOrWhiteSpace(webboardUrl) ? "" : $" --webboard {webboardUrl}");

            foreach (var exe in cand)
            {
                ServerStatus = "AI 서버 시작 중… (모델 로드)";
                try
                {
                    _procLog = new StreamWriter(Path.Combine(LogDir, "server_stdout.txt"), false, new UTF8Encoding(false)) { AutoFlush = true };
                    var psi = new ProcessStartInfo(exe, args)
                    {
                        UseShellExecute = false, CreateNoWindow = true,
                        RedirectStandardOutput = true, RedirectStandardError = true,
                        WorkingDirectory = ServerDir,
                        StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                    };
                    psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                    psi.EnvironmentVariables["PYTHONUTF8"] = "1";
                    _proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
                    _proc.OutputDataReceived += (s, e) => { if (e.Data != null) lock (_lock) _procLog?.WriteLine(e.Data); };
                    _proc.ErrorDataReceived += (s, e) => { if (e.Data != null) lock (_lock) _procLog?.WriteLine("[stderr] " + e.Data); };
                    _proc.Start();
                    _proc.BeginOutputReadLine();
                    _proc.BeginErrorReadLine();
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogWarning($"[AiLink] '{exe}' 실행 실패: {e.Message}");
                    _proc = null; _procLog?.Dispose(); _procLog = null;
                    continue;
                }

                // torch 로드에 수 초 걸림 → 최대 60초 대기
                float t0 = Time.realtimeSinceStartup;
                while (Time.realtimeSinceStartup - t0 < 60f && _lastRxRealtime < 0 && !_proc.HasExited)
                {
                    SendHello();
                    PumpReceived();
                    yield return new WaitForSecondsRealtime(0.5f);
                }
                if (_lastRxRealtime >= 0) { ServerStatus = "AI 서버 실행됨"; UnityEngine.Debug.Log($"[AiLink] {ServerStatus} ({Time.realtimeSinceStartup - t0:F1}s)"); yield break; }
                if (_proc.HasExited) UnityEngine.Debug.LogWarning($"[AiLink] '{exe}' 가 바로 종료됨 (code {_proc.ExitCode}). SimLogs/ai/server_stdout.txt 참고");
                KillServer();
            }
            ServerStatus = "AI 서버 실행 실패 — SimLogs/ai/server_stdout.txt 확인";
            UnityEngine.Debug.LogError("[AiLink] " + ServerStatus);
        }

        void SendHello()
        {
            if (bio == null || !bio.Loaded) return;
            bool auto = string.IsNullOrWhiteSpace(baselineDir);
            // 앞부분 빠른 입력 (기록 재생일 때만): 자동 기준 → 기록 0 s ~ 시작 위치 전부 (기준 수집까지 끝난 상태로 시작)
            //                                    기준 파일 → 시작 위치 앞 최대 300 s (300 s 스케일 특징 창 채우기)
            _preroll = bio.IsLive || !prerollBeforeStart ? 0f : auto ? bio.startOffsetSec : Mathf.Min(bio.startOffsetSec, 300f);
            Send(JsonUtility.ToJson(new Hello
            {
                session = SessionId, participant = participantId, replay = bio.SessionName, mode = mode, origin = bio.Origin,
                fs_ecg = bio.FsEcg, fs_ppg = bio.FsPpg, replay_offset = bio.IsLive ? 0f : bio.startOffsetSec,
                preroll_dir = _preroll > 0 ? bio.SessionPath : "", preroll_sec = _preroll,
                baseline_dir = auto ? "" : System.IO.Path.GetFullPath(baselineDir),
            }));
        }

        void Send(string json)
        {
            if (_tx == null) return;
            try { var b = Encoding.UTF8.GetBytes(json); _tx.Send(b, b.Length, _dst); }
            catch (Exception) { /* 서버가 아직 없을 때 Windows 는 ICMP 오류를 낼 수 있음 — 무시 */ }
        }

        void RxLoop()
        {
            var any = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    var data = _rx.Receive(ref any);
                    lock (_lock) { _rxQueue.Enqueue(Encoding.UTF8.GetString(data)); while (_rxQueue.Count > 8) _rxQueue.Dequeue(); }
                }
                catch (SocketException) { if (!_running) break; }
                catch (ObjectDisposedException) { break; }
            }
        }

        void PumpReceived()
        {
            while (true)
            {
                string json;
                lock (_lock) { if (_rxQueue.Count == 0) return; json = _rxQueue.Dequeue(); }
                try
                {
                    var r = JsonUtility.FromJson<AiUi>(json);
                    if (r == null || r.type != "ui") continue;
                    _lastRxRealtime = Time.realtimeSinceStartup;
                    if (r.session != SessionId) continue;           // 아직 hello 처리 전 / 이전 세션 패킷
                    SessionAccepted = true;
                    if (Latest != null && (r.ecg_w == null || r.ecg_w.Length == 0)) { r.ecg_w = Latest.ecg_w; r.ppg_w = Latest.ppg_w; }
                    Latest = r; ResultCount++;
                }
                catch (Exception e) { UnityEngine.Debug.LogWarning("[AiLink] 패킷 해석 실패: " + e.Message); }
            }
        }

        void FixedUpdate()
        {
            if (_tx == null || !car || !SessionAccepted) return;
            _canAcc += Time.fixedDeltaTime;
            if (_canAcc < 0.02f - 1e-5f) return;
            _canAcc -= 0.02f;
            _cT.Add(bio ? bio.ElapsedSec : Time.timeSinceLevelLoad);
            _cSpeed.Add(Mathf.Abs(car.SpeedKmh));
            _cSteer.Add(car.SteeringWheelDeg);
            _cAcc.Add(Mathf.Clamp01(car.ThrottleInput));
            _cBrk.Add(Mathf.Clamp01(car.BrakeInput));
            _cGear.Add(car.Gear < 0 ? 1 : car.Gear == 0 ? 2 : 3);   // Can list GearPosition: 0=P 1=R 2=N 3=D
            _cTurn.Add(car.TurnSignal);
        }

        void Update()
        {
            PumpReceived();
            if (_tx == null) return;
            _helloAcc += Time.unscaledDeltaTime;
            if (_helloAcc > 1f) { _helloAcc = 0f; SendHello(); }
            _sendAcc += Time.unscaledDeltaTime;
            if (_sendAcc < 0.1f) return;       // 10 Hz 로 묶어서 전송
            _sendAcc = 0f;
            if (!SessionAccepted) return;      // 서버가 세션을 만들기 전 표본은 BioReplay 에 쌓아 둔다 (버리지 않음)

            int guard = 0;
            while (bio && guard++ < 20 && bio.Drain(out var e, out var p))
                Send(JsonUtility.ToJson(new BioPacket { ecg = e, ppg = p }));

            if (_cT.Count > 0)
            {
                Send(JsonUtility.ToJson(new CanPacket
                {
                    t = _cT.ToArray(), speed = _cSpeed.ToArray(), steer = _cSteer.ToArray(), accel = _cAcc.ToArray(),
                    brake = _cBrk.ToArray(), gear = _cGear.ToArray(), turn = _cTurn.ToArray()
                }));
                _cT.Clear(); _cSpeed.Clear(); _cSteer.Clear(); _cAcc.Clear(); _cBrk.Clear(); _cGear.Clear(); _cTurn.Clear();
            }
        }

        void KillServer()
        {
            try { if (_proc != null && !_proc.HasExited) _proc.Kill(); } catch { }
            _proc = null;
            lock (_lock) { _procLog?.Dispose(); _procLog = null; }
        }

        void OnDisable()
        {
            _running = false;
            try { _rx?.Close(); } catch { }
            try { _tx?.Close(); } catch { }
            _rx = null; _tx = null;
            KillServer();
        }
    }
}
