// DrivingDataLogger.cs
// 주행 신호를 고정 주기(기본 50 Hz)로 CSV에 기록한다.
//   - drive_<참가자>_<시각>.csv        : 주행 신호 (CAN 유사 신호 + 차선 기준 지표)
//   - drive_<참가자>_<시각>_events.csv : 기록 시작/정지, 동기화 마커, 깜빡이, 리셋, 랩 등 이벤트
//   - drive_<참가자>_<시각>_meta.json  : 세션 정보 (시작 시각, 샘플링 주기, 트랙 정보 등)
//   - drive_<참가자>_<시각>_can.jsonl  : 저장소 --can-sim-scenario 형식 CAN (속도·핸들각 + 차량 경고 상태 코드)
//       {"t_us":0,"speed_kmh":..,"steering_deg":..,"LaneSafe_indi":4,"Careless_Warn":0,"LTurnSig_Indi":0,"RTurnSig_Indi":0}
//       첫 줄에 상태 코드 네 개를 모두 넣고, 이후에는 바뀐 코드만 넣는다 (web/dashboard/README.md 'Unity 출력 형식')
//
// [시간 컬럼 두 개를 남기는 이유]
//   t_s          : 시뮬레이션(물리) 시간. 간격이 정확히 1/rate 로 균일 → 신호처리용.
//   wall_unix_ms : PC 벽시계(UTC, Unix epoch ms). 생체센서(ECG/PPG) 기록과 맞출 때 쓰는 기준.
//   Unity의 FixedUpdate는 프레임마다 몰아서 실행되므로 wall_unix_ms 간격은 ±1프레임(≈16 ms) 흔들린다.
//   프레임이 크게 끊기면 t_s가 실제 시간보다 뒤처질 수 있으므로, 두 열의 차이로 이를 점검할 수 있다.
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace CapstoneSim
{
    [DefaultExecutionOrder(100)] // CarController.FixedUpdate 이후에 실행
    public class DrivingDataLogger : MonoBehaviour
    {
        [Header("세션")]
        public string participantId = "P01";
        public string sessionNote = "";
        [Tooltip("기록 주기 [Hz]. 물리 주기(기본 100 Hz)의 약수로 두는 것을 권장")] public float logRateHz = 50f;
        [Tooltip("비워 두면 <프로젝트 폴더>/SimLogs 에 저장")] public string outputFolder = "";
        [Tooltip("차선 이탈 판정에 쓰는 차폭 [m]")] public float vehicleWidth = 1.8f;

        [Header("참조")]
        public CarController car;
        public TrackBuilder track;
        [Tooltip("있으면 AI 판정·생체 replay 위치도 함께 기록")] public AiLink ai;
        public BioSource bio;
        [Tooltip("차선 이탈·부주의 경고, 방향지시등 상태 코드")] public VehicleWarnings warnings;

        public bool IsRecording { get; private set; }
        public string CurrentFile { get; private set; } = "";
        public long SampleCount { get; private set; }
        public int MarkerCount { get; private set; }
        public float LastMarkerRealtime { get; private set; } = -999f;
        public double ElapsedSec => IsRecording ? (Time.fixedTimeAsDouble - _t0) : 0.0;
        public TrackSample LastSample { get; private set; }
        public int Lap { get; private set; }

        public const string Header =
            "wall_unix_ms,t_s,sample,speed_kmh,accel_pedal,brake_pedal,steer_input,steering_wheel_deg,wheel_angle_deg," +
            "yaw_rate_dps,long_accel_mps2,lat_accel_mps2,engine_rpm,gear,turn_signal," +
            "track_s_m,lap,lane,lateral_pos_m,lane_offset_m,heading_error_deg,road_curvature,lane_departure,marker," +
            "ai_level,ai_p_used,bio_replay_t_s,lane_safe_indi,careless_warn,assist_wheel_deg";

        StreamWriter _w, _ev, _can;
        readonly int[] _lastCodes = new int[4];
        string _base;
        long _startUnixMs;
        string _startLocal;
        Stopwatch _sw;
        double _t0, _acc;
        int _pendingMarker;
        float _lastFlush;
        float _prevS = -1f;
        readonly StringBuilder _sb = new StringBuilder(512);
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        void Start()
        {
            // 빌드 실행 시: DrivingSim.exe -participant P03 -note "야간 2회차"
            // (SimBootstrap이 Awake에서 필드를 채운 뒤에 실행되도록 Start에서 처리)
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-participant") participantId = args[i + 1];
                if (args[i] == "-note") sessionNote = args[i + 1];
            }
        }

        public string ResolveFolder()
        {
            string f = string.IsNullOrWhiteSpace(outputFolder)
                ? Path.Combine(Application.dataPath, "..", "SimLogs")
                : outputFolder;
            return Path.GetFullPath(f);
        }

        /// <summary>메인 메뉴에서는 L/M 키를 받지 않는다</summary>
        [HideInInspector] public bool allowInput = true;

        void Update()
        {
            if (!allowInput) return;
            if (SimInput.ToggleRecordPressed())
            {
                if (IsRecording) StopRecording(); else StartRecording();
            }
            if (SimInput.MarkerPressed()) AddMarker();
        }

        // ------------------------------------------------------------------
        public void StartRecording()
        {
            if (IsRecording) return;
            string dir = ResolveFolder();
            Directory.CreateDirectory(dir);
            string id = Sanitize(participantId);
            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", Inv);
            _base = Path.Combine(dir, $"drive_{id}_{stamp}");
            CurrentFile = _base + ".csv";

            var enc = new UTF8Encoding(false);
            _w = new StreamWriter(CurrentFile, false, enc, 1 << 16);
            _ev = new StreamWriter(_base + "_events.csv", false, enc);
            _w.WriteLine(Header);
            _can = new StreamWriter(_base + "_can.jsonl", false, enc, 1 << 16);
            for (int i = 0; i < 4; i++) _lastCodes[i] = int.MinValue;
            _ev.WriteLine("wall_unix_ms,t_s,event,value");

            _startUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            _startLocal = DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:ss.fffzzz", Inv);
            _sw = Stopwatch.StartNew();
            _t0 = Time.fixedTimeAsDouble;
            _acc = 1.0 / logRateHz; // 첫 FixedUpdate에서 바로 한 줄 기록
            SampleCount = 0; MarkerCount = 0; _pendingMarker = 0; Lap = 0; _prevS = -1f;
            IsRecording = true;
            LogEvent("RECORD_START", participantId);
            UnityEngine.Debug.Log($"[DrivingDataLogger] 기록 시작 → {CurrentFile}");
        }

        public void StopRecording()
        {
            if (!IsRecording) return;
            LogEvent("RECORD_STOP", SampleCount.ToString(Inv));
            IsRecording = false;
            long endMs = WallMs();
            _w.Flush(); _w.Dispose(); _w = null;
            _ev.Flush(); _ev.Dispose(); _ev = null;
            _can.Flush(); _can.Dispose(); _can = null;

            var meta = new SessionMeta
            {
                participant_id = participantId,
                note = sessionNote,
                start_unix_ms = _startUnixMs,
                end_unix_ms = endMs,
                start_local = _startLocal,
                duration_s = (endMs - _startUnixMs) / 1000.0,
                sim_duration_s = Time.fixedTimeAsDouble - _t0,
                samples = SampleCount,
                markers = MarkerCount,
                log_rate_hz = logRateHz,
                physics_hz = 1f / Time.fixedDeltaTime,
                unity_version = Application.unityVersion,
                track_length_m = track ? track.Length : 0f,
                lane_width_m = track ? track.LaneWidth : 0f,
                vehicle_width_m = vehicleWidth,
                steer_mode = car ? $"{car.steerMode} (마지막 입력: {car.ActiveSteerSource})" : "",
                columns = Header,
            };
            File.WriteAllText(_base + "_meta.json", JsonUtility.ToJson(meta, true), new UTF8Encoding(false));
            UnityEngine.Debug.Log($"[DrivingDataLogger] 기록 종료: {SampleCount} 샘플, {meta.duration_s:F1} s");
        }

        public void AddMarker()
        {
            MarkerCount++;
            LastMarkerRealtime = Time.realtimeSinceStartup;
            if (IsRecording)
            {
                _pendingMarker = MarkerCount;
                LogEvent("MARKER", MarkerCount.ToString(Inv));
            }
        }

        /// <summary>다른 컴포넌트(깜빡이·리셋 등)가 이벤트를 남길 때 사용</summary>
        public void LogEvent(string name, string value)
        {
            if (!IsRecording || _ev == null) return;
            _ev.WriteLine(string.Format(Inv, "{0},{1:F3},{2},{3}", WallMs(), Time.timeAsDouble - _t0, name, (value ?? "").Replace(',', ';')));
            _ev.Flush();
        }

        long WallMs() => _startUnixMs + (long)Math.Round(_sw.Elapsed.TotalMilliseconds);

        // ------------------------------------------------------------------
        void FixedUpdate()
        {
            if (!car || !track) return;
            var ts = track.Query(car.transform.position, car.transform.forward);
            LastSample = ts;

            // 랩 카운트: 진행거리가 끝에서 처음으로 넘어갈 때
            if (_prevS >= 0f && _prevS > track.Length - 100f && ts.s < 100f)
            {
                Lap++;
                LogEvent("LAP", Lap.ToString(Inv));
            }
            _prevS = ts.s;

            if (!IsRecording) return;
            _acc += Time.fixedDeltaTime;
            double period = 1.0 / logRateHz;
            if (_acc + 1e-9 < period) return;
            _acc -= period;

            float halfFree = (track.LaneWidth - vehicleWidth) * 0.5f;
            int departure = (ts.lane == 0 || Mathf.Abs(ts.laneOffset) > halfFree) ? 1 : 0;

            _sb.Clear();
            _sb.Append(WallMs().ToString(Inv)).Append(',');
            F(Time.fixedTimeAsDouble - _t0, "F3");
            _sb.Append(SampleCount.ToString(Inv)).Append(',');
            F(car.SpeedKmh, "F3");
            F(car.ThrottleInput, "F3");
            F(car.BrakeInput, "F3");
            F(car.SteerInput, "F4");
            F(car.SteeringWheelDeg, "F2");
            F(car.WheelAngleDeg, "F3");
            F(car.YawRateDps, "F3");
            F(car.LongAccel, "F3");
            F(car.LatAccel, "F3");
            F(car.EngineRpm, "F0");
            _sb.Append(car.Gear.ToString(Inv)).Append(',');
            _sb.Append(car.TurnSignal.ToString(Inv)).Append(',');
            F(ts.s, "F2");
            _sb.Append(Lap.ToString(Inv)).Append(',');
            _sb.Append(ts.lane.ToString(Inv)).Append(',');
            F(ts.lateral, "F3");
            F(ts.laneOffset, "F3");
            F(ts.headingErrorDeg, "F3");
            F(ts.curvature, "F5");
            _sb.Append(departure.ToString(Inv)).Append(',');
            _sb.Append(_pendingMarker.ToString(Inv)).Append(',');
            // AI 판정 (연결 안 됨/기준선 측정 중 = -1), 생체 replay 파일 시각
            bool aiLive = ai && ai.Connected && ai.Latest != null && ai.Latest.has_out;
            _sb.Append((aiLive ? ai.Latest.level : -9).ToString(Inv)).Append(',');
            F(aiLive ? ai.Latest.p_used : -1f, "F3");
            _sb.Append(bio && bio.Loaded ? bio.FileTimeSec.ToString("F3", Inv) : "-1").Append(',');
            _sb.Append(warnings ? warnings.LaneSafeIndi.ToString(Inv) : "-1").Append(',');
            _sb.Append(warnings ? warnings.CarelessWarn.ToString(Inv) : "-1").Append(',');
            _sb.Append(car.AssistWheelDeg.ToString("F2", Inv));
            _w.WriteLine(_sb.ToString());
            WriteCan(Time.fixedTimeAsDouble - _t0);
            _pendingMarker = 0;
            SampleCount++;

            if (Time.realtimeSinceStartup - _lastFlush > 1f) { _w.Flush(); _lastFlush = Time.realtimeSinceStartup; }
        }

        static readonly string[] CodeNames = { "LaneSafe_indi", "Careless_Warn", "LTurnSig_Indi", "RTurnSig_Indi" };

        void WriteCan(double t)
        {
            if (_can == null) return;
            var b = new StringBuilder(160);
            b.Append("{\"t_us\":").Append(Math.Round(t * 1e6).ToString("F0", Inv))
             .Append(",\"speed_kmh\":").Append(Mathf.Abs(car.SpeedKmh).ToString("F2", Inv))
             .Append(",\"steering_deg\":").Append(car.SteeringWheelDeg.ToString("F2", Inv));
            if (warnings)
            {
                int[] v = { warnings.LaneSafeIndi, warnings.CarelessWarn, warnings.LTurnSigIndi, warnings.RTurnSigIndi };
                for (int i = 0; i < 4; i++)
                {
                    if (v[i] == _lastCodes[i]) continue;
                    b.Append(",\"").Append(CodeNames[i]).Append("\":").Append(v[i].ToString(Inv));
                    _lastCodes[i] = v[i];
                }
            }
            _can.WriteLine(b.Append('}').ToString());
        }

        void F(double v, string fmt) => _sb.Append(v.ToString(fmt, Inv)).Append(',');

        static string Sanitize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "anon";
            foreach (char c in Path.GetInvalidFileNameChars()) s = s.Replace(c, '_');
            return s.Replace(' ', '_');
        }

        void OnDisable() => StopRecording();
        void OnApplicationQuit() => StopRecording();

        [Serializable]
        class SessionMeta
        {
            public string participant_id;
            public string note;
            public long start_unix_ms;
            public long end_unix_ms;
            public string start_local;
            public double duration_s;
            public double sim_duration_s;
            public long samples;
            public int markers;
            public float log_rate_hz;
            public float physics_hz;
            public string unity_version;
            public float track_length_m;
            public float lane_width_m;
            public float vehicle_width_m;
            public string steer_mode;   // 조향 입력 장치 — 장치에 따라 핸들각 신호 특성이 달라지므로 분석 시 구분
            public string columns;
        }
    }
}
