// VehicleWarnings.cs
// 시뮬레이터 차량 경고: 차선 이탈 경고(LDW), 운전자 부주의 경고, 방향지시등 자동 해제 + 경고음.
// 결과는 회사 CAN 매핑(Can list.xlsx) 상태 코드로 내보낸다 → AiLink(실시간 서버)와 DrivingDataLogger(_can.jsonl).
//   LaneSafe_indi : 0 Off / 1 Yellow / 2 GreenBlink / 3 Gray(비활성) / 4 Green(활성) / 5 YellowWarn(차선 이탈 경고)
//   Careless_Warn : 0 Off / 1 Yellow / 2 YellowBlink / 3 YellowWarn
//   LTurnSig_Indi, RTurnSig_Indi : 0 Off / 1 점멸
// 저장소 src/driver_state/can_assist.py (vehicle_warning_sim_v1) 는 LaneSafe_indi=5 (방향지시등 구간 제외)
// 또는 Careless_Warn≥2 를 '차량 경고'로 보고 can_drowsy 를 만든다. 차량 경고는 졸음 정답이 아니라 보조 입력이다.
//
// [차선 이탈] 속도 ≥ minSpeedKmh 에서 차체가 차선을 밟으면(|차로 편차| > (차로폭 − 차폭)/2) 경고.
//             방향지시등이 켜져 있으면 경고하지 않는다(정상 차로 변경). 한 번 켜지면 최소 holdSec 유지.
// [부주의]    (1) 핸들 조작 없음: 운전자가 핸들을 만지지 않은 시간 ≥ 15 s → 1, ≥ 25 s → 2, ≥ 40 s → 3
//             (2) 반복 이탈: 최근 60 s 안의 차선 이탈 경고 2회 → 2, 3회 이상 → 3
//             (3) 시연: K 키로 demoSec 동안 2
//             조향 보조가 강할 때 손을 떼고 달리면 (1)이 켜진다. 실차의 '핸들을 잡으세요' 경고와 같은 성격이다.
// [깜빡이]    Q/E 로 켠 뒤 옆 차로로 옮겨 차로 중심 근처(±0.6 m)에 1 s 머물면 자동으로 끈다. 20 s 가 지나도 끈다.
using System.Collections.Generic;
using UnityEngine;

namespace CapstoneSim
{
    [DefaultExecutionOrder(50)] // CarController 다음, DrivingDataLogger(100) 앞
    public class VehicleWarnings : MonoBehaviour
    {
        public CarController car;
        public TrackBuilder track;
        public DrivingDataLogger logger;
        public DriverAssist assist;

        [Header("차선 이탈 경고")]
        [Tooltip("이 속도 이상에서만 동작 [km/h]")] public float minSpeedKmh = 60f;
        [Tooltip("차선 이탈 판정 차폭 [m] (0 이면 로거 값)")] public float vehicleWidth = 0f;
        [Tooltip("경고 최소 유지 [s]")] public float holdSec = 1.0f;

        [Header("부주의 경고")]
        public float handsOff1 = 15f, handsOff2 = 25f, handsOff3 = 40f;
        [Tooltip("반복 이탈을 세는 구간 [s]")] public float departWindowSec = 60f;
        [Tooltip("부주의 2단계 이상 최소 유지 [s]")] public float carelessHoldSec = 3f;
        [Tooltip("K 키 시연 지속 [s]")] public float demoSec = 4f;

        [Header("방향지시등")]
        public float signalSettleSec = 1.0f, signalTimeoutSec = 20f;

        [Header("소리")]
        public bool sound = true;

        // ---------- 상태 코드 (CAN) ----------
        public int LaneSafeIndi { get; private set; } = 3;
        public int CarelessWarn { get; private set; }
        public int LTurnSigIndi => car && car.TurnSignal < 0 ? 1 : 0;
        public int RTurnSigIndi => car && car.TurnSignal > 0 ? 1 : 0;

        // ---------- 화면 표시용 ----------
        public bool LaneWarning => LaneSafeIndi == 5;
        public int LaneWarnSide { get; private set; }          // -1 왼쪽 선 / +1 오른쪽 선
        public float HandsOffSec { get; private set; }
        float _steerSeen = -1f;
        public int RecentDepartures => _departs.Count;
        public string CarelessReason { get; private set; } = "";

        float _laneOnTime = -999f, _laneLastDepart = -999f;
        readonly Queue<float> _departs = new Queue<float>();
        float _demoUntil = -999f, _carelessHoldUntil = -999f;
        int _signalLane, _signalDir;
        float _signalStart, _settleTime;

        AudioSource _audio;
        AudioClip _ldw, _chime, _tick, _tock;
        float _nextLdwBeep;
        bool _blinkPhase;

        void Start()
        {
            _audio = gameObject.AddComponent<AudioSource>();
            _audio.playOnAwake = false;
            _ldw = Tone(new[] { 1250f }, 0.11f, 0.22f);
            _chime = Tone(new[] { 880f, 660f }, 0.5f, 0.2f);
            _tick = Click(2400f); _tock = Click(1800f);
        }

        float Width => vehicleWidth > 0f ? vehicleWidth : (logger ? logger.vehicleWidth : 1.8f);

        void Update()
        {
            if (!car) return;
            if (SimInput.CarelessDemoPressed()) _demoUntil = Time.time + demoSec;
            Sounds();
        }

        void FixedUpdate()
        {
            if (!car || !track) return;
            float now = Time.time;
            var ts = track.Query(car.transform.position, car.transform.forward);
            float speed = Mathf.Abs(car.SpeedKmh);
            bool auto = car.externalControl;

            // ---------------- 방향지시등 자동 해제 ----------------
            if (car.TurnSignal != 0)
            {
                if (_signalDir != car.TurnSignal) { _signalDir = car.TurnSignal; _signalLane = ts.lane; _signalStart = now; _settleTime = -1f; }
                bool moved = ts.lane != 0 && _signalLane != 0 && ts.lane != _signalLane;
                if (moved && Mathf.Abs(ts.laneOffset) < 0.6f) { if (_settleTime < 0f) _settleTime = now; }
                else _settleTime = -1f;
                bool done = _settleTime >= 0f && now - _settleTime >= signalSettleSec;
                if (done || now - _signalStart > signalTimeoutSec)
                {
                    car.TurnSignal = 0;
                    if (logger) logger.LogEvent("TURN_SIGNAL", done ? "0 auto_lane_change" : "0 auto_timeout");
                }
            }
            if (car.TurnSignal == 0) _signalDir = 0;

            // ---------------- 차선 이탈 경고 ----------------
            int lane;
            if (auto || speed < minSpeedKmh) lane = 3;
            else
            {
                float halfFree = (track.LaneWidth - Width) * 0.5f;
                bool touching = ts.lane == 0 || Mathf.Abs(ts.laneOffset) > halfFree;
                bool cued = car.TurnSignal != 0;
                if (touching && !cued)
                {
                    if (LaneSafeIndi != 5)
                    {
                        _laneOnTime = now;
                        _departs.Enqueue(now);
                        if (logger) logger.LogEvent("LANE_WARN", ts.laneOffset.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    _laneLastDepart = now;
                    LaneWarnSide = ts.lane == 0 ? (ts.lateral < 0f ? -1 : 1) : (ts.laneOffset < 0f ? -1 : 1);
                }
                bool keep = now - _laneLastDepart < 0.3f || now - _laneOnTime < holdSec;
                lane = (touching && !cued) || (LaneSafeIndi == 5 && keep) ? 5 : 4;
                if (lane != 5 && LaneSafeIndi == 5 && logger) logger.LogEvent("LANE_WARN_OFF", "");
            }
            LaneSafeIndi = lane;
            while (_departs.Count > 0 && now - _departs.Peek() > departWindowSec) _departs.Dequeue();

            // ---------------- 부주의 경고 ----------------
            int c = 0; string why = "";
            // 핸들 조작 없음 시간: 30 km/h 이상으로 달린 시간만 센다 (정지·출발 직후·자동 주행은 제외)
            if (auto || car.LastDriverSteerTime > _steerSeen) { _steerSeen = car.LastDriverSteerTime; HandsOffSec = 0f; }
            else if (speed >= 30f) HandsOffSec += Time.fixedDeltaTime;
            if (!auto && speed >= 30f)
            {
                int h = HandsOffSec >= handsOff3 ? 3 : HandsOffSec >= handsOff2 ? 2 : HandsOffSec >= handsOff1 ? 1 : 0;
                if (h > c) { c = h; why = "핸들 조작 없음"; }
                int d = _departs.Count >= 3 ? 3 : _departs.Count >= 2 ? 2 : 0;
                if (d > c) { c = d; why = "차선 이탈 반복"; }
            }
            if (now < _demoUntil && c < 2) { c = 2; why = "시연"; }
            if (c >= 2) _carelessHoldUntil = now + carelessHoldSec;
            else if (CarelessWarn >= 2 && now < _carelessHoldUntil) { c = CarelessWarn; why = CarelessReason; }
            if (c != CarelessWarn && logger) logger.LogEvent("CARELESS_WARN", c + " " + why);
            CarelessWarn = c;
            CarelessReason = why;
        }

        // ---------------------------------------------------------------- 소리
        void Sounds()
        {
            if (!sound || !_audio) return;
            float t = Time.time;
            if (LaneWarning && t >= _nextLdwBeep) { _audio.PlayOneShot(_ldw); _nextLdwBeep = t + 0.28f; }
            if (CarelessWarn != _lastCareless)
            {
                if (CarelessWarn >= 2 && CarelessWarn > _lastCareless) _audio.PlayOneShot(_chime);
                _lastCareless = CarelessWarn;
            }
            // 깜빡이 '딸깍' (SimHUD 의 점멸과 같은 2 Hz 위상)
            bool phase = ((int)(Time.realtimeSinceStartup * 2f)) % 2 == 0;
            if (car.TurnSignal != 0 && phase != _blinkPhase) _audio.PlayOneShot(phase ? _tick : _tock, 0.6f);
            _blinkPhase = phase;
        }
        int _lastCareless;

        static AudioClip Tone(float[] hz, float len, float amp)
        {
            int sr = 44100, n = (int)(sr * len), seg = n / hz.Length;
            var d = new float[n];
            for (int i = 0; i < n; i++)
            {
                float f = hz[Mathf.Min(hz.Length - 1, i / Mathf.Max(1, seg))];
                float env = Mathf.Min(1f, i / 300f) * Mathf.Min(1f, (n - i) / 600f);
                d[i] = amp * env * Mathf.Sin(2 * Mathf.PI * f * i / sr);
            }
            var c = AudioClip.Create("warn" + hz[0], n, 1, sr, false);
            c.SetData(d, 0);
            return c;
        }

        static AudioClip Click(float hz)
        {
            int sr = 44100, n = sr / 50;
            var d = new float[n];
            for (int i = 0; i < n; i++) d[i] = 0.35f * Mathf.Exp(-i / 120f) * Mathf.Sin(2 * Mathf.PI * hz * i / sr);
            var c = AudioClip.Create("click" + hz, n, 1, sr, false);
            c.SetData(d, 0);
            return c;
        }
    }
}
