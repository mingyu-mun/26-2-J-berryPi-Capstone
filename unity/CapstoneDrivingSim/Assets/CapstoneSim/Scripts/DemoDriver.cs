// DemoDriver.cs
// 시연 영상용 '운전자' 시나리오. 생체 기록 재생 위치(BioSource.FileTimeSec, 기록 기준 초)에 맞춰
// 정상 운전 → 졸음 징후(흔들림 증가·핸들 놓침) → 졸음 운전(차선 쪽으로 서서히 흐르다 놀라서 꺾음) 을 재현한다.
// 실측 기록 rec_20261118 의 자기보고 라벨(0~9분 정상, 9~24분 졸음, 24분~ 정상)과 실시간 서버 판정 시각에 맞췄다.
//
//   · 사람 입력 대신 CarController 의 '운전자 핸들각·페달'을 넣는다 (scripted). 조향 보조·차선 이탈 경고·부주의 경고는
//     사람이 운전할 때와 똑같이 동작하므로, 화면과 CAN 기록·서버 판정이 실제 경로를 그대로 탄다.
//   · 운전자 모델: 0.2초마다 차로 중심을 보고 핸들을 고치는 사람 (반응이 띄엄띄엄이라 자연스럽게 흔들림)
//       핸들각 = 조향비 × [atan(L·κ) 피드포워드 − (헤딩 오차 + atan(k·(편차 − 목표 편차)/(v+2)))] + 잡음
//   · 사건 (기록 초 기준, Inspector 에서 바꿀 수 있음)
//       LaneChange : 깜빡이를 켜고 옆 차로로 이동 (경고 없어야 정상)
//       HandsOff   : 핸들에서 손을 뗀 채 유지 (부주의 1~2단계) — 차는 보조/관성으로 감
//       Drift      : 졸음으로 핸들 고치기를 멈추고 한쪽으로 서서히 흐름 → 차선을 밟음(차선 이탈 경고) → 놀라서 급히 되돌림
//   · 시작하면 조향 보조를 '끔'으로 바꾸고(졸음 운전이 차선을 벗어나도록), 끝나면 원래 세기로 되돌린다.
//   · F8 로 켜고 끈다. 운전 키(W·S·A·D·방향키)를 누르면 바로 사람에게 넘긴다.
using System;
using System.Collections.Generic;
using UnityEngine;

namespace CapstoneSim
{
    public enum DemoEventKind { LaneChange, HandsOff, Drift }

    [Serializable]
    public class DemoEvent
    {
        public float t;            // 기록 기준 초
        public DemoEventKind kind;
        public int side = 1;       // +1 오른쪽, -1 왼쪽 (Drift: 흐르는 쪽, LaneChange: 옮길 쪽은 자동)
        public float dur = 6f;     // 지속 [s] (Drift 는 최대 지속)
        public DemoEvent() { }
        public DemoEvent(float t, DemoEventKind k, int side, float dur) { this.t = t; kind = k; this.side = side; this.dur = dur; }
    }

    [DefaultExecutionOrder(-60)] // DriverAssist(-50) · CarController 보다 먼저
    public class DemoDriver : MonoBehaviour
    {
        public CarController car;
        public TrackBuilder track;
        public BioSource bio;
        public DriverAssist assist;
        public SimBootstrap sim;

        [Tooltip("목표 속도 [km/h]")] public float targetKmh = 100f;
        [Tooltip("핸들 고치는 주기 [s] (사람의 띄엄띄엄한 조작)")] public float correctEvery = 0.2f;
        [Tooltip("횡편차 이득")] public float gainK = 0.9f;
        [Tooltip("정상 구간 차로 안 흔들림 폭 [m]")] public float weaveNormal = 0.12f;
        [Tooltip("졸음 구간 차로 안 흔들림 폭 [m]")] public float weaveDrowsy = 0.35f;
        [Tooltip("졸음 흐름 때 더하는 핸들각 [deg]")] public float driftDeg = 3.5f;

        [Header("구간 (기록 기준 초)")]
        [Tooltip("이 시각부터 흔들림이 커짐 (자기보고 졸음 시작)")] public float drowsyFrom = 540f;
        [Tooltip("이 시각 이후 정상으로 돌아옴 (자기보고 졸음 끝)")] public float drowsyTo = 1440f;

        public List<DemoEvent> events = DefaultEvents();

        public bool Running { get; private set; }
        public string Phase { get; private set; } = "";
        public float FileTime => bio && bio.Loaded ? bio.FileTimeSec : _t;

        AssistLevel _savedAssist;
        float _t, _nextCorrect, _cmd, _weavePhase;
        DemoEvent _active;
        float _activeStart, _startleUntil, _driftBase;
        int _laneTarget;
        readonly HashSet<DemoEvent> _done = new HashSet<DemoEvent>();
        System.Random _rng = new System.Random(20261118);

        /// <summary>rec_20261118 실시간 서버(causal) 판정에 맞춘 기본 사건표. 주석의 단계는 생체만으로 낸 판정</summary>
        public static List<DemoEvent> DefaultEvents()
        {
            var e = new List<DemoEvent>
            {
                // 정상 (0~540 s): 깜빡이 켠 차로 변경 — 경고가 나오지 않음을 보여 줌
                new DemoEvent(150f, DemoEventKind.LaneChange, 0, 5f),
                new DemoEvent(330f, DemoEventKind.LaneChange, 0, 5f),
                new DemoEvent(480f, DemoEventKind.LaneChange, 0, 5f),
                // 졸음 시작 (자기보고 540 s~): 손 놓침·흔들림. 생체 판정이 아직 정상/주의라 CAN 경고가 있어도 단계는 안 오름
                new DemoEvent(600f, DemoEventKind.HandsOff, 0, 20f),
                new DemoEvent(700f, DemoEventKind.Drift, 1, 7f),
                new DemoEvent(820f, DemoEventKind.HandsOff, 0, 28f),
                new DemoEvent(900f, DemoEventKind.Drift, -1, 7f),
            };
            // 졸음 (생체 판정 '졸음' 무렵부터): 30~40초마다 차선 쪽으로 흐름 → 졸음 확정
            int side = 1;
            for (float t = 960f; t < 1430f; t += (side > 0 ? 32f : 38f))
            {
                e.Add(new DemoEvent(t, DemoEventKind.Drift, side, 8f));
                side = -side;
            }
            e.Add(new DemoEvent(1200f, DemoEventKind.HandsOff, 0, 26f));
            // 정상 회복 (1440 s~)
            e.Add(new DemoEvent(1560f, DemoEventKind.LaneChange, 0, 5f));
            e.Sort((a, b) => a.t.CompareTo(b.t));
            return e;
        }

        public void Begin()
        {
            if (!car || !track) return;
            Running = true;
            car.scripted = true;
            if (assist) { _savedAssist = assist.level; assist.level = AssistLevel.Off; }
            _t = 0f; _cmd = car.SteeringWheelDeg; _active = null; _done.Clear();
            // 이미 지난 사건은 건너뜀 (재생 위치 중간에서 시작할 때)
            foreach (var ev in events) if (ev.t + ev.dur < FileTime) _done.Add(ev);
        }

        public void End()
        {
            if (!Running) return;
            Running = false;
            if (car) { car.scripted = false; car.scriptedThrottle = car.scriptedBrake = 0f; car.TurnSignal = 0; }
            if (assist) assist.level = _savedAssist;
            Phase = "";
        }

        void OnDisable() => End();

        void Update()
        {
            bool driving = sim && sim.InSession && !(sim.Menu && sim.Menu.Visible);
            if (SimInput.DemoTogglePressed() && (Running || driving)) { if (Running) End(); else Begin(); }
            // 사람이 운전 키를 누르면 바로 넘김
            if (Running && (SimInput.KeyboardSteer() != 0f || SimInput.Throttle() > 0.1f || SimInput.Brake() > 0.1f)) End();
        }

        void FixedUpdate()
        {
            if (!Running || !car || !track) return;
            float dt = Time.fixedDeltaTime;
            _t += dt;
            float now = FileTime;
            var ts = track.Query(car.transform.position, car.transform.forward);
            float v = Mathf.Max(1f, car.SpeedKmh / 3.6f);

            // ---- 속도 ----
            float err = targetKmh - car.SpeedKmh;
            car.scriptedThrottle = Mathf.Clamp01(err * 0.08f);
            car.scriptedBrake = err < -6f ? Mathf.Clamp01(-err * 0.03f) : 0f;

            // ---- 사건 ----
            if (_active != null && now - _activeStart > _active.dur + (_active.kind == DemoEventKind.Drift ? 2f : 0f)) FinishActive();
            if (_active == null)
                foreach (var ev in events)
                    if (!_done.Contains(ev) && now >= ev.t && now < ev.t + ev.dur) { StartEvent(ev, ts, now); break; }
                    else if (!_done.Contains(ev) && now >= ev.t + ev.dur) _done.Add(ev);

            bool drowsy = now >= drowsyFrom && now < drowsyTo;
            float weave = drowsy ? weaveDrowsy : weaveNormal;
            _weavePhase += dt * 2f * Mathf.PI / (drowsy ? 7f : 11f);
            float targetOff = weave * Mathf.Sin(_weavePhase);
            bool handsOn = true;
            string phase = drowsy ? "졸음 구간 운전" : "정상 운전";

            if (_active != null)
            {
                float a = now - _activeStart;
                switch (_active.kind)
                {
                    case DemoEventKind.LaneChange:
                    {
                        // 옆 차로 중심을 목표로 (깜빡이는 VehicleWarnings 가 도착하면 자동으로 끔)
                        int curLane = ts.lane == 0 ? _laneTarget : ts.lane;
                        float centerShift = (_laneTarget - curLane) * track.LaneWidth;
                        targetOff = Mathf.Clamp(centerShift, -track.LaneWidth, track.LaneWidth) * Mathf.Clamp01(a / 2.0f);
                        phase = "차로 변경 (깜빡이)";
                        break;
                    }
                    case DemoEventKind.HandsOff:
                        handsOn = false;                       // 사람 입력 없음 → 핸들각 유지 (고치지 않음)
                        phase = "핸들 놓침";
                        break;
                    case DemoEventKind.Drift:
                    {
                        bool crossed = Mathf.Abs(ts.laneOffset) > 1.05f || ts.lane == 0;
                        if (crossed && _startleUntil < 0f) _startleUntil = now + 1.2f;
                        if (_startleUntil > 0f && now < _startleUntil)
                        {
                            // 놀라서 급히 되돌림 (차로 중심 쪽으로 크게)
                            _cmd = Correct(ts, v, 0f, 2.2f);
                            phase = "놀라서 핸들 꺾음";
                        }
                        else if (_startleUntil > 0f) { FinishActive(); }
                        else
                        {
                            handsOn = false;
                            _cmd = Mathf.MoveTowards(_cmd, _driftBase + _active.side * driftDeg, 1.2f * dt);   // 서서히 한쪽으로
                            phase = "졸음 — 차선 쪽으로 흐름";
                        }
                        car.scriptedHumanDeg = _cmd;
                        car.scriptedHandsOn = _startleUntil > 0f;
                        Phase = phase;
                        return;
                    }
                }
            }

            if (handsOn && Time.time >= _nextCorrect)
            {
                _nextCorrect = Time.time + correctEvery * (drowsy ? 1.8f : 1f);
                _cmd = Correct(ts, v, targetOff, 1f) + (float)(_rng.NextDouble() - 0.5) * (drowsy ? 2.0f : 0.8f);
            }
            car.scriptedHumanDeg = _cmd;
            car.scriptedHandsOn = handsOn;
            Phase = phase;
        }

        float Correct(TrackSample ts, float v, float targetOff, float gainScale)
        {
            float kAhead = track.CurvatureAt(ts.s + v * 0.6f);
            float ff = Mathf.Atan(car.wheelBase * 0.5f * (ts.curvature + kAhead)) * Mathf.Rad2Deg;
            float e = Mathf.Clamp(ts.laneOffset - targetOff, -0.6f, 0.6f);   // 큰 편차에서도 급조향하지 않게
            float fb = -(ts.headingErrorDeg + Mathf.Atan2(gainK * gainScale * e, v + 2f) * Mathf.Rad2Deg);
            return (-ff + fb) * car.steeringRatio;
        }

        void StartEvent(DemoEvent ev, TrackSample ts, float now)
        {
            _active = ev; _activeStart = now; _startleUntil = -1f; _driftBase = _cmd;
            if (ev.kind == DemoEventKind.LaneChange)
            {
                int lane = ts.lane == 0 ? 2 : ts.lane;
                _laneTarget = lane == 1 ? 2 : 1;
                car.TurnSignal = _laneTarget > lane ? 1 : -1;
            }
        }

        void FinishActive()
        {
            if (_active != null) _done.Add(_active);
            _active = null; _startleUntil = -1f;
        }
    }
}
