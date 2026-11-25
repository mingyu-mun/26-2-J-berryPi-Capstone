// CarController.cs
// WheelCollider 기반 승용차 모델.
// 목적: 실제 차량 동역학의 정밀 재현이 아니라, 사람이 운전할 때
// 'CAN에서 읽을 법한 신호(속도·핸들각·페달·요레이트·가속도·RPM·기어)'가 그럴듯하게 나오도록 하는 것.
//
// [조향 모델 — v2]
//   상태변수는 '핸들 각도(steering wheel angle)' 하나다.
//     바퀴 조향각 δ = 핸들각 / 기어비  (기본 15:1, 핸들 최대 ±450° → 바퀴 ±30°)
//   입력 장치별로 핸들각을 움직이는 방식이 다르다.
//     키보드  : 누르고 있는 시간에 따라 회전 속도가 점점 빨라짐(짧게 톡 → 미세 조향),
//               손을 떼면 복원 토크처럼 중앙으로 천천히 돌아옴, 고속일수록 회전 속도↓
//     마우스  : 화면 가로 위치 = 핸들 위치 (절대 위치 방식, 아날로그라 가장 부드러움)
//     게임패드: 스틱 = 핸들 위치, 중앙 부근은 둔하게(expo 커브) 해서 고속 미세 조향 쉬움
//   공통: 속도가 오를수록 낼 수 있는 최대 핸들각을 줄인다(속도 감응형) → 고속 불안정 방지.
//   조향 보조(2026-11-26): 운전자 핸들각(_humanSw)과 별도로 DriverAssist 가 정한 보정각을 더한다.
//     최종 핸들각 = 운전자 핸들각 + clamp(assistGain × assistTargetDeg, ±assistCapDeg)
//     assistTargetDeg 는 차로 중심 추종 조향(곡률 + 편차·헤딩 되먹임)이라, 운전자가 꺾으면 그만큼 더해지고
//     보조는 벗어난 만큼 되돌린다. 보정각에 상한이 있어 운전자가 그보다 크게 꺾으면 차로를 벗어날 수 있다
//     (LKAS 의 토크 상한과 같은 역할).
//   바퀴는 애커먼 기하(안쪽 바퀴를 더 꺾음)로 조향하고, 안티롤바로 차체 롤을 줄인다.
using UnityEngine;

namespace CapstoneSim
{
    public enum SteerMode { Auto, Keyboard, Mouse, Gamepad }

    [RequireComponent(typeof(Rigidbody))]
    public class CarController : MonoBehaviour
    {
        [Header("휠")]
        public WheelCollider wheelFL, wheelFR, wheelRL, wheelRR;
        public Transform visualFL, visualFR, visualRL, visualRR;

        [Header("실내 핸들 (선택) — 있으면 핸들각만큼 회전")]
        public Transform steeringWheelVisual;
        [Tooltip("핸들 회전축(핸들 로컬 좌표). 0이면 자동 추정")] public Vector3 steeringWheelAxis = Vector3.zero;
        public bool invertSteeringWheelVisual = false;

        [Header("구동/제동")]
        [Tooltip("후륜 1개당 최대 구동 토크 [N·m]")] public float maxMotorTorque = 900f;
        [Tooltip("바퀴 1개당 최대 제동 토크 [N·m]")] public float maxBrakeTorque = 1100f;
        [Tooltip("최고 속도 제한 [km/h]")] public float topSpeedKmh = 160f;
        [Tooltip("후진 최고 속도 [km/h]")] public float reverseSpeedKmh = 15f;
        [Tooltip("페달 변화 속도 [1/s] (0→1 걸리는 시간의 역수)")] public float pedalRate = 3f;

        [Header("조향 — 공통")]
        public SteerMode steerMode = SteerMode.Auto;
        [Tooltip("핸들 최대 회전각(한쪽) [deg]")] public float steeringLockDeg = 450f;
        [Tooltip("스티어링 기어비 (바퀴각 = 핸들각 / 기어비)")] public float steeringRatio = 15f;
        [Tooltip("100 km/h 이상에서 허용하는 최대 핸들각 [deg] (속도 감응형)")] public float highSpeedLockDeg = 60f;
        [Tooltip("핸들이 돌 수 있는 최대 속도 [deg/s]")] public float maxSteerRate = 720f;
        [Tooltip("축거 [m] (애커먼 계산용)")] public float wheelBase = 2.8f;
        [Tooltip("윤거 [m] (애커먼 계산용)")] public float trackWidth = 1.6f;

        [Header("조향 — 키보드")]
        [Tooltip("저속에서 핸들 회전 속도 [deg/s]")] public float kbRateLowSpeed = 540f;
        [Tooltip("고속(100 km/h)에서 핸들 회전 속도 [deg/s]")] public float kbRateHighSpeed = 70f;
        [Tooltip("최고 회전 속도까지 걸리는 시간 [s] (짧게 누르면 미세 조향)")] public float kbRampTime = 0.35f;
        [Tooltip("손을 뗐을 때 저속 복원 속도 [deg/s]")] public float kbReturnLowSpeed = 360f;
        [Tooltip("손을 뗐을 때 고속 복원 속도 [deg/s]")] public float kbReturnHighSpeed = 45f;

        [Header("조향 — 마우스")]
        [Tooltip("화면 중앙에서 이 비율(화면 폭 기준)만큼 가면 최대 조향")] public float mouseRange = 0.35f;
        [Range(0f, 1f)] public float mouseExpo = 0.3f;
        [Tooltip("마우스 입력 평활 시간상수 [s]")] public float mouseSmoothing = 0.05f;

        [Header("조향 — 게임패드")]
        [Range(0f, 0.3f)] public float stickDeadzone = 0.08f;
        [Range(0f, 1f)] public float stickExpo = 0.55f;
        public float stickSmoothing = 0.05f;

        [Header("안정성")]
        [Tooltip("안티롤바 강성 [N] (좌우 서스펜션 압축 차이에 비례한 힘)")] public float antiRoll = 9000f;
        [Tooltip("다운포스 계수 [N/(m/s)²]")] public float downforce = 1.5f;

        // ---------- 외부 제어 (자동 테스트용 자율주행) ----------
        [HideInInspector] public bool externalControl;
        [HideInInspector] public float extThrottle, extBrake, extSteerWheelDeg;

        // ---------- 조향 보조 (DriverAssist 가 매 물리 스텝 채움) ----------
        [HideInInspector] public float assistTargetDeg, assistGain, assistCapDeg;

        // ---------- 로거가 읽어 가는 값 ----------
        public float ThrottleInput { get; private set; }     // 0~1 (실제 페달 위치)
        public float BrakeInput { get; private set; }        // 0~1
        public float SteeringWheelDeg { get; private set; }  // 핸들 각도 [deg], +오른쪽
        public float SteerInput => SteeringWheelDeg / steeringLockDeg; // -1~1
        public float WheelAngleDeg { get; private set; }     // 차량 중심 기준 앞바퀴 조향각 [deg]
        public float SpeedKmh { get; private set; }          // 전진 + / 후진 -
        public float YawRateDps { get; private set; }        // [deg/s], +오른쪽(시계방향)
        public float LongAccel { get; private set; }         // 종가속도 [m/s²]
        public float LatAccel { get; private set; }          // 횡가속도 [m/s²], +오른쪽
        public float EngineRpm { get; private set; }
        public int Gear { get; private set; }                // -1 = R, 0 = N(정지), 1~6
        public int TurnSignal { get; set; }                  // -1 = 좌, 0 = 없음, +1 = 우
        public SteerMode ActiveSteerSource { get; private set; } = SteerMode.Keyboard;
        public float CurrentSteerLockDeg { get; private set; }
        /// <summary>운전자 입력만으로 만든 핸들각 [deg] (보조 보정 제외). 부주의(핸들 조작 없음) 판정에 쓴다</summary>
        public float HumanSteeringWheelDeg => _humanSw;
        /// <summary>조향 보조가 더한 핸들각 [deg]</summary>
        public float AssistWheelDeg { get; private set; }
        /// <summary>마지막으로 운전자가 조향 입력을 준 시각 (Time.time)</summary>
        public float LastDriverSteerTime { get; private set; }

        Rigidbody _rb;
        Vector3 _prevVel;
        bool _hasPrev;
        float _kbHold;
        float _humanSw;
        bool _initialized;

        // 바퀴 비주얼 보정값 (외부 모델의 피벗/축 방향이 제각각이므로)
        readonly Quaternion[] _rotOffset = new Quaternion[4];
        readonly Vector3[] _localCenter = new Vector3[4];
        Quaternion _swInitLocalRot;
        Vector3 _swAxis;

        static readonly float[] GearUpper = { 20f, 40f, 65f, 95f, 125f, 999f };

        WheelCollider[] Wheels => new[] { wheelFL, wheelFR, wheelRL, wheelRR };
        Transform[] Visuals => new[] { visualFL, visualFR, visualRL, visualRR };
        WheelCollider[] _ws = new WheelCollider[0];
        Transform[] _vs = new Transform[0];

        void Awake() { _rb = GetComponent<Rigidbody>(); }

        void Start() { if (!_initialized) Initialize(); }

        /// <summary>휠·비주얼 연결이 끝난 뒤 호출 (SimBootstrap이 호출. 안 하면 Start에서 자동)</summary>
        public void Initialize()
        {
            _initialized = true;
            if (wheelFL) wheelFL.ConfigureVehicleSubsteps(5f, 12, 15);

            var ws = _ws = Wheels; var vs = _vs = Visuals;
            for (int i = 0; i < 4; i++)
            {
                if (!ws[i]) continue;
                SetupTire(ws[i]);
                if (!vs[i]) continue;
                // 휠콜라이더 기준 자세 대비 비주얼의 회전 차이와, 비주얼 피벗→바퀴 중심 오프셋을 기억
                _rotOffset[i] = Quaternion.Inverse(ws[i].transform.rotation) * vs[i].rotation;
                _localCenter[i] = vs[i].InverseTransformPoint(RendererCenter(vs[i], ws[i].transform.position));
            }

            if (steeringWheelVisual)
            {
                _swInitLocalRot = steeringWheelVisual.localRotation;
                _swAxis = steeringWheelAxis.sqrMagnitude > 1e-6f ? steeringWheelAxis.normalized : GuessDiscAxis(steeringWheelVisual);
                // 축이 차량 전방을 향하도록 통일 → +각도 = 운전자 시점에서 시계방향(우회전)
                Vector3 worldAxis = steeringWheelVisual.rotation * _swAxis;
                if (Vector3.Dot(worldAxis, transform.forward) < 0f) _swAxis = -_swAxis;
            }
        }

        static void SetupTire(WheelCollider wc)
        {
            var f = wc.forwardFriction;
            f.extremumSlip = 0.4f; f.extremumValue = 1.0f; f.asymptoteSlip = 0.8f; f.asymptoteValue = 0.6f; f.stiffness = 1.5f;
            wc.forwardFriction = f;
            var s = wc.sidewaysFriction;
            s.extremumSlip = 0.25f; s.extremumValue = 1.0f; s.asymptoteSlip = 0.5f; s.asymptoteValue = 0.75f; s.stiffness = 2.0f;
            wc.sidewaysFriction = s;
        }

        // ------------------------------------------------------------------
        void FixedUpdate()
        {
            float dt = Time.fixedDeltaTime;
            Vector3 vel = _rb.linearVelocity;
            float vFwd = Vector3.Dot(vel, transform.forward);
            float speedKmh = vFwd * 3.6f;
            float spd01 = Mathf.InverseLerp(0f, 100f, Mathf.Abs(speedKmh));

            // ----- 페달 (키보드는 0/1이므로 램프로 부드럽게) -----
            float thrT = externalControl ? Mathf.Clamp01(extThrottle) : SimInput.Throttle();
            float brkT = externalControl ? Mathf.Clamp01(extBrake) : SimInput.Brake();
            ThrottleInput = Mathf.MoveTowards(ThrottleInput, thrT, pedalRate * dt);
            BrakeInput = Mathf.MoveTowards(BrakeInput, brkT, pedalRate * dt);

            // ----- 핸들 -----
            UpdateSteering(dt, spd01);
            ApplyAckermann(SteeringWheelDeg / steeringRatio);

            // ----- 구동/제동/후진 -----
            float motor = 0f, brake = 0f;
            bool reversing = false;
            if (BrakeInput > 0.05f && ThrottleInput < 0.05f && speedKmh < 1.0f)
            {
                reversing = true; // 정지 상태에서 브레이크를 계속 누르면 후진
                motor = speedKmh > -reverseSpeedKmh ? -BrakeInput * maxMotorTorque * 0.5f : 0f;
            }
            else
            {
                motor = speedKmh < topSpeedKmh ? ThrottleInput * maxMotorTorque : 0f;
                brake = BrakeInput * maxBrakeTorque;
                if (ThrottleInput < 0.02f && BrakeInput < 0.02f) brake = 60f; // 엔진 브레이크 느낌
            }
            wheelRL.motorTorque = motor; wheelRR.motorTorque = motor;
            wheelFL.brakeTorque = brake; wheelFR.brakeTorque = brake;
            wheelRL.brakeTorque = brake; wheelRR.brakeTorque = brake;

            // ----- 안정성: 안티롤바 + 다운포스 -----
            AntiRoll(wheelFL, wheelFR);
            AntiRoll(wheelRL, wheelRR);
            _rb.AddForce(-transform.up * downforce * vFwd * vFwd);

            // ----- 측정값 -----
            SpeedKmh = speedKmh;
            YawRateDps = _rb.angularVelocity.y * Mathf.Rad2Deg;
            if (_hasPrev)
            {
                Vector3 la = transform.InverseTransformDirection((vel - _prevVel) / dt);
                LongAccel = Mathf.Lerp(LongAccel, la.z, 0.2f); // 수치미분 잡음 완화
                LatAccel = Mathf.Lerp(LatAccel, la.x, 0.2f);
            }
            _prevVel = vel; _hasPrev = true;

            // ----- 기어/RPM (자동변속 모사) -----
            float spd = Mathf.Abs(speedKmh);
            if (reversing || speedKmh < -0.5f) { Gear = -1; EngineRpm = 900f + spd * 120f; }
            else if (spd < 0.5f) { Gear = 0; EngineRpm = 750f + ThrottleInput * 1500f; }
            else
            {
                int g = 0; while (g < GearUpper.Length - 1 && spd > GearUpper[g]) g++;
                float lo = g == 0 ? 0f : GearUpper[g - 1];
                float hi = g == GearUpper.Length - 1 ? topSpeedKmh : GearUpper[g];
                Gear = g + 1;
                EngineRpm = Mathf.Lerp(1300f, 3200f + 1200f * ThrottleInput, Mathf.InverseLerp(lo, hi, spd));
            }
        }

        void UpdateSteering(float dt, float spd01)
        {
            // 속도 감응형 최대 핸들각 (부드러운 S자 보간)
            float s = spd01 * spd01 * (3f - 2f * spd01);
            float lockNow = Mathf.Lerp(steeringLockDeg, Mathf.Min(highSpeedLockDeg, steeringLockDeg), s);
            CurrentSteerLockDeg = lockNow;

            if (externalControl)
            {
                float tgt = Mathf.Clamp(extSteerWheelDeg, -lockNow, lockNow);
                float e = Mathf.Lerp(SteeringWheelDeg, tgt, 1f - Mathf.Exp(-dt / 0.05f));
                SteeringWheelDeg = Mathf.MoveTowards(SteeringWheelDeg, e, maxSteerRate * dt);
                _humanSw = SteeringWheelDeg; AssistWheelDeg = 0f;
                LastDriverSteerTime = Time.time;
                return;
            }

            float kb = SimInput.KeyboardSteer();
            bool hasPad = SimInput.GamepadSteer(out float stick);

            // 어떤 장치로 조향할지 결정
            SteerMode src = steerMode;
            if (src == SteerMode.Auto)
            {
                if (Mathf.Abs(kb) > 0f) src = SteerMode.Keyboard;
                else if (hasPad && Mathf.Abs(stick) > 0.15f) src = SteerMode.Gamepad;
                else src = ActiveSteerSource == SteerMode.Mouse ? SteerMode.Keyboard : ActiveSteerSource;
            }
            ActiveSteerSource = src;

            float sw = _humanSw;
            float before = _humanSw;
            switch (src)
            {
                case SteerMode.Keyboard:
                {
                    if (kb != 0f)
                    {
                        _kbHold += dt;
                        float rate = Mathf.Lerp(kbRateLowSpeed, kbRateHighSpeed, spd01)
                                     * Mathf.Clamp01(0.25f + _kbHold / Mathf.Max(0.01f, kbRampTime));
                        // 반대쪽으로 꺾여 있으면 복원 속도를 더해 빠르게 되돌림
                        if (sw * kb < 0f) rate += Mathf.Lerp(kbReturnLowSpeed, kbReturnHighSpeed, spd01);
                        sw = Mathf.MoveTowards(sw, kb * lockNow, rate * dt);
                    }
                    else
                    {
                        _kbHold = 0f;
                        sw = Mathf.MoveTowards(sw, 0f, Mathf.Lerp(kbReturnLowSpeed, kbReturnHighSpeed, spd01) * dt);
                    }
                    break;
                }
                case SteerMode.Mouse:
                {
                    SimInput.MouseSteer(mouseRange, out float m);
                    float target = Shape(m, 0.02f, mouseExpo) * lockNow;
                    sw = Mathf.Lerp(sw, target, 1f - Mathf.Exp(-dt / Mathf.Max(0.001f, mouseSmoothing)));
                    break;
                }
                case SteerMode.Gamepad:
                {
                    float target = Shape(stick, stickDeadzone, stickExpo) * lockNow;
                    sw = Mathf.Lerp(sw, target, 1f - Mathf.Exp(-dt / Mathf.Max(0.001f, stickSmoothing)));
                    break;
                }
            }
            // 핸들 회전 속도 제한 + 범위 제한 (운전자 핸들각)
            sw = Mathf.MoveTowards(_humanSw, sw, maxSteerRate * dt);
            _humanSw = Mathf.Clamp(sw, -steeringLockDeg, steeringLockDeg);
            // 운전자가 핸들을 만졌는지: 키를 누르고 있거나 핸들각이 3°/s 이상 움직임 (키보드 손 뗌 복원은 제외)
            if (kb != 0f || (src != SteerMode.Keyboard && Mathf.Abs(_humanSw - before) > 3f * dt))
                LastDriverSteerTime = Time.time;

            // 조향 보조 보정각 (상한 있음)
            float assist = 0f;
            if (assistGain > 0f && assistCapDeg > 0f)
                assist = Mathf.Clamp(assistGain * assistTargetDeg, -assistCapDeg, assistCapDeg);
            AssistWheelDeg = Mathf.MoveTowards(AssistWheelDeg, assist, maxSteerRate * 0.25f * dt);
            float final = Mathf.Clamp(_humanSw + AssistWheelDeg, -steeringLockDeg, steeringLockDeg);
            SteeringWheelDeg = Mathf.MoveTowards(SteeringWheelDeg, final, maxSteerRate * dt);
        }

        /// <summary>데드존 + expo 커브: 중앙 부근은 둔하게, 끝으로 갈수록 민감하게</summary>
        static float Shape(float x, float deadzone, float expo)
        {
            float a = Mathf.Abs(x);
            if (a <= deadzone) return 0f;
            a = (a - deadzone) / (1f - deadzone);
            a = (1f - expo) * a + expo * a * a * a;
            return Mathf.Sign(x) * Mathf.Clamp01(a);
        }

        void ApplyAckermann(float deltaDeg)
        {
            WheelAngleDeg = deltaDeg;
            if (Mathf.Abs(deltaDeg) < 0.01f) { wheelFL.steerAngle = 0f; wheelFR.steerAngle = 0f; return; }
            // 회전 중심까지 거리 R = L / tan δ, 안쪽 바퀴는 R - T/2, 바깥쪽은 R + T/2 기준
            float R = wheelBase / Mathf.Tan(Mathf.Abs(deltaDeg) * Mathf.Deg2Rad);
            float inner = Mathf.Atan(wheelBase / (R - trackWidth * 0.5f)) * Mathf.Rad2Deg;
            float outer = Mathf.Atan(wheelBase / (R + trackWidth * 0.5f)) * Mathf.Rad2Deg;
            float sg = Mathf.Sign(deltaDeg);
            if (deltaDeg > 0f) { wheelFR.steerAngle = inner * sg; wheelFL.steerAngle = outer * sg; } // 우회전: 오른쪽이 안쪽
            else { wheelFL.steerAngle = inner * sg; wheelFR.steerAngle = outer * sg; }
        }

        void AntiRoll(WheelCollider l, WheelCollider r)
        {
            if (!l || !r) return;
            float tl = 1f, tr = 1f;
            bool gl = l.GetGroundHit(out var hl), gr = r.GetGroundHit(out var hr);
            if (gl) tl = (-l.transform.InverseTransformPoint(hl.point).y - l.radius) / l.suspensionDistance;
            if (gr) tr = (-r.transform.InverseTransformPoint(hr.point).y - r.radius) / r.suspensionDistance;
            float f = (tl - tr) * antiRoll;
            if (gl) _rb.AddForceAtPosition(l.transform.up * -f, l.transform.position);
            if (gr) _rb.AddForceAtPosition(r.transform.up * f, r.transform.position);
        }

        // ------------------------------------------------------------------
        void Update()
        {
            if (SimInput.SteerModeTogglePressed())
                steerMode = (SteerMode)(((int)steerMode + 1) % 4);

            var ws = _ws; var vs = _vs;
            for (int i = 0; i < ws.Length; i++)
            {
                if (!ws[i] || !vs[i]) continue;
                ws[i].GetWorldPose(out var pos, out var rot);
                vs[i].rotation = rot * _rotOffset[i];
                vs[i].position = pos - vs[i].TransformVector(_localCenter[i]);
            }

            if (steeringWheelVisual)
            {
                float a = invertSteeringWheelVisual ? -SteeringWheelDeg : SteeringWheelDeg;
                steeringWheelVisual.localRotation = _swInitLocalRot * Quaternion.AngleAxis(a, _swAxis);
            }
        }

        /// <summary>하위 렌더러 전체의 경계상자 중심 (렌더러가 없으면 fallback)</summary>
        static Vector3 RendererCenter(Transform t, Vector3 fallback)
        {
            var rs = t.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return fallback;
            var b = rs[0].bounds;
            for (int i = 1; i < rs.Length; i++) b.Encapsulate(rs[i].bounds);
            return b.center;
        }

        /// <summary>원판 모양 메시의 회전축 추정: 메시 경계상자에서 가장 얇은 축</summary>
        static Vector3 GuessDiscAxis(Transform t)
        {
            MeshFilter best = null; float bestSize = -1f;
            foreach (var mf in t.GetComponentsInChildren<MeshFilter>())
            {
                if (!mf.sharedMesh) continue;
                float sz = mf.sharedMesh.bounds.size.sqrMagnitude;
                if (sz > bestSize) { bestSize = sz; best = mf; }
            }
            if (!best) return Vector3.forward;
            Vector3 e = Vector3.Scale(best.sharedMesh.bounds.size, best.transform.lossyScale);
            Vector3 axisInMesh = e.x <= e.y && e.x <= e.z ? Vector3.right : (e.y <= e.z ? Vector3.up : Vector3.forward);
            // 메시 로컬 축 → 핸들 Transform 로컬 축
            Vector3 world = best.transform.TransformDirection(axisInMesh);
            return (Quaternion.Inverse(t.rotation) * world).normalized;
        }

        /// <summary>현재 전방으로 지정 속도 부여 [m/s] (자동 테스트에서 순간이동 후 사용)</summary>
        public void SetForwardSpeed(float mps)
        {
            _rb.linearVelocity = transform.forward * mps;
            _prevVel = _rb.linearVelocity;
            foreach (var w in Wheels) if (w) w.brakeTorque = 0f;
        }

        /// <summary>차량을 지정 자세로 순간이동하고 모든 운동 상태 초기화</summary>
        public void Teleport(Vector3 pos, Quaternion rot)
        {
            _rb.linearVelocity = Vector3.zero;
            _rb.angularVelocity = Vector3.zero;
            _rb.position = pos; _rb.rotation = rot;
            transform.SetPositionAndRotation(pos, rot);
            ThrottleInput = BrakeInput = 0f;
            SteeringWheelDeg = 0f; _humanSw = 0f; AssistWheelDeg = 0f;
            _hasPrev = false; LongAccel = LatAccel = 0f;
            foreach (var w in Wheels)
                if (w) { w.motorTorque = 0f; w.brakeTorque = maxBrakeTorque; }
        }
    }
}
