// DriverAssist.cs
// 조향 보조(차로 유지 보조, LKA 유사). 시뮬레이터 난이도를 낮추기 위한 것이며 실차 LKAS 를 재현하지 않는다.
//
//   목표 핸들각 = 차로 중심을 따라가는 Stanley 조향 (CarAutopilot 과 같은 식)
//     δ = atan(L·κ)(곡률 피드포워드) − (ψ_e + atan(k·e / (v + v0)))   → 핸들각 = δ × 조향비
//   CarController 가 운전자 핸들각에 '보정각 = clamp(gain × 목표, ±cap)' 을 더한다.
//   · 강(Strong): 손을 떼도 코너를 돈다. 운전자 입력은 절반만 반영하고(humanSteerScale), 커브 앞에서
//                 횡가속도 maxLatAccel 이하가 되도록 속도를 자동으로 줄인다. 120 km/h 상한.
//                 운전자가 크게 꺾으면(보정 상한 이상) 차로를 벗어날 수 있다.
//   · 약(Weak)  : 곡선 유지에 필요한 만큼만 돕는다. 졸음 시연(차선 이탈)을 보여 주기 쉽다.
//   곡률 피드포워드는 차량 앞 lookaheadSec 초 지점의 곡률을 함께 써서 커브에 들어가기 전부터 돌기 시작한다.
//   · 끔(Off)
//   깜빡이를 켜면(차로 변경) 보조가 잠시 꺼지고, 저속(30 km/h 미만)·도로 밖·자동 주행 중에는 동작하지 않는다.
//   G 키로 끔 → 약 → 강 순서로 바꾼다.
using UnityEngine;

namespace CapstoneSim
{
    public enum AssistLevel { Off, Weak, Strong }

    [DefaultExecutionOrder(-50)] // CarController.FixedUpdate 보다 먼저 목표값을 채운다
    public class DriverAssist : MonoBehaviour
    {
        public CarController car;
        public TrackBuilder track;

        [Tooltip("시작 세기 (G 키로 전환)")] public AssistLevel level = AssistLevel.Strong;
        [Tooltip("횡편차 이득 k")] public float gainK = 1.0f;
        [Tooltip("이 속도 미만에서는 보조 안 함 [km/h]")] public float minSpeedKmh = 30f;

        [Header("세기별 (이득, 보정각 상한 [deg 핸들각])")]
        public float weakGain = 0.6f, weakCapDeg = 12f;
        public float strongGain = 1.0f, strongCapDeg = 60f;
        [Tooltip("강 모드 횡편차 이득 k")] public float strongGainK = 2.0f;
        [Tooltip("강 모드에서 운전자 핸들각 반영 비율")] public float strongHumanScale = 0.5f;
        [Tooltip("강 모드 커브 속도: 이 횡가속도 이하 [m/s²]")] public float maxLatAccel = 2.7f;
        [Tooltip("강 모드 최고 속도 [km/h] (0 = 제한 없음)")] public float strongMaxKmh = 120f;
        [Tooltip("곡률 피드포워드를 미리 보는 시간 [s]")] public float lookaheadSec = 0.6f;

        [Tooltip("깜빡이를 끈 뒤 보조를 다시 켜기까지 [s]")] public float resumeAfterSignalSec = 1.0f;

        /// <summary>지금 보정이 걸리고 있는지 (화면 표시용)</summary>
        public bool Engaged { get; private set; }
        public string Reason { get; private set; } = "";
        /// <summary>강 모드 속도 상한 [km/h] (0 = 없음)</summary>
        public float SpeedLimitKmh { get; private set; }

        float _weight;          // 0~1, 켜고 끌 때 부드럽게
        float _signalOffTime = -999f;

        public static string LevelName(AssistLevel l) => l == AssistLevel.Strong ? "강" : l == AssistLevel.Weak ? "약" : "끔";

        void Update()
        {
            if (SimInput.AssistTogglePressed())
                level = (AssistLevel)(((int)level + 1) % 3);
        }

        void FixedUpdate()
        {
            if (!car || !track) return;
            if (car.TurnSignal != 0) _signalOffTime = Time.time;

            string why = "";
            if (car.externalControl) why = "자동 주행";
            else if (level == AssistLevel.Off) why = "끔";
            else if (Mathf.Abs(car.SpeedKmh) < minSpeedKmh) why = "저속";
            else if (car.TurnSignal != 0 || Time.time - _signalOffTime < resumeAfterSignalSec) why = "차로 변경";

            var ts = track.Query(car.transform.position, car.transform.forward);
            if (why == "" && ts.lane == 0) why = "도로 밖";

            float want = why == "" ? 1f : 0f;
            _weight = Mathf.MoveTowards(_weight, want, Time.fixedDeltaTime / 0.5f);
            Engaged = _weight > 0.05f;
            Reason = why;

            bool strong = level == AssistLevel.Strong;
            if (car.externalControl) { car.assistGain = 0f; car.humanSteerScale = 1f; car.assistMaxKmh = 0f; return; }

            float v = Mathf.Max(1f, car.SpeedKmh / 3.6f);
            // 곡률: 현재와 lookaheadSec 앞 지점의 평균 (커브 진입 전부터 조향)
            float kAhead = track.CurvatureAt(ts.s + v * lookaheadSec);
            float k = 0.5f * (ts.curvature + kAhead);
            float ff = Mathf.Atan(car.wheelBase * k) * Mathf.Rad2Deg;   // 부호는 CarAutopilot 과 같음
            float kGain = strong ? strongGainK : gainK;
            float fb = -(ts.headingErrorDeg + Mathf.Atan2(kGain * ts.laneOffset, v + 2f) * Mathf.Rad2Deg);
            car.assistTargetDeg = (-ff + fb) * car.steeringRatio;

            car.assistGain = (strong ? strongGain : weakGain) * _weight;
            car.assistCapDeg = level == AssistLevel.Off ? 0f : (strong ? strongCapDeg : weakCapDeg);
            car.humanSteerScale = strong ? Mathf.Lerp(1f, strongHumanScale, _weight) : 1f;

            // 강: 커브 앞 자동 감속 (앞 3초 구간의 최대 곡률 기준) + 최고 속도
            float vmax = 0f;
            if (strong && why == "")
            {
                float kmax = track.CurvatureAhead(ts.s, 0f, Mathf.Max(30f, v * 3f));
                vmax = kmax > 1e-5f ? Mathf.Sqrt(maxLatAccel / kmax) * 3.6f : 999f;
                if (strongMaxKmh > 0f) vmax = Mathf.Min(vmax, strongMaxKmh);
            }
            car.assistMaxKmh = vmax;
            SpeedLimitKmh = vmax;
        }
    }
}
