// CarAutopilot.cs
// 메인 메뉴 배경의 '모의 주행'. 차로 중심을 따라 일정 속도로 달린다 (사람 입력 대신 CarController.ext* 값을 넣음).
//   조향: Stanley 방식  δ = −(ψ_e + atan(k·e / (v + v0)))  +  곡률 피드포워드 atan(L·κ)
//     ψ_e : 헤딩 오차 [rad]   e : 차로 중심 대비 횡편차 [m]   v : 속도 [m/s]   k, v0 : 이득·저속 보정
//     L : 축거 [m]   κ : 도로 곡률 [1/m]   δ : 앞바퀴 조향각 → 핸들각 = δ × 조향비
//   속도: 목표 속도와의 차이에 비례한 가속/감속 (정지 상태에서 브레이크를 계속 밟으면 후진하는 차량 모델이므로 브레이크는 과속일 때만)
using UnityEngine;

namespace CapstoneSim
{
    public class CarAutopilot : MonoBehaviour
    {
        public CarController car;
        public TrackBuilder track;
        [Tooltip("목표 속도 [km/h]")] public float targetKmh = 70f;
        [Tooltip("횡편차 이득 k")] public float gainK = 1.2f;

        void OnEnable() { if (car) car.externalControl = true; }

        void OnDisable()
        {
            if (!car) return;
            car.extThrottle = 0f; car.extBrake = 0f; car.extSteerWheelDeg = 0f;
            car.externalControl = false;
        }

        void FixedUpdate()
        {
            if (!car || !track) return;
            car.externalControl = true;
            var ts = track.Query(car.transform.position, car.transform.forward);
            float v = Mathf.Max(1f, car.SpeedKmh / 3.6f);
            float ff = -Mathf.Atan(car.wheelBase * ts.curvature) * Mathf.Rad2Deg;
            float fb = -(ts.headingErrorDeg + Mathf.Atan2(gainK * ts.laneOffset, v + 2f) * Mathf.Rad2Deg);
            car.extSteerWheelDeg = (ff + fb) * car.steeringRatio;
            float err = targetKmh - car.SpeedKmh;
            car.extThrottle = Mathf.Clamp01(err * 0.08f);
            car.extBrake = err < -6f && car.SpeedKmh > 5f ? Mathf.Clamp01(-err * 0.05f) : 0f;
        }
    }
}
