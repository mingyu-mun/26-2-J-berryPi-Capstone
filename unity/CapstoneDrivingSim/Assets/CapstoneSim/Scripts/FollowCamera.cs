// FollowCamera.cs
// 3인칭 추적 시점 ↔ 운전석 1인칭 시점 전환 카메라.
// 데이터 수집 시에는 실제 운전과 비슷한 '운전석 시점'을 권장한다.
using UnityEngine;

namespace CapstoneSim
{
    public class FollowCamera : MonoBehaviour
    {
        public enum Mode { Driver, Chase, Showcase }

        public Transform target;
        public Mode mode = Mode.Driver;

        [Header("추적 시점")]
        public float distance = 6.5f;
        public float height = 2.4f;
        public float followSharpness = 6f;

        [Header("운전석 시점 (차량 로컬 좌표, 좌핸들)")]
        public Vector3 driverEye = new Vector3(-0.37f, 1.2f, 0.05f);
        public float driverPitch = 4f;

        public void ToggleMode() => mode = mode == Mode.Driver ? Mode.Chase : Mode.Driver;

        [Header("메인 메뉴 쇼케이스 (차량 주위를 천천히 회전)")]
        public float showcaseRadius = 7.5f;
        public float showcaseHeight = 1.7f;
        public float showcaseSpeedDeg = 7f;
        [Tooltip("화면에서 차량을 오른쪽으로 비키게 하는 정도 (메뉴 패널이 왼쪽에 있으므로)")] public float showcaseScreenShift = 2.4f;
        float _orbitDeg = 140f;

        void LateUpdate()
        {
            if (!target) return;
            if (mode == Mode.Showcase)
            {
                _orbitDeg += showcaseSpeedDeg * Time.unscaledDeltaTime;
                Vector3 c = target.position + Vector3.up * 0.7f;
                Vector3 dir = Quaternion.Euler(0f, _orbitDeg, 0f) * target.forward; dir.y = 0f; dir.Normalize();
                Vector3 pos = c + dir * showcaseRadius + Vector3.up * showcaseHeight;
                Vector3 look = c - pos; look.y = 0f;
                Vector3 right = Vector3.Cross(Vector3.up, look.normalized);
                transform.SetPositionAndRotation(pos, Quaternion.LookRotation(c - right * showcaseScreenShift - pos + Vector3.down * 0.3f, Vector3.up));
                return;
            }
            if (mode == Mode.Driver)
            {
                transform.SetPositionAndRotation(
                    target.TransformPoint(driverEye),
                    target.rotation * Quaternion.Euler(driverPitch, 0f, 0f));
                return;
            }

            Vector3 fwd = target.forward; fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-4f) fwd = Vector3.forward;
            fwd.Normalize();
            Vector3 desired = target.position - fwd * distance + Vector3.up * height;
            float k = 1f - Mathf.Exp(-followSharpness * Time.deltaTime);
            transform.position = Vector3.Lerp(transform.position, desired, k);
            transform.rotation = Quaternion.LookRotation(target.position + fwd * 4f + Vector3.up * 1f - transform.position, Vector3.up);
        }
    }
}
