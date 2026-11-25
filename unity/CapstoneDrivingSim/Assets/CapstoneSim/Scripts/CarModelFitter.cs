// CarModelFitter.cs
// 에셋 스토어 등에서 받은 '차량 3D 모델 프리팹'을 시뮬레이터 차량(EgoCar)에 입힌다.
//   1) 프리팹 안의 스크립트·물리(Rigidbody, Collider, WheelCollider)를 모두 제거 → 외형만 사용
//   2) 크기(cm 단위 FBX 등)와 위치를 자동 보정: 길이 ≈ 4.6 m, 바닥 = 지면, 중심 = 원점
//   3) 이름으로 바퀴 4개와 실내 핸들을 찾음 ("wheel", "tire", "rim", "steer" 등)
//      → 바퀴 위치/반경으로 WheelCollider를 배치하고, 핸들은 핸들각만큼 회전시킴
// 자동 인식이 실패하면 SimBootstrap의 'Model Wheel Names' 칸에 오브젝트 이름을 직접 적으면 된다.
using System.Collections.Generic;
using UnityEngine;

namespace CapstoneSim
{
    public class FittedCarModel
    {
        public Transform model;
        public Bounds bounds;                                   // EgoCar 로컬 기준 (생성 시 EgoCar는 원점·무회전)
        public readonly Transform[] wheels = new Transform[4];  // FL, FR, RL, RR
        public readonly Vector3[] wheelCenters = new Vector3[4];
        public float wheelRadius = 0.34f;
        public bool wheelsFound;
        public Transform steeringWheel;
        public Vector3 driverEye;
        public readonly List<string> log = new List<string>();
    }

    public static class CarModelFitter
    {
        static readonly string[] WheelWords = { "wheel", "tire", "tyre", "rim", "바퀴", "휠" };
        static readonly string[] WheelExclude = { "steer", "spare", "핸들", "wheelhouse", "wheel_arch", "wheelarch", "fender" };
        static readonly string[] SteerWords = { "steer", "핸들" };

        public static FittedCarModel Fit(GameObject prefab, Transform root, float yawOffset, float targetLength,
                                         string[] wheelNames, string steeringName, GameObject wheelPrefab = null)
        {
            var res = new FittedCarModel();

            // 비활성 부모 아래에서 생성 → 프리팹 스크립트의 Awake/OnEnable 이 실행되지 않음
            var holder = new GameObject("ModelHolder");
            holder.SetActive(false);
            var go = Object.Instantiate(prefab, holder.transform);
            go.name = "Model_" + prefab.name;
            Strip(go);
            go.transform.SetParent(root, false);
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = Quaternion.Euler(0f, yawOffset, 0f);
            Object.Destroy(holder);
            res.model = go.transform;
            int conv = MaterialFixer.FixHierarchy(go);
            if (conv > 0) res.log.Add($"Built-in/HDRP 머티리얼 {conv}개를 URP/Lit 로 변환 (실행 중에만 적용)");

            // ---- 크기 보정 ----
            var b = BoundsOf(go.transform, root);
            float len = Mathf.Max(b.size.x, b.size.z);
            if (len < 2.5f || len > 8f)
            {
                float f = targetLength / Mathf.Max(0.001f, len);
                go.transform.localScale *= f;
                res.log.Add($"모델 길이 {len:F2} → {targetLength:F1} m 로 스케일 x{f:F3}");
                b = BoundsOf(go.transform, root);
            }
            if (b.size.x > b.size.z * 1.2f)
                res.log.Add("⚠ 모델이 옆으로 누워 있는 것 같습니다 (폭 > 길이). 'Car Model Yaw Offset' 을 90 또는 -90 으로 바꿔 보세요.");

            // ---- 원점 정렬: 바닥 y=0, 가로·세로 중심 0 ----
            go.transform.localPosition += new Vector3(-b.center.x, -b.min.y, -b.center.z);
            b = BoundsOf(go.transform, root);
            res.bounds = b;

            // ---- 바퀴 ----
            FindWheels(res, root, wheelNames, wheelPrefab);

            // ---- 핸들 ----
            res.steeringWheel = !string.IsNullOrWhiteSpace(steeringName) ? FindDeep(go.transform, steeringName) : FindSteering(go.transform, root, b);
            if (res.steeringWheel) res.log.Add($"핸들 인식: {res.steeringWheel.name}");
            else res.log.Add("핸들 오브젝트를 찾지 못함 (실내가 없는 모델이거나 이름이 다름) — HUD 핸들 아이콘만 표시");

            // ---- 운전자 눈 위치 ----
            if (res.steeringWheel)
            {
                var c = root.InverseTransformPoint(BoundsOf(res.steeringWheel, null).center);
                res.driverEye = new Vector3(c.x, c.y + 0.30f, c.z - 0.55f);
            }
            else
            {
                res.driverEye = new Vector3(-b.size.x * 0.2f, b.size.y * 0.88f, b.center.z - b.size.z * 0.06f);
            }
            return res;
        }

        // ------------------------------------------------------------------
        static void Strip(GameObject go)
        {
            // 스크립트끼리 RequireComponent 로 묶여 있을 수 있어 여러 번 시도
            for (int pass = 0; pass < 5; pass++)
            {
                var mbs = go.GetComponentsInChildren<MonoBehaviour>(true);
                if (mbs.Length == 0) break;
                foreach (var mb in mbs) if (mb) Object.DestroyImmediate(mb);
            }
            foreach (var j in go.GetComponentsInChildren<Joint>(true)) Object.DestroyImmediate(j);
            foreach (var w in go.GetComponentsInChildren<WheelCollider>(true)) Object.DestroyImmediate(w);
            foreach (var c in go.GetComponentsInChildren<Collider>(true)) Object.DestroyImmediate(c);
            foreach (var r in go.GetComponentsInChildren<Rigidbody>(true)) Object.DestroyImmediate(r);
            foreach (var a in go.GetComponentsInChildren<Animator>(true)) a.enabled = false;
            foreach (var a in go.GetComponentsInChildren<AudioSource>(true)) a.enabled = false;
        }

        static bool IsVisualRenderer(Renderer r) => r is MeshRenderer || r is SkinnedMeshRenderer;

        /// <summary>t 이하 렌더러 경계상자를 space 로컬 좌표로 (space가 null이면 월드)</summary>
        public static Bounds BoundsOf(Transform t, Transform space)
        {
            bool has = false; var b = new Bounds();
            foreach (var r in t.GetComponentsInChildren<Renderer>())
            {
                if (!IsVisualRenderer(r)) continue;
                var wb = r.bounds;
                // 월드 AABB의 8 꼭짓점을 space 로 변환해 감쌈
                for (int i = 0; i < 8; i++)
                {
                    var p = wb.center + Vector3.Scale(wb.extents, new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1));
                    if (space) p = space.InverseTransformPoint(p);
                    if (!has) { b = new Bounds(p, Vector3.zero); has = true; } else b.Encapsulate(p);
                }
            }
            return b;
        }

        static bool HasWord(string name, string[] words)
        {
            string n = name.ToLowerInvariant();
            foreach (var w in words) if (n.Contains(w)) return true;
            return false;
        }

        static Transform FindDeep(Transform t, string name)
        {
            if (t.name == name) return t;
            foreach (Transform c in t) { var f = FindDeep(c, name); if (f) return f; }
            return null;
        }

        /// <summary>바퀴가 별도 프리팹으로 제공되는 모델용: 각 바퀴 자리(빈 오브젝트)에 바퀴 프리팹을 붙임</summary>
        static void AttachWheelPrefabs(FittedCarModel res, GameObject wheelPrefab)
        {
            if (!wheelPrefab) return;
            int n = 0;
            for (int i = 0; i < 4; i++)
            {
                var slot = res.wheels[i];
                if (!slot) continue;
                var holder = new GameObject("WheelHolder"); holder.SetActive(false);
                var w = Object.Instantiate(wheelPrefab, holder.transform);
                w.name = "WheelVisual_" + wheelPrefab.name;
                Strip(w);
                w.transform.SetParent(slot, false);
                w.transform.localPosition = Vector3.zero;
                w.transform.localRotation = Quaternion.identity;
                w.transform.localScale = Vector3.one;
                Object.Destroy(holder);
                MaterialFixer.FixHierarchy(w);
                n++;
            }
            res.log.Add($"바퀴 프리팹 '{wheelPrefab.name}' 을 {n}개 바퀴 자리에 장착");
        }

        static void FindWheels(FittedCarModel res, Transform root, string[] names, GameObject wheelPrefab)
        {
            var b = res.bounds;
            var model = res.model;

            // 1) 이름을 직접 지정한 경우
            if (names != null && names.Length == 4 && !string.IsNullOrWhiteSpace(names[0]))
            {
                bool ok = true;
                for (int i = 0; i < 4; i++)
                {
                    res.wheels[i] = FindDeep(model, names[i]);
                    if (!res.wheels[i]) { ok = false; res.log.Add($"⚠ 지정한 바퀴 이름 '{names[i]}' 을 찾지 못함"); }
                }
                if (ok) { AttachWheelPrefabs(res, wheelPrefab); FinishWheels(res, root, "이름 지정"); return; }
            }

            // 2) 자동 인식: 바퀴 관련 이름을 가진 렌더러 → 바퀴 하나 크기를 넘지 않는 범위에서 가장 위 조상까지
            var tops = new HashSet<Transform>();
            foreach (var r in model.GetComponentsInChildren<Renderer>())
            {
                if (!IsVisualRenderer(r)) continue;
                Transform t = r.transform, top = null;
                while (t && t != model)
                {
                    if (HasWord(t.name, WheelWords) && !HasWord(t.name, WheelExclude))
                    {
                        var tb = BoundsOf(t, root);
                        if (tb.size.x < b.size.x * 0.45f && tb.size.z < b.size.z * 0.4f) top = t; // 바퀴 1개 크기 이내
                        else break;
                    }
                    t = t.parent;
                }
                if (top) tops.Add(top);
            }

            // 3) 사분면(앞/뒤 × 좌/우)별로 가장 큰 덩어리를 바퀴로 채택, 같은 자리의 나머지 부품(림·타이어 분리 모델)은 그 아래로 묶음
            var quad = new List<Transform>[4];
            for (int i = 0; i < 4; i++) quad[i] = new List<Transform>();
            foreach (var t in tops)
            {
                var c = BoundsOf(t, root);
                if (c.center.y > b.size.y * 0.5f || c.extents.y < 0.12f) continue; // 너무 높거나 작으면 제외
                int q = (c.center.z > 0f ? 0 : 2) + (c.center.x < 0f ? 0 : 1);    // 0 FL, 1 FR, 2 RL, 3 RR
                quad[q].Add(t);
            }
            for (int q = 0; q < 4; q++)
            {
                if (quad[q].Count == 0) continue;
                Transform best = null; float vol = -1f;
                foreach (var t in quad[q]) { var s = BoundsOf(t, root).size; float v = s.x * s.y * s.z; if (v > vol) { vol = v; best = t; } }
                var bc = BoundsOf(best, root);
                foreach (var t in quad[q])
                    if (t != best && Vector3.Distance(BoundsOf(t, root).center, bc.center) < bc.extents.y * 0.6f)
                        t.SetParent(best, true);
                res.wheels[q] = best;
            }
            AttachWheelPrefabs(res, wheelPrefab);
            FinishWheels(res, root, "자동 인식");
        }

        static void FinishWheels(FittedCarModel res, Transform root, string how)
        {
            res.wheelsFound = res.wheels[0] && res.wheels[1] && res.wheels[2] && res.wheels[3];
            if (!res.wheelsFound)
            {
                res.log.Add("⚠ 바퀴 4개를 모두 찾지 못함 → 바퀴는 회전하지 않고 물리 바퀴는 차체 크기로 추정 배치. " +
                            "'Model Wheel Names' 에 FL, FR, RL, RR 순서로 바퀴 오브젝트 이름을 적어 주세요.");
                var b = res.bounds;
                float x = b.size.x * 0.5f - 0.25f, z = b.size.z * 0.31f;
                res.wheelCenters[0] = new Vector3(-x, 0.34f, z); res.wheelCenters[1] = new Vector3(x, 0.34f, z);
                res.wheelCenters[2] = new Vector3(-x, 0.34f, -z); res.wheelCenters[3] = new Vector3(x, 0.34f, -z);
                for (int i = 0; i < 4; i++) res.wheels[i] = null;
                return;
            }
            float rsum = 0f;
            string[] tag = { "FL", "FR", "RL", "RR" };
            var names = new List<string>();
            int empty = 0;
            for (int i = 0; i < 4; i++)
            {
                var wb = BoundsOf(res.wheels[i], root);
                if (wb.size.sqrMagnitude < 1e-6f)
                {
                    // 메시가 없는 빈 자리표시 오브젝트 → 피벗 위치를 바퀴 중심으로, 반경은 기본값
                    res.wheelCenters[i] = root.InverseTransformPoint(res.wheels[i].position);
                    rsum += 0.33f; empty++;
                }
                else
                {
                    res.wheelCenters[i] = wb.center;
                    rsum += wb.extents.y;
                }
                names.Add($"{tag[i]}={res.wheels[i].name}");
            }
            if (empty > 0)
                res.log.Add($"⚠ 바퀴 {empty}개에 보이는 메시가 없음 (빈 자리표시 오브젝트). 바퀴가 별도 프리팹으로 제공되는 에셋이면 'Model Wheel Prefab' 에 넣으세요.");
            res.wheelRadius = Mathf.Clamp(rsum / 4f, 0.2f, 0.6f);
            res.log.Add($"바퀴 {how}: {string.Join(", ", names)} (반경 {res.wheelRadius:F2} m)");
        }

        static Transform FindSteering(Transform model, Transform root, Bounds car)
        {
            Transform best = null; float bestScore = -1f;
            foreach (var t in model.GetComponentsInChildren<Transform>())
            {
                if (!HasWord(t.name, SteerWords)) continue;
                var tb = BoundsOf(t, root);
                if (tb.size == Vector3.zero || tb.size.magnitude > 0.9f) continue; // 핸들 크기(지름 ~0.4 m) 범위
                if (tb.center.y < car.size.y * 0.35f) continue;                  // 바퀴 조향 부품(너클 등) 제외
                float score = tb.size.magnitude;
                if (score > bestScore) { bestScore = score; best = t; }
            }
            return best;
        }
    }
}
