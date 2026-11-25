// TrackBuilder.cs
// 외부 에셋 없이 '스타디움형(직선 2 + 반원 2) 편도 2차로 고속도로 순환로'를 코드로 생성한다.
// 장시간 단조 주행(졸음·피로 유발) 시나리오에 맞춘 형태이며,
// 차량 위치 → 차선 중심 대비 횡방향 편차(lane offset) 계산 기능을 함께 제공한다.
using System.Collections.Generic;
using UnityEngine;

namespace CapstoneSim
{
    /// <summary>차량 위치를 트랙 기준으로 투영한 결과</summary>
    public struct TrackSample
    {
        public float s;              // 트랙 진행 거리 [m] (0 ~ Length)
        public float lateral;        // 도로 중앙선(1·2차로 경계) 기준 횡위치 [m], +가 오른쪽
        public int lane;             // 1 = 왼쪽(1차로), 2 = 오른쪽(2차로), 0 = 도로 밖
        public float laneOffset;     // 현재 차로 중심 대비 편차 [m], +가 오른쪽
        public float headingErrorDeg;// 차량 진행방향 - 도로 접선 방향 [deg], +가 오른쪽으로 틀어짐
        public float curvature;      // 도로 곡률 [1/m] (직선 0, 곡선 1/R)
        public Vector3 point;        // 투영된 중앙선 위 점
        public Vector3 tangent;      // 도로 진행 방향 단위벡터
    }

    public class TrackBuilder : MonoBehaviour
    {
        [Header("트랙 형상")]
        [Tooltip("직선 구간 길이 [m]")] public float straightLength = 1500f;
        [Tooltip("곡선 구간 반경 [m] (고속도로 설계 기준 최소 반경은 수백 m 수준)")] public float curveRadius = 300f;
        [Tooltip("중앙선 샘플 간격 [m]")] public float sampleSpacing = 2f;

        [Header("도로 단면")]
        public float laneWidth = 3.5f;
        public float shoulderWidth = 1.5f;
        public float roadHeight = 0.03f;

        [Header("노면 텍스처")]
        [Tooltip("노면 텍스처 1장이 덮는 크기 [m]")] public float uvTileMeters = 4f;

        [Header("주변 물체 (속도감)")]
        public float poleSpacing = 50f;
        public bool buildPoles = true;
        public bool buildGuardrails = true;
        [Tooltip("가드레일 지주 간격 [m]")] public float guardrailPostSpacing = 4f;
        public bool buildTrees = true;
        public int treeCount = 900;
        [Tooltip("나무를 심는 도로 바깥 거리 범위 [m]")] public Vector2 treeDistance = new Vector2(14f, 140f);

        public float Length { get; private set; }
        public float LaneWidth => laneWidth;

        readonly List<Vector3> _pts = new List<Vector3>();
        readonly List<Vector3> _tan = new List<Vector3>();
        readonly List<float> _s = new List<float>();
        readonly List<float> _curv = new List<float>();
        bool _built;

        // ------------------------------------------------------------------
        // 중앙선(1·2차로 경계선) 매개화: s(진행거리) → 위치, 접선, 곡률
        // 구간 순서: 오른쪽 직선(+Z) → 위쪽 반원 → 왼쪽 직선(-Z) → 아래쪽 반원
        // 위에서 내려다보면 반시계 방향으로 돈다(항상 좌회전 곡선).
        // ------------------------------------------------------------------
        void Evaluate(float s, out Vector3 p, out Vector3 t, out float k)
        {
            float L = straightLength, R = curveRadius;
            float arc = Mathf.PI * R;
            float y = roadHeight;
            s = Mathf.Repeat(s, 2f * L + 2f * arc);

            if (s < L)
            {   // 오른쪽 직선: x = +R, z: -L/2 → +L/2
                p = new Vector3(R, y, -L / 2f + s); t = Vector3.forward; k = 0f; return;
            }
            s -= L;
            if (s < arc)
            {   // 위쪽 반원: 중심 (0, +L/2), 각도 0 → π
                float th = s / R;
                p = new Vector3(R * Mathf.Cos(th), y, L / 2f + R * Mathf.Sin(th));
                t = new Vector3(-Mathf.Sin(th), 0f, Mathf.Cos(th)); k = 1f / R; return;
            }
            s -= arc;
            if (s < L)
            {   // 왼쪽 직선: x = -R, z: +L/2 → -L/2
                p = new Vector3(-R, y, L / 2f - s); t = Vector3.back; k = 0f; return;
            }
            s -= L;
            {   // 아래쪽 반원: 중심 (0, -L/2), 각도 π → 2π
                float th = Mathf.PI + s / R;
                p = new Vector3(R * Mathf.Cos(th), y, -L / 2f + R * Mathf.Sin(th));
                t = new Vector3(-Mathf.Sin(th), 0f, Mathf.Cos(th)); k = 1f / R;
            }
        }

        /// <summary>진행방향 t에 대한 오른쪽 수평 법선 (Unity 왼손 좌표계)</summary>
        static Vector3 RightOf(Vector3 t) => new Vector3(t.z, 0f, -t.x);

        public struct Materials
        {
            public Material road, line, pole, rail, trunk, leaf;
        }

        public void Build(Materials mats)
        {
            Material roadMat = mats.road, lineMat = mats.line, poleMat = mats.pole;
            if (_built) return;
            _built = true;

            Length = 2f * straightLength + 2f * Mathf.PI * curveRadius;
            int n = Mathf.CeilToInt(Length / sampleSpacing);
            float ds = Length / n;
            _pts.Clear(); _tan.Clear(); _s.Clear(); _curv.Clear();
            for (int i = 0; i < n; i++)
            {
                Evaluate(i * ds, out var p, out var t, out var k);
                _pts.Add(p); _tan.Add(t); _s.Add(i * ds); _curv.Add(k);
            }

            float half = laneWidth + shoulderWidth; // 중앙선에서 도로 끝까지

            // --- 노면 (콜라이더 포함) ---
            var road = new GameObject("Road");
            road.transform.SetParent(transform, false);
            var roadMesh = Strip(-half, half, 0f, 0f, Length, true);
            road.AddComponent<MeshFilter>().sharedMesh = roadMesh;
            road.AddComponent<MeshRenderer>().sharedMaterial = roadMat;
            road.AddComponent<MeshCollider>().sharedMesh = roadMesh;

            // --- 차선 도색 (콜라이더 없음) ---
            var lines = new GameObject("LaneMarkings");
            lines.transform.SetParent(transform, false);
            var combine = new List<CombineInstance>();
            const float lw = 0.15f, lift = 0.01f;
            // 좌·우 바깥 실선
            combine.Add(new CombineInstance { mesh = Strip(-laneWidth - lw / 2, -laneWidth + lw / 2, lift, 0f, Length, false) });
            combine.Add(new CombineInstance { mesh = Strip(laneWidth - lw / 2, laneWidth + lw / 2, lift, 0f, Length, false) });
            // 1·2차로 경계 점선 (도색 8 m / 간격 12 m — 국내 고속도로 차선 규격)
            for (float s0 = 0f; s0 < Length - 8f; s0 += 20f)
                combine.Add(new CombineInstance { mesh = Strip(-lw / 2, lw / 2, lift, s0, 8f, false) });
            var lineMesh = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            lineMesh.CombineMeshes(combine.ToArray(), true, false);
            lines.AddComponent<MeshFilter>().sharedMesh = lineMesh;
            lines.AddComponent<MeshRenderer>().sharedMaterial = lineMat;

            // --- 갓길 기둥 (속도감용, 충돌체 없음) ---
            if (buildPoles && poleSpacing > 1f)
            {
                var poles = new GameObject("RoadsidePoles");
                poles.transform.SetParent(transform, false);
                for (float s0 = 0f; s0 < Length; s0 += poleSpacing)
                {
                    Evaluate(s0, out var p, out var t, out _);
                    var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    Destroy(pole.GetComponent<Collider>());
                    pole.name = "Pole";
                    pole.transform.SetParent(poles.transform, false);
                    pole.transform.position = p + RightOf(t) * (half + 1.0f) + Vector3.up * 1.0f;
                    pole.transform.localScale = new Vector3(0.15f, 1.0f, 0.15f);
                    pole.GetComponent<MeshRenderer>().sharedMaterial = poleMat;
                }
            }

            if (buildGuardrails && mats.rail) BuildGuardrails(half + 0.6f, mats.rail);
            if (buildTrees && mats.trunk && mats.leaf) BuildTrees(half, mats.trunk, mats.leaf);
        }

        // ------------------------------------------------------------------
        // 가드레일: 양쪽 도로 끝 바깥에 W빔(판) + 지주. 시각용(충돌체 없음)
        void BuildGuardrails(float offset, Material mat)
        {
            var root = new GameObject("Guardrails");
            root.transform.SetParent(transform, false);
            var parts = new List<CombineInstance>();
            foreach (float side in new[] { -1f, 1f })
                parts.Add(new CombineInstance { mesh = Wall(side * offset, 0.45f, 0.80f), transform = Matrix4x4.identity });

            var tmp = GameObject.CreatePrimitive(PrimitiveType.Cube);
            var cube = tmp.GetComponent<MeshFilter>().sharedMesh;
            Destroy(tmp);
            for (float s0 = 0f; s0 < Length; s0 += guardrailPostSpacing)
            {
                Evaluate(s0, out var p, out var t, out _);
                var r = RightOf(t);
                var rot = Quaternion.LookRotation(t, Vector3.up);
                foreach (float side in new[] { -1f, 1f })
                {
                    var pos = p + r * side * (offset + 0.12f) + Vector3.up * 0.4f;
                    parts.Add(new CombineInstance { mesh = cube, transform = Matrix4x4.TRS(pos, rot, new Vector3(0.1f, 0.8f, 0.1f)) });
                }
            }
            var m = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32, name = "Guardrails" };
            m.CombineMeshes(parts.ToArray(), true, true);
            m.RecalculateBounds();
            root.AddComponent<MeshFilter>().sharedMesh = m;
            root.AddComponent<MeshRenderer>().sharedMaterial = mat;
        }

        /// <summary>중앙선 기준 횡위치 c 에 세운 높이 [h0, h1] 의 양면 벽 메시</summary>
        Mesh Wall(float c, float h0, float h1)
        {
            int segs = Mathf.CeilToInt(Length / sampleSpacing);
            int row = segs + 1;
            var v = new Vector3[row * 4];
            var tri = new int[segs * 12];
            for (int i = 0; i <= segs; i++)
            {
                Evaluate(Length * i / segs, out var p, out var t, out _);
                var b = p + RightOf(t) * c;
                b.y = 0f;
                // 앞면용 2개 + 뒷면용 2개 (법선이 서로 상쇄되지 않도록 정점을 따로 둠)
                v[i * 2] = b + Vector3.up * h0; v[i * 2 + 1] = b + Vector3.up * h1;
                v[row * 2 + i * 2] = v[i * 2]; v[row * 2 + i * 2 + 1] = v[i * 2 + 1];
                if (i < segs)
                {
                    int o = i * 2, k = i * 12, o2 = row * 2 + i * 2;
                    tri[k] = o; tri[k + 1] = o + 2; tri[k + 2] = o + 1;            // -r 방향 면
                    tri[k + 3] = o + 1; tri[k + 4] = o + 2; tri[k + 5] = o + 3;
                    tri[k + 6] = o2; tri[k + 7] = o2 + 1; tri[k + 8] = o2 + 2;     // +r 방향 면
                    tri[k + 9] = o2 + 1; tri[k + 10] = o2 + 3; tri[k + 11] = o2 + 2;
                }
            }
            var m = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            m.vertices = v; m.triangles = tri;
            m.RecalculateNormals(); m.RecalculateBounds();
            return m;
        }

        // ------------------------------------------------------------------
        // 나무: 침엽수(원뿔) 70% + 활엽수(타원체) 30%. 메시를 합쳐 드로우콜 2개로 처리
        void BuildTrees(float roadHalf, Material trunkMat, Material leafMat)
        {
            var rng = new System.Random(1234);
            var tmpCyl = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            var tmpSph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            var cyl = tmpCyl.GetComponent<MeshFilter>().sharedMesh;
            var sph = tmpSph.GetComponent<MeshFilter>().sharedMesh;
            Destroy(tmpCyl); Destroy(tmpSph);
            var cone = Cone(10);

            var trunks = new List<CombineInstance>();
            var leaves = new List<CombineInstance>();
            for (int i = 0; i < treeCount; i++)
            {
                float s0 = (float)rng.NextDouble() * Length;
                float side = rng.NextDouble() < 0.5 ? -1f : 1f;
                float d = roadHalf + Mathf.Lerp(treeDistance.x, treeDistance.y, Mathf.Pow((float)rng.NextDouble(), 1.5f));
                if (side < 0f) d = Mathf.Min(d, curveRadius - 30f); // 안쪽(인필드)은 반대편 도로를 넘지 않게
                Evaluate(s0, out var p, out var t, out _);
                var basePos = p + RightOf(t) * side * d; basePos.y = 0f;
                float h = Mathf.Lerp(6f, 14f, (float)rng.NextDouble());
                var yaw = Quaternion.Euler(0f, (float)rng.NextDouble() * 360f, 0f);
                bool conifer = rng.NextDouble() < 0.7;
                float trunkH = conifer ? h * 0.25f : h * 0.45f;
                trunks.Add(new CombineInstance { mesh = cyl, transform = Matrix4x4.TRS(basePos + Vector3.up * trunkH * 0.5f, yaw, new Vector3(0.35f, trunkH * 0.5f, 0.35f)) });
                if (conifer)
                    leaves.Add(new CombineInstance { mesh = cone, transform = Matrix4x4.TRS(basePos + Vector3.up * trunkH * 0.8f, yaw, new Vector3(h * 0.32f, h * 0.8f, h * 0.32f)) });
                else
                    leaves.Add(new CombineInstance { mesh = sph, transform = Matrix4x4.TRS(basePos + Vector3.up * (trunkH + h * 0.25f), yaw, new Vector3(h * 0.55f, h * 0.6f, h * 0.55f)) });
            }
            MakeCombined("Trees_Trunks", trunks, trunkMat);
            MakeCombined("Trees_Leaves", leaves, leafMat);
        }

        void MakeCombined(string name, List<CombineInstance> parts, Material mat)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            var m = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32, name = name };
            m.CombineMeshes(parts.ToArray(), true, true);
            m.RecalculateBounds();
            go.AddComponent<MeshFilter>().sharedMesh = m;
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
        }

        /// <summary>밑면 반지름 1, 높이 1 의 원뿔 (밑면 중심이 원점)</summary>
        static Mesh Cone(int n)
        {
            var v = new List<Vector3>(); var tri = new List<int>();
            for (int i = 0; i < n; i++)
            {
                float a0 = 2f * Mathf.PI * i / n, a1 = 2f * Mathf.PI * (i + 1) / n;
                var p0 = new Vector3(Mathf.Cos(a0), 0f, Mathf.Sin(a0));
                var p1 = new Vector3(Mathf.Cos(a1), 0f, Mathf.Sin(a1));
                int k = v.Count;
                v.Add(p0); v.Add(Vector3.up); v.Add(p1);  // 옆면 (위에서 볼 때 바깥쪽을 향하도록)
                tri.Add(k); tri.Add(k + 1); tri.Add(k + 2);
            }
            var m = new Mesh();
            m.SetVertices(v); m.SetTriangles(tri, 0);
            m.RecalculateNormals(); m.RecalculateBounds();
            return m;
        }

        /// <summary>
        /// 중앙선 기준 횡위치 [a, b] 사이를 진행거리 [s0, s0+len] 구간만큼 덮는 띠 메시.
        /// </summary>
        Mesh Strip(float a, float b, float lift, float s0, float len, bool uv)
        {
            int segs = Mathf.Max(1, Mathf.CeilToInt(len / sampleSpacing));
            var v = new Vector3[(segs + 1) * 2];
            var uvs = new Vector2[v.Length];
            var tri = new int[segs * 6];
            for (int i = 0; i <= segs; i++)
            {
                float s = s0 + len * i / segs;
                Evaluate(s, out var p, out var t, out _);
                var r = RightOf(t);
                v[i * 2] = p + r * a + Vector3.up * lift;
                v[i * 2 + 1] = p + r * b + Vector3.up * lift;
                uvs[i * 2] = new Vector2(a / uvTileMeters, s / uvTileMeters);
                uvs[i * 2 + 1] = new Vector2(b / uvTileMeters, s / uvTileMeters);
                if (i < segs)
                {
                    int k = i * 6, o = i * 2;
                    // 위에서 봤을 때 시계방향 → 법선이 +Y
                    tri[k] = o; tri[k + 1] = o + 2; tri[k + 2] = o + 1;
                    tri[k + 3] = o + 1; tri[k + 4] = o + 2; tri[k + 5] = o + 3;
                }
            }
            var m = new Mesh { indexFormat = UnityEngine.Rendering.IndexFormat.UInt32 };
            m.vertices = v; m.triangles = tri;
            if (uv) m.uv = uvs;
            m.RecalculateNormals(); m.RecalculateBounds();
            return m;
        }

        /// <summary>진행 거리 s 에서 앞으로 [from, to] m 구간의 최대 곡률 [1/m] (조향 보조의 커브 감속·피드포워드용)</summary>
        public float CurvatureAhead(float s, float from, float to)
        {
            int n = _curv.Count;
            if (n == 0 || Length <= 0f) return 0f;
            float ds = Length / n, kmax = 0f;
            for (float d = from; d <= to; d += ds)
            {
                int i = (int)(Mathf.Repeat(s + d, Length) / ds) % n;
                kmax = Mathf.Max(kmax, Mathf.Abs(_curv[i]));
            }
            return kmax;
        }

        /// <summary>진행 거리 s 의 곡률 [1/m] (부호 포함)</summary>
        public float CurvatureAt(float s)
        {
            int n = _curv.Count;
            if (n == 0 || Length <= 0f) return 0f;
            return _curv[(int)(Mathf.Repeat(s, Length) / (Length / n)) % n];
        }

        /// <summary>월드 좌표 위치·전방벡터를 트랙에 투영</summary>
        public TrackSample Query(Vector3 worldPos, Vector3 forward)
        {
            var res = new TrackSample();
            if (_pts.Count == 0) return res;

            int best = 0; float bestD = float.MaxValue;
            for (int i = 0; i < _pts.Count; i++)
            {
                float dx = worldPos.x - _pts[i].x, dz = worldPos.z - _pts[i].z;
                float d = dx * dx + dz * dz;
                if (d < bestD) { bestD = d; best = i; }
            }
            var p = _pts[best]; var t = _tan[best]; var r = RightOf(t);
            var d3 = worldPos - p; d3.y = 0f;
            float along = Vector3.Dot(d3, t);
            res.lateral = Vector3.Dot(d3, r);
            res.s = Mathf.Repeat(_s[best] + along, Length);
            res.point = p; res.tangent = t; res.curvature = _curv[best];

            float half = laneWidth;
            if (res.lateral < -half - shoulderWidth || res.lateral > half + shoulderWidth) res.lane = 0;
            else res.lane = res.lateral < 0f ? 1 : 2;
            float laneCenter = res.lateral < 0f ? -laneWidth / 2f : laneWidth / 2f;
            res.laneOffset = res.lateral - laneCenter;

            var f = forward; f.y = 0f;
            res.headingErrorDeg = f.sqrMagnitude > 1e-6f ? Vector3.SignedAngle(t, f.normalized, Vector3.up) : 0f;
            return res;
        }

        /// <summary>진행거리 s 지점, 지정 차로 중심의 스폰 자세</summary>
        public void SpawnPose(float s, int lane, out Vector3 pos, out Quaternion rot)
        {
            Evaluate(s, out var p, out var t, out _);
            float off = lane == 1 ? -laneWidth / 2f : laneWidth / 2f;
            pos = p + RightOf(t) * off;
            rot = Quaternion.LookRotation(t, Vector3.up);
        }

        /// <summary>바닥(잔디) 평면이 트랙 전체를 덮도록 하는 크기 [m]</summary>
        public Vector2 GroundSize()
        {
            float margin = 3000f; // 지평선 끝까지 지면이 보이도록 넉넉하게
            return new Vector2(2f * curveRadius + margin, straightLength + 2f * curveRadius + margin);
        }
    }
}
