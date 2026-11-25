// AutoTestRunner.cs
// 자동 테스트 모드 (에디터, 또는 빌드한 실행 파일). <프로젝트>/SimLogs/autotest/active.txt 가 있을 때만 동작한다.
//   - 차량을 자율주행(차로 유지 + 속도 유지)으로 몰면서
//   - 정해진 시점에 여러 각도의 스크린샷(JPG)을 찍고
//   - FPS, 바퀴 정렬, 차선 편차, 스핀 여부, 콘솔 에러/경고를 report.txt 로 남긴 뒤 Play 를 종료한다.
// SimAutoTester(에디터 스크립트)가 request.txt 를 보고 이 모드로 Play 를 시작한다.
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace CapstoneSim
{
    public class AutoTestRunner : MonoBehaviour
    {
        public static string AutoTestDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "SimLogs", "autotest"));
        public static string FlagPath => Path.Combine(AutoTestDir, "active.txt");

        public static bool Active { get; private set; }
        static float _ovBioOffset = float.NaN; static string _ovMode, _ovPreroll;
        static readonly Dictionary<string, string> _args = new Dictionary<string, string>();
        /// <summary>request.txt 로 넘어온 임의 인자 (없으면 null)</summary>
        public static string Arg(string k) => _args.TryGetValue(k, out var v) ? v : null;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { Active = false; _ovBioOffset = float.NaN; _ovMode = _ovPreroll = null; _args.Clear(); }

        /// <summary>SimBootstrap 이 호출: 자동 테스트 요청의 bio_offset / ai_mode / ai_preroll / replay / baseline 적용</summary>
        public static void ApplyOverrides(BioSource bio, AiLink ai)
        {
            if (!Active) return;
            if (bio && !float.IsNaN(_ovBioOffset)) bio.startOffsetSec = _ovBioOffset;
            if (bio is BioReplay br && !string.IsNullOrEmpty(Arg("replay")))
            {
                var e = ReplayLibrary.Find(Arg("replay"));
                if (e != null) br.sessionDir = e.dir;
            }
            if (ai && Arg("baseline") != null)
            {
                var e = ReplayLibrary.Find(Arg("baseline"));
                ai.baselineDir = e != null ? e.dir : "";     // baseline=auto → 자동 수집
            }
            if (ai && !string.IsNullOrEmpty(_ovMode)) ai.mode = _ovMode;
            if (ai && !string.IsNullOrEmpty(_ovPreroll)) ai.prerollBeforeStart = _ovPreroll == "1" || _ovPreroll == "true";
        }

        static readonly List<string> Logs = new List<string>();
        static readonly Dictionary<string, int> LogCount = new Dictionary<string, int>();
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        string _id = "run";
        float _duration = 30f, _speedKmh = 100f;
        string _outDir;
        SimBootstrap _sim;
        Camera _cap;
        RenderTexture _rt;
        Texture2D _tex;
        readonly List<string> _shots = new List<string>();
        readonly List<float> _frameMs = new List<float>();
        float _maxAbsOffset, _maxYaw, _sumSpeed; int _speedN, _departures, _prevDep;
        bool _driving;
        readonly StringBuilder _diag = new StringBuilder();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void Boot()
        {
            // 에디터, 또는 실행 파일을 -jberry-autotest 인자로 띄웠을 때만 (팀원 PC 에 남은 active.txt 로 잘못 켜지지 않게)
            bool allowed = Application.isEditor || System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-jberry-autotest") >= 0;
            if (!allowed || !File.Exists(FlagPath)) return;
            Logs.Clear(); LogCount.Clear();
            Application.logMessageReceived -= OnLog;
            Application.logMessageReceived += OnLog;
            var go = new GameObject("AutoTestRunner");
            DontDestroyOnLoad(go);
            var r = go.AddComponent<AutoTestRunner>();
            Active = true;
            foreach (var line in File.ReadAllLines(FlagPath))
            {
                int i = line.IndexOf('='); if (i < 0) continue;
                string k = line.Substring(0, i).Trim(), v = line.Substring(i + 1).Trim();
                _args[k] = v;
                if (k == "id") r._id = v;
                if (k == "duration") float.TryParse(v, NumberStyles.Float, Inv, out r._duration);
                if (k == "speed") float.TryParse(v, NumberStyles.Float, Inv, out r._speedKmh);
                if (k == "bio_offset") float.TryParse(v, NumberStyles.Float, Inv, out _ovBioOffset);
                if (k == "ai_mode") _ovMode = v;
                if (k == "ai_preroll") _ovPreroll = v;
            }
        }

        static void OnLog(string msg, string stack, LogType type)
        {
            string firstStack = string.IsNullOrEmpty(stack) ? "" : stack.Split('\n')[0];
            string key = $"[{type}] {msg}" + (type == LogType.Log ? "" : $"  @ {firstStack}");
            if (LogCount.ContainsKey(key)) { LogCount[key]++; return; }
            LogCount[key] = 1; Logs.Add(key);
        }

        System.Diagnostics.Process _spawn;

        /// <summary>spawn_py=<스크립트와 인자> : 시험 동안 .venv Python 으로 보조 프로세스 실행 (예: 실측 모드용 rpi_sender.py)</summary>
        void SpawnHelper()
        {
            string cmd = Arg("spawn_py");
            if (string.IsNullOrEmpty(cmd)) return;
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string py = SimPaths.AiPython ?? "python";
            try
            {
                var psi = new System.Diagnostics.ProcessStartInfo(File.Exists(py) ? py : "python", "-u " + cmd)
                { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = root, RedirectStandardOutput = true, RedirectStandardError = true };
                _spawn = System.Diagnostics.Process.Start(psi);
                _spawn.OutputDataReceived += (o, e) => { if (e.Data != null) lock (_spawnOut) _spawnOut.Add(e.Data); };
                _spawn.ErrorDataReceived += (o, e) => { if (e.Data != null) lock (_spawnOut) _spawnOut.Add("[stderr] " + e.Data); };
                _spawn.BeginOutputReadLine(); _spawn.BeginErrorReadLine();
                _diag.AppendLine($"[보조 프로세스] {cmd}");
            }
            catch (System.Exception e) { _diag.AppendLine($"[보조 프로세스 실패] {cmd}: {e.Message}"); }
        }
        readonly List<string> _spawnOut = new List<string>();

        IEnumerator Start()
        {
            SpawnHelper();
            Application.runInBackground = true;
            _outDir = Path.Combine(AutoTestDir, _id);
            Directory.CreateDirectory(_outDir);

            float waitStart = Time.realtimeSinceStartup;
            while ((_sim = FindFirstObjectByType<SimBootstrap>()) == null || _sim.Car == null)
            {
                if (Time.realtimeSinceStartup - waitStart > 10f) { Finish("씬에서 SimBootstrap(DrivingSim)을 찾지 못함"); yield break; }
                yield return null;
            }
            var car = _sim.Car;

            _rt = new RenderTexture(1280, 720, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { antiAliasing = 4 };
            _tex = new Texture2D(1280, 720, TextureFormat.RGB24, false);
            var cgo = new GameObject("AutoTestCaptureCam");
            cgo.transform.SetParent(transform);
            _cap = cgo.AddComponent<Camera>();
            _cap.enabled = false;
            _cap.nearClipPlane = 0.05f; _cap.farClipPlane = 6000f;
            _cap.targetTexture = _rt;

            // ---- 1) 정지 상태: 차량 외형 점검 샷 ----
            yield return new WaitForSeconds(1.5f);
            Diagnose(car, "정차 1.5 s");
            var t = car.transform;
            Vector3 c = t.TransformPoint(CarCenterLocal(car));
            Shot("01_side_left", c - t.right * 6f + Vector3.up * 0.6f, c, 45f);
            Shot("02_front34", c + t.forward * 6f + t.right * 4f + Vector3.up * 1.6f, c, 45f);
            Shot("03_rear34", c - t.forward * 6f - t.right * 4f + Vector3.up * 2.0f, c, 45f);
            if (car.wheelFL) ShotWheel(car, "04_wheel_FL");
            ShotDriver("05_driver_start");

            // ---- 2) 자율주행 ----
            if (Arg("drive") != "0") car.externalControl = true;
            _driving = true;
            yield return new WaitForSeconds(9f);
            Diagnose(car, "직선 주행 9 s");
            ShotDriver("06_driver_cruise");
            ShotChase("07_chase_cruise");

            // ---- 3) 곡선 구간으로 순간이동 후 주행 ----
            TeleportToCurve(car);
            yield return new WaitForFixedUpdate();
            car.SetForwardSpeed(_speedKmh / 3.6f * 0.9f);
            yield return new WaitForSeconds(7f);
            Diagnose(car, "곡선 진입 7 s");
            ShotDriver("08_driver_curve");
            ShotChase("09_chase_curve");

            // ---- 4) 전체 트랙 조감 (안개 끄고) ----
            bool fog = RenderSettings.fog; RenderSettings.fog = false;
            Shot("10_overview", new Vector3(0f, 1500f, -1f), Vector3.zero, 75f);
            RenderSettings.fog = fog;

            // ---- 5) HUD·패널 포함 Game 뷰: 남은 시간 동안 10초마다 캡처 ----
            int hudN = 0;
            do
            {
                string hudName = $"11_gameview_{hudN:00}_{Time.timeSinceLevelLoad:F0}s.png";
                ScreenCapture.CaptureScreenshot(Path.Combine(_outDir, hudName));
                _shots.Add(hudName + " (Game 뷰)");
                hudN++;
                float remain = _duration - Time.timeSinceLevelLoad;
                if (remain <= 1f) break;
                yield return new WaitForSeconds(Mathf.Min(10f, remain - 0.5f));
            } while (hudN < 12);
            yield return new WaitForSeconds(0.5f);
            Finish(null);
        }

        void FixedUpdate()
        {
            if (!_driving || _sim == null || _sim.Car == null || Arg("drive") == "0") return;   // drive=0: 러너는 운전하지 않음 (메뉴 모의 주행 확인용)
            var car = _sim.Car;
            var ts = _sim.Track.Query(car.transform.position, car.transform.forward);
            float v = Mathf.Max(1f, car.SpeedKmh / 3.6f);

            // Stanley 조향 + 곡률 피드포워드 (모든 곡선이 좌회전이므로 피드포워드는 음수)
            float ff = -Mathf.Atan(car.wheelBase * ts.curvature) * Mathf.Rad2Deg;
            float fb = -(ts.headingErrorDeg + Mathf.Atan2(1.2f * ts.laneOffset, v + 2f) * Mathf.Rad2Deg);
            car.extSteerWheelDeg = (ff + fb) * car.steeringRatio;

            float err = _speedKmh - car.SpeedKmh;
            car.extThrottle = Mathf.Clamp01(err * 0.08f);
            car.extBrake = err < -6f ? Mathf.Clamp01(-err * 0.05f) : 0f;

            _maxAbsOffset = Mathf.Max(_maxAbsOffset, Mathf.Abs(ts.laneOffset));
            _maxYaw = Mathf.Max(_maxYaw, Mathf.Abs(car.YawRateDps));
            _sumSpeed += car.SpeedKmh; _speedN++;
            int dep = Mathf.Abs(ts.laneOffset) > (_sim.Track.LaneWidth - 1.8f) * 0.5f || ts.lane == 0 ? 1 : 0;
            if (dep == 1 && _prevDep == 0) _departures++;
            _prevDep = dep;
        }

        readonly List<string> _aiTrace = new List<string>();
        float _aiTraceT = -9f;

        void Update()
        {
            _frameMs.Add(Time.unscaledDeltaTime * 1000f);
            var ai = _sim ? _sim.Ai : null;
            if (ai != null && ai.Latest != null && ai.SessionAccepted && Time.unscaledTime - _aiTraceT >= 1f)
            {
                _aiTraceT = Time.unscaledTime;
                var r = ai.Latest;
                string st = r.has_out
                    ? $"lv {r.level}({r.level_name}){(r.candidate ? " 후보" : "")}{(r.frozen ? " 보류" : "")} p {r.p_used:F2} ch {r.channel} q {r.q_ecg:F2} HR {r.hr_ecg:F0}/{r.hr_ppg:F0} 창끝 {r.win_end:F0}s"
                    : $"기준 {r.base_status} {r.clean_windows}/{r.required_clean}";
                _aiTrace.Add($"{Time.timeSinceLevelLoad,5:F0}s  t_rel {r.t_rel,6:F0}s  {st}  창 {r.n_windows} 처리 {r.proc_ms:F0}ms" +
                             (_sim.Bio ? $"  [replay {_sim.Bio.FileTimeSec:F0}s {_sim.Bio.SegmentLabel}]" : ""));
            }
        }

        // ------------------------------------------------------------------
        Vector3 CarCenterLocal(CarController car)
        {
            var b = CarModelFitter.BoundsOf(car.transform, car.transform);
            return b.size == Vector3.zero ? Vector3.up * 0.7f : b.center;
        }

        void ShotWheel(CarController car, string name)
        {
            Vector3 wp; Quaternion wq;
            car.wheelFL.GetWorldPose(out wp, out wq);
            var t = car.transform;
            Shot(name, wp - t.right * 2.0f + t.forward * 0.6f + Vector3.up * 0.4f, wp, 40f);
        }

        void TeleportToCurve(CarController car)
        {
            Vector3 p; Quaternion r;
            _sim.Track.SpawnPose(_sim.straightLength - 80f, 2, out p, out r);
            car.Teleport(p + Vector3.up * 0.3f, r);
        }

        void ShotDriver(string name)
        {
            var t = _sim.Car.transform;
            var eye = _sim.Cam ? _sim.Cam.driverEye : new Vector3(-0.37f, 1.2f, 0.05f);
            Shot(name, t.TransformPoint(eye), t.rotation * Quaternion.Euler(4f, 0f, 0f), 65f);
        }

        void ShotChase(string name)
        {
            var t = _sim.Car.transform;
            Vector3 f = t.forward; f.y = 0f; f.Normalize();
            Vector3 pos = t.position - f * 7.5f + Vector3.up * 2.6f;
            Shot(name, pos, t.position + f * 6f + Vector3.up * 0.8f, 60f);
        }

        void Shot(string name, Vector3 pos, Vector3 lookAt, float fov) =>
            Shot(name, pos, Quaternion.LookRotation(lookAt - pos, Vector3.up), fov);

        void Shot(string name, Vector3 pos, Quaternion rot, float fov)
        {
            try
            {
                _cap.transform.SetPositionAndRotation(pos, rot);
                _cap.fieldOfView = fov;
                _cap.Render();
                var prev = RenderTexture.active;
                RenderTexture.active = _rt;
                _tex.ReadPixels(new Rect(0, 0, _rt.width, _rt.height), 0, 0);
                _tex.Apply();
                RenderTexture.active = prev;
                File.WriteAllBytes(Path.Combine(_outDir, name + ".jpg"), _tex.EncodeToJPG(85));
                _shots.Add(name + ".jpg");
            }
            catch (System.Exception e) { _shots.Add($"{name} 실패: {e.Message}"); }
        }

        void Diagnose(CarController car, string label)
        {
            var t = car.transform;
            _diag.AppendLine($"[{label}] 위치 {Fmt(t.position)}  속도 {car.SpeedKmh:F1} km/h  기울기(roll/pitch) {Mathf.DeltaAngle(0, t.eulerAngles.z):F1}°/{Mathf.DeltaAngle(0, t.eulerAngles.x):F1}°");
            var ts = _sim.Track.Query(t.position, t.forward);
            _diag.AppendLine($"    차로 {ts.lane}  편차 {ts.laneOffset:F2} m  헤딩오차 {ts.headingErrorDeg:F1}°  핸들 {car.SteeringWheelDeg:F1}°");
            var ws = new[] { car.wheelFL, car.wheelFR, car.wheelRL, car.wheelRR };
            var vs = new[] { car.visualFL, car.visualFR, car.visualRL, car.visualRR };
            string[] tag = { "FL", "FR", "RL", "RR" };
            for (int i = 0; i < 4; i++)
            {
                if (!ws[i]) continue;
                ws[i].GetWorldPose(out var wp, out _);
                bool g = ws[i].GetGroundHit(out var hit);
                string vis = "비주얼 없음";
                if (vs[i])
                {
                    var vb = CarModelFitter.BoundsOf(vs[i], null);
                    vis = $"비주얼 중심 오차 {Vector3.Distance(vb.center, wp) * 100f:F1} cm, 비주얼 반경 {vb.extents.y:F2} m";
                }
                _diag.AppendLine($"    {tag[i]}: 접지 {(g ? "O" : "X")}  휠 반경 {ws[i].radius:F2} m  서스펜션 압축 {(g ? (-ws[i].transform.InverseTransformPoint(hit.point).y - ws[i].radius) / ws[i].suspensionDistance : -1f):F2}  {vis}");
            }
        }

        void Finish(string error)
        {
            _driving = false;
            var sb = new StringBuilder();
            sb.AppendLine($"# 자동 테스트 리포트: {_id}");
            sb.AppendLine($"Unity {Application.unityVersion} / RP: {(UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline ? UnityEngine.Rendering.GraphicsSettings.currentRenderPipeline.name : "Built-in")} / 화면 {Screen.width}x{Screen.height}");
            if (error != null) sb.AppendLine("오류: " + error);
            if (_frameMs.Count > 10)
            {
                _frameMs.Sort();
                float avg = 0f; foreach (var f in _frameMs) avg += f; avg /= _frameMs.Count;
                sb.AppendLine($"프레임: 평균 {1000f / avg:F1} FPS, 하위 5% {1000f / _frameMs[(int)(_frameMs.Count * 0.95f)]:F1} FPS ({_frameMs.Count} 프레임)");
            }
            if (_sim != null && _sim.Car != null)
            {
                var car = _sim.Car;
                var b = CarModelFitter.BoundsOf(car.transform, car.transform);
                sb.AppendLine($"차량: 모델 {(_sim.carModelPrefab ? _sim.carModelPrefab.name : "기본 도형")}, 크기 {Fmt(b.size)} m, 축거 {car.wheelBase:F2} m, 윤거 {car.trackWidth:F2} m");
                sb.AppendLine($"자율주행: 평균 속도 {(_speedN > 0 ? _sumSpeed / _speedN : 0f):F1} km/h, 최대 |차선 편차| {_maxAbsOffset:F2} m, 차선 이탈 {_departures}회, 최대 |요레이트| {_maxYaw:F1}°/s");
            }
            if (_sim != null && _sim.Ai != null)
            {
                sb.AppendLine($"AI: 서버 상태 '{_sim.Ai.ServerStatus}', 결과 {_sim.Ai.ResultCount}개 수신, 연결 {(_sim.Ai.Connected ? "O" : "X")}");
                sb.AppendLine($"Bio replay: {(_sim.Bio ? _sim.Bio.Status : "없음")}");
            }
            sb.AppendLine();
            sb.AppendLine("## AI 판정 흐름 (1초마다)");
            foreach (var line in _aiTrace) sb.AppendLine("    " + line);
            sb.AppendLine();
            sb.AppendLine("## Python 서버 출력 (마지막 20줄)");
            try
            {
                string so = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "SimLogs", "ai", "server_stdout.txt"));
                if (File.Exists(so))
                {
                    using (var fs = new FileStream(so, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                    using (var rd = new StreamReader(fs))
                    {
                        var lines = rd.ReadToEnd().Split('\n');
                        for (int i = Mathf.Max(0, lines.Length - 20); i < lines.Length; i++) sb.AppendLine("    " + lines[i].TrimEnd());
                    }
                }
                else sb.AppendLine("    (없음 — 외부 서버를 썼거나 실행 실패)");
            }
            catch (System.Exception e) { sb.AppendLine("    읽기 실패: " + e.Message); }
            sb.AppendLine();
            if (_spawnOut.Count > 0)
            {
                sb.AppendLine("## 보조 프로세스 출력 (마지막 10줄)");
                lock (_spawnOut) for (int i = Mathf.Max(0, _spawnOut.Count - 10); i < _spawnOut.Count; i++) sb.AppendLine("    " + _spawnOut[i]);
                sb.AppendLine();
            }
            sb.AppendLine("## 진단");
            sb.Append(_diag);
            sb.AppendLine();
            sb.AppendLine("## 스크린샷");
            foreach (var s in _shots) sb.AppendLine("- " + s);
            sb.AppendLine();
            sb.AppendLine("## 콘솔 로그 (중복은 횟수로 표시)");
            foreach (var k in Logs) sb.AppendLine($"- {k}" + (LogCount[k] > 1 ? $"  (x{LogCount[k]})" : ""));
            File.WriteAllText(Path.Combine(_outDir ?? AutoTestDir, "report.txt"), sb.ToString(), new UTF8Encoding(false));
            Application.logMessageReceived -= OnLog;
            try { if (_spawn != null && !_spawn.HasExited) _spawn.Kill(); } catch { }
            try { File.Delete(FlagPath); } catch { }
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        static string Fmt(Vector3 v) => $"({v.x:F2}, {v.y:F2}, {v.z:F2})";

        void OnDestroy()
        {
            try { if (_spawn != null && !_spawn.HasExited) _spawn.Kill(); } catch { }
            if (_rt) _rt.Release();
        }
    }
}
