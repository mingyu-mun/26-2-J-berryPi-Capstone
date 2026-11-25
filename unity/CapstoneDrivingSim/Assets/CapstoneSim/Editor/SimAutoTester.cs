// SimAutoTester.cs  (Editor 전용)
// 외부(예: Claude)가 <프로젝트>/SimLogs/autotest/request.txt 를 쓰면
//   1) (refresh=true 이면) 에셋 새로고침 → 스크립트 재컴파일
//   2) 컴파일 결과를 compile_status.txt 로 기록 (에러가 있으면 여기서 중단)
//   3) SampleScene 을 열고 자동 테스트 모드로 Play → AutoTestRunner 가 스크린샷·리포트를 남기고 종료
// request.txt 형식 (key=value 줄):
//   id=run01
//   refresh=true
//   duration=30
//   speed=100
//   bio_offset=900  ai_mode=causal  ai_preroll=1   ← 실측 기록 재생 시작 위치, 추론 모드, 앞부분 빠른 입력 여부
//   set.<필드명>=<값>   ← DrivingSim(SimBootstrap) 의 Inspector 값을 바꾸고 씬 저장
//      예) set.modelWheelPrefab=Assets/SportCar/Prefabs/Wheels/Sport/Sport_Wheel_1.prefab
//          set.driverEyeAdjust=0,0.05,-0.1    set.night=true    set.modelWheelNames=FL,FR,RL,RR
// 메뉴 [Capstone > 자동 테스트 실행] 으로 직접 실행할 수도 있다.
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEditor.Compilation;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CapstoneSim.EditorTools
{
    [InitializeOnLoad]
    public static class SimAutoTester
    {
        static string Dir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "SimLogs", "autotest"));
        static string RequestPath => Path.Combine(Dir, "request.txt");
        static string StatusPath => Path.Combine(Dir, "status.txt");
        const string ScenePath = "Assets/Scenes/SampleScene.unity";
        const string StageKey = "CapstoneSim.AutoTest.Stage";

        static double _next;
        static readonly List<string> CompileMsgs = new List<string>();

        static SimAutoTester()
        {
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
            CompilationPipeline.assemblyCompilationFinished -= OnAssemblyCompiled;
            CompilationPipeline.assemblyCompilationFinished += OnAssemblyCompiled;
            CompilationPipeline.compilationFinished -= OnCompilationFinished;
            CompilationPipeline.compilationFinished += OnCompilationFinished;
            EditorApplication.playModeStateChanged -= OnPlayMode;
            EditorApplication.playModeStateChanged += OnPlayMode;
            if (SessionState.GetString(StageKey, "") == "") Status("idle (에디터 스크립트 로드됨)");
        }

        [MenuItem("Capstone/자동 테스트 실행 (스크린샷·리포트)", priority = 40)]
        static void RunFromMenu()
        {
            Directory.CreateDirectory(Dir);
            File.WriteAllText(RequestPath, $"id=manual_{System.DateTime.Now:MMdd_HHmmss}\nrefresh=false\nduration=30\nspeed=100\n");
            _next = 0;
        }

        static void OnAssemblyCompiled(string asm, CompilerMessage[] msgs)
        {
            foreach (var m in msgs)
                if (m.type == CompilerMessageType.Error)
                    CompileMsgs.Add($"{Path.GetFileName(asm)}: {m.message} ({m.file}:{m.line})");
        }

        static void OnCompilationFinished(object ctx)
        {
            Directory.CreateDirectory(Dir);
            var sb = new StringBuilder();
            sb.AppendLine($"time={System.DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"errors={CompileMsgs.Count}");
            foreach (var m in CompileMsgs) sb.AppendLine(m);
            File.WriteAllText(Path.Combine(Dir, "compile_status.txt"), sb.ToString(), new UTF8Encoding(false));
            CompileMsgs.Clear();
        }

        static void OnPlayMode(PlayModeStateChange s)
        {
            if (s == PlayModeStateChange.EnteredEditMode)
            {
                string flag = Path.Combine(Dir, "active.txt");
                if (File.Exists(flag)) File.Delete(flag);
                if (SessionState.GetString(StageKey, "") == "playing")
                {
                    SessionState.SetString(StageKey, "");
                    Status("done");
                }
            }
        }

        static void Poll()
        {
            if (EditorApplication.timeSinceStartup < _next) return;
            _next = EditorApplication.timeSinceStartup + 1.0;
            if (!File.Exists(RequestPath)) return;
            if (EditorApplication.isCompiling || EditorApplication.isUpdating) return;
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                if (SessionState.GetString(StageKey, "") != "playing") Status("waiting: Play 모드를 정지하면 자동 테스트를 시작합니다");
                return;
            }

            var req = Parse(File.ReadAllText(RequestPath));
            string id = req.TryGetValue("id", out var v) ? v : "run";
            bool refresh = req.TryGetValue("refresh", out v) && v.ToLower() == "true";
            string stage = SessionState.GetString(StageKey, "");

            if (refresh && stage != "refreshed:" + id)
            {
                SessionState.SetString(StageKey, "refreshed:" + id);
                Status($"refreshing ({id})");
                AssetDatabase.Refresh();
                return; // 컴파일이 일어나면 도메인 리로드 후 다시 Poll 로 들어옴
            }

            if (EditorUtility.scriptCompilationFailed)
            {
                File.Delete(RequestPath);
                SessionState.SetString(StageKey, "");
                Status($"failed ({id}): 스크립트 컴파일 에러 — compile_status.txt 참고");
                return;
            }

            // 시뮬레이터 씬 준비
            if (Object.FindFirstObjectByType<SimBootstrap>() == null)
            {
                var active = EditorSceneManager.GetActiveScene();
                if (active.isDirty)
                {
                    File.Delete(RequestPath);
                    Status($"failed ({id}): 현재 씬에 저장 안 된 변경이 있어 SampleScene 을 열 수 없음");
                    return;
                }
                EditorSceneManager.OpenScene(ScenePath);
                if (Object.FindFirstObjectByType<SimBootstrap>() == null)
                {
                    File.Delete(RequestPath);
                    Status($"failed ({id}): SampleScene 에 DrivingSim(SimBootstrap) 이 없음");
                    return;
                }
            }

            // Inspector 값 변경 요청 적용
            string setLog = ApplySettings(req);
            if (setLog.Length > 0) File.WriteAllText(Path.Combine(Dir, "settings_applied.txt"), setLog, new UTF8Encoding(false));

            // 자동 테스트 플래그 → AutoTestRunner 가 Play 시작 시 읽음
            Directory.CreateDirectory(Dir);
            var flag = new StringBuilder();
            flag.AppendLine("id=" + id);
            // id·refresh·set.* 를 뺀 나머지 키(duration, speed, bio_offset, ai_mode, ai_preroll …)는 그대로 전달
            foreach (var kv in req)
                if (kv.Key != "id" && kv.Key != "refresh" && !kv.Key.StartsWith("set."))
                    flag.AppendLine(kv.Key + "=" + kv.Value);
            File.WriteAllText(Path.Combine(Dir, "active.txt"), flag.ToString());
            File.Delete(RequestPath);
            SessionState.SetString(StageKey, "playing");
            Status($"playing ({id})");
            EditorApplication.EnterPlaymode();
        }

        static string ApplySettings(Dictionary<string, string> req)
        {
            var log = new StringBuilder();
            var sim = Object.FindFirstObjectByType<SimBootstrap>();
            if (!sim) return "";
            var so = new SerializedObject(sim);
            bool any = false;
            foreach (var kv in req)
            {
                if (!kv.Key.StartsWith("set.")) continue;
                string name = kv.Key.Substring(4), val = kv.Value;
                var prop = so.FindProperty(name);
                if (prop == null) { log.AppendLine($"X {name}: 필드 없음"); continue; }
                try
                {
                    var inv = System.Globalization.CultureInfo.InvariantCulture;
                    switch (prop.propertyType)
                    {
                        case SerializedPropertyType.ObjectReference:
                            prop.objectReferenceValue = string.IsNullOrEmpty(val) || val == "null" ? null : AssetDatabase.LoadAssetAtPath<Object>(val);
                            if (!string.IsNullOrEmpty(val) && val != "null" && prop.objectReferenceValue == null) { log.AppendLine($"X {name}: 에셋 없음 {val}"); continue; }
                            break;
                        case SerializedPropertyType.Float: prop.floatValue = float.Parse(val, inv); break;
                        case SerializedPropertyType.Integer: prop.intValue = int.Parse(val, inv); break;
                        case SerializedPropertyType.Enum: prop.enumValueIndex = int.Parse(val, inv); break;
                        case SerializedPropertyType.Boolean: prop.boolValue = val.ToLower() == "true" || val == "1"; break;
                        case SerializedPropertyType.String: prop.stringValue = val; break;
                        case SerializedPropertyType.Vector3:
                            var p = val.Split(',');
                            prop.vector3Value = new Vector3(float.Parse(p[0], inv), float.Parse(p[1], inv), float.Parse(p[2], inv));
                            break;
                        default:
                            if (prop.isArray)
                            {
                                var items = val.Split(',');
                                prop.arraySize = items.Length;
                                for (int i = 0; i < items.Length; i++) prop.GetArrayElementAtIndex(i).stringValue = items[i].Trim();
                                break;
                            }
                            log.AppendLine($"X {name}: 지원하지 않는 형식 {prop.propertyType}"); continue;
                    }
                    any = true;
                    log.AppendLine($"O {name} = {val}");
                }
                catch (System.Exception e) { log.AppendLine($"X {name}: {e.Message}"); }
            }
            if (any)
            {
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorSceneManager.MarkSceneDirty(sim.gameObject.scene);
                EditorSceneManager.SaveScene(sim.gameObject.scene);
            }
            return log.ToString();
        }

        static Dictionary<string, string> Parse(string text)
        {
            var d = new Dictionary<string, string>();
            foreach (var raw in text.Split('\n'))
            {
                int i = raw.IndexOf('=');
                if (i > 0) d[raw.Substring(0, i).Trim()] = raw.Substring(i + 1).Trim();
            }
            return d;
        }

        static void Status(string s)
        {
            try
            {
                Directory.CreateDirectory(Dir);
                File.WriteAllText(StatusPath, $"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss} {s}\n");
            }
            catch { }
        }
    }
}
