// CapstoneBuild.cs  (Editor 전용)
// Unity 없이 실행하는 Windows 실행 파일 만들기.
//   메뉴 Capstone > 실행 파일 빌드 (Windows)            : Build/JberryDrivingSim/JberryDrivingSim.exe
//   메뉴 Capstone > 배포용 빌드 (휴대용 AI 엔진 + zip)   : 빌드 → 휴대용 Python AI 엔진 설치 → zip
//   메뉴 Capstone > 배포 zip 만들기                     : 이미 만든 빌드 폴더를 zip 으로
// 외부(Claude)에서 SimLogs/autotest/build_request.txt 를 써도 같은 일을 한다 (key=value 줄):
//   build=1                 실행 파일 빌드
//   setup=portable|system   빌드 폴더 안에 AI 엔진 설치 (portable = 휴대용 Python, 배포용)
//   test=<id>               빌드한 실행 파일을 자동 테스트 모드로 띄움. t.<키>=<값> 은 자동 테스트 인자 (t.duration=40 …)
//   pack=1                  zip 만들기 (Build/JberryDrivingSim_<날짜>.zip)
// 진행: SimLogs/autotest/build_status.txt
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

namespace CapstoneSim.EditorTools
{
    [InitializeOnLoad]
    public static class CapstoneBuild
    {
        public const string ExeName = "JberryDrivingSim";
        const string ScenePath = "Assets/Scenes/SampleScene.unity";
        const string KeepDir = "Assets/CapstoneSim/Resources/ShaderKeep";

        static string Root => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        public static string OutDir => Path.Combine(Root, "Build", ExeName);
        static string ExePath => Path.Combine(OutDir, ExeName + ".exe");
        static string TestDir => Path.Combine(Root, "SimLogs", "autotest");
        static string ReqPath => Path.Combine(TestDir, "build_request.txt");
        static string StatusPath => Path.Combine(TestDir, "build_status.txt");

        static readonly string[] Order = { "build", "setup", "test", "pack" };
        static Dictionary<string, string> _req;
        static string _stage = "";
        static double _next, _t0;
        static bool _refreshed;
        static Process _proc;
        static Task _zip;
        static string _zipOut, _zipErr;
        static readonly StringBuilder _log = new StringBuilder();

        static CapstoneBuild()
        {
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
        }

        [MenuItem("Capstone/실행 파일 빌드 (Windows)", priority = 50)]
        static void MenuBuild() => Begin(new Dictionary<string, string> { { "build", "1" } });

        [MenuItem("Capstone/배포용 빌드 (휴대용 AI 엔진 + zip)", priority = 51)]
        static void MenuAll() => Begin(new Dictionary<string, string> { { "build", "1" }, { "setup", "portable" }, { "pack", "1" } });

        [MenuItem("Capstone/배포 zip 만들기", priority = 52)]
        static void MenuPack() => Begin(new Dictionary<string, string> { { "pack", "1" } });

        // ---------------------------------------------------------------- 진행
        static void Begin(Dictionary<string, string> req)
        {
            if (_stage != "") { UnityEngine.Debug.LogWarning("[Build] 이미 진행 중: " + _stage); return; }
            _req = req;
            _log.Clear();
            Log("요청: " + string.Join(", ", req.Select(kv => kv.Key + "=" + kv.Value)));
            Advance("");
        }

        static void Advance(string done)
        {
            int i = done == "" ? 0 : System.Array.IndexOf(Order, done) + 1;
            for (; i < Order.Length; i++)
                if (_req.ContainsKey(Order[i])) { _stage = Order[i]; _t0 = EditorApplication.timeSinceStartup; Log("단계: " + _stage); return; }
            _stage = "";
            _refreshed = false;
            Log("끝");
            Status("done");
        }

        static void Fail(string why)
        {
            Log("실패: " + why);
            Status("failed: " + why);
            UnityEngine.Debug.LogError("[Build] " + why);
            _stage = "";
            _refreshed = false;
            try { if (_proc != null && !_proc.HasExited) _proc.Kill(); } catch { }
        }

        static void Poll()
        {
            if (EditorApplication.timeSinceStartup < _next) return;
            _next = EditorApplication.timeSinceStartup + 1.0;
            if (_stage == "")
            {
                if (!File.Exists(ReqPath) || EditorApplication.isCompiling) return;
                if (EditorApplication.isPlayingOrWillChangePlaymode) { Status("waiting: Play 모드를 정지하면 빌드를 시작합니다"); return; }
                // 바뀐 스크립트를 먼저 가져오고(필요하면 재컴파일·도메인 리로드), 다음 확인 때 시작 → 옛 코드로 빌드되지 않게
                if (!_refreshed) { _refreshed = true; AssetDatabase.Refresh(); return; }
                if (EditorApplication.isUpdating) return;
                Dictionary<string, string> req;
                try { req = Parse(File.ReadAllText(ReqPath)); File.Delete(ReqPath); } catch { return; }
                Begin(req);
                return;
            }
            try
            {
                switch (_stage)
                {
                    case "build": DoBuild(); break;
                    case "setup": DoSetup(); break;
                    case "test": DoTest(); break;
                    case "pack": DoPack(); break;
                }
            }
            catch (System.Exception e) { Fail(e.Message); }
        }

        // ---------------------------------------------------------------- 1) 빌드
        static void DoBuild()
        {
            if (EditorApplication.isPlaying) { Fail("Play 중에는 빌드할 수 없습니다"); return; }
            Status("building");
            EnsureShaderKeep();

            PlayerSettings.companyName = "J_Berry PI";
            PlayerSettings.productName = "J_Berry Driving Sim";
            PlayerSettings.runInBackground = true;              // 창이 뒤에 있어도 생체 재생·AI 통신 계속
            PlayerSettings.fullScreenMode = FullScreenMode.FullScreenWindow;
            PlayerSettings.defaultIsNativeResolution = true;
            PlayerSettings.resizableWindow = true;

            Directory.CreateDirectory(OutDir);
            var opt = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = ExePath,
                target = BuildTarget.StandaloneWindows64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None,
            };
            var rep = BuildPipeline.BuildPlayer(opt);
            var sum = rep.summary;
            if (sum.result != BuildResult.Succeeded)
            {
                var errs = rep.steps.SelectMany(s => s.messages).Where(m => m.type == LogType.Error || m.type == LogType.Exception).Select(m => m.content).Take(5);
                Fail($"빌드 실패 ({sum.result}, 오류 {sum.totalErrors}개): " + string.Join(" | ", errs));
                return;
            }
            Log($"빌드 성공: {sum.totalSize / 1048576f:F0} MB, {sum.totalTime.TotalSeconds:F0} s");
            CopyTools();
            WriteLaunchers();
            Advance("build");
        }

        /// <summary>런타임에 Shader.Find 로 만드는 URP Lit 재질이 빌드에서 빠지지 않도록, 쓰는 키워드 조합의 재질을 Resources 에 둔다</summary>
        static void EnsureShaderKeep()
        {
            var lit = Shader.Find("Universal Render Pipeline/Lit");
            if (!lit) { Log("URP Lit 셰이더 없음 — 건너뜀"); return; }
            Directory.CreateDirectory(KeepDir);
            string texPath = KeepDir + "/keep_tex.asset";
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texPath);
            if (!tex)
            {
                tex = new Texture2D(4, 4, TextureFormat.RGBA32, false);
                AssetDatabase.CreateAsset(tex, texPath);
            }
            int made = 0;
            for (int surf = 0; surf < 3; surf++)          // 0 불투명, 1 알파 컷, 2 반투명
                for (int bits = 0; bits < 16; bits++)     // 노멀맵, 금속광택맵, AO맵, 발광
                {
                    string path = $"{KeepDir}/keep_{surf}_{bits:00}.mat";
                    if (File.Exists(Path.Combine(Root, path))) continue;
                    var m = new Material(lit);
                    if ((bits & 1) != 0) { m.SetTexture("_BumpMap", tex); m.EnableKeyword("_NORMALMAP"); }
                    if ((bits & 2) != 0) { m.SetTexture("_MetallicGlossMap", tex); m.EnableKeyword("_METALLICSPECGLOSSMAP"); }
                    if ((bits & 4) != 0) { m.SetTexture("_OcclusionMap", tex); m.EnableKeyword("_OCCLUSIONMAP"); }
                    if ((bits & 8) != 0)
                    {
                        m.SetColor("_EmissionColor", Color.white);
                        m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
                        m.EnableKeyword("_EMISSION");
                    }
                    if (surf == 1)
                    {
                        m.SetFloat("_AlphaClip", 1f); m.EnableKeyword("_ALPHATEST_ON"); m.renderQueue = (int)RenderQueue.AlphaTest;
                    }
                    else if (surf == 2)
                    {
                        m.SetFloat("_Surface", 1f); m.SetFloat("_Blend", 0f);
                        m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha); m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha); m.SetFloat("_ZWrite", 0f);
                        m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT"); m.SetOverrideTag("RenderType", "Transparent");
                        m.renderQueue = (int)RenderQueue.Transparent;
                    }
                    AssetDatabase.CreateAsset(m, path);
                    made++;
                }
            if (made > 0) { AssetDatabase.SaveAssets(); Log($"셰이더 보존용 재질 {made}개 생성 ({KeepDir})"); }
        }

        /// <summary>실행 파일 옆에 AI 서버·도구·문서를 복사. 이미 설치된 AI 엔진(.venv, python)은 그대로 둠</summary>
        static void CopyTools()
        {
            string src = Path.Combine(Root, "Tools"), dst = Path.Combine(OutDir, "Tools");
            CopyDir(Path.Combine(src, "jberry_ai"), Path.Combine(dst, "jberry_ai"), new[] { ".venv", "python", "__pycache__" });
            CopyDir(Path.Combine(src, "rpi_bridge"), Path.Combine(dst, "rpi_bridge"), new[] { "__pycache__" });
            foreach (var f in new[] { "convert_recording.py", "gen_synthetic_bio.py", "check_log.py" })
                if (File.Exists(Path.Combine(src, f))) File.Copy(Path.Combine(src, f), Path.Combine(dst, f), true);
            CopyDir(Path.Combine(Root, "docs"), Path.Combine(OutDir, "docs"), new string[0]);
            Log("Tools·docs 복사");
        }

        static void CopyDir(string src, string dst, string[] skip)
        {
            if (!Directory.Exists(src)) return;
            Directory.CreateDirectory(dst);
            foreach (var f in Directory.GetFiles(src))
            {
                if (f.EndsWith(".pyc") || f.EndsWith(".meta")) continue;
                File.Copy(f, Path.Combine(dst, Path.GetFileName(f)), true);
            }
            foreach (var d in Directory.GetDirectories(src))
            {
                string n = Path.GetFileName(d);
                if (skip.Contains(n)) continue;
                CopyDir(d, Path.Combine(dst, n), skip);
            }
        }

        static void WriteLaunchers()
        {
            var cp = new UTF8Encoding(false);
            File.WriteAllText(Path.Combine(OutDir, "AI_엔진_설치.bat"),
                "@echo off\r\nchcp 65001 >nul\r\necho J_Berry Driving Sim - AI 엔진 설치 (처음 한 번, 수 분 걸립니다)\r\n" +
                "powershell -NoProfile -ExecutionPolicy Bypass -File \"%~dp0Tools\\jberry_ai\\setup_ai.ps1\" %*\r\npause\r\n", cp);
            File.WriteAllText(Path.Combine(OutDir, "읽어보기.txt"),
                "J_Berry Driving Sim (운전자 상태 분석 시뮬레이터)\r\n\r\n" +
                "실행: JberryDrivingSim.exe 를 더블클릭합니다. Unity 는 필요 없습니다.\r\n" +
                "  · 창 모드 전환: Alt+Enter\r\n" +
                "  · AI 엔진: 메인 메뉴 아래 상태 줄이 'AI 엔진 준비됨' 이면 바로 씁니다.\r\n" +
                "    '없습니다' 로 나오면 [AI 엔진 설치] 를 누르거나 AI_엔진_설치.bat 을 실행합니다 (인터넷 필요, 수 분).\r\n\r\n" +
                "폴더\r\n" +
                "  JberryDrivingSim_Data/StreamingAssets/Replays  리플레이 기록 (meta.json · ecg.csv · ppg.csv)\r\n" +
                "  SimLogs/                                       주행 로그·AI 판정 기록 (실행하면 생김)\r\n" +
                "  Tools/jberry_ai/                               AI 서버 (S1-EP + 상태 머신 v2)\r\n" +
                "  docs/                                          사용 설명서\r\n\r\n" +
                "주의\r\n" +
                "  · 들어 있는 실측 생체 기록과 SimLogs 는 팀 내부용입니다. 밖으로 공유하지 마세요.\r\n" +
                "  · 모델 가중치는 WESAD 로 학습해 비상업적 용도로만 쓸 수 있습니다.\r\n", cp);
        }

        // ---------------------------------------------------------------- 2) 빌드 폴더에 AI 엔진 설치
        static string BuildSetupStatus => Path.Combine(OutDir, "SimLogs", "ai", "setup_status.txt");

        static void DoSetup()
        {
            if (_proc == null)
            {
                bool portable = _req["setup"] != "system";
                string ps1 = Path.Combine(OutDir, "Tools", "jberry_ai", "setup_ai.ps1");
                if (!File.Exists(ps1)) { Fail("빌드 폴더에 setup_ai.ps1 없음 — 먼저 build=1"); return; }
                if (portable && File.Exists(Path.Combine(OutDir, "Tools", "jberry_ai", "python", "python.exe")) && ReadStatus(BuildSetupStatus).StartsWith("ok"))
                { Log("휴대용 AI 엔진이 이미 있음"); Advance("setup"); return; }
                Directory.CreateDirectory(Path.GetDirectoryName(BuildSetupStatus));
                File.WriteAllText(BuildSetupStatus, "running: 시작\n");
                var psi = new ProcessStartInfo("powershell.exe", $"-NoProfile -ExecutionPolicy Bypass -File \"{ps1}\"" + (portable ? " -Portable" : ""))
                { UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = Path.GetDirectoryName(ps1) };
                _proc = Process.Start(psi);
                Log("AI 엔진 설치 시작 (" + (portable ? "휴대용 Python" : "PC Python") + ")");
                return;
            }
            string st = ReadStatus(BuildSetupStatus);
            Status($"setup: {st}  ({EditorApplication.timeSinceStartup - _t0:F0} s)");
            if (!_proc.HasExited)
            {
                if (EditorApplication.timeSinceStartup - _t0 > 3600) Fail("AI 엔진 설치가 60분을 넘김");
                return;
            }
            _proc = null;
            if (st.StartsWith("ok")) { Log("AI 엔진 설치 완료"); Advance("setup"); }
            else Fail("AI 엔진 설치 실패: " + st + " (Build/JberryDrivingSim/SimLogs/ai/setup_log.txt)");
        }

        // ---------------------------------------------------------------- 3) 실행 파일 자동 테스트
        static void DoTest()
        {
            string id = _req["test"];
            float dur = 40f;
            if (_req.TryGetValue("t.duration", out var dv)) float.TryParse(dv, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out dur);
            if (_proc == null)
            {
                if (!File.Exists(ExePath)) { Fail("실행 파일 없음 — 먼저 build=1"); return; }
                string tdir = Path.Combine(OutDir, "SimLogs", "autotest");
                Directory.CreateDirectory(tdir);
                var sb = new StringBuilder("id=" + id + "\n");
                foreach (var kv in _req) if (kv.Key.StartsWith("t.")) sb.Append(kv.Key.Substring(2) + "=" + kv.Value + "\n");
                File.WriteAllText(Path.Combine(tdir, "active.txt"), sb.ToString());
                string plog = Path.Combine(OutDir, "SimLogs", "player_log.txt");
                var psi = new ProcessStartInfo(ExePath, $"-jberry-autotest -screen-fullscreen 0 -screen-width 1600 -screen-height 900 -logFile \"{plog}\"")
                { UseShellExecute = true, WorkingDirectory = OutDir };       // 탐색기에서 더블클릭한 것과 같게 (에디터의 표준 입출력을 물려받지 않음)
                _proc = Process.Start(psi);
                Log($"실행 파일 자동 테스트 시작: {id} ({dur:F0} s)");
                return;
            }
            Status($"test: {id} 실행 중 ({EditorApplication.timeSinceStartup - _t0:F0} s)");
            if (!_proc.HasExited)
            {
                if (EditorApplication.timeSinceStartup - _t0 > dur + 150) { try { _proc.Kill(); } catch { } _proc = null; Fail("실행 파일이 제시간에 끝나지 않음 (강제 종료)"); }
                return;
            }
            Log($"실행 파일 종료 (코드 {_proc.ExitCode}) → Build/{ExeName}/SimLogs/autotest/{id}/");
            _proc = null;
            Advance("test");
        }

        // ---------------------------------------------------------------- 4) zip
        static readonly string[] ZipSkipDirs = { "SimLogs", ".venv", "__pycache__" };

        static void DoPack()
        {
            if (_zip == null)
            {
                if (!File.Exists(ExePath)) { Fail("실행 파일 없음 — 먼저 build=1"); return; }
                _zipOut = Path.Combine(Root, "Build", $"{ExeName}_{System.DateTime.Now:yyyyMMdd}.zip");
                _zipErr = null;
                string src = OutDir, dst = _zipOut;
                _zip = Task.Run(() =>
                {
                    try
                    {
                        if (File.Exists(dst)) File.Delete(dst);
                        using (var fs = new FileStream(dst, FileMode.CreateNew))
                        using (var za = new ZipArchive(fs, ZipArchiveMode.Create))
                            AddDir(za, src, ExeName);
                    }
                    catch (System.Exception e) { _zipErr = e.Message; }
                });
                Log("zip 만드는 중: " + _zipOut);
                return;
            }
            Status($"pack: zip 만드는 중 ({EditorApplication.timeSinceStartup - _t0:F0} s)");
            if (!_zip.IsCompleted) return;
            _zip = null;
            if (_zipErr != null) { Fail("zip 실패: " + _zipErr); return; }
            Log($"zip 완료: {_zipOut} ({new FileInfo(_zipOut).Length / 1048576f:F0} MB)");
            Advance("pack");
        }

        static void AddDir(ZipArchive za, string dir, string prefix)
        {
            foreach (var f in Directory.GetFiles(dir))
            {
                var e = za.CreateEntry(prefix + "/" + Path.GetFileName(f), System.IO.Compression.CompressionLevel.Optimal);
                using (var es = e.Open())
                using (var fsr = File.OpenRead(f)) fsr.CopyTo(es);
            }
            foreach (var d in Directory.GetDirectories(dir))
            {
                string n = Path.GetFileName(d);
                if (ZipSkipDirs.Contains(n) || n.EndsWith("_DoNotShip") || n.EndsWith("_ButDontShipItWithYourGame")) continue;
                AddDir(za, d, prefix + "/" + n);
            }
        }

        // ---------------------------------------------------------------- 도우미
        static string ReadStatus(string p)
        {
            try { return File.Exists(p) ? File.ReadAllText(p).Trim() : ""; } catch { return ""; }
        }

        static void Log(string s)
        {
            _log.AppendLine($"[{System.DateTime.Now:HH:mm:ss}] {s}");
            UnityEngine.Debug.Log("[Build] " + s);
        }

        static void Status(string s)
        {
            Directory.CreateDirectory(TestDir);
            File.WriteAllText(StatusPath, $"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss} {s}\n\n{_log}", new UTF8Encoding(false));
        }

        static Dictionary<string, string> Parse(string text)
        {
            var d = new Dictionary<string, string>();
            foreach (var raw in text.Split('\n'))
            {
                string line = raw.Trim();
                int c = line.IndexOf('#'); if (c >= 0) line = line.Substring(0, c).Trim();
                int i = line.IndexOf('=');
                if (i > 0) d[line.Substring(0, i).Trim()] = line.Substring(i + 1).Trim();
            }
            return d;
        }
    }
}
