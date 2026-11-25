// AiEnvInstaller.cs  (Editor 전용)
// 메뉴 [Capstone > AI 추론 환경 설치 (Python venv)] → Tools/jberry_ai/setup_env.py 실행
//   Tools/jberry_ai/.venv 를 만들고 numpy·scipy·pandas·scikit-learn·pyyaml·safetensors·torch(CPU) 를 설치한 뒤
//   selftest.py 로 저장소 모델을 실측 기록에 한 번 돌려 본다. (처음엔 내려받기 때문에 수 분 걸림)
//   진행: SimLogs/ai/setup_log.txt, 결과: SimLogs/ai/setup_status.txt (ok / fail: …)
// 외부(Claude)에서 SimLogs/ai/setup_request.txt 를 만들어도 같은 설치가 시작된다.
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace CapstoneSim.EditorTools
{
    [InitializeOnLoad]
    public static class AiEnvInstaller
    {
        static string Root => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        static string AiDir => Path.Combine(Root, "SimLogs", "ai");
        static string RequestPath => Path.Combine(AiDir, "setup_request.txt");
        static string Script => Path.Combine(Root, "Tools", "jberry_ai", "setup_env.py");
        static double _next;

        static AiEnvInstaller()
        {
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
        }

        [MenuItem("Capstone/AI 추론 환경 설치 (Python venv)", priority = 30)]
        static void InstallFromMenu()
        {
            if (Start()) EditorUtility.DisplayDialog("Capstone",
                "AI 추론 환경 설치를 시작했습니다 (백그라운드).\n처음에는 torch 내려받기로 수 분 걸립니다.\n\n진행: SimLogs/ai/setup_log.txt\n결과: SimLogs/ai/setup_status.txt", "확인");
        }

        static void Poll()
        {
            if (EditorApplication.timeSinceStartup < _next) return;
            _next = EditorApplication.timeSinceStartup + 2.0;
            if (!File.Exists(RequestPath)) return;
            try { File.Delete(RequestPath); } catch { return; }
            Start();
        }

        static bool Start()
        {
            Directory.CreateDirectory(AiDir);
            if (!File.Exists(Script)) { UnityEngine.Debug.LogError("[AiEnv] setup_env.py 없음: " + Script); return false; }
            foreach (var exe in new[] { "python", "py" })
            {
                try
                {
                    var psi = new ProcessStartInfo(exe, $"-u \"{Script}\"")
                    {
                        UseShellExecute = false, CreateNoWindow = true,
                        WorkingDirectory = Path.GetDirectoryName(Script),
                    };
                    psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                    psi.EnvironmentVariables["PYTHONUTF8"] = "1";
                    Process.Start(psi);
                    File.WriteAllText(Path.Combine(AiDir, "setup_launcher.txt"), $"{System.DateTime.Now:yyyy-MM-dd HH:mm:ss} started with '{exe}'\n", new UTF8Encoding(false));
                    UnityEngine.Debug.Log($"[AiEnv] 설치 시작 ({exe}) — SimLogs/ai/setup_log.txt");
                    return true;
                }
                catch (System.Exception e) { UnityEngine.Debug.LogWarning($"[AiEnv] '{exe}' 실행 실패: {e.Message}"); }
            }
            File.WriteAllText(Path.Combine(AiDir, "setup_status.txt"), "fail: python / py 를 찾을 수 없음 — python.org 에서 Python 3.12 설치\n", new UTF8Encoding(false));
            UnityEngine.Debug.LogError("[AiEnv] Python 을 찾을 수 없음");
            return false;
        }
    }
}
