// SimPaths.cs
// 에디터와 실행 파일(빌드)에서 같은 규칙으로 쓰는 경로.
//   루트 = Application.dataPath 의 상위 폴더
//     에디터: 프로젝트 폴더          빌드: JberryDrivingSim.exe 가 있는 폴더
//   AI 엔진 Python: Tools/jberry_ai/python/python.exe (휴대용, 배포 zip 에 포함) → Tools/jberry_ai/.venv (각자 설치)
using System.IO;
using UnityEngine;

namespace CapstoneSim
{
    public static class SimPaths
    {
        public static string Root => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        public static string AiDir => Path.Combine(Root, "Tools", "jberry_ai");
        public static string LogDir => Path.Combine(Root, "SimLogs");
        public static string AiLogDir => Path.Combine(LogDir, "ai");
        public static string PortablePython => Path.Combine(AiDir, "python", "python.exe");
        public static string VenvPython => Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer
            ? Path.Combine(AiDir, ".venv", "Scripts", "python.exe")
            : Path.Combine(AiDir, ".venv", "bin", "python");

        /// <summary>AI 서버를 돌릴 Python. 없으면 null</summary>
        public static string AiPython
        {
            get
            {
                if (File.Exists(PortablePython)) return PortablePython;
                if (File.Exists(VenvPython)) return VenvPython;
                return null;
            }
        }

        /// <summary>설치 결과 (setup_status.txt 첫 줄: ok / running… / fail: …). 파일이 없으면 ""</summary>
        public static string SetupStatus
        {
            get
            {
                try
                {
                    string p = Path.Combine(AiLogDir, "setup_status.txt");
                    return File.Exists(p) ? File.ReadAllText(p).Trim() : "";
                }
                catch { return ""; }
            }
        }

        /// <summary>AI 엔진을 쓸 수 있는지: Python 이 있고, 설치가 실패·진행 중이 아님</summary>
        public static bool AiReady
        {
            get
            {
                if (AiPython == null) return false;
                string s = SetupStatus;
                return s == "" || s.StartsWith("ok");
            }
        }
    }
}
