// AiInstaller.cs
// 실행 파일에서도 쓰는 AI 엔진 설치기. 메인 메뉴의 [AI 엔진 설치] 버튼이 부른다.
//   Windows: powershell 로 Tools/jberry_ai/setup_ai.ps1 실행
//     · PC 에 Python 3.10~3.13 이 있으면 Tools/jberry_ai/.venv 를 만들고 패키지 설치
//     · 없으면 휴대용 Python(공식 embeddable zip)을 Tools/jberry_ai/python 에 받아서 설치
//   진행 상황: SimLogs/ai/setup_status.txt (running: … / ok / fail: …), 전체 출력: SimLogs/ai/setup_log.txt
using System.Diagnostics;
using System.IO;
using System.Text;

namespace CapstoneSim
{
    public static class AiInstaller
    {
        static Process _proc;

        public static bool Running
        {
            get { try { return _proc != null && !_proc.HasExited; } catch { return false; } }
        }

        public static string Script => Path.Combine(SimPaths.AiDir, "setup_ai.ps1");

        /// <summary>설치 시작. portable=true 면 PC 의 Python 이 있어도 휴대용 Python 으로 설치</summary>
        public static bool Start(bool portable = false)
        {
            if (Running) return true;
            Directory.CreateDirectory(SimPaths.AiLogDir);
            string status = Path.Combine(SimPaths.AiLogDir, "setup_status.txt");
            if (!File.Exists(Script))
            {
                File.WriteAllText(status, "fail: setup_ai.ps1 이 없습니다 (Tools/jberry_ai)\n", new UTF8Encoding(false));
                return false;
            }
            try
            {
                File.WriteAllText(status, "running: 설치 준비\n", new UTF8Encoding(false));
                var psi = new ProcessStartInfo("powershell.exe",
                    $"-NoProfile -ExecutionPolicy Bypass -File \"{Script}\"" + (portable ? " -Portable" : ""))
                {
                    UseShellExecute = false, CreateNoWindow = true, WorkingDirectory = SimPaths.AiDir,
                };
                _proc = Process.Start(psi);
                return true;
            }
            catch (System.Exception e)
            {
                File.WriteAllText(status, $"fail: powershell 실행 실패 ({e.Message})\n", new UTF8Encoding(false));
                return false;
            }
        }

        /// <summary>setup_log.txt 마지막 줄 (진행 표시용, 짧게)</summary>
        public static string LastLogLine()
        {
            try
            {
                string p = Path.Combine(SimPaths.AiLogDir, "setup_log.txt");
                if (!File.Exists(p)) return "";
                using (var fs = new FileStream(p, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    long n = System.Math.Min(4096, fs.Length);
                    fs.Seek(-n, SeekOrigin.End);
                    var buf = new byte[n];
                    fs.Read(buf, 0, (int)n);
                    var lines = Encoding.UTF8.GetString(buf).Replace("\r", "").Split('\n');
                    for (int i = lines.Length - 1; i >= 0; i--)
                    {
                        string l = lines[i].Trim();
                        if (l.Length == 0) continue;
                        return l.Length > 70 ? l.Substring(0, 70) + "…" : l;
                    }
                }
            }
            catch { }
            return "";
        }
    }
}
