// LaunchProbe.cs
// 실행 파일이 켜질 때 받은 실행 인자·경로를 SimLogs/last_launch.txt 에 남긴다 (문제가 생겼을 때 확인용, 에디터에서는 안 씀).
using System.IO;
using System.Text;
using UnityEngine;

namespace CapstoneSim
{
    public static class LaunchProbe
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void Write()
        {
            if (Application.isEditor) return;
            try
            {
                Directory.CreateDirectory(SimPaths.LogDir);
                var sb = new StringBuilder();
                sb.AppendLine($"time={System.DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                sb.AppendLine($"version={Application.version} unity={Application.unityVersion}");
                sb.AppendLine($"args={string.Join(" | ", System.Environment.GetCommandLineArgs())}");
                sb.AppendLine($"dataPath={Application.dataPath}");
                sb.AppendLine($"root={SimPaths.Root}");
                sb.AppendLine($"consoleLogPath={Application.consoleLogPath}");
                sb.AppendLine($"autotest_flag={File.Exists(AutoTestRunner.FlagPath)} ({AutoTestRunner.FlagPath})");
                sb.AppendLine($"ai_python={SimPaths.AiPython ?? "(없음)"}");
                File.WriteAllText(Path.Combine(SimPaths.LogDir, "last_launch.txt"), sb.ToString(), new UTF8Encoding(false));
            }
            catch { }
        }
    }
}
