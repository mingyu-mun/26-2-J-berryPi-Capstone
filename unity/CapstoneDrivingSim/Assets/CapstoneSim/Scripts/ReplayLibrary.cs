// ReplayLibrary.cs
// 리플레이·기준(baseline) 기록 목록. 세션 폴더(meta.json + ecg.csv + ppg.csv [+ labels.csv]) 를 찾는다.
//   1) <프로젝트>/Assets/StreamingAssets/Replays/<이름>/   ← 유니티 안에 넣는 기록 (빌드에도 포함)
//   2) <프로젝트>/SimLogs/bio/<이름>/                     ← convert_recording.py 기본 출력
// meta.json 의 "role": "baseline" 이면 기준 측정 파일로 분류한다 (convert_recording.py --role baseline).
// 원신호는 개인 생체 데이터이므로 두 폴더 모두 .gitignore 대상이다.
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

namespace CapstoneSim
{
    public class ReplayEntry
    {
        public string name, dir, source, role, subject;
        public double fsEcg, fsPpg;
        public float durationSec;
        public List<(float s, float e, string label)> segments = new List<(float, float, string)>();
        public bool IsBaseline => role == "baseline";
        public bool HasLabels => segments.Count > 0;
    }

    public static class ReplayLibrary
    {
        public static string StreamingDir => Path.Combine(Application.streamingAssetsPath, "Replays");
        public static string SimLogsDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "SimLogs", "bio"));

        static List<ReplayEntry> _cache;
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() => _cache = null;

        public static List<ReplayEntry> All(bool refresh = false)
        {
            if (_cache != null && !refresh) return _cache;
            var list = new List<ReplayEntry>();
            var seen = new HashSet<string>();
            var roots = new[] { StreamingDir, SimLogsDir };
            var srcs = new[] { "StreamingAssets", "SimLogs" };
            for (int k = 0; k < roots.Length; k++)
            {
                string root = roots[k], src = srcs[k];
                if (!Directory.Exists(root)) continue;
                foreach (var d in Directory.GetDirectories(root))
                {
                    var e = Read(d, src);
                    if (e != null && seen.Add(e.name)) list.Add(e);
                }
            }
            list.Sort((a, b) => a.IsBaseline != b.IsBaseline ? (a.IsBaseline ? 1 : -1) : string.CompareOrdinal(a.name, b.name));
            _cache = list;
            return list;
        }

        public static string DefaultReplayDir()
        {
            foreach (var e in All()) if (!e.IsBaseline) return e.dir;
            return null;
        }

        public static ReplayEntry Find(string nameOrDir)
        {
            if (string.IsNullOrEmpty(nameOrDir)) return null;
            foreach (var e in All()) if (e.name == nameOrDir || Path.GetFullPath(e.dir) == SafeFull(nameOrDir)) return e;
            return null;
        }
        static string SafeFull(string p) { try { return Path.GetFullPath(p); } catch { return p; } }

        static ReplayEntry Read(string dir, string src)
        {
            string meta = Path.Combine(dir, "meta.json");
            string ecg = Path.Combine(dir, "ecg.csv");
            if (!File.Exists(meta) || !File.Exists(ecg) || !File.Exists(Path.Combine(dir, "ppg.csv"))) return null;
            try
            {
                string mj = File.ReadAllText(meta);
                var e = new ReplayEntry { name = Path.GetFileName(dir), dir = dir, source = src };
                var fsm = Regex.Match(mj, "\"fs\"\\s*:\\s*\\{([^}]*)\\}");
                if (fsm.Success) { e.fsEcg = Num(fsm.Groups[1].Value, "ecg"); e.fsPpg = Num(fsm.Groups[1].Value, "ppg"); }
                e.role = Str(mj, "role") ?? "";
                e.subject = Str(mj, "subject_id") ?? "";
                double dur = Num(mj, "duration_s");
                if (double.IsNaN(dur) && e.fsEcg > 0) dur = EstimateRows(ecg) / e.fsEcg;     // 예전 변환본: 줄 수로 추정
                e.durationSec = double.IsNaN(dur) ? -1f : (float)dur;
                string lab = Path.Combine(dir, "labels.csv");
                if (File.Exists(lab))
                    foreach (var line in File.ReadLines(lab))
                    {
                        var c = line.Split(',');
                        if (c.Length >= 3 && float.TryParse(c[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var s0)
                            && float.TryParse(c[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var s1))
                            e.segments.Add((s0, s1, c[2]));
                    }
                return e;
            }
            catch { return null; }
        }

        static double Num(string json, string key)
        {
            var m = Regex.Match(json, "\"" + key + "\"\\s*:\\s*([-0-9.eE+]+)");
            return m.Success ? double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) : double.NaN;
        }
        static string Str(string json, string key)
        {
            var m = Regex.Match(json, "\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
            return m.Success ? m.Groups[1].Value : null;
        }
        /// <summary>파일 앞 64 KB 의 줄 길이로 전체 줄 수 추정 (888k 줄을 다 읽지 않으려고)</summary>
        static double EstimateRows(string path)
        {
            var fi = new FileInfo(path);
            using (var fs = File.OpenRead(path))
            {
                var buf = new byte[(int)System.Math.Min(65536, fi.Length)];
                int n = fs.Read(buf, 0, buf.Length), lines = 0;
                for (int i = 0; i < n; i++) if (buf[i] == (byte)'\n') lines++;
                return lines <= 1 ? double.NaN : fi.Length / (n / (double)lines) - 1;
            }
        }

        public static void OpenFolder()
        {
            Directory.CreateDirectory(StreamingDir);
            Application.OpenURL("file:///" + StreamingDir.Replace('\\', '/'));
        }
    }
}
