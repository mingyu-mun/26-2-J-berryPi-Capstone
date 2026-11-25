// BioReplay.cs
// 실측 ECG/PPG 기록을 '센서처럼' 실제 시간에 맞춰 흘려보낸다 (Replay).
//   입력: 저장소 docs/data-handoff.md 의 세션 폴더 형식
//         <폴더>/meta.json  (fs.ecg, fs.ppg …)
//         <폴더>/ecg.csv, ppg.csv  (value 열 하나, 각자 원래 샘플링 주기 — 예: ECG ≈435 Hz, PPG ≈100 Hz)
//         <폴더>/labels.csv  (선택: start_s,end_s,label — 화면에 '참고 라벨'로만 표시, 모델 입력 아님)
//   기본 폴더: <프로젝트>/Assets/StreamingAssets/Replays/<기록> (메인 메뉴에서 선택), 없으면 SimLogs/bio/<기록>
// 두 채널은 각자의 주기로 PC 벽시계(Stopwatch)에 맞춰 나온다. AiLink 가 Drain 으로 꺼내 추론 서버에 보낸다.
// 원신호 값은 바꾸지 않는다 (필터·전처리는 서버가 저장소 코드로 한다).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

namespace CapstoneSim
{
    public class BioReplay : BioSource
    {
        [Tooltip("세션 폴더 (meta.json + ecg.csv + ppg.csv). 비우면 리플레이 라이브러리의 첫 기록 (StreamingAssets/Replays → SimLogs/bio)")]
        public string sessionDir = "";
        public bool Finished { get; private set; }
        public override string Origin => "recorded";
        float _duration = -1f, _fileTime;
        string _segLabel = "";
        public override float DurationSec => _duration;
        /// <summary>현재 재생 위치 (기록 기준 초)</summary>
        public override float FileTimeSec => _fileTime;
        /// <summary>Play 시작 후 흘려보낸 시간 (세션 시각, 초)</summary>
        public override float ElapsedSec => _sw == null ? 0f : (float)_sw.Elapsed.TotalSeconds;
        public override string SegmentLabel => _segLabel;
        /// <summary>labels.csv 구간 (메뉴 타임라인·화면 표시용)</summary>
        public System.Collections.Generic.List<(float start, float end, string label)> Segments => _segments;

        float[] _ecg, _ppg;
        readonly List<(float start, float end, string label)> _segments = new List<(float, float, string)>();
        Stopwatch _sw;
        long _e0, _p0, _eIdx, _pIdx;            // 시작 표본 번호, 다음에 내보낼 표본 번호
        readonly List<float> _outE = new List<float>(4096), _outP = new List<float>(1024);
        bool _begun;

        public override void Begin() { if (_begun) return; _begun = true; Load(); }
        void Start() { Begin(); }

        public string ResolveDir()
        {
            if (!string.IsNullOrWhiteSpace(sessionDir)) return System.IO.Path.GetFullPath(sessionDir);
            return ReplayLibrary.DefaultReplayDir() ?? "";
        }

        static float[] ReadValues(string path)
        {
            var list = new List<float>(1 << 20);
            var inv = CultureInfo.InvariantCulture;
            int col = 0;
            bool header = true;
            foreach (var line in File.ReadLines(path))
            {
                if (line.Length == 0) continue;
                var c = line.Split(',');
                if (header)
                {
                    header = false;
                    int k = Array.FindIndex(c, s => s.Trim().Trim('"').ToLowerInvariant() == "value");
                    if (k >= 0) { col = k; continue; }
                    if (!float.TryParse(c[0], NumberStyles.Float, inv, out _)) continue;
                }
                if (c.Length > col && float.TryParse(c[col], NumberStyles.Float, inv, out var v)) list.Add(v);
                else list.Add(float.NaN);           // 결측은 결측 그대로
            }
            return list.ToArray();
        }

        static double JsonNumber(string json, string key)
        {
            var m = Regex.Match(json, "\"" + key + "\"\\s*:\\s*([-0-9.eE+]+)");
            return m.Success ? double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) : double.NaN;
        }

        void Load()
        {
            string dir = ResolveDir();
            SessionPath = dir;
            SessionName = Path.GetFileName(dir);
            string meta = Path.Combine(dir, "meta.json");
            if (!File.Exists(meta)) { Status = $"생체 기록 없음: {dir}"; UnityEngine.Debug.LogWarning("[BioReplay] " + Status); return; }
            try
            {
                string mj = File.ReadAllText(meta);
                // "fs": {"ecg": 435.3, "ppg": 100.6}
                var fsm = Regex.Match(mj, "\"fs\"\\s*:\\s*\\{([^}]*)\\}");
                FsEcg = fsm.Success ? JsonNumber("{" + fsm.Groups[1].Value + "}", "ecg") : double.NaN;
                FsPpg = fsm.Success ? JsonNumber("{" + fsm.Groups[1].Value + "}", "ppg") : double.NaN;
                if (!(FsEcg > 1) || !(FsPpg > 1)) { Status = "meta.json 에 fs.ecg / fs.ppg 없음"; return; }
                var sw = Stopwatch.StartNew();
                _ecg = ReadValues(Path.Combine(dir, "ecg.csv"));
                _ppg = ReadValues(Path.Combine(dir, "ppg.csv"));
                UnityEngine.Debug.Log($"[BioReplay] CSV 읽기 {sw.ElapsedMilliseconds} ms");
            }
            catch (Exception e) { Status = "생체 기록 읽기 실패: " + e.Message; UnityEngine.Debug.LogError("[BioReplay] " + Status); return; }

            _duration = (float)Math.Min(_ecg.Length / FsEcg, _ppg.Length / FsPpg);
            var inv = CultureInfo.InvariantCulture;
            string lab = Path.Combine(dir, "labels.csv");
            if (File.Exists(lab))
                foreach (var line in File.ReadLines(lab))
                {
                    var c = line.Split(',');
                    if (c.Length >= 3 && float.TryParse(c[0], NumberStyles.Float, inv, out var s0) && float.TryParse(c[1], NumberStyles.Float, inv, out var s1))
                        _segments.Add((s0, s1, c[2]));
                }

            startOffsetSec = Mathf.Clamp(startOffsetSec, 0f, Mathf.Max(0f, DurationSec - 1f));
            _e0 = _eIdx = (long)(startOffsetSec * FsEcg);
            _p0 = _pIdx = (long)(startOffsetSec * FsPpg);
            _sw = Stopwatch.StartNew();
            Loaded = true;
            Status = $"{SessionName} · ECG {FsEcg:F1} Hz · PPG {FsPpg:F1} Hz · {DurationSec / 60f:F1}분";
            UnityEngine.Debug.Log($"[BioReplay] 로드: {dir} (ECG {_ecg.Length}, PPG {_ppg.Length}), 시작 {startOffsetSec:F0}s");
        }

        void Update()
        {
            if (!Loaded || Finished) return;
            double t = _sw.Elapsed.TotalSeconds;
            long eT = Math.Min(_e0 + (long)(t * FsEcg), _ecg.Length);
            long pT = Math.Min(_p0 + (long)(t * FsPpg), _ppg.Length);
            for (; _eIdx < eT; _eIdx++) _outE.Add(_ecg[_eIdx]);
            for (; _pIdx < pT; _pIdx++) _outP.Add(_ppg[_pIdx]);
            if (_eIdx >= _ecg.Length || _pIdx >= _ppg.Length) { Finished = true; Status += " · 재생 끝"; }
            _fileTime = (float)(_eIdx / FsEcg);
            _segLabel = "";
            foreach (var s in _segments) if (_fileTime >= s.start && _fileTime < s.end) { _segLabel = s.label; break; }
        }

        /// <summary>쌓인 표본을 꺼냄 (각 채널 원래 주기). maxEcg 로 한 번에 꺼낼 양을 제한</summary>
        public override bool Drain(out float[] ecg, out float[] ppg, int maxEcg = 2200)
        {
            ecg = ppg = null;
            if (_outE.Count == 0 && _outP.Count == 0) return false;
            int ne = Math.Min(_outE.Count, maxEcg);
            int np = Math.Min(_outP.Count, Math.Max(1, (int)Math.Ceiling(ne * FsPpg / FsEcg) + 1));
            ecg = _outE.GetRange(0, ne).ToArray(); _outE.RemoveRange(0, ne);
            ppg = _outP.GetRange(0, np).ToArray(); _outP.RemoveRange(0, np);
            return true;
        }

        public int Pending => _outE.Count;
    }
}
