// SimPanels.cs
// 주행 화면 정보 패널 (웹보드 규약 v1.2 표시 규칙: 상태는 상태 머신 v2 출력만, 값이 없으면 '–')
//   왼쪽 위   : 운전자 상태 — level(-1 측정 불가 / 0 정상 / 1 주의 / 2 졸음), 졸음 확률·임계값, 판정 이력
//   오른쪽 위 : 세션 — 데이터 출처(생체 기록 재생 / 실측, CAN 합성), 재생 위치 또는 수신 상태
//   위 가운데 : 경고 배너 (주의 / 졸음 / 졸음 확정) + 경고음
//   졸음 확정 = 상태 머신(생체)이 '졸음'이고 차량 경고(차선 이탈·부주의)가 함께 있을 때 (서버 confirmed, 2026-11-26)
//   아래      : ECG · PPG 파형 (최근 6초, 공통 전처리). 개인 기준 파형을 같은 배율로 겹쳐 그림
using UnityEngine;

namespace CapstoneSim
{
    /// <summary>주행 화면 배치 (SimHUD 와 함께 씀)</summary>
    public static class HudLayout
    {
        public const float M = 28f;                 // 화면 가장자리 여백
        public const float ClusterW = 470f, ClusterH = 176f;
        public const float HelpH = 30f;
        public const float DockH = 200f;
        public const float StateW = 450f, SessionW = 400f;
        public static float HelpY => SimTheme.VH - 12f - HelpH;
        public static float DockY => HelpY - 12f - DockH;
        public static float DockX => M + ClusterW + 16f;
        public static float ClusterY => SimTheme.VH - M - ClusterH;
    }

    public class SimPanels : MonoBehaviour
    {
        public BioSource bio;
        public AiLink ai;
        public SimBootstrap sim;
        public bool show = true;
        [Tooltip("졸음 확정 시 경고음")] public bool beep = true;

        const int GW = 720, GH = 130;
        Texture2D _texE, _texP;
        Color32[] _pix;
        int _lastLevel = -9;
        float[] _lastE;
        AudioSource _audio;
        AudioClip _beepLo, _beepHi;

        static readonly Color32 Bg = new Color32(0, 0, 0, 0), Grid = new Color32(255, 255, 255, 16), Mid = new Color32(255, 255, 255, 26);
        static readonly Color32 BaseC32 = new Color32(255, 255, 255, 105);
        static readonly Color EcgC = new Color(0.40f, 0.95f, 0.55f), PpgC = new Color(1f, 0.56f, 0.32f);

        void Start()
        {
            _texE = NewTex(); _texP = NewTex();
            _pix = new Color32[GW * GH];
            _audio = gameObject.AddComponent<AudioSource>();
            _audio.playOnAwake = false;
            _beepLo = Tone(660f, 0.25f); _beepHi = Tone(1100f, 0.12f);
            Clear(_texE); Clear(_texP);
        }

        static Texture2D NewTex() => new Texture2D(GW, GH, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };

        static AudioClip Tone(float hz, float len)
        {
            int sr = 44100, n = (int)(sr * len);
            var d = new float[n];
            for (int i = 0; i < n; i++)
            {
                float env = Mathf.Min(1f, i / 400f) * Mathf.Min(1f, (n - i) / 800f);
                d[i] = 0.25f * env * Mathf.Sin(2 * Mathf.PI * hz * i / sr);
            }
            var c = AudioClip.Create("beep" + hz, n, 1, sr, false);
            c.SetData(d, 0);
            return c;
        }

        void Update()
        {
            if (SimInput.HudTogglePressed()) show = !show;
            var r = Live ? ai.Latest : null;

            int lv = r != null && r.has_out ? r.level : -9;
            if (lv == 2 && r.confirmed) lv = 3;              // 졸음 확정
            if (beep && _audio && lv != _lastLevel)
            {
                if (lv == 1) _audio.PlayOneShot(_beepLo);
                if (lv == 2) StartCoroutine(DoubleBeep());
                if (lv == 3) StartCoroutine(TripleBeep());
            }
            _lastLevel = lv;

            if (r != null && r.ecg_w != null && r.ecg_w.Length > 1 && r.ecg_w != _lastE)
            {
                _lastE = r.ecg_w;
                Plot(_texE, r.ecg_w, r.ecg_b, r.wave_fs, EcgC);
                Plot(_texP, r.ppg_w, r.ppg_b, r.wave_fs, PpgC);
            }
        }

        bool Live => ai != null && ai.Connected && ai.Latest != null && ai.SessionAccepted;

        System.Collections.IEnumerator DoubleBeep()
        {
            for (int i = 0; i < 2; i++) { _audio.PlayOneShot(_beepLo); yield return new WaitForSecondsRealtime(0.3f); }
        }

        System.Collections.IEnumerator TripleBeep()
        {
            for (int i = 0; i < 3; i++) { _audio.PlayOneShot(_beepHi); yield return new WaitForSecondsRealtime(0.18f); }
        }

        void Clear(Texture2D tex)
        {
            for (int i = 0; i < _pix.Length; i++) _pix[i] = Bg;
            tex.SetPixels32(_pix); tex.Apply(false);
        }

        /// <summary>현재 파형과 기준 파형을 같은 세로 범위로 그림. 기준(반투명 흰색)을 먼저, 현재(채널 색)를 위에.
        /// 표본을 GW 열로 나눠 열마다 최소~최대를 세로선으로 그림 (min-max 다운샘플링)</summary>
        void Plot(Texture2D tex, float[] cur, float[] bas, float fs, Color col)
        {
            for (int i = 0; i < _pix.Length; i++) _pix[i] = Bg;
            int n = cur == null ? 0 : cur.Length;
            if (fs > 0 && n > 0)
                for (float s = fs; s < n; s += fs)
                {
                    int x = (int)(s * GW / n);
                    for (int y = 0; y < GH; y++) _pix[y * GW + x] = Grid;
                }
            for (int x = 0; x < GW; x++) _pix[(GH / 2) * GW + x] = Mid;
            if (n < 2) { tex.SetPixels32(_pix); tex.Apply(false); return; }
            bool hasB = bas != null && bas.Length == n;
            // 세로 범위: 기준 파형이 있으면 기준 범위 + 위아래 35% 여유 (진폭 비교 기준을 고정, 잡음 스파이크는 잘림)
            //            없으면 현재 파형의 1~99 백분위
            float mn, mx;
            if (hasB)
            {
                mn = float.MaxValue; mx = float.MinValue;
                for (int i = 0; i < n; i++) { mn = Mathf.Min(mn, bas[i]); mx = Mathf.Max(mx, bas[i]); }
                float e = (mx - mn) * 0.35f; mn -= e; mx += e;
            }
            else
            {
                var srt = (float[])cur.Clone(); System.Array.Sort(srt);
                mn = srt[(int)(n * 0.01f)]; mx = srt[Mathf.Min(n - 1, (int)(n * 0.99f))];
            }
            float pad = (mx - mn) * 0.06f + 1e-3f, lo = mn - pad, span = Mathf.Max(1e-3f, mx - mn + 2 * pad);
            if (hasB) Trace(bas, n, lo, span, BaseC32, 3);
            Trace(cur, n, lo, span, col, 2);
            tex.SetPixels32(_pix);
            tex.Apply(false);
        }

        void Trace(float[] buf, int n, float lo, float span, Color32 col, int thick)
        {
            int prevY = -1;
            for (int x = 0; x < GW; x++)
            {
                int a = (int)((long)x * n / GW), b = Mathf.Min(n, Mathf.Max(a + 1, (int)((long)(x + 1) * n / GW)));
                float cmin = float.MaxValue, cmax = float.MinValue;
                for (int i = a; i < b; i++) { if (buf[i] < cmin) cmin = buf[i]; if (buf[i] > cmax) cmax = buf[i]; }
                int y0 = Mathf.Clamp((int)((cmin - lo) / span * (GH - 1)), 0, GH - 1);
                int y1 = Mathf.Clamp((int)((cmax - lo) / span * (GH - 1)), 0, GH - 1);
                if (prevY >= 0) { y0 = Mathf.Min(y0, prevY); y1 = Mathf.Max(y1, prevY); }
                for (int y = y0; y <= y1; y++)
                    for (int t = 0; t < thick && x + t < GW; t++) _pix[y * GW + x + t] = col;
                prevY = Mathf.Clamp((int)((buf[b - 1] - lo) / span * (GH - 1)), 0, GH - 1);
            }
        }

        static string V(float v, string fmt) => v < 0 ? "–" : v.ToString(fmt);
        static string MmSs(float s) => SimTheme.MmSs(s);

        void OnGUI()
        {
            if (!show) return;
            var m = SimTheme.Begin();
            DrawState();
            DrawSession();
            DrawBio();
            DrawBanner();
            SimTheme.End(m);
        }

        // ---------------------------------------------------------------- 아래: ECG · PPG
        void DrawBio()
        {
            var r = Live ? ai.Latest : null;
            float x = HudLayout.DockX, y = HudLayout.DockY, w = SimTheme.VW - HudLayout.M - x;
            float cw = (w - 16f) * 0.5f;
            bool has = r != null && r.has_out;
            WaveCard(new Rect(x, y, cw, HudLayout.DockH), "ECG", EcgC, _texE, r?.ecg_w, r?.ecg_b,
                     has ? r.hr_ecg : -1, r != null ? r.ecg_hrb : -1, has ? r.q_ecg : -1);
            WaveCard(new Rect(x + cw + 16f, y, cw, HudLayout.DockH), "PPG", PpgC, _texP, r?.ppg_w, r?.ppg_b,
                     has ? r.hr_ppg : -1, r != null ? r.ppg_hrb : -1, has ? r.q_ppg : -1);
        }

        void WaveCard(Rect R, string name, Color col, Texture2D tex, float[] wave, float[] bas, float hr, float hrBase, float q)
        {
            SimTheme.Panel(R);
            float x = R.x + 16, y = R.y + 14;
            SimTheme.Plate(new Rect(x, y, 78, 34), name, col, 20);
            // 심박: 현재(크게) · 기준
            float hx = x + 94;
            string hs = V(hr, "F0");
            SimTheme.Display(new Rect(hx, y - 4, 120, 40), hs, 32, SimTheme.Text);
            hx += SimTheme.DisplayWidth(hs, 32) + 6;
            SimTheme.Label(new Rect(hx, y + 12, 50, 20), "bpm", 13, SimTheme.Sub, true);
            hx += 46;
            if (hrBase > 0)
            {
                string d = hr > 0 ? $"  ({(hr - hrBase >= 0 ? "+" : "")}{hr - hrBase:F0})" : "";
                SimTheme.Label(new Rect(hx, y + 10, 200, 22), $"기준 {hrBase:F0}{d}", 15, SimTheme.Sub, true);
            }
            // 범례
            bool hasB = bas != null && bas.Length > 1;
            float lx = R.xMax - 16;
            if (q >= 0) { string qs = $"이상 간격 {q * 100f:F0}%"; float qw = SimTheme.TextWidth(qs, 13); lx -= qw; SimTheme.Label(new Rect(lx, y + 8, qw + 4, 20), qs, 13, q > 0.2f ? SimTheme.Amber : SimTheme.Dim); lx -= 18; }
            LegendItem(ref lx, y + 18, "현재", col);
            if (hasB) LegendItem(ref lx, y + 18, "기준", new Color(1, 1, 1, 0.6f));

            var g = new Rect(R.x + 16, R.y + 60, R.width - 32, R.height - 74);
            SimTheme.Fill(g, new Color(0, 0, 0, 0.5f), 6f);
            GUI.DrawTexture(new Rect(g.x + 4, g.y + 4, g.width - 8, g.height - 8), tex);
            SimTheme.Outline(g, SimTheme.A(SimTheme.Ink, 0.9f), 1.5f, 6f);
            if (wave == null || wave.Length < 2)
                SimTheme.Label(g, Live ? "파형 대기" : "AI 서버에 연결되면 표시합니다", 14, SimTheme.Dim, false, TextAnchor.MiddleCenter);
            else
                SimTheme.Label(new Rect(g.x + 10, g.yMax - 22, 200, 18), "6초", 11, SimTheme.Dim);
        }

        static void LegendItem(ref float rx, float cy, string text, Color c)
        {
            float tw = SimTheme.TextWidth(text, 13, true);
            rx -= tw;
            SimTheme.Label(new Rect(rx, cy - 10, tw + 4, 20), text, 13, SimTheme.Sub, true);
            rx -= 24;
            SimTheme.Fill(new Rect(rx, cy - 1.5f, 18, 3), c, 1f);
            rx -= 14;
        }

        // ---------------------------------------------------------------- 왼쪽 위: 운전자 상태
        void DrawState()
        {
            const float PW = HudLayout.StateW;
            var lr = ai != null && Live ? ai.Latest : null;
            bool warm = lr != null && lr.base_source != null && lr.base_source.StartsWith("file") && lr.feat_ready >= 0 && lr.feat_ready < 1f;
            float ph = lr == null ? 170 : (lr.has_out ? 386 : 334) + (warm ? 22 : 0);
            var P = new Rect(HudLayout.M, HudLayout.M, PW, ph);
            SimTheme.Panel(P);
            float x = P.x + 18, y = P.y + 16, w = PW - 36;
            SimTheme.Plate(new Rect(x, y, 176, 32), "DRIVER STATE", SimTheme.Accent, 16);
            SimTheme.Label(new Rect(x, y, w, 32), "S1-EP · 상태 머신 v2", 12, SimTheme.Dim, false, TextAnchor.MiddleRight);
            if (ai == null) return;
            var r = ai.Latest;
            y += 48;
            if (!Live)
            {
                SimTheme.Display(new Rect(x, y, w, 46), "연결 대기", 34, SimTheme.Text);
                SimTheme.Label(new Rect(x, y + 50, w, 50), ai.ServerStatus, 15, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
                return;
            }

            bool ready = r.base_status == "ready" || r.base_status == "retry_required" && r.base_a >= 0;
            if (!r.has_out)
            {
                string head, sub; float prog;
                if (r.base_status == "waiting")
                {
                    head = "착석 대기"; prog = Mathf.Clamp01(r.t_rel / Mathf.Max(1, r.base_wait));
                    sub = $"처음 {r.base_wait:F0}초는 기준에서 뺍니다   {r.t_rel:F0} / {r.base_wait:F0}초";
                }
                else if (!ready)
                {
                    head = "기준 수집"; prog = Mathf.Clamp01(r.clean_windows / (float)Mathf.Max(1, r.required_clean));
                    sub = $"깨끗한 30초 창 {r.clean_windows} / {r.required_clean}   경과 {MmSs(r.t_rel)} (최대 {MmSs(r.base_max)})";
                }
                else
                {
                    head = "첫 판정 대기"; prog = 1f;
                    sub = $"기준 {BaseText(r)}";
                }
                SimTheme.Display(new Rect(x, y, w, 52), head, 38, SimTheme.Text);
                y += 62;
                SimTheme.Bar(new Rect(x, y, w, 14), prog, SimTheme.Accent);
                SimTheme.Label(new Rect(x, y + 20, w, 40), sub, 14, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
                y += 62;
            }
            else
            {
                // 상태 배지 + 졸음 확률
                var c = SimTheme.LevelColor(r.level);
                string lvName = string.IsNullOrEmpty(r.level_name) ? "–" : r.level == 2 && r.confirmed ? "졸음 확정" : r.level_name;
                float bw = Mathf.Max(132f, SimTheme.DisplayWidth(lvName, 40) + 40f);
                SimTheme.Plate(new Rect(x, y, bw, 76), lvName, c, 40);
                float px = x + bw + 20;
                SimTheme.Label(new Rect(px, y + 2, 160, 20), "졸음 확률", 13, SimTheme.Sub, true);
                SimTheme.Display(new Rect(px, y + 20, 160, 52), V(r.p_used, "F2"), 42, SimTheme.Text);
                float cx = x + w;
                if (r.frozen) { float fw = SimTheme.TextWidth("판정 보류", 12, true) + 18; cx -= fw; SimTheme.Chip(new Rect(cx, y + 4, fw, 24), "판정 보류", SimTheme.Gray); cx -= 6; }
                if (r.candidate) { float fw = SimTheme.TextWidth("졸음 후보", 12, true) + 18; cx -= fw; SimTheme.Chip(new Rect(cx, y + 4, fw, 24), "졸음 후보", SimTheme.Amber); }
                y += 90;
                var bar = new Rect(x, y, w, 16);
                Color pcol = r.p_used >= r.th_on ? SimTheme.Red : r.p_used >= r.th_off ? SimTheme.Amber : SimTheme.Green;
                SimTheme.Bar(bar, Mathf.Max(0, r.p_used), pcol);
                Tick(bar, r.th_on, "on", 1f); Tick(bar, r.th_off, "off", 0.55f);
                y += 38;
                string conf = r.level == 2 ? (r.confirmed ? "   ·   졸음 확정 (AI 모델 졸음 + 차량 경고)" : "   ·   차량 경고 대기") : "";
                SimTheme.Label(new Rect(x, y, w, 20), $"채널 {r.channel}   EP {V(r.p_ep, "F2")}   PPG {V(r.p_ppg, "F2")}   신뢰도 {V(r.confidence, "F2")}{conf}", 13, SimTheme.Sub);
                y += 28;
            }

            // 판정 이력
            SimTheme.Label(new Rect(x, y, w, 18), "최근 15분", 13, SimTheme.Sub, true);
            var g = new Rect(x, y + 22, w, 70);
            SimTheme.Fill(g, new Color(0, 0, 0, 0.45f), 6f);
            SimTheme.Outline(g, SimTheme.A(SimTheme.Ink, 0.9f), 1.5f, 6f);
            SimTheme.Fill(new Rect(g.x, g.y + g.height * (1 - r.th_on), g.width, 1), new Color(1, 1, 1, 0.35f));
            SimTheme.Fill(new Rect(g.x, g.y + g.height * (1 - r.th_off), g.width, 1), new Color(1, 1, 1, 0.15f));
            if (r.hist_t != null && r.hist_t.Length > 0)
            {
                int n = r.hist_t.Length, maxN = 90, s0 = Mathf.Max(0, n - maxN);
                for (int i = s0; i < n; i++)
                {
                    float hx = g.x + 4 + (g.width - 8) * (i - s0 + 0.5f) / maxN, p = r.hist_p[i];
                    if (p < 0) { SimTheme.Fill(new Rect(hx - 1.5f, g.yMax - 5, 3, 3), SimTheme.Gray); continue; }
                    var d = new Rect(hx - 2.5f, g.y + 4 + (g.height - 8) * (1 - Mathf.Clamp01(p)) - 2.5f, 5, 5);
                    SimTheme.Fill(d, SimTheme.LevelColor(r.hist_lv[i]), 1.5f);
                }
            }
            y += 102;
            if (r.z_names != null && r.z_names.Length > 0)
            {
                var sb = new System.Text.StringBuilder("기준과 가장 다른 지표   ");
                for (int i = 0; i < r.z_names.Length; i++) sb.Append($"{r.z_names[i]} {(r.z_vals[i] >= 0 ? "+" : "")}{r.z_vals[i]:F1}   ");
                SimTheme.Label(new Rect(x, y, w, 20), sb.ToString(), 12, SimTheme.Sub);
                y += 22;
            }
            if (warm) { SimTheme.Label(new Rect(x, y, w, 20), $"5분 지표 채우는 중 {r.feat_ready * 100f:F0}% · 그동안 확률이 낮게 나올 수 있음", 12, SimTheme.Amber); y += 22; }

            string mode = r.mode == "exact" ? "batch 동일" : "실시간";
            string err = string.IsNullOrEmpty(r.error) ? "" : $"   ⚠ {r.error}";
            SimTheme.Label(new Rect(x, P.yMax - 30, w, 18), $"{mode} · 기준 {BaseText(r)} · 처리 {V(r.proc_ms, "F0")} ms{err}", 12, string.IsNullOrEmpty(r.error) ? SimTheme.Dim : SimTheme.Amber);
        }

        static void Tick(Rect bar, float v, string label, float a)
        {
            float tx = bar.x + bar.width * Mathf.Clamp01(v);
            SimTheme.Fill(new Rect(tx - 1.5f, bar.y - 4, 3, bar.height + 8), new Color(1, 1, 1, a));
        }

        static string BaseText(AiUi r)
        {
            string src = r.base_source != null && r.base_source.StartsWith("file:") ? "파일 " : "";
            return r.base_a >= 0 ? $"{src}{r.base_a:F0}–{r.base_b:F0}초{(r.base_forced ? " (강제)" : "")}" : r.base_status;
        }

        // ---------------------------------------------------------------- 오른쪽 위: 세션
        void DrawSession()
        {
            const float PW = HudLayout.SessionW;
            var P = new Rect(SimTheme.VW - HudLayout.M - PW, HudLayout.M, PW, 128);
            SimTheme.Panel(P);
            float x = P.x + 18, y = P.y + 16, w = PW - 36;
            bool liveSrc = bio != null && bio.IsLive;
            SimTheme.Plate(new Rect(x, y, 112, 32), liveSrc ? "LIVE" : "REPLAY", liveSrc ? SimTheme.Accent : SimTheme.Berry, 16);
            float cw = SimTheme.TextWidth("CAN 합성", 12, true) + 20;
            SimTheme.Chip(new Rect(x + w - cw, y + 4, cw, 24), "CAN 합성", SimTheme.Amber);
            if (bio == null || !bio.Loaded)
            {
                SimTheme.Label(new Rect(x, y + 44, w, 50), bio ? bio.Status : "입력 없음", 14, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
                return;
            }
            y += 46;
            if (bio is BioReplay br)
            {
                SimTheme.Label(new Rect(x, y, w, 22), br.SessionName, 15, SimTheme.Text, true);
                SimTheme.Label(new Rect(x, y, w, 22), $"{MmSs(br.FileTimeSec)} / {MmSs(br.DurationSec)}", 15, SimTheme.Text, true, TextAnchor.MiddleRight);
                float dur = Mathf.Max(1f, br.DurationSec);
                var tl = new Rect(x, y + 30, w, 10);
                SimTheme.Fill(tl, new Color(0, 0, 0, 0.45f), 4f);
                foreach (var g in br.Segments)
                {
                    Color c = g.label.Contains("졸음") ? SimTheme.Red : g.label.Contains("정상") ? SimTheme.Green : SimTheme.Gray;
                    SimTheme.Fill(new Rect(tl.x + tl.width * g.start / dur, tl.y, Mathf.Max(1, tl.width * (g.end - g.start) / dur - 2), tl.height), SimTheme.A(c, 0.75f), 3f);
                }
                SimTheme.Outline(tl, SimTheme.Ink, 1.5f, 4f);
                float px = tl.x + tl.width * Mathf.Clamp01(br.FileTimeSec / dur);
                var knob = new Rect(px - 4, tl.y - 5, 8, 20);
                SimTheme.Fill(knob, Color.white, 2f); SimTheme.Outline(knob, SimTheme.Ink, 1.5f, 2f);
                string lab = string.IsNullOrEmpty(br.SegmentLabel) ? "" : $"라벨(자기보고) {br.SegmentLabel}   ·   ";
                SimTheme.Label(new Rect(x, y + 46, w, 18), $"{lab}ECG {br.FsEcg:F0} Hz · PPG {br.FsPpg:F0} Hz", 12, SimTheme.Dim);
            }
            else if (bio is LiveBioSource lb)
            {
                SimTheme.Label(new Rect(x, y, w, 22), lb.Device, 15, SimTheme.Text, true);
                SimTheme.Label(new Rect(x, y, w, 22), lb.Receiving ? "수신 중" : "수신 끊김", 14, lb.Receiving ? SimTheme.Green : SimTheme.Amber, true, TextAnchor.MiddleRight);
                SimTheme.Label(new Rect(x, y + 28, w, 20), $"{lb.PacketRate:F0} 패킷/초 · 손실 {lb.Lost} · 지터 {lb.JitterMs:F1} ms · {lb.LastAgeSec:F1}초 전", 13, SimTheme.Sub);
            }
        }

        // ---------------------------------------------------------------- 위 가운데: 경고 배너
        void DrawBanner()
        {
            if (!Live || !ai.Latest.has_out) return;
            int lv = ai.Latest.level;
            if (lv < 1) return;
            float left = HudLayout.M + HudLayout.StateW + 24, right = SimTheme.VW - HudLayout.M - HudLayout.SessionW - 24, w = Mathf.Min(720f, right - left);
            if (w < 360) { w = 600; left = SimTheme.VW * 0.5f - 300; }
            else left = (left + right) * 0.5f - w * 0.5f;
            bool drowsy = lv == 2;
            bool confirmed = drowsy && ai.Latest.confirmed;
            float pulse = drowsy ? 0.8f + 0.2f * Mathf.Sin(Time.realtimeSinceStartup * 6f) : 1f;
            Color c = drowsy ? SimTheme.Red : SimTheme.Amber;
            var r = new Rect(left, HudLayout.M, w, 76);
            SimTheme.Panel(r);
            SimTheme.Fill(new Rect(r.x + 2, r.y + 2, r.width - 4, r.height - 4), SimTheme.A(c, 0.22f * pulse), 8f);
            string tag = confirmed ? "졸음 확정" : drowsy ? "졸음" : "주의";
            float tw = Mathf.Max(96f, SimTheme.DisplayWidth(tag, 30) + 32f);
            SimTheme.Plate(new Rect(r.x + 14, r.y + 14, tw, 48), tag, SimTheme.A(c, pulse > 0.9f ? 1f : 0.85f), 30);
            SimTheme.Display(new Rect(r.x + tw + 32, r.y, r.width - tw - 44, r.height), confirmed ? "(AI 모델 졸음 + 차량 경고)" : drowsy ? "졸음 상태입니다. 쉬어 가세요" : "졸음 징후가 있습니다", w < 560 ? 20 : 25, SimTheme.Text);
        }
    }
}
