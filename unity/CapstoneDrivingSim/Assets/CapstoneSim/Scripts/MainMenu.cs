// MainMenu.cs
// 시뮬레이터 메인 메뉴 (SimTheme 디자인). Play → 홈 → [리플레이 모드 | 실측 모드] 설정 → 주행 시작.
//   리플레이 : 유니티 안(StreamingAssets/Replays)의 실측 생체 기록 재생. 개인 기준(baseline) 은 자동 수집 또는 기준 측정 파일.
//   실측(데모): 라즈베리파이 → UDP 로 들어오는 센서 데이터를 바로 판정 (Serial / MQTT / WebSocket 은 틀만)
// 주행 중 Esc → 세션 메뉴 (계속 / 메인 메뉴로 / 종료). 배경은 차량 주위를 도는 쇼케이스 카메라.
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace CapstoneSim
{
    /// <summary>메뉴에서 고른 세션 설정</summary>
    public class SessionConfig
    {
        public bool live;
        public string replayDir = "";
        public float startOffset;
        public bool preroll = true;
        public string baselineDir = "";            // 비우면 자동 수집
        public string aiMode = "causal";
        public string participantId = "P01";
        public LiveBioSource.Transport transport = LiveBioSource.Transport.Udp;
        public int udpPort = 5010;
        public string serialPort = "COM3", mqttHost = "raspberrypi.local", mqttTopic = "jberry/bio", wsUrl = "ws://raspberrypi.local:8765/bio";
        public int serialBaud = 921600, mqttPort = 1883;

        public static SessionConfig FromInspector(SimBootstrap s) => new SessionConfig
        {
            replayDir = string.IsNullOrWhiteSpace(s.bioSessionDir) ? (ReplayLibrary.DefaultReplayDir() ?? "") : s.bioSessionDir,
            startOffset = s.bioStartOffsetSec, preroll = s.aiPrerollBeforeStart, aiMode = s.aiMode,
            baselineDir = s.baselineSessionDir, participantId = s.participantId,
        };

        public void ApplyTo(LiveBioSource lb)
        {
            if (lb.Listening && (lb.transport != transport || lb.udpPort != udpPort)) lb.StopListening();
            lb.transport = transport; lb.udpPort = udpPort; lb.serialPort = serialPort; lb.serialBaud = serialBaud;
            lb.mqttHost = mqttHost; lb.mqttPort = mqttPort; lb.mqttTopic = mqttTopic; lb.wsUrl = wsUrl;
        }
    }

    public class MainMenu : MonoBehaviour
    {
        public SimBootstrap sim;
        public const string Version = "v0.6 · 2026-11";

        enum Page { Hidden, Home, Replay, Live, Pause }
        Page _page = Page.Hidden;
        public bool Visible => _page != Page.Hidden;

        SessionConfig _cfg;
        List<ReplayEntry> _lib = new List<ReplayEntry>();
        bool _venvOk;
        string _setupStatus = "";

        // ---------------------------------------------------------------- 상태 전환
        public void Show(string page)
        {
            if (_cfg == null) _cfg = SessionConfig.FromInspector(sim);
            Refresh();
            _page = page == "replay" ? Page.Replay : page == "live" ? Page.Live : Page.Home;
            if (_page == Page.Replay && string.IsNullOrEmpty(_cfg.replayDir) && _lib.Count > 0) _cfg.replayDir = _lib[0].dir;
            if (_page == Page.Live) _cfg.live = true;
        }
        public void Hide() => _page = Page.Hidden;
        public void ShowPause() => _page = Page.Pause;

        void Refresh()
        {
            _lib = ReplayLibrary.All(true);
            _venvOk = SimPaths.AiReady;
            _setupStatus = SimPaths.SetupStatus;
        }

        void Update()
        {
            if (!SimInput.BackPressed()) return;
            if (_page == Page.Replay || _page == Page.Live) { StopLiveTest(); _page = Page.Home; }
            else if (_page == Page.Pause) _page = Page.Hidden;
            else if (_page == Page.Hidden && sim.InSession && sim.Ai) _page = Page.Pause;
        }

        void StopLiveTest()
        {
            if (sim.Live && !sim.InSession) sim.Live.StopListening();
        }

        // ---------------------------------------------------------------- 그리기
        void OnGUI()
        {
            GUI.depth = -10;
            if (_cfg == null) _cfg = SessionConfig.FromInspector(sim);   // Play 중 재컴파일되면 직렬화되지 않는 설정이 비므로 다시 만듦
            if (_lib == null) _lib = new List<ReplayEntry>();
            var m = SimTheme.Begin();
            if (_page == Page.Pause) DrawPause();
            else if (_page != Page.Hidden)
            {
                DrawBackdrop();
                if (_page == Page.Home) DrawHome();
                else if (_page == Page.Replay) DrawReplay();
                else if (_page == Page.Live) DrawLive();
                DrawStatusBar();
            }
            if (sim.InSession && sim.Ai != null && !sim.Ai.SessionAccepted) DrawLoading();
            SimTheme.End(m);
        }

        void DrawBackdrop()
        {
            float W = SimTheme.VW, H = SimTheme.VH;
            SimTheme.GradientH(new Rect(0, 0, W * 0.62f, H), new Color(0.05f, 0.06f, 0.06f, 0.88f));
            SimTheme.GradientUp(new Rect(0, H - 260, W, 260), new Color(0.04f, 0.05f, 0.05f, 0.85f));
            // 타이틀: 굵은 기울임 외곽선 글자 두 줄 + 라임 배지
            SimTheme.Label(new Rect(76, 36, 700, 22), "J_BERRY PI × DH오토웨어 산학 캡스톤", 14, SimTheme.Sub, true);
            SimTheme.Display(new Rect(68, 56, 900, 66), "Driver State", 58, SimTheme.Text);
            float tw = SimTheme.DisplayWidth("Simulator", 58);
            SimTheme.Display(new Rect(68, 112, 900, 66), "Simulator", 58, SimTheme.Text);
            SimTheme.Plate(new Rect(68 + tw + 18, 116, 92, 62), "AI", SimTheme.Accent, 36);
            SimTheme.Label(new Rect(76, 184, 1000, 24), "ECG·PPG와 차량 신호로 졸음을 판정하는 On-Device AI 운전 시뮬레이터", 16, SimTheme.Sub);
        }

        void DrawStatusBar()
        {
            float W = SimTheme.VW, y = SimTheme.VH - 52;
            SimTheme.Fill(new Rect(72, y - 14, W - 144, 1), SimTheme.Line);
            int nRep = 0, nBase = 0;
            foreach (var e in _lib) if (e.IsBaseline) nBase++; else nRep++;
            float x = 72;
            DrawEngineStatus(x, y);
            x += 470;
            SimTheme.Dot(new Rect(x, y, 300, 24), $"리플레이 기록 {nRep}개 · 기준 파일 {nBase}개", nRep > 0 ? SimTheme.Accent : SimTheme.Amber);
            x += 320;
            SimTheme.Label(new Rect(x, y, 300, 24), Version, 14, SimTheme.Dim);
            SimTheme.Label(new Rect(W - 372, y, 300, 24), _page == Page.Home ? "" : "Esc  뒤로", 14, SimTheme.Dim, false, TextAnchor.MiddleRight);
        }

        float _nextPoll;
        /// <summary>AI 엔진 상태 + 없으면 [AI 엔진 설치] 버튼. 설치 중에는 진행 상황</summary>
        void DrawEngineStatus(float x, float y)
        {
            if (!_venvOk && Time.realtimeSinceStartup > _nextPoll) { _nextPoll = Time.realtimeSinceStartup + 1.5f; Refresh(); }
            if (AiInstaller.Running || (_setupStatus.StartsWith("running") && SetupLogFresh()))
            {
                string step = _setupStatus.StartsWith("running:") ? _setupStatus.Substring(8).Trim() : "";
                string tail = AiInstaller.LastLogLine();
                SimTheme.Dot(new Rect(x, y, 440, 24), $"AI 엔진 설치 중 {Spinner()}  {step}", SimTheme.Amber);
                if (!string.IsNullOrEmpty(tail)) SimTheme.Label(new Rect(x + 16, y - 24, 700, 20), tail, 12, SimTheme.Dim);
                return;
            }
            if (_venvOk)
            {
                SimTheme.Dot(new Rect(x, y, 440, 24), "AI 엔진 준비됨 (S1-EP · 상태 머신 v2 · CPU)", SimTheme.Green);
                return;
            }
            string why = _setupStatus.StartsWith("fail") ? "설치 실패 · " + _setupStatus.Substring(5).Trim() : "AI 엔진이 없습니다";
            SimTheme.Dot(new Rect(x, y, 300, 24), why, SimTheme.Amber);
            if (SimTheme.Button(new Rect(x + 300, y - 8, 160, 40), "AI 엔진 설치", 15, SimTheme.Kind.Primary)) { AiInstaller.Start(); Refresh(); }
        }

        /// <summary>다른 곳(에디터 메뉴 등)에서 돌린 설치가 아직 진행 중인지: 로그가 90초 안에 바뀌었는지</summary>
        static bool SetupLogFresh()
        {
            try
            {
                var p = Path.Combine(SimPaths.AiLogDir, "setup_log.txt");
                return File.Exists(p) && (System.DateTime.Now - File.GetLastWriteTime(p)).TotalSeconds < 90;
            }
            catch { return false; }
        }

        static string Spinner() => "◐◓◑◒"[(int)(Time.realtimeSinceStartup * 4f) % 4].ToString();

        // ---------------------------------------------------------------- 홈
        void DrawHome()
        {
            float x = 72, y = 244, w = 600;
            SimTheme.Header(new Rect(x, y, w, 24), "모드 선택");
            y += 40;
            if (SimTheme.Card(new Rect(x, y, w, 170), "REPLAY", "리플레이 모드",
                    "실측 ECG·PPG 기록을 실제 속도로 재생하며 주행합니다. 개인 기준은 자동으로 잡거나 기준 측정 파일에서 고릅니다.",
                    "사용 가능", SimTheme.Green, false))
            { _cfg.live = false; Show("replay"); }
            y += 186;
            if (SimTheme.Card(new Rect(x, y, w, 170), "LIVE", "실측 모드",
                    "라즈베리파이가 보낸 센서 데이터를 받아 바로 판정합니다. 지금은 UDP만 동작하고 Serial·MQTT·WebSocket은 틀만 있습니다.",
                    "DEMO", SimTheme.Amber, false))
            { Show("live"); }
            y += 190;
            if (SimTheme.Button(new Rect(x, y, 190, 46), "기록 폴더 열기", 15)) ReplayLibrary.OpenFolder();
            if (SimTheme.Button(new Rect(x + 205, y, 190, 46), "목록 새로고침", 15)) Refresh();
            if (SimTheme.Button(new Rect(x + w - 150, y, 150, 46), "종료", 15, SimTheme.Kind.Ghost)) Quit();

            // 빠른 시연: 첫 기록의 졸음 구간(16:00)부터, 자동 기준 + 앞부분 빠른 입력
            y += 70;
            string demoDir = ReplayLibrary.DefaultReplayDir();
            var qr = new Rect(x, y, w, 92);
            SimTheme.Panel(qr);
            SimTheme.Label(new Rect(qr.x + 22, qr.y + 14, 300, 20), "QUICK DEMO", 13, SimTheme.Berry, true);
            if (demoDir != null)
            {
                SimTheme.Label(new Rect(qr.x + 22, qr.y + 36, w - 290, 22), Path.GetFileName(demoDir), 15, SimTheme.Text, true);
                SimTheme.Label(new Rect(qr.x + 22, qr.y + 60, w - 290, 20), "16:00부터 재생 · 약 1분 뒤 졸음 확정", 13, SimTheme.Sub);
            }
            else SimTheme.Label(new Rect(qr.x + 22, qr.y + 40, w - 290, 22), "리플레이 기록이 없습니다", 14, SimTheme.Sub);
            if (SimTheme.Button(new Rect(qr.xMax - 250, qr.y + 22, 230, 48), "졸음 감지 시연 ▶", 16, SimTheme.Kind.Primary, demoDir != null && _venvOk))
                sim.StartSession(new SessionConfig { replayDir = demoDir, startOffset = 960f, preroll = true, baselineDir = "", aiMode = "causal", participantId = _cfg.participantId });
        }

        // ---------------------------------------------------------------- 리플레이 설정
        Vector2 _repScroll;
        void DrawReplay()
        {
            float x0 = 72, y0 = 228, W = Mathf.Min(SimTheme.VW - 144, 1560), H = SimTheme.VH - y0 - 92;
            SimTheme.Panel(new Rect(x0, y0, W, H), 10f, 0.9f);
            float pad = 28, colW1 = 420, colW2 = 460, cy = y0 + pad;
            float c1 = x0 + pad, c2 = c1 + colW1 + 28, c3 = c2 + colW2 + 28, colW3 = x0 + W - pad - c3;
            float bottom = y0 + H - 180;

            // --- 1열: 기록 목록 ---
            SimTheme.Header(new Rect(c1, cy, colW1, 24), "리플레이 기록");
            float ly = cy + 36;
            int shown = 0;
            ReplayEntry sel = null;
            foreach (var e in _lib)
            {
                if (e.IsBaseline) continue;
                bool on = SamePath(e.dir, _cfg.replayDir);
                if (on) sel = e;
                if (ly + 78 > bottom - 50) continue;
                var r = new Rect(c1, ly, colW1, 72);
                if (EntryRow(r, e, on)) { _cfg.replayDir = e.dir; _cfg.startOffset = Mathf.Min(_cfg.startOffset, Mathf.Max(0, e.durationSec - 60)); }
                ly += 80; shown++;
            }
            if (shown == 0)
                SimTheme.Label(new Rect(c1, ly, colW1, 120), "기록이 없습니다.\n측정 zip을 Tools/convert_recording.py로 변환해\nAssets/StreamingAssets/Replays/에 넣으세요.", 14, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
            if (SimTheme.Button(new Rect(c1, bottom - 46, 200, 40), "폴더 열기", 14)) ReplayLibrary.OpenFolder();
            if (SimTheme.Button(new Rect(c1 + 214, bottom - 46, 200, 40), "새로고침", 14)) Refresh();

            // --- 2열: 기준 + 계산 방식 ---
            float by = DrawBaselinePicker(c2, cy, colW2, bottom - 160, sel != null ? sel.name : null);
            DrawAiMode(c2, by + 22, colW2);

            // --- 3열: 요약 ---
            SimTheme.Header(new Rect(c3, cy, colW3, 24), "세션 요약");
            _cfg.participantId = SimTheme.TextField(new Rect(c3, cy + 36, Mathf.Min(260, colW3), 58), "참가자 ID (로그 파일명)", _cfg.participantId);
            float sy = cy + 112;
            string baseTxt = string.IsNullOrEmpty(_cfg.baselineDir) ? "자동 (기록 앞부분)" : "기준 파일 · " + Path.GetFileName(_cfg.baselineDir);
            Kv(c3, ref sy, colW3, "기록", sel != null ? $"{sel.name}  ({SimTheme.MmSs(sel.durationSec)})" : "–");
            Kv(c3, ref sy, colW3, "시작 위치", SimTheme.MmSs(_cfg.startOffset));
            Kv(c3, ref sy, colW3, "개인 기준", baseTxt);
            Kv(c3, ref sy, colW3, "계산 방식", _cfg.aiMode == "exact" ? "batch 동일 (약 140초 지연)" : "실시간 (약 5초 지연)");
            Kv(c3, ref sy, colW3, "앞부분 넣기", PrerollText(sel));
            sy += 6;
            SimTheme.Label(new Rect(c3, sy, colW3, 120), ExpectText(sel), 14, SimTheme.Sub, false, TextAnchor.UpperLeft, true);

            // --- 아래: 타임라인 + 시작 ---
            float ty = bottom + 6;
            SimTheme.Fill(new Rect(c1, ty - 10, W - 2 * pad, 1), SimTheme.Line);
            SimTheme.Header(new Rect(c1, ty + 4, 400, 24), "재생 시작 위치");
            float dur = sel != null ? sel.durationSec : 0f;
            SimTheme.Label(new Rect(c1 + 200, ty + 4, 600, 24), sel != null && sel.HasLabels ? "색 띠: 자기보고 라벨 (초록 정상 · 빨강 졸음)" : "", 13, SimTheme.Dim);
            var tl = new Rect(c1, ty + 44, W - 2 * pad - 300, 34);
            _cfg.startOffset = Mathf.Round(SimTheme.Timeline(tl, _cfg.startOffset, Mathf.Max(1, dur), sel?.segments, 4242) / 10f) * 10f;
            for (int k = 0; k <= 6; k++)
                SimTheme.Label(new Rect(tl.x + tl.width * k / 6f - 30, tl.yMax + 2, 60, 18), SimTheme.MmSs(dur * k / 6f), 12, SimTheme.Dim, false, TextAnchor.MiddleCenter);
            _cfg.preroll = SimTheme.Toggle(new Rect(c1, ty + 104, 760, 50), "앞부분을 AI에 빠르게 넣기", _cfg.preroll,
                string.IsNullOrEmpty(_cfg.baselineDir) ? "기준 수집(7~8분)을 건너뛰고 바로 판정합니다."
                                                        : "시작 지점 앞 5분을 넣어 5분 지표를 미리 채웁니다.");
            bool can = sel != null && _venvOk;
            if (SimTheme.Button(new Rect(x0 + W - pad - 280, ty + 40, 280, 64), "주행 시작 ▶", 22, SimTheme.Kind.Primary, can))
            { _cfg.live = false; sim.StartSession(_cfg); }
            if (!_venvOk) SimTheme.Label(new Rect(x0 + W - pad - 280, ty + 108, 280, 40), "아래 상태 줄에서 AI 엔진을 설치하세요", 13, SimTheme.Amber, false, TextAnchor.UpperCenter);
        }

        string PrerollText(ReplayEntry sel)
        {
            if (!_cfg.preroll || _cfg.startOffset <= 0) return "없음";
            return string.IsNullOrEmpty(_cfg.baselineDir) ? $"0~{SimTheme.MmSs(_cfg.startOffset)} (기준 수집 포함)"
                                                          : $"{SimTheme.MmSs(Mathf.Max(0, _cfg.startOffset - 300))}~{SimTheme.MmSs(_cfg.startOffset)} (5분 지표용)";
        }

        string ExpectText(ReplayEntry sel)
        {
            bool auto = string.IsNullOrEmpty(_cfg.baselineDir);
            if (auto && (!_cfg.preroll || _cfg.startOffset < 1))
                return "실제 측정과 순서가 같습니다. 60초 착석 → 기준 수집(보통 7~8분) → 10초마다 판정.";
            if (auto) return "기준 수집을 마친 상태로 시작합니다. 몇 초 안에 첫 판정이 나옵니다.";
            return "기준 파일로 바로 판정합니다. 앞부분 넣기를 끄면 처음 5분은 확률이 낮게 나올 수 있습니다.";
        }

        float DrawBaselinePicker(float x, float y, float w, float maxY, string replayName)
        {
            SimTheme.Header(new Rect(x, y, w, 24), "개인 기준");
            y += 36;
            if (SimTheme.Radio(new Rect(x, y, w, 72), "자동 수집", "입력 앞부분에서 자동으로 고릅니다 (60초 대기 후 깨끗한 30초 창 20개, 최대 10분)", string.IsNullOrEmpty(_cfg.baselineDir)))
                _cfg.baselineDir = "";
            y += 80;
            int n = 0;
            foreach (var e in _lib)
            {
                if (!e.IsBaseline) continue;
                if (y + 64 > maxY) break;
                string sub = $"{SimTheme.MmSs(e.durationSec)} · ECG {e.fsEcg:F0} Hz · PPG {e.fsPpg:F0} Hz · {(e.source == "StreamingAssets" ? "내장" : "SimLogs")}";
                if (SimTheme.Radio(new Rect(x, y, w, 64), "기준 파일 · " + e.name, sub, SamePath(e.dir, _cfg.baselineDir)))
                    _cfg.baselineDir = e.dir;
                y += 72; n++;
            }
            if (n == 0)
            {
                SimTheme.Label(new Rect(x + 4, y, w - 8, 60), "기준 측정 파일이 없습니다. 안정 상태로 5~10분 잰 기록을\nconvert_recording.py --role baseline으로 변환해 넣으세요.", 13, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
                y += 60;
            }
            return y;
        }

        void DrawAiMode(float x, float y, float w)
        {
            SimTheme.Header(new Rect(x, y, w, 24), "AI 계산 방식");
            int s = SimTheme.Segmented(new Rect(x, y + 34, w, 42), new[] { "실시간 (causal)", "batch 동일 (exact)" }, _cfg.aiMode == "exact" ? 1 : 0);
            _cfg.aiMode = s == 1 ? "exact" : "causal";
            SimTheme.Label(new Rect(x, y + 82, w, 40), s == 1 ? "학습 때와 같은 구간(창 중심 ±150초)으로 계산합니다. 판정이 140초 늦게 나옵니다." : "지난 데이터만 씁니다. 판정이 5초 뒤에 나오고, 값이 학습 때와 조금 다를 수 있습니다.", 13, SimTheme.Sub, false, TextAnchor.UpperLeft, true);
        }

        bool EntryRow(Rect r, ReplayEntry e, bool on)
        {
            bool hov = r.Contains(Event.current.mousePosition);
            SimTheme.Fill(r, on ? SimTheme.PanelHi : hov ? new Color(1, 1, 1, 0.06f) : new Color(0, 0, 0, 0.22f), 8f);
            SimTheme.Outline(r, on ? SimTheme.A(SimTheme.Accent, 0.9f) : SimTheme.Line, on ? 2f : 1f, 8f);
            SimTheme.Label(new Rect(r.x + 16, r.y + 10, r.width - 120, 26), e.name, 17, SimTheme.Text, true);
            SimTheme.Label(new Rect(r.x + 16, r.y + 38, r.width - 32, 22), $"{SimTheme.MmSs(e.durationSec)}  ·  ECG {e.fsEcg:F0} Hz  ·  PPG {e.fsPpg:F0} Hz", 13, SimTheme.Sub);
            float cx = r.xMax - 12;
            string src = e.source == "StreamingAssets" ? "내장" : "SimLogs";
            float w1 = SimTheme.TextWidth(src, 11, true) + 18; cx -= w1;
            SimTheme.Chip(new Rect(cx, r.y + 12, w1, 20), src, SimTheme.Accent);
            if (e.HasLabels) { float w2 = SimTheme.TextWidth("라벨", 11, true) + 18; cx -= w2 + 6; SimTheme.Chip(new Rect(cx, r.y + 12, w2, 20), "라벨", SimTheme.Berry); }
            return GUI.Button(r, GUIContent.none, GUIStyle.none);
        }

        static void Kv(float x, ref float y, float w, string k, string v)
        {
            SimTheme.Label(new Rect(x, y, 96, 26), k, 14, SimTheme.Sub);
            SimTheme.Label(new Rect(x + 100, y, w - 100, 26), v, 15, SimTheme.Text, true);
            y += 30;
        }

        static bool SamePath(string a, string b)
        {
            if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b)) return false;
            try { return Path.GetFullPath(a).TrimEnd('\\', '/') == Path.GetFullPath(b).TrimEnd('\\', '/'); } catch { return a == b; }
        }

        // ---------------------------------------------------------------- 실측 모드 (데모)
        void DrawLive()
        {
            float x0 = 72, y0 = 228, W = Mathf.Min(SimTheme.VW - 144, 1560), H = SimTheme.VH - y0 - 92;
            SimTheme.Panel(new Rect(x0, y0, W, H), 10f, 0.9f);
            float pad = 28, colW1 = 440, colW2 = 520, cy = y0 + pad;
            float c1 = x0 + pad, c2 = c1 + colW1 + 28, c3 = c2 + colW2 + 28, colW3 = x0 + W - pad - c3;
            SimTheme.Chip(new Rect(c1 + colW1 - 74, cy, 74, 24), "DEMO", SimTheme.Amber);

            // --- 1열: 전송 방식 ---
            SimTheme.Header(new Rect(c1, cy, colW1, 24), "전송 방식");
            int t = (int)_cfg.transport;
            int nt = SimTheme.Segmented(new Rect(c1, cy + 36, colW1, 44), new[] { "UDP", "Serial", "MQTT", "WebSocket" }, t);
            if (nt != t) { StopLiveTest(); _cfg.transport = (LiveBioSource.Transport)nt; }
            float fy = cy + 98;
            switch (_cfg.transport)
            {
                case LiveBioSource.Transport.Udp:
                    int.TryParse(SimTheme.TextField(new Rect(c1, fy, 200, 58), "수신 포트", _cfg.udpPort.ToString()), out _cfg.udpPort); break;
                case LiveBioSource.Transport.Serial:
                    _cfg.serialPort = SimTheme.TextField(new Rect(c1, fy, 200, 58), "포트", _cfg.serialPort);
                    int.TryParse(SimTheme.TextField(new Rect(c1 + 214, fy, 200, 58), "보레이트", _cfg.serialBaud.ToString()), out _cfg.serialBaud); break;
                case LiveBioSource.Transport.Mqtt:
                    _cfg.mqttHost = SimTheme.TextField(new Rect(c1, fy, 260, 58), "브로커", _cfg.mqttHost);
                    int.TryParse(SimTheme.TextField(new Rect(c1 + 274, fy, 140, 58), "포트", _cfg.mqttPort.ToString()), out _cfg.mqttPort);
                    _cfg.mqttTopic = SimTheme.TextField(new Rect(c1, fy + 68, colW1, 58), "토픽", _cfg.mqttTopic); fy += 68; break;
                default:
                    _cfg.wsUrl = SimTheme.TextField(new Rect(c1, fy, colW1, 58), "주소", _cfg.wsUrl); break;
            }
            fy += 76;
            bool impl = LiveBioSource.Implemented(_cfg.transport);
            var live = sim.Live;
            bool listening = live && live.Listening;
            if (!listening)
            {
                if (SimTheme.Button(new Rect(c1, fy, 200, 44), "수신 시험 시작", 15, SimTheme.Kind.Normal, impl))
                { var lb = sim.EnsureLive(); _cfg.ApplyTo(lb); lb.StartListening(); }
            }
            else if (SimTheme.Button(new Rect(c1, fy, 200, 44), "수신 중지", 15)) live.StopListening();
            if (!impl) SimTheme.Chip(new Rect(c1 + 214, fy + 10, 120, 24), "틀만 있음", SimTheme.Gray);
            fy += 64;
            SimTheme.Label(new Rect(c1, fy, colW1, 230), TransportNote(_cfg.transport), 13, SimTheme.Sub, false, TextAnchor.UpperLeft, true);

            // --- 2열: 수신 상태 ---
            SimTheme.Header(new Rect(c2, cy, colW2, 24), "수신 상태");
            string st = live ? live.Status : impl ? "'수신 시험 시작'을 누르세요" : "아직 구현하지 않은 방식입니다";
            Color stc = live && live.Receiving ? SimTheme.Green : impl ? SimTheme.Amber : SimTheme.Gray;
            SimTheme.Dot(new Rect(c2, cy + 36, colW2, 26), st, stc, 16);
            float gy = cy + 76, gw = (colW2 - 24) / 3f;
            Metric(new Rect(c2, gy, gw, 70), "패킷/초", live ? $"{live.PacketRate:F0}" : "–");
            Metric(new Rect(c2 + gw + 12, gy, gw, 70), "손실 패킷", live ? $"{live.Lost}" : "–");
            Metric(new Rect(c2 + 2 * (gw + 12), gy, gw, 70), "지터", live && live.Packets > 2 ? $"{live.JitterMs:F1} ms" : "–");
            gy += 82;
            Metric(new Rect(c2, gy, gw, 70), "마지막 수신", live && live.Packets > 0 ? $"{live.LastAgeSec:F1} s 전" : "–");
            Metric(new Rect(c2 + gw + 12, gy, gw, 70), "ECG / PPG", live && live.Loaded ? $"{live.FsEcg:F0} / {live.FsPpg:F0} Hz" : "–");
            Metric(new Rect(c2 + 2 * (gw + 12), gy, gw, 70), "장치", live && live.Packets > 0 ? live.Device : "–");
            gy += 90;
            SimTheme.Label(new Rect(c2, gy, colW2, 20), "원신호 (필터 전)", 12, SimTheme.Sub, true);
            Spark(new Rect(c2, gy + 24, colW2, 70), live && live.Packets > 0 ? live.Preview(true) : null, new Color(0.31f, 1f, 0.55f));
            Spark(new Rect(c2, gy + 102, colW2, 70), live && live.Packets > 0 ? live.Preview(false) : null, new Color(1f, 0.47f, 0.35f));
            SimTheme.Label(new Rect(c2, gy + 182, colW2, 60), "라즈베리파이 없이 시험하려면\npython Tools/rpi_bridge/rpi_sender.py --session <기록 폴더>", 13, SimTheme.Dim, false, TextAnchor.UpperLeft, true);

            // --- 3열: 기준 + 시작 ---
            float by = DrawBaselinePicker(c3, cy, colW3, y0 + H - 330, null);
            DrawAiMode(c3, by + 12, colW3);
            _cfg.participantId = SimTheme.TextField(new Rect(c3, y0 + H - 190, Mathf.Min(260, colW3), 58), "참가자 ID", _cfg.participantId);
            bool can = impl && _venvOk;
            if (SimTheme.Button(new Rect(c3, y0 + H - pad - 64, colW3, 64), "실측 주행 시작 ▶", 22, SimTheme.Kind.Primary, can))
            { _cfg.live = true; sim.StartSession(_cfg); }
            SimTheme.Label(new Rect(c3, y0 + H - pad - 104, colW3, 36), impl ? "수신 전에도 시작할 수 있습니다. 첫 패킷이 0초입니다." : "UDP를 고르세요", 12, SimTheme.Dim, false, TextAnchor.MiddleLeft, true);
        }

        static string TransportNote(LiveBioSource.Transport t)
        {
            switch (t)
            {
                case LiveBioSource.Transport.Udp:
                    return "Wi-Fi나 이더넷. 브로커가 필요 없고 같은 LAN이면 지연이 수 ms입니다. 패킷이 빠지거나 순서가 바뀔 수 있어 seq로 확인합니다. 판정 창이 30초라 가끔 빠지는 패킷은 영향이 작습니다.";
                case LiveBioSource.Transport.Serial:
                    return "USB 케이블. 지터가 가장 작고 네트워크 설정이 없습니다. 921600 bps면 ECG 500 Hz와 PPG 100 Hz를 보내기에 충분합니다. (틀만 있음)";
                case LiveBioSource.Transport.Mqtt:
                    return "라즈베리파이의 mosquitto 브로커를 거칩니다. LAN에서 QoS 0이면 지연이 수~수십 ms라 10 Hz 묶음 전송에 충분합니다. 웹보드·시뮬레이터·기록기가 같은 데이터를 함께 받을 때 유리합니다. (틀만 있음)";
                default:
                    return "웹보드 규약이 제안하는 방식입니다. 브라우저 대시보드와 서버를 함께 쓸 수 있습니다. TCP라 손실은 없지만 재전송 때 지연이 튈 수 있습니다. (틀만 있음)";
            }
        }

        static void Metric(Rect r, string k, string v)
        {
            SimTheme.Fill(r, new Color(0, 0, 0, 0.3f), 8f);
            SimTheme.Outline(r, SimTheme.Line, 1f, 8f);
            SimTheme.Label(new Rect(r.x + 14, r.y + 8, r.width - 20, 18), k, 12, SimTheme.Sub, true);
            SimTheme.Display(new Rect(r.x + 14, r.y + 28, r.width - 20, 34), v, 20, SimTheme.Text, TextAnchor.MiddleLeft, 1.5f);
        }

        static void Spark(Rect r, float[] d, Color c)
        {
            SimTheme.Fill(r, new Color(0, 0, 0, 0.3f), 8f);
            if (d == null || d.Length < 4) { SimTheme.Label(r, "신호 없음", 12, SimTheme.Dim, false, TextAnchor.MiddleCenter); return; }
            float mn = float.MaxValue, mx = float.MinValue;
            foreach (var v in d) { if (v < mn) mn = v; if (v > mx) mx = v; }
            float span = Mathf.Max(1e-6f, mx - mn);
            int cols = Mathf.Max(8, (int)(r.width / 2));
            float prevY = -1f;
            for (int i = 0; i < cols; i++)
            {
                int a = i * d.Length / cols, b = Mathf.Max(a + 1, (i + 1) * d.Length / cols);
                float lo = float.MaxValue, hi = float.MinValue;
                for (int k = a; k < b && k < d.Length; k++) { lo = Mathf.Min(lo, d[k]); hi = Mathf.Max(hi, d[k]); }
                float y0 = r.yMax - 6 - (hi - mn) / span * (r.height - 12), y1 = r.yMax - 6 - (lo - mn) / span * (r.height - 12);
                if (prevY >= 0) { y0 = Mathf.Min(y0, prevY); y1 = Mathf.Max(y1, prevY); }   // 이전 열과 이어 그림
                SimTheme.Fill(new Rect(r.x + i * 2, y0, 2, Mathf.Max(1.5f, y1 - y0)), c);
                int last = Mathf.Min(b, d.Length) - 1;
                prevY = r.yMax - 6 - (d[last] - mn) / span * (r.height - 12);
            }
        }

        // ---------------------------------------------------------------- 세션 메뉴 / 로딩
        void DrawPause()
        {
            float W = SimTheme.VW, H = SimTheme.VH;
            SimTheme.Fill(new Rect(0, 0, W, H), new Color(0, 0, 0, 0.55f));
            var r = new Rect(W * 0.5f - 280, H * 0.5f - 240, 560, 480);
            SimTheme.Panel(r, 10f, 0.95f);
            SimTheme.Plate(new Rect(r.x + 32, r.y + 26, 132, 34), "PAUSE", SimTheme.Accent, 18);
            SimTheme.Display(new Rect(r.x + 32, r.y + 62, r.width - 64, 44), "일시 정지", 32, SimTheme.Text);
            float y = r.y + 122;
            var c = sim.Session;
            if (c != null)
            {
                Kv(r.x + 32, ref y, r.width - 64, "모드", c.live ? $"실측 (DEMO · {LiveBioSource.TransportNames[(int)c.transport]})" : "리플레이");
                if (!c.live) Kv(r.x + 32, ref y, r.width - 64, "기록", sim.Bio != null ? sim.Bio.SessionName : Path.GetFileName(c.replayDir));
                string bd = sim.Ai != null ? sim.Ai.baselineDir : c.baselineDir;      // 실제 적용된 값 (자동 테스트 덮어쓰기 포함)
                Kv(r.x + 32, ref y, r.width - 64, "개인 기준", string.IsNullOrEmpty(bd) ? "자동" : "파일 · " + Path.GetFileName(bd));
                Kv(r.x + 32, ref y, r.width - 64, "경과", SimTheme.MmSs(Time.realtimeSinceStartup - sim.SessionStartRealtime));
            }
            float by = r.yMax - 192;
            if (SimTheme.Button(new Rect(r.x + 32, by, r.width - 64, 52), "주행 계속", 18, SimTheme.Kind.Primary)) _page = Page.Hidden;
            if (SimTheme.Button(new Rect(r.x + 32, by + 64, r.width - 64, 52), "메인 메뉴 (세션 종료)", 17)) sim.EndSession();
            if (SimTheme.Button(new Rect(r.x + 32, by + 128, r.width - 64, 44), "시뮬레이터 종료", 15, SimTheme.Kind.Ghost)) Quit();
        }

        void DrawLoading()
        {
            var ai = sim.Ai;
            float W = SimTheme.VW;
            var r = new Rect(W * 0.5f - 300, 120, 600, 104);
            SimTheme.Panel(r, 10f, 0.92f);
            bool fail = ai.ServerStatus.Contains("실패") || ai.ServerStatus.Contains("미설치");
            bool waitLive = sim.Bio != null && sim.Bio.IsLive && !sim.Bio.Loaded;
            SimTheme.Label(new Rect(r.x + 24, r.y + 16, r.width - 48, 20), fail ? "AI 엔진 오류" : waitLive ? "실측 데이터 대기" : "AI 엔진 준비 중", 13, fail ? SimTheme.Red : SimTheme.Accent, true);
            SimTheme.Label(new Rect(r.x + 24, r.y + 38, r.width - 48, 28), waitLive ? sim.Bio.Status : ai.ServerStatus, 18, SimTheme.Text, true);
            if (!fail)
            {
                float t = Time.realtimeSinceStartup * 0.6f % 1f;
                var bar = new Rect(r.x + 24, r.y + 80, r.width - 48, 6);
                SimTheme.Fill(bar, new Color(1, 1, 1, 0.08f), 3f);
                SimTheme.Fill(new Rect(bar.x + bar.width * Mathf.Clamp01(t - 0.25f), bar.y, bar.width * 0.25f, bar.height), SimTheme.Accent, 3f);
            }
            else SimTheme.Label(new Rect(r.x + 24, r.y + 72, r.width - 48, 22), "Esc → 메인 메뉴 · 로그: SimLogs/ai/server_stdout.txt", 13, SimTheme.Sub);
        }

        static void Quit()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}
