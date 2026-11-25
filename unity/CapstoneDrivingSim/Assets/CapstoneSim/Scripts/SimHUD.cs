// SimHUD.cs
// 주행 계기판 (SimTheme 디자인, 배치는 HudLayout).
//   왼쪽 아래: 속도·기어·RPM·차로·깜빡이 + 핸들   /   그 위: 주행 기록(REC) 상태
//   맨 아래 오른쪽: 조작키   /   위 가운데: 동기화 마커(M) 플래시 (영상과 로그를 맞추는 기준점, HUD 를 꺼도 표시)
//   위 가운데(AI 배너 아래): 차량 경고 (차선 이탈 / 부주의)   /   REC 위: 조향 보조·차선 이탈 경고 상태
using UnityEngine;

namespace CapstoneSim
{
    public class SimHUD : MonoBehaviour
    {
        public CarController car;
        public DrivingDataLogger logger;
        public SimBootstrap sim;
        public bool show = true;
        public bool showHelp = true;

        Texture2D _wheelIcon;

        void Update()
        {
            if (SimInput.HudTogglePressed()) show = !show;
        }

        void OnGUI()
        {
            if (!_wheelIcon) _wheelIcon = ProceduralTextures.SteeringIcon();
            var m = SimTheme.Begin();
            float W = SimTheme.VW, H = SimTheme.VH;

            if (logger && Time.realtimeSinceStartup - logger.LastMarkerRealtime < 0.3f)
                SimTheme.Fill(new Rect(W * 0.5f - 70, 6, 140, 60), Color.white, 6f);

            if (show && car)
            {
                DrawCluster(W, H);
                DrawRec();
                DrawAdasStatus();
                DrawVehicleWarnings(W);
                if (showHelp) DrawHelp(W);
            }
            SimTheme.End(m);
        }

        void DrawCluster(float W, float H)
        {
            var r = new Rect(HudLayout.M, HudLayout.ClusterY, HudLayout.ClusterW, HudLayout.ClusterH);
            SimTheme.Panel(r);
            float spd = Mathf.Abs(car.SpeedKmh);
            string s = $"{spd:0}";
            SimTheme.Display(new Rect(r.x + 22, r.y + 6, 220, 96), s, 82, SimTheme.Text);
            float nw = SimTheme.DisplayWidth(s, 82);
            SimTheme.Label(new Rect(r.x + 28 + nw, r.y + 62, 80, 26), "km/h", 17, SimTheme.Sub, true);
            // 기어
            string gear = car.Gear == -1 ? "R" : car.Gear == 0 ? "N" : "D" + car.Gear;
            SimTheme.Plate(new Rect(r.x + 236, r.y + 20, 74, 52), gear, car.Gear == -1 ? SimTheme.Amber : SimTheme.Accent, 26);
            // RPM
            float rpm = car.EngineRpm;
            SimTheme.Label(new Rect(r.x + 24, r.y + 106, 50, 18), "RPM", 12, SimTheme.Sub, true);
            SimTheme.Bar(new Rect(r.x + 66, r.y + 109, 190, 12), rpm / 7000f, rpm > 5800 ? SimTheme.Red : SimTheme.Accent);
            SimTheme.Label(new Rect(r.x + 262, r.y + 104, 48, 22), $"{rpm:0}", 13, SimTheme.Text, true, TextAnchor.MiddleRight);
            // 차로·편차·깜빡이·랩
            if (logger)
            {
                var ts = logger.LastSample;
                float off = ts.laneOffset;
                Color oc = Mathf.Abs(off) > 0.85f ? SimTheme.Red : Mathf.Abs(off) > 0.5f ? SimTheme.Amber : SimTheme.Text;
                float y = r.y + 136;
                SimTheme.Label(new Rect(r.x + 24, y, 90, 24), ts.lane == 0 ? "도로 밖" : $"{ts.lane}차로", 16, ts.lane == 0 ? SimTheme.Red : SimTheme.Text, true);
                SimTheme.Label(new Rect(r.x + 96, y, 130, 24), $"편차 {off:+0.00;-0.00} m", 15, oc, true);
                SimTheme.Display(new Rect(r.x + 218, y - 4, 30, 30), "◀", 18, car.TurnSignal < 0 && Blink() ? SimTheme.Green : SimTheme.Dim, TextAnchor.MiddleCenter, 1.5f);
                SimTheme.Display(new Rect(r.x + 246, y - 4, 30, 30), "▶", 18, car.TurnSignal > 0 && Blink() ? SimTheme.Green : SimTheme.Dim, TextAnchor.MiddleCenter, 1.5f);
                SimTheme.Label(new Rect(r.x + 262, y, 48, 24), $"랩 {logger.Lap}", 14, SimTheme.Sub, true, TextAnchor.MiddleRight);
            }
            // 핸들 (패널 오른쪽 칸)
            SimTheme.Fill(new Rect(r.x + 326, r.y + 14, 1.5f, r.height - 28), SimTheme.Line);
            var wr = new Rect(r.x + 344, r.y + 16, 108, 108);
            var gm = GUI.matrix;
            GUIUtility.RotateAroundPivot(car.SteeringWheelDeg, Vector2.Scale(wr.center, new Vector2(SimTheme.S, SimTheme.S)));
            GUI.DrawTexture(wr, _wheelIcon);
            GUI.matrix = gm;
            string src = car.ActiveSteerSource == SteerMode.Mouse ? "마우스" : car.ActiveSteerSource == SteerMode.Gamepad ? "패드" : "키보드";
            string mode = car.steerMode == SteerMode.Auto ? $"자동·{src}" : src;
            SimTheme.Label(new Rect(wr.x - 12, wr.yMax + 8, wr.width + 24, 20), $"{car.SteeringWheelDeg:+0;-0;0}°  {mode}", 13, SimTheme.Sub, true, TextAnchor.MiddleCenter);

            if (car.ActiveSteerSource == SteerMode.Mouse)
            {
                float half = W * car.mouseRange;
                SimTheme.Fill(new Rect(W * 0.5f - half, H - 6, half * 2f, 3), new Color(1, 1, 1, 0.25f));
                SimTheme.Fill(new Rect(W * 0.5f - 1, H - 12, 2, 10), new Color(1, 1, 1, 0.4f));
            }
        }

        void DrawRec()
        {
            if (!logger) return;
            float y = HudLayout.ClusterY - 46;
            if (logger.IsRecording)
            {
                var plate = new Rect(HudLayout.M, y, 84, 36);
                SimTheme.Plate(plate, "REC", Blink() ? SimTheme.Red : SimTheme.A(SimTheme.Red, 0.7f), 18);
                var r = new Rect(plate.xMax + 8, y, HudLayout.ClusterW - 92, 36);
                SimTheme.Panel(r, 8f);
                SimTheme.Label(new Rect(r.x + 14, r.y, r.width - 20, r.height),
                    $"{FormatTime(logger.ElapsedSec)}   샘플 {logger.SampleCount}   마커 {logger.MarkerCount}   {logger.participantId}", 15, SimTheme.Text, true);
            }
            else
            {
                var r = new Rect(HudLayout.M, y + 4, 196, 30);
                SimTheme.Panel(r, 8f, 0.75f);
                SimTheme.Label(r, "L 키로 주행 기록 시작", 13, SimTheme.Sub, true, TextAnchor.MiddleCenter);
            }
        }

        void DrawHelp(float W)
        {
            string[] keys = { "W/↑", "S/↓", "A·D", "L", "M", "Q/E", "G", "K", "F8", "R", "C", "T", "H", "Esc" };
            string[] acts = { "가속", "브레이크", "조향", "기록", "마커", "깜빡이", "조향 보조", "부주의 시연", "시연 주행", "복귀", "시점", "조향 방식", "HUD", "메뉴" };
            float y = HudLayout.HelpY, h = HudLayout.HelpH, total = 0f;
            for (int i = 0; i < keys.Length; i++) total += SimTheme.DisplayWidth(acts[i], 13) + SimTheme.TextWidth(keys[i], 12, true) + 16 + 6 + 18;
            float x = W - HudLayout.M - total + 18;
            for (int i = 0; i < keys.Length; i++)
            {
                float aw = SimTheme.DisplayWidth(acts[i], 13), kw = SimTheme.TextWidth(keys[i], 12, true) + 16;
                var kr = new Rect(x, y + 3, kw, h - 6);
                SimTheme.Fill(new Rect(kr.x, kr.y + 2, kr.width, kr.height), SimTheme.A(SimTheme.Ink, 0.6f), 5f);
                SimTheme.Fill(kr, new Color(0.88f, 0.90f, 0.86f, 0.95f), 5f);
                SimTheme.Outline(kr, SimTheme.Ink, 1.5f, 5f);
                SimTheme.Label(kr, keys[i], 12, SimTheme.Ink, true, TextAnchor.MiddleCenter);
                x += kw + 6;
                SimTheme.Display(new Rect(x, y, aw + 8, h), acts[i], 13, SimTheme.Text, TextAnchor.MiddleLeft, 1.2f);
                x += aw + 18;
            }
        }

        // ---------------------------------------------------------------- 운전 보조 · 차량 경고
        void DrawAdasStatus()
        {
            if (!sim || !sim.Assist || !sim.Warnings) return;
            var a = sim.Assist; var w = sim.Warnings;
            float y = HudLayout.ClusterY - 46 - 42, x = HudLayout.M;
            var r = new Rect(x, y, HudLayout.ClusterW, 34);
            SimTheme.Panel(r, 8f, 0.8f);
            // 조향 보조
            string an = "조향 보조 " + DriverAssist.LevelName(a.level);
            Color ac = a.level == AssistLevel.Off ? SimTheme.Dim : a.Engaged ? SimTheme.Accent : SimTheme.Amber;
            float cw = SimTheme.TextWidth(an, 14, true) + 20;
            SimTheme.Chip(new Rect(r.x + 8, r.y + 5, cw, 24), an, ac, a.Engaged);
            float nx = r.x + 16 + cw;
            if (a.level != AssistLevel.Off && !a.Engaged && a.Reason != "")
            {
                SimTheme.Label(new Rect(nx, r.y, 90, r.height), a.Reason, 13, SimTheme.Sub, true);
                nx += SimTheme.TextWidth(a.Reason, 13, true) + 12;
            }
            // 차선 이탈 경고 (LDW) 상태
            string ln = w.LaneSafeIndi == 5 ? "차선 이탈!" : w.LaneSafeIndi == 4 ? "차선 이탈 경고 켜짐" : "차선 이탈 경고 대기";
            Color lc = w.LaneSafeIndi == 5 ? SimTheme.Red : w.LaneSafeIndi == 4 ? SimTheme.Green : SimTheme.Dim;
            float lw = SimTheme.TextWidth(ln, 14, true) + 20;
            SimTheme.Chip(new Rect(nx, r.y + 5, lw, 24), ln, lc, w.LaneSafeIndi == 5);
            // 시연 시나리오 자동 운전
            if (sim.Demo && sim.Demo.Running)
            {
                var dr = new Rect(r.x, r.y - 30, r.width, 26);
                SimTheme.Panel(dr, 6f, 0.8f);
                SimTheme.Label(new Rect(dr.x + 10, dr.y, dr.width - 20, dr.height),
                    $"시연 자동 운전 (F8·운전 키로 해제)   ·   {sim.Demo.Phase}   ·   기록 {SimTheme.MmSs(sim.Demo.FileTime)}", 13, SimTheme.Berry, true);
            }
            // 핸들 조작 없음 시간 (10초 넘으면 표시)
            if (w.HandsOffSec >= 10f)
                SimTheme.Label(new Rect(r.xMax - 120, r.y, 112, r.height), $"손 뗌 {w.HandsOffSec:0}초", 13,
                    w.HandsOffSec >= w.handsOff2 ? SimTheme.Amber : SimTheme.Sub, true, TextAnchor.MiddleRight);
        }

        void DrawVehicleWarnings(float W)
        {
            if (!sim || !sim.Warnings) return;
            var w = sim.Warnings;
            float bw = 600f, x = W * 0.5f - bw * 0.5f, y = HudLayout.M + 76 + 12;
            if (w.LaneWarning)
            {
                float pulse = 0.75f + 0.25f * Mathf.Sin(Time.realtimeSinceStartup * 12f);
                var r = new Rect(x, y, bw, 66);
                SimTheme.Panel(r);
                SimTheme.Fill(new Rect(r.x + 2, r.y + 2, r.width - 4, r.height - 4), SimTheme.A(SimTheme.Red, 0.25f * pulse), 8f);
                SimTheme.Plate(new Rect(r.x + 12, r.y + 11, 132, 44), "차선 이탈", SimTheme.Red, 22);
                string side = w.LaneWarnSide < 0 ? "◀ 왼쪽" : "오른쪽 ▶";
                SimTheme.Display(new Rect(r.x + 160, r.y, r.width - 170, r.height), $"{side} 차선을 밟았습니다", 22, SimTheme.Text);
                y += 66 + 8;
            }
            int c = w.CarelessWarn;
            if (c >= 1)
            {
                bool blink = c >= 2 && !Blink();
                Color col = c >= 3 ? SimTheme.Red : SimTheme.Amber;
                var r = new Rect(x, y, bw, c >= 2 ? 66 : 48);
                SimTheme.Panel(r);
                if (c >= 2) SimTheme.Fill(new Rect(r.x + 2, r.y + 2, r.width - 4, r.height - 4), SimTheme.A(col, blink ? 0.12f : 0.28f), 8f);
                SimTheme.Plate(new Rect(r.x + 12, r.y + (r.height - 40) * 0.5f, 132, 40), "부주의", col, 21);
                string msg = c >= 3 ? "휴식이 필요합니다" : c == 2 ? "전방을 주시하세요" : "핸들을 잡아 주세요";
                if (!string.IsNullOrEmpty(w.CarelessReason)) msg += $"  ({w.CarelessReason})";
                SimTheme.Display(new Rect(r.x + 160, r.y, r.width - 170, r.height), msg, c >= 2 ? 21 : 18, SimTheme.Text);
            }
        }

        static bool Blink() => ((int)(Time.realtimeSinceStartup * 2f)) % 2 == 0;

        static string FormatTime(double sec)
        {
            int s = (int)sec;
            return $"{s / 3600:00}:{s / 60 % 60:00}:{s % 60:00}";
        }
    }
}
