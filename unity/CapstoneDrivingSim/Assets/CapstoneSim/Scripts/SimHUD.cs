// SimHUD.cs
// 주행 계기판 (SimTheme 디자인, 배치는 HudLayout).
//   왼쪽 아래: 속도·기어·RPM·차로·깜빡이 + 핸들   /   그 위: 주행 기록(REC) 상태
//   맨 아래 오른쪽: 조작키   /   위 가운데: 동기화 마커(M) 플래시 (영상과 로그를 맞추는 기준점, HUD 를 꺼도 표시)
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
            string[] keys = { "W/↑", "S/↓", "A·D", "L", "M", "Q/E", "R", "C", "T", "H", "Esc" };
            string[] acts = { "가속", "브레이크", "조향", "기록", "마커", "깜빡이", "복귀", "시점", "조향 방식", "HUD", "메뉴" };
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

        static bool Blink() => ((int)(Time.realtimeSinceStartup * 2f)) % 2 == 0;

        static string FormatTime(double sec)
        {
            int s = (int)sec;
            return $"{s / 3600:00}:{s / 60 % 60:00}:{s % 60:00}";
        }
    }
}
