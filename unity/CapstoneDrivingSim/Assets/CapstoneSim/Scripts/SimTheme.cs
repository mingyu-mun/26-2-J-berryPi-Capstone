// SimTheme.cs
// 메인 메뉴와 주행 화면이 함께 쓰는 IMGUI 디자인 시스템.
//   스타일: 게임 타이틀풍 — 굵은 기울임 글자 + 검은 외곽선, 라임색 배지, 두꺼운 테두리의 짙은 패널
//   · 해상도 독립: 세로 1080 기준 가상 좌표로 그리고 GUI.matrix 로 확대/축소 (Begin/End)
//   · 글꼴: Windows 맑은 고딕 (없으면 Unity 기본 글꼴). 기울임은 Unity 가 합성
//   · 도우미: Display(외곽선 큰 글자), Plate(배지), Panel, Button, Card, Chip, Bar, Radio, Segmented, Toggle, TextField, Timeline
using System.Collections.Generic;
using UnityEngine;

namespace CapstoneSim
{
    public static class SimTheme
    {
        // ---------------------------------------------------------------- 색
        public static readonly Color Ink = new Color(0.07f, 0.08f, 0.09f);               // 외곽선·테두리
        public static readonly Color PanelBg = new Color(0.11f, 0.13f, 0.14f, 0.9f);
        public static readonly Color PanelHi = new Color(0.17f, 0.20f, 0.21f, 0.95f);
        public static readonly Color Line = new Color(1f, 1f, 1f, 0.12f);
        public static readonly Color Text = new Color(0.97f, 0.98f, 0.95f);
        public static readonly Color Sub = new Color(0.74f, 0.77f, 0.71f);
        public static readonly Color Dim = new Color(0.52f, 0.55f, 0.50f);
        public static readonly Color Accent = new Color(0.55f, 0.78f, 0.20f);             // 라임
        public static readonly Color AccentDark = new Color(0.25f, 0.42f, 0.08f);
        public static readonly Color Berry = new Color(0.42f, 0.74f, 0.96f);              // 보조색 (하늘)
        public static readonly Color Green = new Color(0.55f, 0.82f, 0.24f);
        public static readonly Color Amber = new Color(1f, 0.69f, 0.13f);
        public static readonly Color Red = new Color(0.94f, 0.27f, 0.22f);
        public static readonly Color Gray = new Color(0.55f, 0.58f, 0.55f);

        public static Color LevelColor(int lv) => lv == 0 ? Green : lv == 1 ? Amber : lv == 2 ? Red : Gray;
        public static Color A(Color c, float a) { c.a = a; return c; }

        // ---------------------------------------------------------------- 좌표계
        /// <summary>현재 배율 (실제 픽셀 / 가상 픽셀)</summary>
        public static float S { get; private set; } = 1f;
        /// <summary>가상 화면 크기 (세로 1080 기준)</summary>
        public static float VW { get; private set; } = 1920f;
        public static float VH { get; private set; } = 1080f;

        /// <summary>OnGUI 첫 줄에서 호출. 반환값을 End 에 넘김</summary>
        public static Matrix4x4 Begin()
        {
            var old = GUI.matrix;
            S = Mathf.Clamp(Screen.height / 1080f, 0.5f, 2.5f);
            VW = Screen.width / S; VH = Screen.height / S;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(S, S, 1f));
            Init();
            return old;
        }
        public static void End(Matrix4x4 old) => GUI.matrix = old;

        // ---------------------------------------------------------------- 리소스
        static Font _font;
        static bool _init, _fontTried;
        static Texture2D _gradH, _gradV;
        static readonly Dictionary<long, GUIStyle> Styles = new Dictionary<long, GUIStyle>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { _init = false; _fontTried = false; _font = null; _gradH = _gradV = null; Styles.Clear(); }

        static void Init()
        {
            if (_init && _gradH != null) return;
            _init = true;
            if (!_fontTried)
            {
                _fontTried = true;
                try { _font = Font.CreateDynamicFontFromOSFont(new[] { "Malgun Gothic", "맑은 고딕", "Segoe UI", "Arial" }, 16); }
                catch { _font = null; }
            }
            _gradH = new Texture2D(256, 1, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.DontSave };
            for (int i = 0; i < 256; i++) _gradH.SetPixel(i, 0, new Color(1, 1, 1, 1f - i / 255f));
            _gradH.Apply();
            _gradV = new Texture2D(1, 256, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp, hideFlags = HideFlags.DontSave };
            for (int i = 0; i < 256; i++) _gradV.SetPixel(0, i, new Color(1, 1, 1, i / 255f));   // 아래(0)=투명 → 위=불투명
            _gradV.Apply();
            Styles.Clear();
        }

        /// <summary>글자 스타일 (크기·굵기·기울임·정렬·줄바꿈 별로 캐시)</summary>
        public static GUIStyle T(int size, bool bold = false, TextAnchor anchor = TextAnchor.MiddleLeft, bool wrap = false, bool italic = false)
        {
            long key = size * 10000L + (italic ? 1000 : 0) + (bold ? 100 : 0) + (int)anchor * 2 + (wrap ? 1 : 0);
            if (!Styles.TryGetValue(key, out var st) || st == null)
            {
                var fs = bold ? (italic ? FontStyle.BoldAndItalic : FontStyle.Bold) : (italic ? FontStyle.Italic : FontStyle.Normal);
                st = new GUIStyle(GUI.skin.label) { fontSize = size, fontStyle = fs, alignment = anchor, wordWrap = wrap, richText = true, clipping = TextClipping.Overflow };
                st.padding = new RectOffset(0, 0, 0, 0);
                st.margin = new RectOffset(0, 0, 0, 0);
                if (_font) st.font = _font;
                Styles[key] = st;
            }
            return st;
        }

        public static void Label(Rect r, string text, int size, Color c, bool bold = false, TextAnchor anchor = TextAnchor.MiddleLeft, bool wrap = false)
        {
            var st = T(size, bold, anchor, wrap);
            st.normal.textColor = c;
            GUI.Label(r, text, st);
        }

        public static float TextWidth(string text, int size, bool bold = false, bool italic = false) => T(size, bold, TextAnchor.MiddleLeft, false, italic).CalcSize(new GUIContent(text)).x;
        public static float DisplayWidth(string text, int size) => TextWidth(text, size, true, true) + Mathf.Max(2f, size * 0.08f) * 2f;

        static readonly Vector2[] Dirs = { new Vector2(-1, 0), new Vector2(1, 0), new Vector2(0, -1), new Vector2(0, 1), new Vector2(-0.7f, -0.7f), new Vector2(0.7f, -0.7f), new Vector2(-0.7f, 0.7f), new Vector2(0.7f, 0.7f) };

        /// <summary>타이틀 글자: 굵은 기울임 + 검은 외곽선 + 아래 그림자</summary>
        public static void Display(Rect r, string text, int size, Color fill, TextAnchor anchor = TextAnchor.MiddleLeft, float stroke = -1f)
        {
            if (string.IsNullOrEmpty(text)) return;
            float k = stroke >= 0 ? stroke : Mathf.Max(1.5f, size * 0.07f);
            var st = T(size, true, anchor, false, true);
            bool rt = st.richText; st.richText = false;
            if (k > 0)
            {
                st.normal.textColor = A(Ink, 0.45f);
                GUI.Label(new Rect(r.x + k * 0.6f, r.y + k * 1.8f, r.width, r.height), text, st);
                st.normal.textColor = Ink;
                foreach (var d in Dirs) GUI.Label(new Rect(r.x + d.x * k, r.y + d.y * k, r.width, r.height), text, st);
            }
            st.normal.textColor = fill;
            GUI.Label(r, text, st);
            st.richText = rt;
        }

        // ---------------------------------------------------------------- 도형
        public static void Fill(Rect r, Color c, float radius = 0f)
        {
            if (c.a <= 0f) return;
            GUI.DrawTexture(r, Texture2D.whiteTexture, ScaleMode.StretchToFill, true, 0f, c, 0f, radius);
        }
        public static void Outline(Rect r, Color c, float width = 1f, float radius = 0f)
        {
            GUI.DrawTexture(r, Texture2D.whiteTexture, ScaleMode.StretchToFill, true, 0f, c, width, radius);
        }
        static Rect Inset(Rect r, float d) => new Rect(r.x + d, r.y + d, r.width - 2 * d, r.height - 2 * d);

        /// <summary>짙은 패널: 반투명 바탕 + 검은 2px 테두리 + 안쪽 밝은 선</summary>
        public static void Panel(Rect r, float radius = 10f, float alpha = 0.9f)
        {
            radius = Mathf.Min(radius, 12f);
            Fill(r, A(PanelBg, alpha), radius);
            Outline(r, A(Ink, 0.9f), 2f, radius);
            Outline(Inset(r, 2f), Line, 1f, Mathf.Max(0, radius - 2f));
        }

        /// <summary>배지 (참고 디자인의 라임색 숫자 판): 색 바탕 + 검은 테두리 + 안쪽 흰 선 + 외곽선 글자</summary>
        public static void Plate(Rect r, string text, Color bg, int size = 0, Color? fg = null)
        {
            float rad = Mathf.Min(10f, r.height * 0.24f);
            Fill(new Rect(r.x + 1, r.y + 3, r.width, r.height), A(Ink, 0.35f), rad);           // 그림자
            Fill(r, bg, rad);
            Fill(new Rect(r.x + 3, r.y + 3, r.width - 6, r.height * 0.42f), new Color(1, 1, 1, 0.14f), Mathf.Max(0, rad - 3));   // 윗면 광택
            Outline(r, Ink, 2f, rad);
            Outline(Inset(r, 3f), new Color(1, 1, 1, 0.75f), 1.5f, Mathf.Max(0, rad - 3));
            if (!string.IsNullOrEmpty(text))
            {
                int s = size > 0 ? size : Mathf.RoundToInt(r.height * 0.52f);
                Display(r, text, s, fg ?? Text, TextAnchor.MiddleCenter);
            }
        }

        /// <summary>왼쪽(불투명)→오른쪽(투명) 그라데이션</summary>
        public static void GradientH(Rect r, Color c) { var o = GUI.color; GUI.color = c; GUI.DrawTexture(r, _gradH); GUI.color = o; }
        /// <summary>아래(불투명)→위(투명) 그라데이션</summary>
        public static void GradientUp(Rect r, Color c)
        {
            var o = GUI.color; GUI.color = c;
            GUI.DrawTextureWithTexCoords(r, _gradV, new Rect(0, 1, 1, -1));
            GUI.color = o;
        }

        /// <summary>섹션 머리말: 라임 막대 + 굵은 기울임 글자</summary>
        public static void Header(Rect r, string text, Color? c = null)
        {
            Fill(new Rect(r.x, r.y + r.height * 0.5f - 9, 6, 18), c ?? Accent, 1.5f);
            Outline(new Rect(r.x, r.y + r.height * 0.5f - 9, 6, 18), Ink, 1.5f, 1.5f);
            var st = T(17, true, TextAnchor.MiddleLeft, false, true);
            st.normal.textColor = Text;
            GUI.Label(new Rect(r.x + 16, r.y, r.width - 16, r.height), text, st);
        }

        // ---------------------------------------------------------------- 컨트롤
        public enum Kind { Normal, Primary, Ghost, Danger }

        static bool Hover(Rect r) => r.Contains(Event.current.mousePosition);

        public static bool Button(Rect r, string text, int size = 17, Kind kind = Kind.Normal, bool enabled = true)
        {
            bool hov = enabled && Hover(r);
            if (enabled && (kind == Kind.Primary || kind == Kind.Danger))
            {
                Color bg = kind == Kind.Primary ? Accent : Red;
                if (hov) bg = Color.Lerp(bg, Color.white, 0.15f);
                Plate(r, null, bg);
                Display(r, text, size + 2, Text, TextAnchor.MiddleCenter);
                return GUI.Button(r, GUIContent.none, GUIStyle.none);
            }
            Color fill = kind == Kind.Ghost ? (hov ? new Color(1, 1, 1, 0.08f) : new Color(0, 0, 0, 0)) : hov ? PanelHi : new Color(0.15f, 0.17f, 0.18f, 0.92f);
            if (!enabled) fill = new Color(0.12f, 0.13f, 0.14f, 0.7f);
            Fill(r, fill, 8f);
            if (kind != Kind.Ghost || hov) Outline(r, A(Ink, 0.9f), 2f, 8f);
            if (kind != Kind.Ghost) Outline(Inset(r, 2f), hov ? A(Accent, 0.7f) : Line, 1f, 6f);
            var st = T(size, true, TextAnchor.MiddleCenter, false, true);
            st.normal.textColor = enabled ? Text : Dim;
            GUI.Label(r, text, st);
            return enabled && GUI.Button(r, GUIContent.none, GUIStyle.none);
        }

        /// <summary>큰 카드형 버튼 (메인 메뉴 모드 선택)</summary>
        public static bool Card(Rect r, string tag, string title, string desc, string chip, Color chipColor, bool selected)
        {
            bool hov = Hover(r);
            Fill(r, hov || selected ? PanelHi : PanelBg, 10f);
            Outline(r, A(Ink, 0.9f), 2f, 10f);
            Outline(Inset(r, 2f), hov || selected ? A(Accent, 0.9f) : Line, hov || selected ? 2f : 1f, 8f);
            Label(new Rect(r.x + 26, r.y + 18, r.width - 52, 20), tag, 13, Accent, true);
            Display(new Rect(r.x + 26, r.y + 40, r.width - 52, 44), title, 32, Text);
            Label(new Rect(r.x + 26, r.y + 92, r.width - 100, r.height - 100), desc, 15, Sub, false, TextAnchor.UpperLeft, true);
            if (!string.IsNullOrEmpty(chip))
            {
                float w = TextWidth(chip, 12, true) + 20;
                Chip(new Rect(r.xMax - w - 20, r.y + 18, w, 24), chip, chipColor);
            }
            Display(new Rect(r.xMax - 50, r.y + r.height * 0.5f - 18, 30, 36), "›", 34, hov || selected ? Accent : Dim, TextAnchor.MiddleCenter);
            return GUI.Button(r, GUIContent.none, GUIStyle.none);
        }

        public static void Chip(Rect r, string text, Color c, bool solid = false)
        {
            Fill(r, solid ? c : A(c, 0.18f), 5f);
            Outline(r, solid ? Ink : A(c, 0.75f), 1.5f, 5f);
            Label(r, text, Mathf.RoundToInt(r.height * 0.54f), solid ? Ink : c, true, TextAnchor.MiddleCenter);
        }

        /// <summary>상태 점 + 글자</summary>
        public static void Dot(Rect r, string text, Color c, int size = 14)
        {
            var d = new Rect(r.x, r.y + r.height * 0.5f - 5, 10, 10);
            Fill(d, c, 5f); Outline(d, Ink, 1.5f, 5f);
            Label(new Rect(r.x + 16, r.y, r.width - 16, r.height), text, size, Sub);
        }

        public static void Bar(Rect r, float v, Color c, Color? bg = null)
        {
            float rad = Mathf.Min(4f, r.height * 0.5f);
            Fill(r, bg ?? new Color(0, 0, 0, 0.45f), rad);
            if (v > 0f) Fill(new Rect(r.x, r.y, Mathf.Max(r.height, r.width * Mathf.Clamp01(v)), r.height), c, rad);
            Outline(r, Ink, 1.5f, rad);
        }

        public static bool Radio(Rect r, string title, string sub, bool on, bool enabled = true)
        {
            bool hov = enabled && Hover(r);
            Fill(r, on ? PanelHi : hov ? new Color(1, 1, 1, 0.06f) : new Color(0, 0, 0, 0.22f), 8f);
            Outline(r, on ? A(Accent, 0.9f) : Line, on ? 2f : 1f, 8f);
            var c = new Rect(r.x + 14, r.y + 16, 20, 20);
            Fill(c, new Color(0, 0, 0, 0.4f), 10f);
            Outline(c, on ? Accent : Sub, 2f, 10f);
            if (on) { Fill(new Rect(c.x + 5, c.y + 5, 10, 10), Accent, 5f); }
            Label(new Rect(r.x + 46, r.y + 9, r.width - 56, 22), title, 16, enabled ? Text : Dim, true);
            if (!string.IsNullOrEmpty(sub)) Label(new Rect(r.x + 46, r.y + 32, r.width - 56, r.height - 38), sub, 13, Sub, false, TextAnchor.UpperLeft, true);
            return enabled && GUI.Button(r, GUIContent.none, GUIStyle.none);
        }

        public static int Segmented(Rect r, string[] items, int sel, bool[] enabled = null)
        {
            Fill(r, new Color(0, 0, 0, 0.35f), 8f);
            Outline(r, A(Ink, 0.9f), 2f, 8f);
            float w = r.width / items.Length;
            for (int i = 0; i < items.Length; i++)
            {
                var c = new Rect(r.x + i * w + 4, r.y + 4, w - 8, r.height - 8);
                bool en = enabled == null || enabled[i];
                bool hov = Hover(c);
                if (i == sel) { Fill(c, Accent, 6f); Outline(c, Ink, 1.5f, 6f); }
                else if (hov) Fill(c, new Color(1, 1, 1, 0.07f), 6f);
                var st = T(14, true, TextAnchor.MiddleCenter, false, i == sel);
                st.normal.textColor = i == sel ? Ink : en ? Text : Dim;
                GUI.Label(c, items[i], st);
                if (GUI.Button(c, GUIContent.none, GUIStyle.none)) sel = i;
            }
            return sel;
        }

        public static bool Toggle(Rect r, string text, bool v, string sub = null)
        {
            var sw = new Rect(r.x, r.y + 4, 48, 26);
            Fill(sw, v ? Accent : new Color(0, 0, 0, 0.45f), 6f);
            Outline(sw, Ink, 2f, 6f);
            var knob = new Rect(v ? sw.xMax - 23 : sw.x + 3, sw.y + 3, 20, 20);
            Fill(knob, Color.white, 4f); Outline(knob, Ink, 1.5f, 4f);
            Label(new Rect(r.x + 60, r.y + 2, r.width - 60, 24), text, 16, Text, true);
            if (!string.IsNullOrEmpty(sub)) Label(new Rect(r.x + 60, r.y + 26, r.width - 60, r.height - 26), sub, 13, Sub, false, TextAnchor.UpperLeft, true);
            if (GUI.Button(r, GUIContent.none, GUIStyle.none)) v = !v;
            return v;
        }

        public static string TextField(Rect r, string label, string value)
        {
            Label(new Rect(r.x, r.y, r.width, 18), label, 12, Sub, true);
            var f = new Rect(r.x, r.y + 20, r.width, r.height - 20);
            Fill(f, new Color(0, 0, 0, 0.4f), 6f);
            Outline(f, A(Ink, 0.9f), 2f, 6f);
            Outline(Inset(f, 2f), Line, 1f, 4f);
            var st = T(16, false, TextAnchor.MiddleLeft);
            var ts = new GUIStyle(GUI.skin.textField) { font = st.font, fontSize = 16, alignment = TextAnchor.MiddleLeft, padding = new RectOffset(10, 10, 0, 0) };
            ts.normal.background = ts.focused.background = ts.hover.background = null;
            ts.normal.textColor = ts.focused.textColor = ts.hover.textColor = Text;
            return GUI.TextField(f, value ?? "", ts);
        }

        /// <summary>재생 시작 위치 막대: 라벨 구간을 색으로 칠하고, 클릭·드래그로 위치를 고름</summary>
        public static float Timeline(Rect r, float value, float max, IList<(float s, float e, string label)> segs, int id)
        {
            var track = new Rect(r.x, r.y + r.height * 0.5f - 5, r.width, 10);
            Fill(track, new Color(0, 0, 0, 0.45f), 4f);
            if (max > 0 && segs != null)
                foreach (var g in segs)
                {
                    float x0 = r.x + r.width * Mathf.Clamp01(g.s / max), x1 = r.x + r.width * Mathf.Clamp01(g.e / max);
                    Color c = g.label.Contains("졸음") ? Red : g.label.Contains("주의") || g.label.Contains("피로") ? Amber : g.label.Contains("정상") ? Green : Gray;
                    Fill(new Rect(x0 + 1, r.yMax - 7, Mathf.Max(1, x1 - x0 - 2), 6), A(c, 0.9f), 2f);
                }
            var e = Event.current;
            int ctl = GUIUtility.GetControlID(id, FocusType.Passive, r);
            if ((e.type == EventType.MouseDown && r.Contains(e.mousePosition)) || (e.type == EventType.MouseDrag && GUIUtility.hotControl == ctl))
            {
                GUIUtility.hotControl = ctl;
                value = Mathf.Clamp01((e.mousePosition.x - r.x) / r.width) * max;
                e.Use();
            }
            if (e.type == EventType.MouseUp && GUIUtility.hotControl == ctl) { GUIUtility.hotControl = 0; e.Use(); }
            float hx = r.x + r.width * (max > 0 ? Mathf.Clamp01(value / max) : 0f);
            Fill(new Rect(track.x, track.y, hx - track.x, track.height), Accent, 4f);
            Outline(track, Ink, 1.5f, 4f);
            var knob = new Rect(hx - 10, r.y + r.height * 0.5f - 12, 20, 24);
            Fill(knob, Color.white, 5f); Outline(knob, Ink, 2f, 5f);
            return value;
        }

        public static string MmSs(float s) => s < 0 ? "–" : $"{(int)s / 60:00}:{(int)s % 60:00}";
    }
}
