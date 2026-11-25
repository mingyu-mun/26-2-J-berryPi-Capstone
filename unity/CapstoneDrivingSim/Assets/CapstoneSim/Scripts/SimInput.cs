// SimInput.cs
// 키보드/게임패드 입력을 한 곳에서 읽는다.
// - 새 Input System 패키지(Unity 6 기본값)와 구 Input Manager 둘 다 지원한다.
// - 연속값(가속/브레이크/조향)은 FixedUpdate에서, 단발 키(마커·깜빡이 등)는 Update에서 읽는다.
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace CapstoneSim
{
    public static class SimInput
    {
        /// <summary>가속 페달 목표값 0~1</summary>
        public static float Throttle()
        {
            float v = 0f;
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && (kb.wKey.isPressed || kb.upArrowKey.isPressed)) v = 1f;
            var gp = Gamepad.current;
            if (gp != null) v = Mathf.Max(v, gp.rightTrigger.ReadValue());
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow)) v = 1f;
#endif
            return Mathf.Clamp01(v);
        }

        /// <summary>브레이크 페달 목표값 0~1</summary>
        public static float Brake()
        {
            float v = 0f;
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null && (kb.sKey.isPressed || kb.downArrowKey.isPressed || kb.spaceKey.isPressed)) v = 1f;
            var gp = Gamepad.current;
            if (gp != null) v = Mathf.Max(v, gp.leftTrigger.ReadValue());
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.Space)) v = 1f;
#endif
            return Mathf.Clamp01(v);
        }

        /// <summary>조향 목표값 -1(왼쪽)~+1(오른쪽). analog=true면 게임패드 스틱 값</summary>
        public static float Steer(out bool analog)
        {
            analog = false;
            float v = 0f;
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null)
            {
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) v -= 1f;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) v += 1f;
            }
            var gp = Gamepad.current;
            if (gp != null)
            {
                float x = gp.leftStick.x.ReadValue();
                if (Mathf.Abs(x) > 0.05f) { v = x; analog = true; }
            }
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) v -= 1f;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) v += 1f;
#endif
            return Mathf.Clamp(v, -1f, 1f);
        }

        /// <summary>게임패드 왼쪽 스틱 X (-1~1). 게임패드가 없으면 false</summary>
        public static bool GamepadSteer(out float x)
        {
            x = 0f;
#if ENABLE_INPUT_SYSTEM
            var gp = Gamepad.current;
            if (gp == null) return false;
            x = gp.leftStick.x.ReadValue();
            return true;
#else
            return false;
#endif
        }

        /// <summary>키보드 좌/우 조향 키 상태 (-1, 0, +1)</summary>
        public static float KeyboardSteer()
        {
            float v = 0f;
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb != null)
            {
                if (kb.aKey.isPressed || kb.leftArrowKey.isPressed) v -= 1f;
                if (kb.dKey.isPressed || kb.rightArrowKey.isPressed) v += 1f;
            }
#elif ENABLE_LEGACY_INPUT_MANAGER
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow)) v -= 1f;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) v += 1f;
#endif
            return v;
        }

        /// <summary>마우스 가로 위치를 화면 중앙 기준 -1~1 로 (화면 폭의 ±range 비율에서 끝값)</summary>
        public static bool MouseSteer(float range, out float x)
        {
            x = 0f;
            Vector2 p;
#if ENABLE_INPUT_SYSTEM
            var m = Mouse.current;
            if (m == null) return false;
            p = m.position.ReadValue();
#elif ENABLE_LEGACY_INPUT_MANAGER
            p = Input.mousePosition;
#else
            return false;
#endif
            float half = Mathf.Max(1f, Screen.width * range);
            x = Mathf.Clamp((p.x - Screen.width * 0.5f) / half, -1f, 1f);
            return true;
        }

        // ---------- 단발 입력 (Update에서만 호출) ----------

        /// <summary>T : 조향 입력 방식 전환 (자동 → 키보드 → 마우스 → 게임패드)</summary>
        public static bool SteerModeTogglePressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            return kb != null && kb.tKey.wasPressedThisFrame;
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.T);
#else
            return false;
#endif
        }

        /// <summary>L / 게임패드 Start : 기록 시작·정지</summary>
        public static bool ToggleRecordPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.lKey.wasPressedThisFrame) || (gp != null && gp.startButton.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.L);
#else
            return false;
#endif
        }

        /// <summary>M / 게임패드 A(South) : 동기화 마커</summary>
        public static bool MarkerPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.mKey.wasPressedThisFrame) || (gp != null && gp.buttonSouth.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.M);
#else
            return false;
#endif
        }

        /// <summary>Q / 게임패드 LB : 좌측 깜빡이 토글</summary>
        public static bool LeftSignalPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.qKey.wasPressedThisFrame) || (gp != null && gp.leftShoulder.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.Q);
#else
            return false;
#endif
        }

        /// <summary>E / 게임패드 RB : 우측 깜빡이 토글</summary>
        public static bool RightSignalPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.eKey.wasPressedThisFrame) || (gp != null && gp.rightShoulder.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.E);
#else
            return false;
#endif
        }

        /// <summary>C / 게임패드 Y(North) : 카메라 전환</summary>
        public static bool CameraTogglePressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.cKey.wasPressedThisFrame) || (gp != null && gp.buttonNorth.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.C);
#else
            return false;
#endif
        }

        /// <summary>R / 게임패드 Select : 차량을 차선 위로 복귀</summary>
        public static bool ResetPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.rKey.wasPressedThisFrame) || (gp != null && gp.selectButton.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.R);
#else
            return false;
#endif
        }

        /// <summary>H : HUD 표시 토글</summary>
        public static bool HudTogglePressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            return kb != null && kb.hKey.wasPressedThisFrame;
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.H);
#else
            return false;
#endif
        }

        /// <summary>G / 게임패드 X(West) : 조향 보조 세기 전환 (끔 → 약 → 강)</summary>
        public static bool AssistTogglePressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.gKey.wasPressedThisFrame) || (gp != null && gp.buttonWest.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.G);
#else
            return false;
#endif
        }

        /// <summary>K : 부주의 경고 시연 (몇 초간 강제로 켬)</summary>
        public static bool CarelessDemoPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            return kb != null && kb.kKey.wasPressedThisFrame;
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.K);
#else
            return false;
#endif
        }

        /// <summary>Esc / 게임패드 B : 뒤로 · 세션 메뉴</summary>
        public static bool BackPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current; var gp = Gamepad.current;
            return (kb != null && kb.escapeKey.wasPressedThisFrame) || (gp != null && gp.buttonEast.wasPressedThisFrame);
#elif ENABLE_LEGACY_INPUT_MANAGER
            return Input.GetKeyDown(KeyCode.Escape);
#else
            return false;
#endif
        }
    }
}
