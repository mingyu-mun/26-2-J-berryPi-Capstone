// SimBootstrap.cs
// 빈 GameObject에 이 컴포넌트 하나만 붙이고 Play 하면
// 도로·차량·카메라·조명·HUD·로거가 모두 코드로 생성된다.
// 상단 메뉴 [Capstone > 주행 시뮬레이터 씬 구성] 으로도 추가할 수 있다.
//
// [외형 교체] Inspector 의 'Car Model Prefab' 에 에셋 스토어 차량 프리팹을,
//            'Road Material' 에 아스팔트 머티리얼을 끌어다 놓으면 기본 도형 대신 사용한다.
using UnityEngine;

namespace CapstoneSim
{
    public class SimBootstrap : MonoBehaviour
    {
        [Header("세션 (Play 전에 입력)")]
        public string participantId = "P01";
        public string sessionNote = "";
        [Tooltip("CSV 기록 주기 [Hz]")] public float logRateHz = 50f;
        [Tooltip("비워 두면 <프로젝트 폴더>/SimLogs")] public string outputFolder = "";

        [Header("조향")]
        [Tooltip("Auto: 키보드/게임패드 자동 전환. Mouse: 마우스 가로 위치로 조향 (T 키로도 전환)")]
        public SteerMode steerMode = SteerMode.Auto;

        [Header("차량 외형 (선택) — 비워 두면 기본 도형")]
        [Tooltip("에셋 스토어 등에서 받은 차량 프리팹 (외형만 사용, 물리는 이 시뮬레이터 것을 씀)")]
        public GameObject carModelPrefab;
        [Tooltip("모델 앞쪽이 +Z 방향이 아닐 때 회전 보정 [deg] (보통 0, 90, -90, 180 중 하나)")]
        public float carModelYawOffset = 0f;
        [Tooltip("자동 인식 실패 시 바퀴 오브젝트 이름을 FL, FR, RL, RR 순서로 입력")]
        public string[] modelWheelNames = new string[4];
        [Tooltip("바퀴가 차체와 별도 프리팹으로 제공되는 에셋용 (예: SportCar 의 Sport_Wheel_1). 각 바퀴 자리에 붙임")]
        public GameObject modelWheelPrefab;
        [Tooltip("자동 인식 실패 시 실내 핸들 오브젝트 이름")]
        public string modelSteeringWheelName = "";
        [Tooltip("실내 핸들이 반대로 돌면 체크")] public bool invertSteeringWheelVisual = false;
        [Tooltip("운전석 시점 위치 미세 조정 (자동 계산값에 더함, 차량 로컬 m) — x:좌우 y:높이 z:앞뒤")]
        public Vector3 driverEyeAdjust = Vector3.zero;

        [Header("도로·환경 외형 (선택)")]
        [Tooltip("노면 머티리얼 (차선 표시가 없는 아스팔트 권장). 비우면 코드로 만든 아스팔트 텍스처")]
        public Material roadMaterial;
        [Tooltip("지면 머티리얼. 비우면 코드로 만든 잔디 텍스처")]
        public Material groundMaterial;
        public bool guardrails = true;
        public bool trees = true;
        public bool fog = true;
        [Tooltip("야간 주행 시나리오 (조명 어둡게 + 전조등)")] public bool night = false;

        [Header("생체신호 Replay · AI 실시간 판정")]
        public bool enableBioAi = true;
        [Tooltip("실측 기록 세션 폴더 (meta.json + ecg.csv + ppg.csv [+ labels.csv]). 비우면 <프로젝트>/SimLogs/bio/rec_20261118")]
        public string bioSessionDir = "";
        [Tooltip("기록의 몇 초 지점부터 재생할지 [s]")] public float bioStartOffsetSec = 0f;
        [Tooltip("시작 위치 앞 기록을 AI 서버에 미리 빠르게 넣음 → 기준 수집(약 7~8분)을 건너뛰고 시작 (화면에 표시됨)")]
        public bool aiPrerollBeforeStart = true;
        [Tooltip("causal = 실시간(과거 데이터만, 지연 없음) / exact = 저장소 batch 와 같은 계산(판정이 약 140초 늦음)")]
        public string aiMode = "causal";
        [Tooltip("Play 시 Tools/jberry_ai/server.py (저장소 S1-EP 모델) 자동 실행")] public bool aiAutoLaunchServer = true;
        [Tooltip("비우면 Tools/jberry_ai/.venv 의 Python 사용")] public string pythonExe = "";
        [Tooltip("웹보드 주소 (비우면 전송 안 함). 예: http://127.0.0.1:8000")] public string aiWebboardUrl = "";
        [Tooltip("개인 기준(baseline) 측정 파일 세션 폴더. 비우면 재생 기록 앞부분에서 자동 수집")] public string baselineSessionDir = "";
        [Tooltip("Play 시 메인 메뉴부터 시작 (끄면 위 설정으로 바로 리플레이 주행)")] public bool showMainMenu = true;

        [Header("물리")]
        [Tooltip("물리 갱신 주기 [Hz]. 차량 안정성을 위해 100 Hz 권장")] public float physicsHz = 100f;

        [Header("트랙")]
        public float straightLength = 1500f;
        public float curveRadius = 300f;
        [Tooltip("시작 차로 (1 = 1차로/왼쪽, 2 = 2차로/오른쪽)")] public int startLane = 2;

        public CarController Car { get; private set; }
        public TrackBuilder Track { get; private set; }
        public DrivingDataLogger Logger { get; private set; }
        public FollowCamera Cam { get; private set; }
        public BioSource Bio { get; private set; }
        public AiLink Ai { get; private set; }
        public SimHUD Hud { get; private set; }
        public SimPanels Panels { get; private set; }
        public MainMenu Menu { get; private set; }
        public LiveBioSource Live { get; private set; }
        public bool InSession { get; private set; }
        public SessionConfig Session { get; private set; }
        public float SessionStartRealtime { get; private set; }
        FollowCamera.Mode _driveCamMode = FollowCamera.Mode.Driver;
        Vector3 _spawnPos; Quaternion _spawnRot;
        CarAutopilot _autopilot;

        Bounds _carBounds = new Bounds(new Vector3(0f, 0.75f, 0f), new Vector3(1.8f, 1.5f, 4.4f));

        // 서스펜션 공통값
        const float SuspDist = 0.25f, SuspTarget = 0.5f, Spring = 70000f, Damper = 7000f, CarMass = 1500f;

        void Awake()
        {
            Time.fixedDeltaTime = 1f / Mathf.Max(20f, physicsHz);

            // ---------- 머티리얼 ----------
            var asphalt = ProceduralTextures.Asphalt();
            var grass = ProceduralTextures.Grass();
            var mats = new TrackBuilder.Materials
            {
                road = roadMaterial ? MaterialFixer.Convert(roadMaterial) : MakeMat(new Color(0.9f, 0.9f, 0.9f), 0.25f, asphalt),
                line = MakeMat(new Color(0.92f, 0.92f, 0.88f), 0.3f),
                pole = MakeMat(new Color(0.85f, 0.85f, 0.85f), 0.4f),
                rail = MakeMat(new Color(0.62f, 0.64f, 0.66f), 0.75f, null, 0.8f),
                trunk = MakeMat(new Color(0.30f, 0.22f, 0.15f), 0.05f),
                leaf = MakeMat(new Color(0.16f, 0.30f, 0.14f), 0.05f),
            };

            // ---------- 트랙 ----------
            var trackGo = new GameObject("Track");
            Track = trackGo.AddComponent<TrackBuilder>();
            Track.straightLength = straightLength;
            Track.curveRadius = curveRadius;
            Track.buildGuardrails = guardrails;
            Track.buildTrees = trees;
            Track.Build(mats);

            // ---------- 지면 ----------
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane); // 기본 10 m × 10 m
            ground.name = "Ground";
            var gs = Track.GroundSize();
            ground.transform.localScale = new Vector3(gs.x / 10f, 1f, gs.y / 10f);
            var gm = groundMaterial ? MaterialFixer.Convert(groundMaterial) : MakeMat(Color.white, 0.05f, grass);
            if (!groundMaterial) gm.mainTextureScale = new Vector2(gs.x / 8f, gs.y / 8f); // 8 m 마다 반복
            ground.GetComponent<MeshRenderer>().sharedMaterial = gm;

            // ---------- 차량 ----------
            Car = BuildCar();
            Car.steerMode = steerMode;
            Car.invertSteeringWheelVisual = invertSteeringWheelVisual;
            Track.SpawnPose(0f, startLane, out var spawnPos, out var spawnRot);
            // Rigidbody 보간(interpolation)이 켜져 있으면 transform 만 옮겨서는 안 되고 rb 위치도 함께 옮겨야 함
            Car.Teleport(spawnPos + Vector3.up * 0.3f, spawnRot);
            Car.Initialize();
            _spawnPos = spawnPos; _spawnRot = spawnRot;

            // ---------- 카메라 ----------
            var cam = Camera.main;
            if (!cam)
            {
                var cgo = new GameObject("Main Camera") { tag = "MainCamera" };
                cam = cgo.AddComponent<Camera>();
                cgo.AddComponent<AudioListener>();
            }
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 4000f;
            cam.fieldOfView = 65f;
            Cam = cam.gameObject.GetComponent<FollowCamera>(); // UnityEngine.Object 에 ?? 금지
            if (!Cam) Cam = cam.gameObject.AddComponent<FollowCamera>();
            Cam.target = Car.transform;
            Cam.driverEye = _driverEye;
            Cam.distance = Mathf.Max(6.5f, _carBounds.size.z * 1.4f);

            // ---------- 조명/환경 ----------
            Light sun = null;
            foreach (var l in FindObjectsByType<Light>(FindObjectsSortMode.None))
                if (l.type == LightType.Directional) { sun = l; break; }
            if (!sun)
            {
                sun = new GameObject("Directional Light").AddComponent<Light>();
                sun.type = LightType.Directional;
                sun.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            }
            sun.shadows = LightShadows.Soft;
            if (night)
            {
                sun.intensity = 0.05f;
                RenderSettings.ambientIntensity = 0.15f;
                RenderSettings.fogColor = new Color(0.02f, 0.02f, 0.04f);
                foreach (float x in new[] { -0.6f, 0.6f })
                {
                    var head = new GameObject("Headlight").AddComponent<Light>();
                    head.type = LightType.Spot; head.range = 120f; head.spotAngle = 55f; head.intensity = 5f;
                    head.shadows = LightShadows.None;
                    head.transform.SetParent(Car.transform, false);
                    head.transform.localPosition = new Vector3(x, _carBounds.size.y * 0.45f, _carBounds.max.z + 0.1f);
                    head.transform.localRotation = Quaternion.Euler(4f, 0f, 0f);
                }
            }
            if (fog)
            {
                RenderSettings.fog = true;
                RenderSettings.fogMode = FogMode.ExponentialSquared;
                RenderSettings.fogDensity = night ? 0.012f : 0.0022f;
                if (!night) RenderSettings.fogColor = new Color(0.75f, 0.8f, 0.85f);
            }

            // ---------- 로거 / HUD ----------
            Logger = gameObject.AddComponent<DrivingDataLogger>();
            Logger.participantId = participantId;
            Logger.sessionNote = sessionNote;
            Logger.logRateHz = logRateHz;
            Logger.outputFolder = outputFolder;
            Logger.vehicleWidth = Mathf.Clamp(_carBounds.size.x, 1.6f, 2.1f);
            Logger.car = Car;
            Logger.track = Track;

            Hud = gameObject.AddComponent<SimHUD>();
            Hud.car = Car;
            Hud.logger = Logger;
            Hud.sim = this;

            // ---------- 메인 메뉴 / 세션 ----------
            // 자동 테스트: menu=<home|replay|live> 인자가 있으면 그 메뉴 화면에 머물고, 없으면 바로 리플레이 세션
            Menu = gameObject.AddComponent<MainMenu>();
            Menu.sim = this;
            string menuArg = AutoTestRunner.Active ? AutoTestRunner.Arg("menu") : null;
            if (!enableBioAi) EnterDriveOnly();
            else if (AutoTestRunner.Active && string.IsNullOrEmpty(menuArg))
            {
                var c = SessionConfig.FromInspector(this);
                c.live = AutoTestRunner.Arg("mode") == "live";          // 자동 테스트: mode=live → 실측(UDP) 세션
                StartSession(c);
            }
            else if (showMainMenu || !string.IsNullOrEmpty(menuArg))
            {
                EnterMenu(menuArg);
                if (AutoTestRunner.Active && AutoTestRunner.Arg("live_listen") == "1") EnsureLive().StartListening();
            }
            else StartSession(SessionConfig.FromInspector(this));
            // 자동 테스트: pause_at=<초> 이면 그 시각에 세션 메뉴를 띄워 스크린샷에 담는다
            if (AutoTestRunner.Active && float.TryParse(AutoTestRunner.Arg("pause_at"), System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float pauseAt))
                StartCoroutine(PauseLater(pauseAt));
        }

        System.Collections.IEnumerator PauseLater(float t)
        {
            yield return new WaitForSecondsRealtime(t);
            if (InSession) Menu.ShowPause();
        }

        // ================================================================== 세션 관리
        void EnterDriveOnly()
        {
            InSession = true;
            Hud.enabled = true;
        }

        /// <summary>메인 메뉴로: 차량 정차, 쇼케이스 카메라, HUD 숨김</summary>
        public void EnterMenu(string page = null)
        {
            InSession = false;
            if (Cam && Cam.mode != FollowCamera.Mode.Showcase) { _driveCamMode = Cam.mode; Cam.mode = FollowCamera.Mode.Showcase; }
            // 메뉴 배경: 차로를 따라 달리는 모의 주행 (정지 + 브레이크 유지는 후진으로 이어지므로 쓰지 않음)
            if (!_autopilot) { _autopilot = gameObject.AddComponent<CarAutopilot>(); _autopilot.car = Car; _autopilot.track = Track; }
            _autopilot.enabled = true;
            Logger.allowInput = false;
            Hud.enabled = false;
            Menu.Show(page);
        }

        /// <summary>메뉴에서 '주행 시작': 생체 입력(재생/실시간) + AI 연결 + 화면 패널을 만든다</summary>
        public void StartSession(SessionConfig c)
        {
            if (InSession && Ai) EndSession(false);
            Session = c;
            Menu.Hide();
            if (Cam) Cam.mode = _driveCamMode;
            // 모의 주행을 끄고 맵 첫 지점(직선 구간 시작, 시작 차로)으로 순간이동해 정지 상태로 시작
            if (_autopilot) _autopilot.enabled = false;
            ResetCarToStart();
            Car.externalControl = false;
            Logger.allowInput = true;
            Logger.participantId = string.IsNullOrWhiteSpace(c.participantId) ? participantId : c.participantId;
            Hud.enabled = true;

            if (c.live)
            {
                var lb = EnsureLive();
                c.ApplyTo(lb);
                Bio = lb;
            }
            else
            {
                var br = gameObject.AddComponent<BioReplay>();
                br.sessionDir = c.replayDir;
                br.startOffsetSec = c.startOffset;
                Bio = br;
            }
            Ai = gameObject.AddComponent<AiLink>();
            Ai.autoLaunchServer = aiAutoLaunchServer;
            Ai.pythonExe = pythonExe;
            Ai.mode = c.aiMode;
            Ai.prerollBeforeStart = c.preroll;
            Ai.baselineDir = c.baselineDir ?? "";
            Ai.webboardUrl = aiWebboardUrl;
            Ai.bio = Bio; Ai.car = Car; Ai.track = Track; Ai.participantId = Logger.participantId;
            AutoTestRunner.ApplyOverrides(Bio, Ai);   // 자동 테스트 중이면 재생 위치·모드·기준 파일 덮어쓰기
            Bio.Begin();
            Panels = gameObject.AddComponent<SimPanels>();
            Panels.bio = Bio; Panels.ai = Ai; Panels.sim = this;
            Logger.ai = Ai; Logger.bio = Bio;
            InSession = true;
            SessionStartRealtime = Time.realtimeSinceStartup;
        }

        /// <summary>세션 종료: 기록 정지, AI 서버 종료, 차량 원위치 후 메뉴로</summary>
        public void EndSession(bool toMenu = true)
        {
            Logger.StopRecording();
            Logger.ai = null; Logger.bio = null;
            if (Panels) Destroy(Panels);
            if (Ai) Destroy(Ai);                     // OnDisable 에서 Python 서버 종료
            if (Bio && Bio != Live) Destroy(Bio);
            if (Live) { Destroy(Live); Live = null; }
            Panels = null; Ai = null; Bio = null;
            InSession = false;
            ResetCarToStart();
            if (toMenu) EnterMenu("home");
        }

        void ResetCarToStart()
        {
            Car.Teleport(_spawnPos + Vector3.up * 0.3f, _spawnRot);
            Car.SetForwardSpeed(0f);
            Car.extThrottle = 0f; Car.extBrake = 0f; Car.extSteerWheelDeg = 0f;
            Car.TurnSignal = 0;
            if (Logger) Logger.LogEvent("RESET", "session_start");
        }

        /// <summary>실측 모드 수신기 (메뉴의 수신 시험과 세션이 같은 것을 쓴다)</summary>
        public LiveBioSource EnsureLive()
        {
            if (!Live) Live = gameObject.AddComponent<LiveBioSource>();
            return Live;
        }

        void Update()
        {
            if (!Car) return;
            if (!InSession) return;                       // 메뉴에서는 MainMenu 가 입력을 받는다
            if (Menu && Menu.Visible) return;

            if (SimInput.CameraTogglePressed() && Cam) Cam.ToggleMode();

            if (SimInput.LeftSignalPressed())
            {
                Car.TurnSignal = Car.TurnSignal == -1 ? 0 : -1;
                Logger.LogEvent("TURN_SIGNAL", Car.TurnSignal.ToString());
            }
            if (SimInput.RightSignalPressed())
            {
                Car.TurnSignal = Car.TurnSignal == 1 ? 0 : 1;
                Logger.LogEvent("TURN_SIGNAL", Car.TurnSignal.ToString());
            }

            if (SimInput.ResetPressed())
            {
                var ts = Track.Query(Car.transform.position, Car.transform.forward);
                int lane = ts.lane == 0 ? 2 : ts.lane;
                Track.SpawnPose(ts.s, lane, out var p, out var r);
                Car.Teleport(p + Vector3.up * 0.3f, r);
                Car.TurnSignal = 0;
                Logger.LogEvent("RESET", ts.s.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
            }
        }

        // ------------------------------------------------------------------
        // 차량은 원점·무회전 상태로 조립한 뒤 스폰 위치로 옮긴다 (모델 크기 계산을 단순하게 하려고)
        CarController BuildCar()
        {
            var root = new GameObject("EgoCar");
            var rb = root.AddComponent<Rigidbody>();
            rb.mass = CarMass;
            rb.linearDamping = 0.02f;
            rb.angularDamping = 0.5f;
            rb.interpolation = RigidbodyInterpolation.Interpolate;
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

            var wc = new Vector3[4];
            var vis = new Transform[4];
            float radius;
            Transform steerVisual = null;
            Vector3 eye;

            FittedCarModel fit = carModelPrefab
                ? CarModelFitter.Fit(carModelPrefab, root.transform, carModelYawOffset, 4.6f, modelWheelNames, modelSteeringWheelName, modelWheelPrefab)
                : null;

            if (fit != null)
            {
                foreach (var line in fit.log) Debug.Log("[CarModel] " + line);
                _carBounds = fit.bounds;
                radius = fit.wheelRadius;
                for (int i = 0; i < 4; i++) { wc[i] = fit.wheelCenters[i]; vis[i] = fit.wheels[i]; }
                steerVisual = fit.steeringWheel;
                eye = fit.driverEye;
            }
            else
            {
                var body = MakeMat(new Color(0.15f, 0.35f, 0.75f), 0.6f);
                var glass = MakeMat(new Color(0.1f, 0.12f, 0.15f), 0.8f);
                var tire = MakeMat(new Color(0.05f, 0.05f, 0.05f), 0.1f);
                var trim = MakeMat(new Color(0.08f, 0.08f, 0.09f), 0.3f);

                Prim(PrimitiveType.Cube, "Body", root.transform, new Vector3(0f, 0.6f, 0f), Quaternion.identity, new Vector3(1.8f, 0.6f, 4.4f), body);
                Prim(PrimitiveType.Cube, "Cabin", root.transform, new Vector3(0f, 1.15f, -0.25f), Quaternion.identity, new Vector3(1.6f, 0.5f, 2.2f), glass);
                // 실내 핸들 (원판 + 스포크 — 스포크가 있어야 회전이 보임)
                var sw = new GameObject("SteeringWheel").transform;
                sw.SetParent(root.transform, false);
                sw.localPosition = new Vector3(-0.37f, 1.0f, 0.45f);
                sw.localRotation = Quaternion.Euler(70f, 0f, 0f); // 로컬 Y축이 핸들 회전축
                Prim(PrimitiveType.Cylinder, "Rim", sw, Vector3.zero, Quaternion.identity, new Vector3(0.38f, 0.012f, 0.38f), trim);
                Prim(PrimitiveType.Cube, "Spoke", sw, new Vector3(0f, 0.014f, -0.1f), Quaternion.identity, new Vector3(0.035f, 0.01f, 0.2f), body);
                steerVisual = sw;

                radius = 0.35f;
                wc[0] = new Vector3(-0.8f, radius, 1.4f); wc[1] = new Vector3(0.8f, radius, 1.4f);
                wc[2] = new Vector3(-0.8f, radius, -1.4f); wc[3] = new Vector3(0.8f, radius, -1.4f);
                string[] tag = { "FL", "FR", "RL", "RR" };
                for (int i = 0; i < 4; i++)
                    vis[i] = Prim(PrimitiveType.Cylinder, "Wheel_" + tag[i], root.transform, wc[i], Quaternion.Euler(0f, 0f, 90f), new Vector3(radius * 2f, 0.12f, radius * 2f), tire).transform;
                eye = new Vector3(-0.37f, 1.2f, 0.05f);
            }

            // 차체 충돌체: 바퀴 위쪽 ~ 지붕 아래
            var col = new GameObject("BodyCollider").AddComponent<BoxCollider>();
            col.transform.SetParent(root.transform, false);
            float y0 = Mathf.Max(0.25f, radius * 0.9f), y1 = Mathf.Max(y0 + 0.3f, _carBounds.size.y * 0.85f);
            col.center = new Vector3(0f, (y0 + y1) * 0.5f, _carBounds.center.z);
            col.size = new Vector3(_carBounds.size.x * 0.95f, y1 - y0, _carBounds.size.z * 0.95f);

            var car = root.AddComponent<CarController>();
            // 정지 상태 처짐(sag)을 고려해, 정지했을 때 바퀴 중심이 모델 바퀴 위치에 오도록 휠콜라이더를 올려 둔다
            float sag = CarMass * 9.81f / 4f / Spring;
            float lift = Mathf.Max(0f, SuspDist * (1f - SuspTarget) - sag);
            string[] ids = { "FL", "FR", "RL", "RR" };
            var wheels = new WheelCollider[4];
            for (int i = 0; i < 4; i++) wheels[i] = MakeWheelCollider(root.transform, ids[i], wc[i] + Vector3.up * lift, radius);
            car.wheelFL = wheels[0]; car.wheelFR = wheels[1]; car.wheelRL = wheels[2]; car.wheelRR = wheels[3];
            car.visualFL = vis[0]; car.visualFR = vis[1]; car.visualRL = vis[2]; car.visualRR = vis[3];
            car.steeringWheelVisual = steerVisual;
            car.wheelBase = Mathf.Abs(wc[0].z - wc[2].z);
            car.trackWidth = Mathf.Abs(wc[1].x - wc[0].x);

            // 질량중심을 낮춰 전복 방지. 충돌체를 모두 붙인 뒤 지정해야 덮어써지지 않는다.
            rb.centerOfMass = new Vector3(0f, Mathf.Max(0.3f, _carBounds.size.y * 0.22f), _carBounds.center.z + 0.1f);

            _driverEye = eye + driverEyeAdjust;
            return car;
        }

        Vector3 _driverEye;

        static WheelCollider MakeWheelCollider(Transform parent, string id, Vector3 localPos, float radius)
        {
            var go = new GameObject("WheelCollider_" + id);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            var wc = go.AddComponent<WheelCollider>();
            wc.mass = 20f;
            wc.radius = radius;
            wc.suspensionDistance = SuspDist;
            wc.forceAppPointDistance = 0.15f;
            wc.suspensionSpring = new JointSpring { spring = Spring, damper = Damper, targetPosition = SuspTarget };
            return wc;
        }

        static GameObject Prim(PrimitiveType type, string name, Transform parent, Vector3 lpos, Quaternion lrot, Vector3 lscale, Material mat)
        {
            var g = GameObject.CreatePrimitive(type);
            g.name = name;
            DestroyImmediate(g.GetComponent<Collider>()); // 첫 물리 스텝 전에 확실히 제거
            g.transform.SetParent(parent, false);
            g.transform.localPosition = lpos;
            g.transform.localRotation = lrot;
            g.transform.localScale = lscale;
            g.GetComponent<MeshRenderer>().sharedMaterial = mat;
            return g;
        }

        static Material MakeMat(Color c, float smoothness, Texture tex = null, float metallic = 0f)
        {
            // URP 프로젝트면 URP/Lit, 아니면 Built-in Standard 셰이더
            var shader = Shader.Find("Universal Render Pipeline/Lit");
            if (!shader) shader = Shader.Find("Standard");
            var m = new Material(shader) { color = c };
            if (tex) m.mainTexture = tex;
            if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", smoothness);
            if (m.HasProperty("_Glossiness")) m.SetFloat("_Glossiness", smoothness);
            if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", metallic);
            return m;
        }
    }
}
