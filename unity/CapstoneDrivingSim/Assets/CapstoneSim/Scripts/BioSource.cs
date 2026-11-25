// BioSource.cs
// 생체신호 입력의 공통 틀. AiLink(추론 서버 전송)·SimPanels(화면)·로거는 이 타입만 본다.
//   BioReplay      : 실측 기록 재생 (origin = recorded)
//   LiveBioSource  : 라즈베리파이 등에서 실시간 수신 (origin = live)  — 데모 단계
using UnityEngine;

namespace CapstoneSim
{
    public abstract class BioSource : MonoBehaviour
    {
        [Tooltip("기록의 몇 초 지점부터 재생할지 (재생 입력만 해당)")] public float startOffsetSec = 0f;

        public bool Loaded { get; protected set; }
        public string Status { get; protected set; } = "";
        public string SessionName { get; protected set; } = "";
        /// <summary>재생 입력의 세션 폴더 (서버가 앞부분을 직접 읽을 때 사용). 실시간 입력은 빈 값</summary>
        public string SessionPath { get; protected set; } = "";
        public double FsEcg { get; protected set; }
        public double FsPpg { get; protected set; }
        /// <summary>웹보드 규약 origin: recorded / live</summary>
        public abstract string Origin { get; }
        public bool IsLive => Origin == "live";
        /// <summary>입력이 시작된 뒤 경과 시간 (세션 시각, 초)</summary>
        public abstract float ElapsedSec { get; }
        /// <summary>재생 위치 (기록 기준 초). 실시간이면 ElapsedSec</summary>
        public virtual float FileTimeSec => ElapsedSec;
        public virtual float DurationSec => -1f;
        public virtual string SegmentLabel => "";

        public abstract void Begin();
        /// <summary>쌓인 표본을 꺼냄 (각 채널 원래 주기)</summary>
        public abstract bool Drain(out float[] ecg, out float[] ppg, int maxEcg = 2200);
    }
}
