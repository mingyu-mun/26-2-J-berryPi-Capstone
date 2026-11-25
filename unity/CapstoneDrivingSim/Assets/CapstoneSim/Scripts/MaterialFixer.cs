// MaterialFixer.cs
// 에셋 스토어 에셋 중에는 Built-in(Standard) 이나 HDRP 셰이더로 만들어진 것이 많아
// URP 프로젝트에서는 분홍색으로 보인다. 실행 시점에 URP/Lit 머티리얼로 바꿔 끼운다.
// (원본 머티리얼 파일은 건드리지 않음 — 에디터 Scene 뷰에서는 여전히 분홍색일 수 있다.
//  에디터에서도 정상으로 보이게 하려면 Window > Rendering > Render Pipeline Converter 사용)
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace CapstoneSim
{
    public static class MaterialFixer
    {
        static readonly Dictionary<Material, Material> Cache = new Dictionary<Material, Material>();
        static Shader _urpLit;

        // 'Enter Play Mode Options'에서 도메인 리로드를 끈 프로젝트(Unity 6 기본값)는 static 값이 Play 사이에 남는다.
        // 지난 Play 에서 만든 머티리얼은 Play 종료 때 파괴되므로, Play 시작마다 캐시를 비운다.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetStatics() { Cache.Clear(); }

        public static bool UrpActive
        {
            get
            {
                var rp = GraphicsSettings.currentRenderPipeline;
                return rp != null && rp.GetType().Name.Contains("Universal");
            }
        }

        static bool NeedsConversion(Shader s)
        {
            if (!s) return true;
            string n = s.name;
            return n.StartsWith("Standard") || n.StartsWith("Legacy Shaders/") || n.StartsWith("Mobile/")
                   || n.StartsWith("HDRP/") || n.StartsWith("Autodesk") || n == "Hidden/InternalErrorShader";
        }

        /// <summary>렌더러 하위 전체의 머티리얼을 필요 시 URP/Lit 로 교체. 교체한 개수 반환</summary>
        public static int FixHierarchy(GameObject go)
        {
            if (!UrpActive) return 0;
            int n = 0;
            foreach (var r in go.GetComponentsInChildren<Renderer>(true))
            {
                var mats = r.sharedMaterials;
                bool changed = false;
                for (int i = 0; i < mats.Length; i++)
                {
                    var c = Convert(mats[i]);
                    if (c != mats[i]) { mats[i] = c; changed = true; n++; }
                }
                if (changed) r.sharedMaterials = mats;
            }
            return n;
        }

        /// <summary>단일 머티리얼 변환 (필요 없으면 그대로 반환)</summary>
        public static Material Convert(Material src)
        {
            if (!src || !UrpActive || !NeedsConversion(src.shader)) return src;
            if (Cache.TryGetValue(src, out var done))
            {
                if (done) return done;   // 파괴된 머티리얼이면 다시 만든다
                Cache.Remove(src);
            }
            if (!_urpLit) _urpLit = Shader.Find("Universal Render Pipeline/Lit");
            if (!_urpLit) return src;

            var m = new Material(_urpLit) { name = src.name + " (URP)" };

            // 색 / 기본 텍스처
            Texture baseTex = First(Tex(src, "_MainTex"), Tex(src, "_BaseColorMap"), Tex(src, "_BaseMap"));
            Color baseCol = src.HasProperty("_Color") ? src.GetColor("_Color")
                          : src.HasProperty("_BaseColor") ? src.GetColor("_BaseColor") : Color.white;
            m.SetColor("_BaseColor", baseCol);
            if (baseTex)
            {
                m.SetTexture("_BaseMap", baseTex);
                string stName = src.HasProperty("_MainTex") ? "_MainTex" : src.HasProperty("_BaseColorMap") ? "_BaseColorMap" : "_BaseMap";
                m.SetTextureScale("_BaseMap", src.GetTextureScale(stName));
                m.SetTextureOffset("_BaseMap", src.GetTextureOffset(stName));
            }

            // 금속성 / 매끄러움
            if (src.HasProperty("_Metallic")) m.SetFloat("_Metallic", src.GetFloat("_Metallic"));
            float smooth = src.HasProperty("_Glossiness") ? src.GetFloat("_Glossiness")
                         : src.HasProperty("_Smoothness") ? src.GetFloat("_Smoothness") : 0.5f;
            m.SetFloat("_Smoothness", smooth);
            var mg = Tex(src, "_MetallicGlossMap");
            if (mg) { m.SetTexture("_MetallicGlossMap", mg); m.EnableKeyword("_METALLICSPECGLOSSMAP"); }

            // 노멀맵
            var nrm = First(Tex(src, "_BumpMap"), Tex(src, "_NormalMap"));
            if (nrm)
            {
                m.SetTexture("_BumpMap", nrm);
                m.SetFloat("_BumpScale", src.HasProperty("_BumpScale") ? src.GetFloat("_BumpScale") : 1f);
                m.EnableKeyword("_NORMALMAP");
            }

            // AO
            var ao = Tex(src, "_OcclusionMap");
            if (ao) { m.SetTexture("_OcclusionMap", ao); m.EnableKeyword("_OCCLUSIONMAP"); }

            // 발광 (라이트류)
            if (src.IsKeywordEnabled("_EMISSION") && src.HasProperty("_EmissionColor"))
            {
                m.SetColor("_EmissionColor", src.GetColor("_EmissionColor"));
                var em = Tex(src, "_EmissionMap");
                if (em) m.SetTexture("_EmissionMap", em);
                m.EnableKeyword("_EMISSION");
                m.globalIlluminationFlags = MaterialGlobalIlluminationFlags.RealtimeEmissive;
            }

            // 투명도: Standard 의 _Mode (0 불투명, 1 컷아웃, 2 페이드, 3 투명)
            int mode = src.HasProperty("_Mode") ? Mathf.RoundToInt(src.GetFloat("_Mode")) : 0;
            if (src.shader && src.shader.name.Contains("Transparent")) mode = 3;
            if (mode == 1)
            {
                m.SetFloat("_AlphaClip", 1f);
                m.SetFloat("_Cutoff", src.HasProperty("_Cutoff") ? src.GetFloat("_Cutoff") : 0.5f);
                m.EnableKeyword("_ALPHATEST_ON");
                m.renderQueue = (int)RenderQueue.AlphaTest;
            }
            else if (mode >= 2)
            {
                m.SetFloat("_Surface", 1f);   // Transparent
                m.SetFloat("_Blend", 0f);     // Alpha
                m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
                m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
                m.SetFloat("_ZWrite", 0f);
                m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                m.SetOverrideTag("RenderType", "Transparent");
                m.renderQueue = (int)RenderQueue.Transparent;
            }

            Cache[src] = m;
            return m;
        }

        static Texture Tex(Material m, string prop) => m.HasProperty(prop) ? m.GetTexture(prop) : null;

        // UnityEngine.Object 에는 ?? 를 쓰면 안 되므로 명시적으로 검사
        static Texture First(params Texture[] ts)
        {
            foreach (var t in ts) if (t) return t;
            return null;
        }
    }
}
