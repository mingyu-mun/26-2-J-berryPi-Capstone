// ProceduralTextures.cs
// 외부 텍스처 없이도 노면/잔디가 '단색 판'으로 보이지 않도록 노이즈 텍스처를 코드로 만든다.
// (에셋 스토어 머티리얼을 Inspector에 넣으면 그쪽이 우선한다)
using UnityEngine;

namespace CapstoneSim
{
    public static class ProceduralTextures
    {
        /// <summary>아스팔트: 여러 주파수의 노이즈 + 밝은/어두운 골재 점</summary>
        public static Texture2D Asphalt(int size = 512, int seed = 7)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, true) { name = "ProcAsphalt", wrapMode = TextureWrapMode.Repeat };
            var rng = new System.Random(seed);
            float ox = (float)rng.NextDouble() * 100f, oy = (float)rng.NextDouble() * 100f;
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size, v = (float)y / size;
                    float n = 0.5f * TileNoise(u, v, 8, ox, oy) + 0.3f * TileNoise(u, v, 32, ox, oy) + 0.2f * TileNoise(u, v, 128, ox, oy);
                    float g = 0.17f + 0.10f * n;
                    double r = rng.NextDouble();
                    if (r < 0.02) g += 0.12f;        // 밝은 골재
                    else if (r < 0.05) g -= 0.05f;   // 어두운 틈
                    byte c = (byte)(Mathf.Clamp01(g) * 255f);
                    px[y * size + x] = new Color32(c, c, (byte)Mathf.Min(255, c + 3), 255);
                }
            tex.SetPixels32(px);
            tex.Apply(true);
            tex.anisoLevel = 8;
            return tex;
        }

        /// <summary>잔디: 녹색 계열 저주파 얼룩 + 고주파 잡음</summary>
        public static Texture2D Grass(int size = 256, int seed = 11)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, true) { name = "ProcGrass", wrapMode = TextureWrapMode.Repeat };
            var rng = new System.Random(seed);
            float ox = (float)rng.NextDouble() * 100f, oy = (float)rng.NextDouble() * 100f;
            var px = new Color32[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float u = (float)x / size, v = (float)y / size;
                    float lo = TileNoise(u, v, 4, ox, oy), hi = TileNoise(u, v, 64, ox, oy);
                    float t = 0.6f * lo + 0.4f * hi;
                    var c = Color.Lerp(new Color(0.20f, 0.33f, 0.14f), new Color(0.38f, 0.48f, 0.22f), t);
                    c *= 0.9f + 0.2f * (float)rng.NextDouble();
                    px[y * size + x] = c;
                }
            tex.SetPixels32(px);
            tex.Apply(true);
            tex.anisoLevel = 4;
            return tex;
        }

        /// <summary>HUD용 핸들 아이콘: 원형 림 + 위쪽 표시점 + 가로 스포크</summary>
        public static Texture2D SteeringIcon(int size = 128)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false) { name = "SteeringIcon", wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[size * size];
            float c = (size - 1) * 0.5f, R = size * 0.46f, w = size * 0.07f;
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                {
                    float dx = x - c, dy = y - c, d = Mathf.Sqrt(dx * dx + dy * dy);
                    float a = 0f;
                    if (Mathf.Abs(d - R + w * 0.5f) < w * 0.5f) a = 1f;                       // 림
                    if (Mathf.Abs(dy) < w * 0.35f && d < R - w * 0.5f) a = 0.9f;              // 가로 스포크
                    if (dy < 0 && Mathf.Abs(dx) < w * 0.35f && d < R - w * 0.5f) a = 0.9f;    // 아래 스포크
                    if (d < w * 0.9f) a = 1f;                                                 // 허브
                    bool marker = dy > R - w * 1.6f && Mathf.Abs(dx) < w * 0.6f;              // 12시 표시
                    px[y * size + x] = marker ? new Color32(255, 80, 60, 255) : new Color32(255, 255, 255, (byte)(a * 230));
                }
            tex.SetPixels32(px);
            tex.Apply(false);
            return tex;
        }

        // 이음매 없이 타일링되는 값(value) 노이즈: 격자 좌표를 freq 주기로 감아서 텍스처 경계가 이어지게 함
        static float TileNoise(float u, float v, int freq, float ox, float oy)
        {
            float x = u * freq, y = v * freq;
            int x0 = Mathf.FloorToInt(x), y0 = Mathf.FloorToInt(y);
            float fx = x - x0, fy = y - y0;
            float sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
            float a = Hash(x0 % freq, y0 % freq, ox, oy), b = Hash((x0 + 1) % freq, y0 % freq, ox, oy);
            float c = Hash(x0 % freq, (y0 + 1) % freq, ox, oy), d = Hash((x0 + 1) % freq, (y0 + 1) % freq, ox, oy);
            return Mathf.Lerp(Mathf.Lerp(a, b, sx), Mathf.Lerp(c, d, sx), sy);
        }

        static float Hash(int x, int y, float ox, float oy)
        {
            float h = Mathf.Sin(x * 127.1f + y * 311.7f + ox * 13.3f + oy * 7.7f) * 43758.5453f;
            return h - Mathf.Floor(h);
        }
    }
}
