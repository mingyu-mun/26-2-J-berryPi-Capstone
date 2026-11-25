// CapstoneSimMenu.cs  (Editor 전용)
// 상단 메뉴에 [Capstone] 항목을 추가한다.
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace CapstoneSim.EditorTools
{
    public static class CapstoneSimMenu
    {
        [MenuItem("Capstone/주행 시뮬레이터 씬 구성", priority = 0)]
        static void CreateSimObject()
        {
            var existing = Object.FindFirstObjectByType<SimBootstrap>();
            if (existing)
            {
                Selection.activeGameObject = existing.gameObject;
                EditorUtility.DisplayDialog("Capstone", "이미 이 씬에 DrivingSim 오브젝트가 있습니다.", "확인");
                return;
            }
            var go = new GameObject("DrivingSim");
            go.AddComponent<SimBootstrap>();
            Undo.RegisterCreatedObjectUndo(go, "Create DrivingSim");
            Selection.activeGameObject = go;
            EditorSceneManager.MarkSceneDirty(go.scene);
            Debug.Log("[Capstone] DrivingSim 생성 완료 — 씬을 저장(Ctrl+S)한 뒤 Play 하세요.");
        }

        [MenuItem("Capstone/로그 폴더 열기 (SimLogs)", priority = 20)]
        static void OpenLogFolder()
        {
            string p = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "SimLogs"));
            Directory.CreateDirectory(p);
            EditorUtility.RevealInFinder(p);
        }
    }
}
