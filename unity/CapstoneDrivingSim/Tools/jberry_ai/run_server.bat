@echo off
rem Unity 없이 서버만 직접 띄울 때 (보통은 Unity Play 시 자동 실행됨)
cd /d %~dp0
".venv\Scripts\python.exe" -u server.py --mode causal --log-dir ..\..\SimLogs\ai %*
