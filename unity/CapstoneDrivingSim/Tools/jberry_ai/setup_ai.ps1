# setup_ai.ps1 — AI 엔진(Python + numpy·scipy·pandas·scikit-learn·torch CPU) 설치
#   게임 메인 메뉴의 [AI 엔진 설치] 버튼, 또는 실행 파일 폴더의 AI_엔진_설치.bat 이 실행합니다.
#   1) PC 에 Python 3.10~3.13 이 있으면 → Tools/jberry_ai/.venv 를 만들어 설치 (setup_env.py)
#   2) 없거나 -Portable 이면 → 공식 휴대용 Python(embeddable zip)을 Tools/jberry_ai/python 에 받아 그 안에 설치
#   진행: SimLogs/ai/setup_status.txt (running: … / ok / fail: …), 전체 출력: SimLogs/ai/setup_log.txt
param([switch]$Portable)
$ErrorActionPreference = 'Continue'   # Windows PowerShell 5.1 은 Stop 이면 외부 프로그램의 stderr 출력만으로 중단됨
$ProgressPreference = 'SilentlyContinue'      # Invoke-WebRequest 진행 표시줄이 내려받기를 크게 늦춤
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here '..\..')).Path
$logd = Join-Path $root 'SimLogs\ai'
New-Item -ItemType Directory -Force $logd | Out-Null
$statusFile = Join-Path $logd 'setup_status.txt'
$logFile = Join-Path $logd 'setup_log.txt'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Status($s) { [IO.File]::WriteAllText($statusFile, "$s`n", $utf8); Log "STATUS: $s" }
function Log($s) { $line = (Get-Date -Format '[HH:mm:ss] ') + $s; Write-Host $line; [IO.File]::AppendAllText($logFile, "$line`n", $utf8) }

function PyVersion($exe, $argsPre) {
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { return $null }
    try {
        $v = & $exe @argsPre -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(\d+)$') { return [int]$Matches[1] }
    } catch { }
    return $null
}

try {
    [IO.File]::WriteAllText($logFile, "", $utf8)
    Status 'running: 준비'
    Log "폴더: $here"

    # ---- 1) PC 의 Python 찾기
    $found = $null
    if (-not $Portable) {
        foreach ($m in 12, 11, 13, 10) {
            if ((PyVersion 'py' @("-3.$m")) -eq $m) { $found = @('py', "-3.$m"); break }
        }
        if (-not $found) {
            $minor = PyVersion 'python' @()
            if ($minor -ge 10 -and $minor -le 13) { $found = @('python') }
        }
    }
    if ($found) {
        Log ("PC 의 Python 사용: " + ($found -join ' '))
        $exe = $found[0]; $pre = @(); if ($found.Length -gt 1) { $pre = $found[1..($found.Length - 1)] }
        & $exe @pre -u (Join-Path $here 'setup_env.py')
        exit $LASTEXITCODE
    }

    # ---- 2) 휴대용 Python
    $pyDir = Join-Path $here 'python'
    $py = Join-Path $pyDir 'python.exe'
    if (-not (Test-Path $py)) {
        Status 'running: 휴대용 Python 내려받기'
        $zip = Join-Path $env:TEMP 'jberry_python_embed.zip'
        $ok = $false
        foreach ($ver in '3.12.10', '3.12.8', '3.11.9') {
            try {
                Log "내려받기: python-$ver-embed-amd64.zip"
                Invoke-WebRequest -UseBasicParsing "https://www.python.org/ftp/python/$ver/python-$ver-embed-amd64.zip" -OutFile $zip -ErrorAction Stop
                $ok = $true; break
            } catch { Log "  실패: $($_.Exception.Message)" }
        }
        if (-not $ok) { Status 'fail: 휴대용 Python 을 내려받지 못했습니다 (인터넷 연결 확인)'; exit 1 }
        if (Test-Path $pyDir) { Remove-Item -Recurse -Force $pyDir }
        Expand-Archive -Path $zip -DestinationPath $pyDir -Force -ErrorAction Stop
        Remove-Item $zip -Force

        # python3XX._pth: site 켜기 (pip 패키지 경로) + 상위 폴더(Tools/jberry_ai)를 import 경로에 추가
        $pth = Get-ChildItem $pyDir -Filter 'python3*._pth' -ErrorAction Stop | Select-Object -First 1
        $lines = Get-Content $pth.FullName | Where-Object { $_ -notmatch '^\s*#?\s*import site' }
        $lines += '..'
        $lines += 'import site'
        [IO.File]::WriteAllLines($pth.FullName, $lines, $utf8)
        Log "수정: $($pth.Name)"

        Status 'running: pip 설치'
        $gp = Join-Path $pyDir 'get-pip.py'
        Invoke-WebRequest -UseBasicParsing 'https://bootstrap.pypa.io/get-pip.py' -OutFile $gp -ErrorAction Stop
        & $py $gp --no-warn-script-location 2>&1 | ForEach-Object { Log "    $_" }
        if ($LASTEXITCODE -ne 0) { Status 'fail: pip 설치 실패 (setup_log.txt 참고)'; exit 1 }
        Remove-Item $gp -Force
    } else {
        Log "기존 휴대용 Python 사용: $py"
    }
    & $py -u (Join-Path $here 'setup_env.py') --no-venv
    exit $LASTEXITCODE
}
catch {
    Status ("fail: " + $_.Exception.Message)
    exit 1
}
