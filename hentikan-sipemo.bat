@echo off
setlocal EnableExtensions

rem Selalu bekerja dari folder proyek, termasuk saat file di-double-click.
cd /d "%~dp0"

set "SIPEMO_LAUNCH_PYTHON=%CD%\.venv\Scripts\python.exe"
set "SIPEMO_PID_FILE=%CD%\.runtime\sipemo.pid"

echo.
echo ========================================
echo   Menghentikan SIPEMO
echo ========================================
echo.

if not exist "%SIPEMO_PID_FILE%" (
    echo [INFO] SIPEMO tidak sedang berjalan melalui jalankan-sipemo.bat.
    exit /b 0
)

set "SIPEMO_SERVER_PID="
set /p SIPEMO_SERVER_PID=<"%SIPEMO_PID_FILE%"

if not defined SIPEMO_SERVER_PID (
    echo [PERINGATAN] PID file kosong. Tidak ada proses yang dihentikan.
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
    exit /b 0
)

rem Verifikasi executable dan command line agar PID terpakai ulang tidak salah dibunuh.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$idValue = 0; if (-not [int]::TryParse($env:SIPEMO_SERVER_PID, [ref]$idValue)) { exit 10 }; $process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $idValue) -ErrorAction SilentlyContinue; if ($null -eq $process) { exit 11 }; $expectedLauncher = [IO.Path]::GetFullPath($env:SIPEMO_LAUNCH_PYTHON); if (-not $process.ExecutablePath.Equals($expectedLauncher, [StringComparison]::OrdinalIgnoreCase) -or $process.CommandLine -notlike '*backend\app.py*') { exit 12 }; try { $taskkill = Join-Path $env:SystemRoot 'System32\taskkill.exe'; & $taskkill /PID $idValue /T /F | Out-Null; if ($LASTEXITCODE -ne 0) { exit 13 }; Wait-Process -Id $idValue -Timeout 10 -ErrorAction SilentlyContinue; exit 0 } catch { exit 13 }" >nul 2>&1
set "SIPEMO_STOP_RESULT=%ERRORLEVEL%"

if "%SIPEMO_STOP_RESULT%"=="0" (
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
    echo [SUKSES] SIPEMO dengan PID %SIPEMO_SERVER_PID% telah dihentikan.
    exit /b 0
)

if "%SIPEMO_STOP_RESULT%"=="11" (
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
    echo [INFO] Proses SIPEMO sudah tidak berjalan. PID file lama dibersihkan.
    exit /b 0
)

if "%SIPEMO_STOP_RESULT%"=="10" (
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
    echo [PERINGATAN] Isi PID file tidak valid. Tidak ada proses yang dihentikan.
    exit /b 1
)

if "%SIPEMO_STOP_RESULT%"=="12" (
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
    echo [DIBATALKAN] PID %SIPEMO_SERVER_PID% bukan proses backend SIPEMO yang dikenali.
    echo              Tidak ada proses yang dihentikan.
    exit /b 1
)

echo [GAGAL] Proses SIPEMO tidak dapat dihentikan.
echo         Coba jalankan file ini lagi sebagai Administrator bila diperlukan.
exit /b 1