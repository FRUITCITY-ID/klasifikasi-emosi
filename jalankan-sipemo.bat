@echo off
setlocal EnableExtensions

rem Selalu bekerja dari folder proyek, termasuk saat file di-double-click.
cd /d "%~dp0"

set "SIPEMO_PROJECT_DIR=%CD%"
set "SIPEMO_LAUNCH_PYTHON=%CD%\.venv\Scripts\python.exe"
set "SIPEMO_LAUNCH_SCRIPT=%CD%\backend\app.py"
set "SIPEMO_RUNTIME_DIR=%CD%\.runtime"
set "SIPEMO_PID_FILE=%SIPEMO_RUNTIME_DIR%\sipemo.pid"
set "SIPEMO_STDOUT_LOG=%SIPEMO_RUNTIME_DIR%\sipemo.out.log"
set "SIPEMO_STDERR_LOG=%SIPEMO_RUNTIME_DIR%\sipemo.err.log"

if not defined SIPEMO_PORT set "SIPEMO_PORT=8000"
set "SIPEMO_URL=http://127.0.0.1:%SIPEMO_PORT%"

echo.
echo ========================================
echo   Menjalankan SIPEMO
echo ========================================
echo.

if not exist "%SIPEMO_LAUNCH_PYTHON%" (
    echo [GAGAL] Virtual environment belum tersedia:
    echo         %SIPEMO_LAUNCH_PYTHON%
    echo.
    echo Buat .venv dan instal dependensi sesuai README.md terlebih dahulu.
    goto :failed
)

if not exist "%SIPEMO_LAUNCH_SCRIPT%" (
    echo [GAGAL] Backend tidak ditemukan:
    echo         %SIPEMO_LAUNCH_SCRIPT%
    goto :failed
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$port = 0; if (-not [int]::TryParse($env:SIPEMO_PORT, [ref]$port) -or $port -lt 1 -or $port -gt 65535) { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [GAGAL] SIPEMO_PORT harus berupa angka antara 1 dan 65535.
    goto :failed
)

if not exist "%SIPEMO_RUNTIME_DIR%" mkdir "%SIPEMO_RUNTIME_DIR%"
if errorlevel 1 (
    echo [GAGAL] Folder runtime tidak dapat dibuat.
    goto :failed
)

rem Jika PID tersimpan dan benar-benar milik SIPEMO, jangan jalankan duplikat.
if exist "%SIPEMO_PID_FILE%" (
    set "SIPEMO_EXISTING_PID="
    set /p SIPEMO_EXISTING_PID=<"%SIPEMO_PID_FILE%"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$idValue = 0; if (-not [int]::TryParse($env:SIPEMO_EXISTING_PID, [ref]$idValue)) { exit 1 }; $process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $idValue) -ErrorAction SilentlyContinue; $expectedLauncher = [IO.Path]::GetFullPath($env:SIPEMO_LAUNCH_PYTHON); if ($null -ne $process -and $process.ExecutablePath.Equals($expectedLauncher, [StringComparison]::OrdinalIgnoreCase) -and $process.CommandLine -like '*backend\app.py*') { exit 0 }; exit 1" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] SIPEMO sudah berjalan.
        echo [INFO] Membuka %SIPEMO_URL%
        if /I not "%SIPEMO_NO_BROWSER%"=="1" start "" "%SIPEMO_URL%"
        exit /b 0
    )
    del /q "%SIPEMO_PID_FILE%" >nul 2>&1
)

rem Tolak start jika port dipakai proses lain. Ini mencegah salah membunuh proses.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$port = [int]$env:SIPEMO_PORT; $used = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners() | Where-Object { $_.Port -eq $port } | Select-Object -First 1; if ($null -ne $used) { exit 1 }" >nul 2>&1
if errorlevel 1 (
    echo [GAGAL] Port %SIPEMO_PORT% sedang dipakai proses lain.
    echo         Hentikan proses tersebut atau atur SIPEMO_PORT ke port lain.
    goto :failed
)

echo [INFO] Menyalakan server di %SIPEMO_URL% ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $process = Start-Process -FilePath $env:SIPEMO_LAUNCH_PYTHON -ArgumentList @('-u', 'backend\app.py') -WorkingDirectory $env:SIPEMO_PROJECT_DIR -WindowStyle Hidden -RedirectStandardOutput $env:SIPEMO_STDOUT_LOG -RedirectStandardError $env:SIPEMO_STDERR_LOG -PassThru; Set-Content -LiteralPath $env:SIPEMO_PID_FILE -Value @($process.Id, $env:SIPEMO_PORT) -Encoding Ascii"
if errorlevel 1 (
    echo [GAGAL] Proses server tidak dapat dibuat.
    goto :failed
)

set "SIPEMO_SERVER_PID="
set /p SIPEMO_SERVER_PID=<"%SIPEMO_PID_FILE%"
if not defined SIPEMO_SERVER_PID (
    echo [GAGAL] PID server tidak berhasil disimpan.
    goto :failed
)

echo [INFO] PID server: %SIPEMO_SERVER_PID%
echo [INFO] Menunggu server siap. Pemuatan model dapat memerlukan waktu ...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$deadline = [DateTime]::UtcNow.AddSeconds(90); $idValue = [int]$env:SIPEMO_SERVER_PID; do { if ($null -eq (Get-Process -Id $idValue -ErrorAction SilentlyContinue)) { exit 2 }; try { Invoke-RestMethod -Uri ($env:SIPEMO_URL + '/api/health') -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } } while ([DateTime]::UtcNow -lt $deadline); exit 1" >nul 2>&1
set "SIPEMO_READY_RESULT=%ERRORLEVEL%"

if "%SIPEMO_READY_RESULT%"=="0" goto :ready
if "%SIPEMO_READY_RESULT%"=="2" goto :server_stopped

echo [PERINGATAN] Server masih berjalan, tetapi belum merespons setelah 90 detik.
echo              Periksa log berikut:
echo              %SIPEMO_STDOUT_LOG%
echo              %SIPEMO_STDERR_LOG%
echo.
echo Gunakan hentikan-sipemo.bat jika ingin menghentikannya.
goto :failed

:server_stopped
echo [GAGAL] Server berhenti sebelum siap.
if exist "%SIPEMO_STDERR_LOG%" (
    echo.
    echo Isi log error:
    type "%SIPEMO_STDERR_LOG%"
)
del /q "%SIPEMO_PID_FILE%" >nul 2>&1
goto :failed

:ready
echo [SUKSES] SIPEMO siap digunakan.
echo [INFO] Membuka %SIPEMO_URL%
if /I not "%SIPEMO_NO_BROWSER%"=="1" start "" "%SIPEMO_URL%"
exit /b 0

:failed
echo.
if /I not "%SIPEMO_NO_PAUSE%"=="1" (
    echo Tekan tombol apa saja untuk menutup jendela ini.
    pause >nul
)
exit /b 1