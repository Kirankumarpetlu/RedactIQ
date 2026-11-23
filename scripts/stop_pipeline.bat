@echo off
REM ============================================================================
REM  Stop every Stage 9 pipeline process started by start_pipeline.bat
REM ============================================================================
setlocal EnableDelayedExpansion

if "%PII_WORKER_PORT%"==""  set "PII_WORKER_PORT=8003"
if "%PII_GATEWAY_PORT%"=="" set "PII_GATEWAY_PORT=8080"

echo Stopping Stage 9 pipeline...

for %%P in (%PII_GATEWAY_PORT% %PII_WORKER_PORT%) do (
    set "FOUND="
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%%P "') do (
        if not "%%A"=="0" (
            echo   port %%P: stopping pid %%A
            taskkill /f /pid %%A >nul 2>&1
            set "FOUND=1"
        )
    )
    if not defined FOUND echo   port %%P: not running
)

tasklist /fi "imagename eq ngrok.exe" 2>nul | findstr /i ngrok.exe >nul
if errorlevel 1 (
    echo   ngrok: not running
) else (
    echo   ngrok: stopping
    taskkill /f /im ngrok.exe >nul 2>&1
)

echo Done. The console windows stay open so you can read the logs; close them yourself.
endlocal
