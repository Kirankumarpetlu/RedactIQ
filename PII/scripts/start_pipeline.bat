@echo off
REM ============================================================================
REM  Stage 9 PII pipeline - one-command startup (Windows)
REM ============================================================================
REM  Opens three console windows:
REM    Terminal 1  CPU worker  : port 8003  (loads spaCy + GLiNER2 into memory)
REM    Terminal 2  Gateway     : port 8080  (Swagger UI at /docs)
REM    Terminal 3  ngrok tunnel: public HTTPS URL -> gateway
REM
REM  Then waits for health, opens Swagger in Chrome, prints the public link.
REM
REM  Usage:
REM    scripts\start_pipeline.bat              full stack + ngrok + browser
REM    scripts\start_pipeline.bat --no-ngrok   local only, no public tunnel
REM    scripts\start_pipeline.bat --no-browser do not open Chrome
REM
REM  Environment overrides (set before running):
REM    set PII_VENV=venv_pii
REM    set PII_WORKER_PORT=8003
REM    set PII_GATEWAY_PORT=8080
REM    set PII_CONFIG_PATH=config\pipeline_config.yaml
REM ============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

if "%PII_WORKER_PORT%"==""  set "PII_WORKER_PORT=8003"
if "%PII_GATEWAY_PORT%"=="" set "PII_GATEWAY_PORT=8080"
if "%PII_CONFIG_PATH%"==""  set "PII_CONFIG_PATH=config\pipeline_config.yaml"
set "MODEL_DIR=models\gliner2-primary"

set "USE_NGROK=1"
set "USE_BROWSER=1"
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--no-ngrok"   set "USE_NGROK=0"
if /i "%~1"=="--no-browser" set "USE_BROWSER=0"
shift
goto parse_args
:args_done

REM ---------------------------------------------------------------------------
REM 1. Resolve the virtualenv
REM ---------------------------------------------------------------------------
echo ==^> Resolving Python environment
set "PY="
for %%V in ("%PII_VENV%" ".venv_improved" ".venv" "venv_pii" "venv39" "venv") do (
    if not "%%~V"=="" (
        if exist "%PROJECT_ROOT%\%%~V\Scripts\python.exe" (
            if not defined PY (
                "%PROJECT_ROOT%\%%~V\Scripts\python.exe" -c "import fastapi, uvicorn, gliner2, presidio_analyzer, spacy" >nul 2>&1
                if !errorlevel! equ 0 (
                    set "PY=%PROJECT_ROOT%\%%~V\Scripts\python.exe"
                    echo   OK using %%~V
                )
            )
        )
    )
)
if not defined PY (
    echo ERROR No virtualenv with the full dependency set ^(fastapi, gliner2, presidio, spacy^).
    echo       Follow the setup steps in README.md, or set PII_VENV=^<dir^>.
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 2. Preflight: model weights, spaCy model, config, ports
REM ---------------------------------------------------------------------------
echo ==^> Preflight checks
"%PY%" -m scripts.download_data_assets --group runtime --check-only
if errorlevel 1 (
    echo ERROR Runtime data missing. Run: "%PY%" -m scripts.download_data_assets --group runtime
    exit /b 1
)
if not exist "%MODEL_DIR%\model.safetensors" (
    echo ERROR GLiNER checkpoint missing at %MODEL_DIR%.
    echo       Provision it first:  "%PY%" scripts\provision_models.py --gliner-revision ^<APPROVED_SHA^>
    exit /b 1
)
echo   OK GLiNER checkpoint present

"%PY%" -c "import spacy; spacy.load('en_core_web_sm')" >nul 2>&1
if errorlevel 1 (
    echo ERROR spaCy model en_core_web_sm not installed.
    echo       Run: "%PY%" -m spacy download en_core_web_sm
    exit /b 1
)
echo   OK spaCy en_core_web_sm present

if not exist "%PII_CONFIG_PATH%" (
    echo ERROR Config not found: %PII_CONFIG_PATH%
    exit /b 1
)
echo   OK config %PII_CONFIG_PATH%

for %%P in (%PII_WORKER_PORT% %PII_GATEWAY_PORT%) do (
    netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%%P " >nul 2>&1
    if !errorlevel! equ 0 (
        echo ERROR Port %%P is already in use. Stop that process, or run scripts\stop_pipeline.bat
        exit /b 1
    )
)
echo   OK ports %PII_WORKER_PORT% and %PII_GATEWAY_PORT% free

REM ---------------------------------------------------------------------------
REM 3. Terminal 1 - CPU worker (loads the models; slowest step)
REM ---------------------------------------------------------------------------
echo ==^> Terminal 1: CPU worker on port %PII_WORKER_PORT%
start "TERMINAL 1: CPU WORKER (port %PII_WORKER_PORT%)" cmd /k ^
  "cd /d "%PROJECT_ROOT%" ^&^& set PII_SERVICE_ROLE=cpu-worker ^&^& set PII_API_PORT=%PII_WORKER_PORT% ^&^& set PII_CONFIG_PATH=%PII_CONFIG_PATH% ^&^& "%PY%" -m pii_pipeline"

call :wait_health "http://localhost:%PII_WORKER_PORT%/health" "CPU worker (loading GLiNER ~1.2GB)" 300
if errorlevel 1 exit /b 1

REM ---------------------------------------------------------------------------
REM 4. Terminal 2 - gateway / Swagger
REM ---------------------------------------------------------------------------
echo ==^> Terminal 2: gateway on port %PII_GATEWAY_PORT%
start "TERMINAL 2: GATEWAY + SWAGGER (port %PII_GATEWAY_PORT%)" cmd /k ^
  "cd /d "%PROJECT_ROOT%" ^&^& set PII_SERVICE_ROLE=gateway ^&^& set PII_API_PORT=%PII_GATEWAY_PORT% ^&^& set PII_ENABLED_WORKERS=cpu_worker ^&^& set PII_CONFIG_PATH=%PII_CONFIG_PATH% ^&^& "%PY%" -m pii_pipeline"

call :wait_health "http://localhost:%PII_GATEWAY_PORT%/health" "gateway" 60
if errorlevel 1 exit /b 1

REM ---------------------------------------------------------------------------
REM 5. Warm the lazily-built stages (IndicNER + medical recognizers)
REM ---------------------------------------------------------------------------
echo ==^> Warming IndicNER and the medical recognizers
curl -s -X POST "http://localhost:%PII_GATEWAY_PORT%/analyze" -H "Content-Type: application/json" -d "{\"text\":\"Patient John Doe, DOB 1985-04-12, NPI 1234567893.\",\"language\":\"en\",\"gliner_threshold\":0.95}" >nul
echo   OK pipeline warm - all stages resident

REM ---------------------------------------------------------------------------
REM 6. Terminal 3 - ngrok
REM ---------------------------------------------------------------------------
set "PUBLIC_URL="
if "%USE_NGROK%"=="1" (
    where ngrok >nul 2>&1
    if errorlevel 1 (
        echo   WARN ngrok not installed; skipping the public tunnel.
    ) else (
        echo ==^> Terminal 3: ngrok tunnel -^> localhost:%PII_GATEWAY_PORT%
        start "TERMINAL 3: NGROK TUNNEL (-> %PII_GATEWAY_PORT%)" cmd /k ^
          "cd /d "%PROJECT_ROOT%" ^&^& ngrok http %PII_GATEWAY_PORT% --log=stdout"
        echo     waiting for tunnel...
        for /l %%i in (1,1,30) do (
            if not defined PUBLIC_URL (
                for /f "delims=" %%U in ('curl -s http://127.0.0.1:4040/api/tunnels 2^>nul ^| "%PY%" -c "import sys,orjson;d=orjson.loads(sys.stdin.buffer.read());print(d['tunnels'][0]['public_url'] if d.get('tunnels') else '')" 2^>nul') do (
                    if not "%%U"=="" set "PUBLIC_URL=%%U"
                )
                if not defined PUBLIC_URL ping -n 2 127.0.0.1 >nul
            )
        )
    )
)

REM ---------------------------------------------------------------------------
REM 7. Open Swagger in Chrome
REM ---------------------------------------------------------------------------
if "%USE_BROWSER%"=="1" (
    echo ==^> Opening Swagger UI in Chrome
    start "" chrome "http://localhost:%PII_GATEWAY_PORT%/docs" 2>nul || start "" "http://localhost:%PII_GATEWAY_PORT%/docs"
    if defined PUBLIC_URL start "" chrome "!PUBLIC_URL!/docs" 2>nul
)

REM ---------------------------------------------------------------------------
REM 8. Summary
REM ---------------------------------------------------------------------------
echo.
echo ------------------------------------------------------------------
echo  Stage 9 PII pipeline is up
echo ------------------------------------------------------------------
echo   Swagger UI (local) : http://localhost:%PII_GATEWAY_PORT%/docs
echo   Health             : http://localhost:%PII_GATEWAY_PORT%/health
echo   Live metrics       : http://localhost:%PII_GATEWAY_PORT%/pipeline/metrics
echo   CPU worker         : http://localhost:%PII_WORKER_PORT%/health
if defined PUBLIC_URL (
    echo   Public Swagger     : !PUBLIC_URL!/docs      ^<-- share this
    echo   ngrok inspector    : http://127.0.0.1:4040
)
echo.
echo   Stop everything    : scripts\stop_pipeline.bat
echo ------------------------------------------------------------------
endlocal
exit /b 0

REM ---------------------------------------------------------------------------
REM Helper: poll a health URL until it responds or the timeout expires
REM ---------------------------------------------------------------------------
:wait_health
set "_URL=%~1"
set "_LABEL=%~2"
set /a _TIMEOUT=%~3
set /a _ELAPSED=0
echo     waiting for %_LABEL%...
:wait_loop
curl -sf "%_URL%" >nul 2>&1
if %errorlevel% equ 0 (
    echo     ready ^(%_ELAPSED%s^)
    exit /b 0
)
ping -n 3 127.0.0.1 >nul
set /a _ELAPSED+=2
if %_ELAPSED% geq %_TIMEOUT% (
    echo ERROR %_LABEL% did not come up within %_TIMEOUT%s. Check its console window.
    exit /b 1
)
goto wait_loop
