@echo off
setlocal

for %%V in (WEIGHTS_HOST WEIGHTS_PATH WEIGHTS_FILE) do (
    if not defined %%V (set "MSG=%%V not set" & goto :die)
)
if /i "%WEIGHTS_FILE%"=="weights.bin" (set "MSG=WEIGHTS_FILE cannot be weights.bin (export target)" & goto :die)

rem repo root is two levels up from this script; outputs land there
pushd "%~dp0..\.." || (set "MSG=cannot cd to repo root" & goto :die)

where python >nul 2>&1 || (set "MSG=python not found (venv not active?)" & goto :die)
python -c "import torch, numpy, h5py, chess" >nul 2>&1 || (set "MSG=missing python deps: pip install torch numpy h5py chess tqdm" & goto :die)

set "PORT_OPT="
if defined WEIGHTS_PORT set "PORT_OPT=-P%WEIGHTS_PORT%"
scp %PORT_OPT% "%WEIGHTS_HOST%:%WEIGHTS_PATH%/%WEIGHTS_FILE%" "%WEIGHTS_FILE%" || (set "MSG=scp failed" & goto :die)
call :nonempty "%WEIGHTS_FILE%" || (set "MSG=copy of %WEIGHTS_HOST%:%WEIGHTS_PATH%/%WEIGHTS_FILE% missing or empty" & goto :die)

python tools\nnue\train_torch.py --import "%WEIGHTS_FILE%" --export weights.bin --quant || (set "MSG=train_torch.py failed" & goto :die)
call :nonempty weights.bin || (set "MSG=weights.bin not produced" & goto :die)

python tools\nnue\pack_weights.py || (set "MSG=pack_weights.py failed" & goto :die)
python tools\nnue\golds.py weights.bin || (set "MSG=golds.py failed" & goto :die)

echo %~nx0: done
popd
exit /b 0

rem equivalent of bash [[ -s file ]]: exists and size > 0
:nonempty
if not exist "%~1" exit /b 1
if %~z1 EQU 0 exit /b 1
exit /b 0

:die
>&2 echo %~nx0: %MSG%
popd 2>nul
exit /b 1
