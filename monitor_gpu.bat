@echo off
echo ========================================
echo GPU MONITORING - Press Ctrl+C to stop
echo ========================================
echo.
echo This will show GPU stats every 1 second
echo Look for "GPU-Util" percentage to spike
echo when PyTorch/LSTM is training!
echo.
echo ----------------------------------------
echo.

:loop
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
timeout /t 1 /nobreak >nul
goto loop
