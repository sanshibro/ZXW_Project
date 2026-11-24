@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
pushd "%~dp0.."
echo [1/2] Starting Local Streamlit Server at http://localhost:8501 ...
start "LocalGammaServer" /min "D:\soft_exe\Aconda\python.exe" -m streamlit run "demo\app.py" --server.port 8501 --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false
timeout /t 3 /nobreak >nul
echo [2/2] Starting Key-Pinned Public HTTPS Tunnel (Serveo)...
ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -i "%USERPROFILE%\.ssh\id_ed25519_github" -R 80:127.0.0.1:8501 serveo.net
popd
pause
