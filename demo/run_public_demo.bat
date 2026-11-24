@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
pushd "%~dp0.."
echo [1/2] Starting Local Streamlit Server at http://localhost:8501 ...
start "LocalGammaServer" /min "D:\soft_exe\Aconda\python.exe" -m streamlit run "demo\app.py" --server.port 8501 --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false
timeout /t 3 /nobreak >nul
echo [2/2] Starting Cloudflare Public HTTPS Tunnel (look for https://xxxx.trycloudflare.com below)...
"C:\Users\Administrator\AppData\Local\hermes\tools\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8501
popd
pause
