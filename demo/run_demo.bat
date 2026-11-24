@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
pushd "%~dp0.."
echo Starting Dual-Mode Gamma Imaging Interactive Workbench...
echo Browser will open at http://localhost:8501
"D:\soft_exe\Aconda\python.exe" -m streamlit run "demo\app.py" --browser.gatherUsageStats false
popd
pause
