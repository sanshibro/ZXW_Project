@echo off
chcp 65001 >nul
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0.."
echo 正在启动【像素化阵列准直-康普顿双模智能伽马成像】交互式可视化工作台...
"D:\soft_exe\Aconda\python.exe" -m streamlit run demo/app.py --server.headless false
pause
