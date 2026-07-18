@echo off
chcp 65001 >nul
title 手写数字识别测试工具

echo.
echo   拖拽数据集文件夹到这里，按 Enter 开始测试
echo.

set /p DATADIR="数据集路径: "
set DATADIR=%DATADIR:"=%

if not exist "%DATADIR%" (
    echo [ERROR] 路径无效!
    pause
    exit /b
)

python "%~dp0test_model.py" "%DATADIR%"
pause
