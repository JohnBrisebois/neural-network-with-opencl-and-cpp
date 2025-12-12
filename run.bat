@echo off
echo Compiling...

g++ nnGPU.cpp -o nn.exe ^ -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\include" ^ -L "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64" ^ -lOpenCL
if %errorlevel% neq 0 (
    echo Compilation failed.
    pause
    exit /b %errorlevel%
)

echo Running program...
nn.exe

pause
