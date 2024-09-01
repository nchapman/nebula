@echo off

REM Create the directory if it doesn't exist
if not exist libtorch (
    mkdir libtorch
)

REM Download the file using PowerShell's Invoke-WebRequest
powershell -Command "Invoke-WebRequest -OutFile ./libtorch/libtorch-2.0.0.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"
if %errorlevel% neq 0 (
    echo "Failed to download libtorch"
    exit /b 1
)

REM Unzip the file
powershell -Command "Expand-Archive -Path ./libtorch/libtorch-2.0.0.zip -DestinationPath ./libtorch/ -Force"
if %errorlevel% neq 0 (
    echo "Failed to unzip libtorch"
    exit /b 1
)

REM Set environment variables
set LIBTORCH=%cd%\libtorch
set LD_LIBRARY_PATH=%LIBTORCH%\lib;%LD_LIBRARY_PATH%

echo "Successfully built dependencies for Windows"
