# Ensure script stops on errors
$ErrorActionPreference = "Stop"

# Create the directory if it doesn't exist
$libtorchDir = "./libtorch"
if (-not (Test-Path -Path $libtorchDir)) {
    New-Item -ItemType Directory -Path $libtorchDir
}

# Download the libtorch ZIP file
$libtorchZip = "$libtorchDir/libtorch-2.0.0.zip"
$libtorchUrl = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"

try {
    Invoke-WebRequest -Uri $libtorchUrl -OutFile $libtorchZip
    Write-Host "Downloaded libtorch successfully."
} catch {
    Write-Host "Failed to download libtorch"
    exit 1
}

# Unzip the file
try {
    Expand-Archive -Path $libtorchZip -DestinationPath $libtorchDir -Force
    Write-Host "Unzipped libtorch successfully."
} catch {
    Write-Host "Failed to unzip libtorch"
    exit 1
}

# Set environment variables
$LIBTORCH = Resolve-Path -Path $libtorchDir
$env:LIBTORCH = $LIBTORCH
$env:LD_LIBRARY_PATH = "$LIBTORCH\lib;$env:LD_LIBRARY_PATH"

Write-Host "Successfully built dependencies for Windows"
