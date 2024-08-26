#!/bin/bash

get_linux_distribution() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$ID"
    else
        echo ""
    fi
}

build_dependencies_for_ubuntu() {
    sudo apt-get install -y clang
    sudo apt-get install -y llvm
    sudo apt-get install -y libssl-dev
    sudo apt-get install -y libclang-dev
    if [ $? -ne 0 ]; then
        echo "Failed to install clang"
        exit 1
    fi

    sudo apt-get install -y espeak-ng
    if [ $? -ne 0 ]; then
        echo "Failed to install espeak-ng"
        exit 1
    fi

    mkdir -p ./libtorch/
    wget -O ./libtorch/libtorch-2.0.0.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip
    if [ $? -ne 0 ]; then
        echo "Failed to download libtorch"
        exit 1
    fi

    sudo apt-get install -y unzip
    if [ $? -ne 0 ]; then
        echo "Failed to install unzip"
        exit 1
    fi

    unzip ./libtorch/libtorch-2.0.0.zip -d ./libtorch/
    if [ $? -ne 0 ]; then
        echo "Failed to unzip libtorch"
        exit 1
    fi

    export LIBTORCH=$(realpath ./libtorch/)
    export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

    echo "Successfully built dependencies for Ubuntu"
}

build_dependencies_for_mac_os() {
    brew install llvm
    if [ $? -ne 0 ]; then
        echo "Failed to install clang"
        exit 1
    fi

    brew install espeak-ng
    if [ $? -ne 0 ]; then
        echo "Failed to install espeak-ng"
        exit 1
    fi

    mkdir -p ./libtorch/
    curl -o ./libtorch/libtorch-2.0.0.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip
    if [ $? -ne 0 ]; then
        echo "Failed to download libtorch"
        exit 1
    fi

    brew install unzip
    if [ $? -ne 0 ]; then
        echo "Failed to install unzip"
        exit 1
    fi

    unzip ./libtorch/libtorch-2.0.0.zip -d ./libtorch/
    if [ $? -ne 0 ]; then
        echo "Failed to unzip libtorch"
        exit 1
    fi

    export LIBTORCH=$(cd "$(dirname "./libtorch/")" && pwd)/$(basename "./libtorch/")
    cd -
    export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

    echo "Successfully built dependencies for MacOS"
}

if [[ "$OSTYPE" == "darwin"* ]]; then
    build_dependencies_for_mac_os
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    distribution=$(get_linux_distribution)
    if [ "$distribution" == "ubuntu" ]; then
        build_dependencies_for_ubuntu
    fi
fi
