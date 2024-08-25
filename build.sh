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

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running on macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    distribution=$(get_linux_distribution)
    if [ "$distribution" == "ubuntu" ]; then
        build_dependencies_for_ubuntu
    fi
fi
