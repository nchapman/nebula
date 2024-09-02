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

    sudo apt-get install -y unzip
    if [ $? -ne 0 ]; then
        echo "Failed to install unzip"
        exit 1
    fi

    sudo apt-get install -y espeak-ng
    sudo apt-get install -y make autoconf automake libtool pkg-config
    sudo apt-get install -y gcc
    sudo apt-get install -y libsonic-dev
    sudo apt-get install -y ronn
    sudo apt-get install -y kramdown
    sudo apt-get install -y libpcaudio-dev
    mkdir -p ./espeak-ng/
    wget -O ./espeak-ng/espeak-ng.zip https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.1.zip
    unzip -o ./espeak-ng/espeak-ng.zip -d ./espeak-ng/
    cd ./espeak-ng/espeak-ng-1.51.1/
    chmod +x autogen.sh
    ./autogen.sh
    chmod +x configure
    ./configure --prefix=/usr
    make
    cd ../..
    sudo ln -sf /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1.1.49 /lib/libespeak-ng.so
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

    unzip -o ./libtorch/libtorch-2.0.0.zip -d ./libtorch/
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
    sudo ln -sf /opt/homebrew/Cellar/llvm/*/lib/libclang.dylib /usr/local/lib/libclang.dylib
    if [ $? -ne 0 ]; then
        echo "Failed to install clang"
        exit 1
    fi

    brew install unzip
    if [ $? -ne 0 ]; then
        echo "Failed to install unzip"
        exit 1
    fi

    brew install wget
    brew install espeak-ng
    brew install make autoconf automake libtool pkg-config
    brew install gcc
    brew install ronn
    brew install pcaudiolib
    mkdir -p ./espeak-ng/
    wget -O ./espeak-ng/espeak-ng.zip https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.1.zip
    unzip -o ./espeak-ng/espeak-ng.zip -d ./espeak-ng/
    cd ./espeak-ng/espeak-ng-1.51.1/
    mv CHANGELOG.md ChangeLog.md
    chmod +x autogen.sh
    ./autogen.sh
    chmod +x configure
    ./configure --exec-prefix=/usr/local/ --datarootdir=/usr/local --sysconfdir=/usr/local --sharedstatedir=/usr/local --localstatedir=/usr/local --includedir=/usr/local
    make
    sudo make LIBDIR=/usr/local/lib install
    cd ../..
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

    unzip -o ./libtorch/libtorch-2.0.0.zip -d ./libtorch/
    if [ $? -ne 0 ]; then
        echo "Failed to unzip libtorch"
        exit 1
    fi

    export LIBTORCH=$(cd "$(dirname "./libtorch/")" && pwd)/$(basename "./libtorch/")
    cd ..
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
