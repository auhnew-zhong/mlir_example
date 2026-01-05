#!/bin/bash

# MLIR Project Build Script
# This script configures and builds the MLIR code generation project

set -e  # Exit on any error

# LLVM Installation Path (optional)
LLVM_DIR_DEFAULT="/home/auhnewzhong/llvm-project/install"
USE_LLVM_HINTS="OFF"
if [ -d "$LLVM_DIR_DEFAULT" ]; then
    export LLVM_DIR="$LLVM_DIR_DEFAULT"
    export PATH="$LLVM_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"
    USE_LLVM_HINTS="ON"
fi

echo "=== MLIR Project Build Script ==="
if [ "$USE_LLVM_HINTS" = "ON" ]; then
    echo "LLVM Installation: $LLVM_DIR"
else
    echo "LLVM installation hint not found at $LLVM_DIR_DEFAULT"
    echo "Proceeding without LLVM/MLIR hints (will build simplified version)"
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake..."
if [ "$USE_LLVM_HINTS" = "ON" ]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
        -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir"
else
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug
fi

echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable location: $PWD/mlir-example"
