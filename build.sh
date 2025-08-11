#!/bin/bash

# MLIR Project Build Script
# This script configures and builds the MLIR code generation project

set -e  # Exit on any error

# LLVM Installation Path
export LLVM_DIR="/home/auhnewzhong/llvm-project/install"
export PATH="$LLVM_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"

echo "=== MLIR Project Build Script ==="
echo "LLVM Installation: $LLVM_DIR"

# Check if LLVM installation exists
if [ ! -d "$LLVM_DIR" ]; then
    echo "Error: LLVM installation not found at $LLVM_DIR"
    exit 1
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
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir"

echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable location: $PWD/mlir-example"
