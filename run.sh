#!/bin/bash

# MLIR Project Run Script
# This script runs the compiled MLIR example program

set -e  # Exit on any error

# LLVM Installation Path
export LLVM_DIR="/home/auhnewzhong/llvm-project/install"
export PATH="$LLVM_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"

echo "=== MLIR Project Run Script ==="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Error: Build directory not found. Please run ./build.sh first."
    exit 1
fi

# Check if executable exists
EXECUTABLE="build/mlir-example"
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found at $EXECUTABLE"
    echo "Please run ./build.sh to build the project first."
    exit 1
fi

echo "Running MLIR example..."
echo "----------------------------------------"

# Run the executable
./$EXECUTABLE

echo "----------------------------------------"
echo "MLIR example completed successfully!"

# Optionally show generated files
if [ -f "output.mlir" ]; then
    echo ""
    echo "Generated MLIR IR:"
    echo "=================="
    cat output.mlir
fi

if [ -f "output.ll" ]; then
    echo ""
    echo "Generated LLVM IR:"
    echo "=================="
    head -20 output.ll
    if [ $(wc -l < output.ll) -gt 20 ]; then
        echo "... (truncated, see output.ll for full content)"
    fi
fi
