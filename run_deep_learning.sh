#!/bin/bash

# Deep Learning MLIR Example Run Script
# This script runs the deep learning MLIR demonstration

set -e  # Exit on any error

# LLVM Installation Path
export LLVM_DIR="/home/auhnewzhong/llvm-project/install"
export PATH="$LLVM_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$LLVM_DIR/lib:$LD_LIBRARY_PATH"

echo "=== Deep Learning MLIR Example Run Script ==="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Error: Build directory not found. Please run ./build.sh first."
    exit 1
fi

# Check if deep learning executable exists
DL_EXECUTABLE="build/mlir-dl-example"
if [ ! -f "$DL_EXECUTABLE" ]; then
    echo "Error: Deep learning executable not found at $DL_EXECUTABLE"
    echo "Please run ./build.sh to build the project first."
    exit 1
fi

echo "Running Deep Learning MLIR example..."
echo "----------------------------------------"

# Run the deep learning executable
./$DL_EXECUTABLE

echo "----------------------------------------"
echo "Deep Learning MLIR example completed successfully!"

# Show generated files
if [ -f "deep_learning_output.mlir" ]; then
    echo ""
    echo "Generated Deep Learning MLIR IR (first 30 lines):"
    echo "=================================================="
    head -30 deep_learning_output.mlir
    if [ $(wc -l < deep_learning_output.mlir) -gt 30 ]; then
        echo "... (see deep_learning_output.mlir for full content)"
    fi
fi

echo ""
echo "Deep Learning files generated:"
echo "- deep_learning_output.mlir: Complete neural network MLIR representation"
echo ""
echo "Documentation available:"
echo "- docs/deep_learning/deep_learning_mlir_design.md: Detailed design document"
echo "- docs/deep_learning/tutorial.md: Practical tutorial"
echo "- examples/deep_learning/: Example MLIR files"
