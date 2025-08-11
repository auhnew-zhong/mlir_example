# MLIR Code Generation Project

This project demonstrates MLIR (Multi-Level Intermediate Representation) code generation, compilation, and execution using LLVM infrastructure.

## Project Structure

```
mlir_example/
├── README.md                 # This file
├── CMakeLists.txt           # CMake build configuration
├── build.sh                 # Build script
├── run.sh                   # Run script
├── src/                     # Source code
│   ├── main.cpp            # Main MLIR generation program
│   ├── dialect/            # Custom dialect definitions
│   └── passes/             # Custom transformation passes
├── examples/               # Example MLIR files
├── docs/                   # Design documentation
│   ├── design.md          # Detailed design document
│   └── architecture.md    # Architecture overview
└── tests/                  # Test cases

```

## Prerequisites

- LLVM/MLIR installed at: `/home/auhnewzhong/llvm-project/install`
- CMake 3.20+
- C++17 compatible compiler

## Quick Start

1. Build the project:
   ```bash
   ./build.sh
   ```

2. Run the basic MLIR example:
   ```bash
   ./run.sh
   ```

3. Run the deep learning MLIR example:
   ```bash
   ./run_deep_learning.sh
   ```

## Features

### Basic MLIR Features
- MLIR IR generation from high-level constructs
- Custom dialect support
- Transformation passes
- Code generation to LLVM IR
- JIT compilation and execution
- Comprehensive documentation

### Deep Learning Features
- Neural network model representation in MLIR
- Convolutional Neural Network (CNN) support
- Tensor operations with Linalg dialect
- Activation functions (ReLU, Softmax)
- Optimization passes for deep learning
- Performance analysis and benchmarking
- Multi-level optimization strategies

## Environment Setup

The project automatically configures the LLVM environment using the specified installation path.
