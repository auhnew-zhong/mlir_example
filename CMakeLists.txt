cmake_minimum_required(VERSION 3.20)
project(mlir-example)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to find LLVM and MLIR
find_package(LLVM CONFIG)
find_package(MLIR CONFIG)

# Check if MLIR was found
if(MLIR_FOUND AND LLVM_FOUND)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
    message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
    message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
    
    # Set LLVM definitions and include directories
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    
    # Include directories
    include_directories(${LLVM_INCLUDE_DIRS})
    include_directories(${MLIR_INCLUDE_DIRS})
    
    # Add definitions
    separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
    add_definitions(${LLVM_DEFINITIONS_LIST})
    
    # LLVM/MLIR libraries
    llvm_map_components_to_libnames(llvm_libs
        Analysis
        Core
        ExecutionEngine
        InstCombine
        Object
        OrcJIT
        RuntimeDyld
        ScalarOpts
        Support
        TransformUtils
        native
    )
    
    # MLIR libraries
    set(mlir_libs
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRFunctionInterfaces
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRArithDialect
        MLIRFuncDialect
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation
    )
    
    # Add executable with full MLIR support
    add_executable(mlir-example
        src/main.cpp
    )
    
    # Link libraries
    target_link_libraries(mlir-example
        ${mlir_libs}
        ${llvm_libs}
    )
    
    message(STATUS "Building with full MLIR support")
else()
    message(WARNING "MLIR not found. Building simplified version.")
    
    # Add executable with simple version
    add_executable(mlir-example
        src/simple_main.cpp
    )
    
    # Add deep learning example executable
    add_executable(mlir-dl-example
        src/deep_learning/dl_main.cpp
    )
    
    message(STATUS "Building with simplified MLIR demonstration")
    message(STATUS "Building deep learning MLIR example")
endif()

# Set output directory
set_target_properties(mlir-example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
