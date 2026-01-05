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
        MLIRControlFlowToLLVM
        MLIRExecutionEngine
        MLIRFuncToLLVM
        MLIRFunctionInterfaces
        MLIRIR
        MLIRArithToLLVM
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRReconcileUnrealizedCasts
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIRArithDialect
        MLIRControlFlowDialect
        MLIRFuncDialect
        MLIRBuiltinToLLVMIRTranslation
        MLIRLLVMToLLVMIRTranslation
    )

    set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/src/dialect/toy/ToyOps.td)
    mlir_tablegen(ToyOps.h.inc -gen-op-decls)
    mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)
    mlir_tablegen(ToyOpsDialect.h.inc -gen-dialect-decls -dialect=toy)
    mlir_tablegen(ToyOpsDialect.cpp.inc -gen-dialect-defs -dialect=toy)
    add_public_tablegen_target(ToyOpsIncGen)

    add_mlir_dialect_library(MLIRToyDialect
        src/dialect/toy/ToyDialect.cpp
        DEPENDS
        ToyOpsIncGen
        LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
    )
    target_include_directories(MLIRToyDialect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    
    # Add executable with full MLIR support
    add_executable(mlir-example
        src/main.cpp
    )
    
    # Link libraries
    target_link_libraries(mlir-example
        ${mlir_libs}
        ${llvm_libs}
        MLIRToyDialect
    )
    target_include_directories(mlir-example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    
    message(STATUS "Building with full MLIR support")
else()
    message(WARNING "MLIR not found. Building simplified version.")
    
    # Add executable with simple version
    add_executable(mlir-example
        src/simple_main.cpp
    )
    
    message(STATUS "Building with simplified MLIR demonstration")
endif()

# Always build deep learning MLIR example
add_executable(mlir-dl-example
    src/deep_learning/dl_main.cpp
)

# Set output directory
set_target_properties(mlir-example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
