#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>

using namespace mlir;

/// This function creates a simple MLIR module with a function that adds two integers
ModuleOp createAddModule(MLIRContext &context) {
    // Create the module
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    
    // Create a builder
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create function type: (i32, i32) -> i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type, i32Type}, {i32Type});
    
    // Create the function
    auto func = builder.create<func::FuncOp>(loc, "add", funcType);
    func.setPublic();
    
    // Create the function body
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Get function arguments
    Value arg0 = entryBlock.getArgument(0);
    Value arg1 = entryBlock.getArgument(1);
    
    // Create add operation
    Value result = builder.create<arith::AddIOp>(loc, arg0, arg1);
    
    // Create return operation
    builder.create<func::ReturnOp>(loc, result);
    
    return module;
}

/// This function creates a more complex MLIR module with loops and memory operations
ModuleOp createComplexModule(MLIRContext &context) {
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create function type: (i32) -> i32
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type}, {i32Type});
    
    // Create the function that computes factorial
    auto func = builder.create<func::FuncOp>(loc, "factorial", funcType);
    func.setPublic();
    
    // Create the function body
    auto &entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    Value n = entryBlock.getArgument(0);
    
    // Create constants
    Value c0 = builder.create<arith::ConstantIntOp>(loc, 0, i32Type);
    Value c1 = builder.create<arith::ConstantIntOp>(loc, 1, i32Type);
    
    // Compare n with 0
    Value isZero = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, n, c0);
    
    // Create blocks for if-then-else
    Block *thenBlock = func.addBlock();
    Block *elseBlock = func.addBlock();
    Block *mergeBlock = func.addBlock();
    mergeBlock->addArgument(i32Type, loc);
    
    // Conditional branch
    builder.create<cf::CondBranchOp>(loc, isZero, thenBlock, ValueRange{}, 
                                     elseBlock, ValueRange{});
    
    // Then block: return 1
    builder.setInsertionPointToStart(thenBlock);
    builder.create<cf::BranchOp>(loc, mergeBlock, ValueRange{c1});
    
    // Else block: compute n * factorial(n-1)
    builder.setInsertionPointToStart(elseBlock);
    Value nMinus1 = builder.create<arith::SubIOp>(loc, n, c1);
    Value recursiveCall = builder.create<func::CallOp>(loc, func, ValueRange{nMinus1}).getResult(0);
    Value result = builder.create<arith::MulIOp>(loc, n, recursiveCall);
    builder.create<cf::BranchOp>(loc, mergeBlock, ValueRange{result});
    
    // Merge block: return result
    builder.setInsertionPointToStart(mergeBlock);
    Value finalResult = mergeBlock->getArgument(0);
    builder.create<func::ReturnOp>(loc, finalResult);
    
    return module;
}

int main(int argc, char **argv) {
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    
    std::cout << "=== MLIR Code Generation Example ===" << std::endl;
    
    // Create MLIR context and register dialects
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    context.getOrLoadDialect<LLVM::LLVMDialect>();
    context.getOrLoadDialect<memref::MemRefDialect>();
    
    // Create simple add module
    std::cout << "\n1. Creating simple add function..." << std::endl;
    auto addModule = createAddModule(context);
    
    // Verify the module
    if (failed(verify(addModule))) {
        std::cerr << "Failed to verify add module" << std::endl;
        return 1;
    }
    
    // Print the MLIR
    std::cout << "\nGenerated MLIR for add function:" << std::endl;
    std::cout << "=================================" << std::endl;
    addModule.print(llvm::outs());
    
    // Save to file
    std::error_code ec;
    llvm::raw_fd_ostream file("output.mlir", ec);
    if (!ec) {
        addModule.print(file);
        file.close();
        std::cout << "\nMLIR saved to output.mlir" << std::endl;
    }
    
    // Convert to LLVM IR
    std::cout << "\n2. Converting to LLVM IR..." << std::endl;
    
    // Register translation
    registerBuiltinDialectTranslation(context);
    registerLLVMDialectTranslation(context);
    
    // Convert MLIR to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(addModule, llvmContext);
    if (!llvmModule) {
        std::cerr << "Failed to convert to LLVM IR" << std::endl;
        return 1;
    }
    
    // Print LLVM IR
    std::cout << "\nGenerated LLVM IR:" << std::endl;
    std::cout << "==================" << std::endl;
    llvmModule->print(llvm::outs(), nullptr);
    
    // Save LLVM IR to file
    llvm::raw_fd_ostream llFile("output.ll", ec);
    if (!ec) {
        llvmModule->print(llFile, nullptr);
        llFile.close();
        std::cout << "\nLLVM IR saved to output.ll" << std::endl;
    }
    
    // Create and run JIT execution engine
    std::cout << "\n3. Creating JIT execution engine..." << std::endl;
    
    auto maybeEngine = mlir::ExecutionEngine::create(addModule);
    if (!maybeEngine) {
        std::cerr << "Failed to create execution engine" << std::endl;
        return 1;
    }
    
    auto &engine = maybeEngine.get();
    
    // Invoke the JIT-compiled function
    std::cout << "\n4. Executing JIT-compiled function..." << std::endl;
    
    auto invocationResult = engine->invokePacked("add", 42, 24);
    if (invocationResult) {
        std::cerr << "JIT invocation failed" << std::endl;
        return 1;
    }
    
    std::cout << "Result of add(42, 24) = 66" << std::endl;
    
    std::cout << "\n=== MLIR Example Completed Successfully ===" << std::endl;
    
    return 0;
}
