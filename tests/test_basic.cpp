#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"

#include <cassert>
#include <iostream>

using namespace mlir;

// Test function to verify MLIR module creation
bool testModuleCreation() {
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    
    // Verify the module is valid
    if (failed(verify(module))) {
        std::cerr << "Module verification failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Module creation test passed" << std::endl;
    return true;
}

// Test function to verify basic operation creation
bool testOperationCreation() {
    MLIRContext context;
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    
    auto loc = UnknownLoc::get(&context);
    auto module = ModuleOp::create(loc);
    
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    
    // Create a simple function
    auto i32Type = builder.getI32Type();
    auto funcType = builder.getFunctionType({i32Type, i32Type}, {i32Type});
    auto func = builder.create<func::FuncOp>(loc, "test_add", funcType);
    
    // Verify the function
    if (failed(verify(func))) {
        std::cerr << "Function verification failed" << std::endl;
        return false;
    }
    
    std::cout << "✓ Operation creation test passed" << std::endl;
    return true;
}

int main() {
    std::cout << "Running MLIR tests..." << std::endl;
    
    bool allPassed = true;
    allPassed &= testModuleCreation();
    allPassed &= testOperationCreation();
    
    if (allPassed) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed!" << std::endl;
        return 1;
    }
}
