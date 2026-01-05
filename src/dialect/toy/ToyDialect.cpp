#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ToyOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "ToyOps.h.inc"

#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"

#include "ToyOpsDialect.cpp.inc"

namespace toy {
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
      >();
}
} // namespace toy
