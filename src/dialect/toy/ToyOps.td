include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
// MLIR 20 moved/renamed some interfaces; keep only side-effects here.

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "toy";
}

// A simple function to hold toy ops.
// We keep the dialect minimal for learning: constant + add only.

def Toy_ConstantOp : Op<Toy_Dialect, "constant", [ConstantLike, NoMemoryEffect]> {
  let summary = "constant scalar";
  let description = "Creates an f64 constant from a FloatAttr.";

  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);

  let assemblyFormat = "$value attr-dict";
}

def Toy_AddOp : Op<Toy_Dialect, "add", [NoMemoryEffect]> {
  let summary = "add two scalars";
  let description = "Adds two f64 values.";

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}
