// Simple addition function example
// This file demonstrates basic MLIR syntax for arithmetic operations

func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @main() -> i32 {
  %c42 = arith.constant 42 : i32
  %c24 = arith.constant 24 : i32
  %result = call @add(%c42, %c24) : (i32, i32) -> i32
  return %result : i32
}
