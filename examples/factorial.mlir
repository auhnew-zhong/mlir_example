// Factorial function with control flow
// This file demonstrates MLIR control flow constructs

func.func @factorial(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  
  %is_zero = arith.cmpi eq, %n, %c0 : i32
  cf.cond_br %is_zero, ^bb1, ^bb2
  
^bb1:  // Base case: return 1
  cf.br ^bb3(%c1 : i32)
  
^bb2:  // Recursive case: n * factorial(n-1)
  %n_minus_1 = arith.subi %n, %c1 : i32
  %recursive_result = call @factorial(%n_minus_1) : (i32) -> i32
  %result = arith.muli %n, %recursive_result : i32
  cf.br ^bb3(%result : i32)
  
^bb3(%final_result: i32):
  return %final_result : i32
}
