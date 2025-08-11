; ModuleID = 'mlir_example'
source_filename = "mlir_example"

define i32 @add(i32 %arg0, i32 %arg1) {
entry:
  %0 = add i32 %arg0, %arg1
  ret i32 %0
}

define i32 @main() {
entry:
  %0 = call i32 @add(i32 42, i32 24)
  ret i32 %0
}

define i32 @factorial(i32 %n) {
entry:
  %is_zero = icmp eq i32 %n, 0
  br i1 %is_zero, label %base_case, label %recursive_case

base_case:
  ret i32 1

recursive_case:
  %n_minus_1 = sub i32 %n, 1
  %recursive_result = call i32 @factorial(i32 %n_minus_1)
  %result = mul i32 %n, %recursive_result
  ret i32 %result
}
