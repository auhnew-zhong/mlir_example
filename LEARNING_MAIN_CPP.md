# main.cpp 学习指南（对应 docs/design.md）

本指南以 [docs/design.md](file:///home/auhnewzhong/mlir_example/docs/design.md) 的章节为主线，带你阅读 [main.cpp](file:///home/auhnewzhong/mlir_example/src/main.cpp#L1-L283)。目标是让你能独立完成三件事：

- 用 C++ API 构造 MLIR（Module / Func / Block / Op）
- 把高层方言（func/arith/cf）lower 到 LLVM Dialect，再导出 LLVM IR
- 用 ExecutionEngine 做 JIT 并调用函数验证结果

## 0. 你应该先能跑通什么

- 构建：`./build.sh`
- 运行：`./run.sh`

`./run.sh` 成功时会依次看到：

- “Generated MLIR for add function”（用 func + arith 构造的 `@add`）
- “Generated MLIR for toy dialect”（用自定义 `toy` 方言构造的标量常量与加法）
- “Generated LLVM IR”（将 `@add` lowering 并导出到 LLVM IR）
- “Result of add(42, 24) = 66”（JIT 执行结果）

## 1. 对照阅读方式

阅读时建议把 `main.cpp` 分成四块，对应你在编译器里最常见的四条链路：

- Dialect/Context 初始化（“世界观”）
- IR 构造（“前端/IR builder”）
- Lowering + 导出 LLVM IR（“后端前半段”）
- ExecutionEngine JIT（“后端最后一公里”）

下面章节按 [docs/design.md](file:///home/auhnewzhong/mlir_example/docs/design.md) 的编号来组织。

## 3. Implementation Details

### 3.1 MLIR Context Setup

对应代码：

- [main.cpp:L139-L153](file:///home/auhnewzhong/mlir_example/src/main.cpp#L139-L153)

读到这里你应该理解什么：

- `MLIRContext` 是 MLIR 的“全局容器”：类型、属性、方言、符号等都依赖它管理。
- `context.getOrLoadDialect<...>()` 的意义：只有加载了对应 dialect，才能创建/解析该 dialect 的 op。
- 你现在加载的 dialect：
  - `arith`：整数算术（`arith.addi` 等）
  - `func`：函数（`func.func`、`func.call`、`func.return`）
  - `cf`：控制流（`cf.cond_br`、`cf.br`）
  - `llvm`：LLVM Dialect（lowering 的目标）
  - `memref`：内存引用（当前主线示例里没重度使用，但常见于后续扩展）
  - `toy`：你自定义的教学 dialect（目前是标量 `toy.constant`/`toy.add`）

你可以改什么（安全、小范围）：

- 增删 dialect 的加载：当你删掉一个 dialect 后，如果代码里还创建了它的 op，会在构造/验证阶段报错。
- 把 `getOrLoadDialect` 换成 `loadDialect`：行为接近，但前者更常用、语义更直接。

### 3.2 IR Generation Strategy

本项目把“IR 构造”封装成了两个函数：

- 简单加法：`createAddModule`  
  - [main.cpp:L46-L79](file:///home/auhnewzhong/mlir_example/src/main.cpp#L46-L79)
- 更复杂的控制流/递归例子：`createComplexModule`（main 当前没有调用，但非常适合你练手）  
  - [main.cpp:L81-L137](file:///home/auhnewzhong/mlir_example/src/main.cpp#L81-L137)

#### 3.2.1 你需要抓住的“构造套路”

以 `createAddModule` 为例，你应该能一眼看出以下固定套路：

- `ModuleOp::create(loc)`：创建顶层容器
- `OpBuilder` + `setInsertionPoint...`：控制后续 op 插入的位置
- `builder.getI32Type()` / `getFunctionType(...)`：显式构造类型
- `builder.create<func::FuncOp>(..., "add", funcType)`：创建函数
- `func.addEntryBlock()`：创建入口 block 并拿到参数
- `builder.create<arith::AddIOp>(...)`：创建算术 op
- `builder.create<func::ReturnOp>(...)`：返回

读到这里你应该理解什么：

- MLIR IR 是“操作（op）+ region + block + value”的组合：函数是一个 op，函数体是 region，region 里有 block，block 里是 op。
- “插入点”决定新 op 放到哪里；如果插入点错了，IR 结构会不合法。
- Value 的 SSA 关系由 builder 返回的 `Value` 串起来，不靠你手写 `%0`。

你可以改什么（安全、小范围）：

- 修改 `createAddModule` 里创建的算术 op（`AddIOp`→`MulIOp`/`SubIOp`），观察 MLIR 文本和 LLVM IR 的变化。
- 修改函数签名类型（i32→i64），并同步修改常量/调用处。

#### 3.2.2 自定义 toy dialect 的“最小用法”

对应代码：

- [main.cpp:L178-L195](file:///home/auhnewzhong/mlir_example/src/main.cpp#L178-L195)

这里做的事情很简单：

- 创建一个临时 `toyModule`
- 插入两个 `toy.constant`（f64）
- 插入一个 `toy.add`
- `verify(toyModule)` 后打印

读到这里你应该理解什么：

- 自定义 dialect 只要正确注册到 context，就能像内建 dialect 一样用 `builder.create<toy::...Op>()` 构造。
- 目前 toy 模块只用于“展示 IR”，没有参与 lowering/JIT（主线 JIT 的模块是 `addModule`）。

你可以改什么（安全、小范围）：

- 改两个常量值，观察打印出来的 MLIR 文本变化。

### 3.3 Code Generation Pipeline

对应代码：

- Lower 到 LLVM Dialect：  
  - [main.cpp:L197-L212](file:///home/auhnewzhong/mlir_example/src/main.cpp#L197-L212)
- 导出 LLVM IR：  
  - [main.cpp:L214-L235](file:///home/auhnewzhong/mlir_example/src/main.cpp#L214-L235)
- JIT（ExecutionEngine）：  
  - [main.cpp:L237-L281](file:///home/auhnewzhong/mlir_example/src/main.cpp#L237-L281)

#### 3.3.1 Lowering：为什么要 clone？

你会看到：

- `auto loweredModule = cast<ModuleOp>(addModule->clone());`

读到这里你应该理解什么：

- 你希望同时保留：
  - 原始的高层 IR（func/arith）
  - lower 后的 LLVM Dialect IR
- 所以把 lowering 放到 clone 上更直观（并且便于 debug 比较）。

#### 3.3.2 PassManager：这条 pipeline 在做什么

你会看到 pass 顺序：

- `createConvertFuncToLLVMPass()`
- `createArithToLLVMConversionPass()`
- `createConvertControlFlowToLLVMPass()`
- `createReconcileUnrealizedCastsPass()`

读到这里你应该理解什么：

- “Lower 到 LLVM Dialect”不是直接变成 LLVM IR，而是把 IR 从 func/arith/cf 等 dialect 变成 `llvm.*` dialect。
- `reconcile-unrealized-casts` 常用于清理转换过程中引入的 cast 占位符，否则 verify 或后续导出阶段会失败。

你可以改什么（安全、小范围）：

- 把 pass 顺序调乱：通常会失败（这会让你理解 pipeline 的依赖关系）。
- 先只跑一部分 pass：观察打印出来的 IR 会停留在哪个 dialect 层级。

#### 3.3.3 导出 LLVM IR：为什么要 register translation

你会看到：

- `registerBuiltinDialectTranslation(context);`
- `registerLLVMDialectTranslation(context);`
- `translateModuleToLLVMIR(loweredModule, llvmContext);`

读到这里你应该理解什么：

- MLIR 到 LLVM IR 的导出是“Dialect Translation Interface”机制驱动的。
- 只有 LLVM Dialect 和 builtin dialect（module 等）都注册了翻译接口，`translateModuleToLLVMIR` 才能工作。

#### 3.3.4 JIT：为什么又出现一套 lowering

你会看到 ExecutionEngineOptions 里自定义了 `llvmModuleBuilder`，里面又跑了一遍 lowering + translate：

- [main.cpp:L240-L254](file:///home/auhnewzhong/mlir_example/src/main.cpp#L240-L254)

读到这里你应该理解什么：

- `ExecutionEngine` 默认假设输入模块“已经能直接导出 LLVM IR”（或它能做默认导出）。为了让 `addModule` 这种高层模块也能 JIT，这里显式告诉它“怎么把 MLIR 变成 LLVM IR”。
- 你前面在第 2 步做的 “Converting to LLVM IR...” 是为了展示导出结果；这里是为了让 JIT 真的能跑。

你可以改什么（安全、小范围）：

- 把 `llvmModuleBuilder` 里的 pass 改成和第 2 步一致/不一致，观察是否还能 JIT。

## 4. Build System Design（和 main.cpp 的关系）

建议你只记住两点：

- CMake 里链接的 MLIR/LLVM 库决定了你能不能：
  - 构造对应 dialect 的 op（编译期）
  - 运行 pass / translation（链接期）
- 一旦出现“undefined reference / missing translation interface”，大概率是：
  - 没链接某个转换库，或
  - 没注册某个 translation

## 练习（2～3 个小改动，每个都能用 ./run.sh 验证）

### 练习 1：把 add 变成 mul，并用 JIT 验证结果

目标：

- 让 `@add` 仍叫 `add`，但内部做乘法
- 运行 `./run.sh` 后，最后结果从 `66` 变成 `1008`

改动位置：

- 把 [createAddModule](file:///home/auhnewzhong/mlir_example/src/main.cpp#L46-L79) 里这一行：
  - `builder.create<arith::AddIOp>(...)`
  - 改成 `builder.create<arith::MulIOp>(...)`

验证方式：

- `./build.sh && ./run.sh`
- 观察：
  - MLIR 文本中变成 `arith.muli`
  - LLVM IR 中变成 `mul i32`
  - 输出 `Result of add(42, 24) = 1008`

### 练习 2：打印 Lowered MLIR（LLVM Dialect）以理解“中间层”

目标：

- 在导出 LLVM IR 之前，把 `loweredModule`（LLVM Dialect 版本）打印出来

改动位置：

- 在 [main.cpp:L214-L235](file:///home/auhnewzhong/mlir_example/src/main.cpp#L214-L235) 之前插入：
  - `loweredModule.print(llvm::outs());`

验证方式：

- `./build.sh && ./run.sh`
- 观察输出中会出现 `llvm.func`、`llvm.add` 等 LLVM Dialect 的 op（不是 LLVM IR 文本）。

### 练习 3（进阶）：给 toy 方言加一个 toy.mul，并在 toy module 里用它

目标：

- 增加新 op：`toy.mul`（f64, f64 -> f64）
- `./run.sh` 的 toy 模块输出里能看到 `toy.mul`

改动步骤（只改 3 处）：

- 1) 在 [ToyOps.td](file:///home/auhnewzhong/mlir_example/src/dialect/toy/ToyOps.td) 里增加一个和 `toy.add` 类似的 op 定义（把名字改成 `mul`）。
- 2) 重新构建：`./build.sh`（会自动重新 tablegen）
- 3) 在 [main.cpp:L178-L195](file:///home/auhnewzhong/mlir_example/src/main.cpp#L178-L195) 的 toy module 里创建 `toy.mul` 并打印（例如先 mul 再 add）。

验证方式：

- `./build.sh && ./run.sh`
- 观察 “Generated MLIR for toy dialect” 输出里出现 `toy.mul`。

