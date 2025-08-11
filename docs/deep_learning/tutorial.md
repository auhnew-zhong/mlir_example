# 深度学习与MLIR实践教程

## 1. 快速入门

### 1.1 环境准备
确保您已经按照主README文档设置了MLIR开发环境。

### 1.2 编译深度学习示例
```bash
# 编译包含深度学习示例的项目
./build.sh

# 运行深度学习示例
./build/mlir-dl-example
```

## 2. 核心概念

### 2.1 张量类型系统
MLIR中的张量类型提供了强类型检查和形状推断：

```mlir
// 静态形状张量
%input : tensor<32x784xf32>  // 批次大小32，特征维度784

// 动态形状张量
%dynamic : tensor<?x?xf32>   // 运行时确定形状

// 多维张量
%conv_input : tensor<1x28x28x1xf32>  // NHWC格式：批次x高x宽x通道
```

### 2.2 Linalg方言
Linalg方言是深度学习操作的核心，提供了高级线性代数抽象：

```mlir
// 矩阵乘法
%result = linalg.matmul ins(%A, %B : tensor<MxKxf32>, tensor<KxNxf32>) 
                        outs(%C : tensor<MxNxf32>) -> tensor<MxNxf32>

// 通用操作（逐元素）
%relu = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
^bb0(%in: f32, %out: f32):
  %c0 = arith.constant 0.0 : f32
  %max = arith.maximumf %in, %c0 : f32
  linalg.yield %max : f32
} -> tensor<?x?xf32>
```

## 3. 实践示例

### 3.1 构建简单神经网络

#### 步骤1：定义线性层
```mlir
func.func @linear(%input: tensor<32x784xf32>, 
                  %weight: tensor<784x128xf32>, 
                  %bias: tensor<128xf32>) -> tensor<32x128xf32> {
  // 矩阵乘法：input @ weight
  %mm = linalg.matmul ins(%input, %weight : tensor<32x784xf32>, tensor<784x128xf32>) 
                      outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  
  // 添加偏置
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias, %mm : tensor<128xf32>, tensor<32x128xf32>) 
    outs(%init : tensor<32x128xf32>) {
  ^bb0(%b: f32, %x: f32, %out: f32):
    %sum = arith.addf %b, %x : f32
    linalg.yield %sum : f32
  } -> tensor<32x128xf32>
  
  return %result : tensor<32x128xf32>
}
```

#### 步骤2：定义激活函数
```mlir
func.func @relu(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %c0 : f32
    linalg.yield %max : f32
  } -> tensor<?x?xf32>
  
  return %result : tensor<?x?xf32>
}
```

#### 步骤3：组合完整网络
```mlir
func.func @mlp(%input: tensor<32x784xf32>) -> tensor<32x10xf32> {
  // 第一层：784 -> 128
  %h1 = call @linear(%input, %w1, %b1) : 
    (tensor<32x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
  %a1 = call @relu(%h1) : (tensor<32x128xf32>) -> tensor<32x128xf32>
  
  // 第二层：128 -> 64
  %h2 = call @linear(%a1, %w2, %b2) : 
    (tensor<32x128xf32>, tensor<128x64xf32>, tensor<64xf32>) -> tensor<32x64xf32>
  %a2 = call @relu(%h2) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  
  // 输出层：64 -> 10
  %output = call @linear(%a2, %w3, %b3) : 
    (tensor<32x64xf32>, tensor<64x10xf32>, tensor<10xf32>) -> tensor<32x10xf32>
  
  return %output : tensor<32x10xf32>
}
```

### 3.2 卷积神经网络

#### 卷积层实现
```mlir
func.func @conv2d(%input: tensor<1x28x28x1xf32>, 
                  %filter: tensor<3x3x1x32xf32>) -> tensor<1x26x26x32xf32> {
  %result = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>)
    outs(%init : tensor<1x26x26x32xf32>) -> tensor<1x26x26x32xf32>
  
  return %result : tensor<1x26x26x32xf32>
}
```

#### 池化层实现
```mlir
func.func @max_pool(%input: tensor<1x26x26x32xf32>) -> tensor<1x13x13x32xf32> {
  %result = linalg.pooling_nchw_max {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %kernel : tensor<1x26x26x32xf32>, tensor<2x2xf32>)
    outs(%init : tensor<1x13x13x32xf32>) -> tensor<1x13x13x32xf32>
  
  return %result : tensor<1x13x13x32xf32>
}
```

## 4. 优化技术

### 4.1 算子融合
将多个连续操作融合为单个高效kernel：

```mlir
// 融合前：分离的操作
%conv = call @conv2d(%input, %filter) : (...) -> tensor<1x26x26x32xf32>
%bias = call @add_bias(%conv, %bias_tensor) : (...) -> tensor<1x26x26x32xf32>
%relu = call @relu(%bias) : (...) -> tensor<1x26x26x32xf32>

// 融合后：单个操作
%fused = call @conv_bias_relu(%input, %filter, %bias_tensor) : 
  (...) -> tensor<1x26x26x32xf32>
```

### 4.2 内存布局优化
根据目标硬件选择最优的数据布局：

```mlir
// NCHW布局（适合GPU）
%nchw_tensor : tensor<1x32x26x26xf32>

// NHWC布局（适合CPU）
%nhwc_tensor : tensor<1x26x26x32xf32>

// 布局转换
%converted = tensor.transpose %nchw_tensor [0, 2, 3, 1] : 
  tensor<1x32x26x26xf32> to tensor<1x26x26x32xf32>
```

### 4.3 并行化
利用多核CPU或GPU并行计算：

```mlir
// 批次并行处理
scf.parallel (%i) = (%c0) to (%batch_size) step (%c1) {
  %batch_slice = tensor.extract_slice %input[%i, 0, 0, 0] [1, 28, 28, 1] [1, 1, 1, 1] : 
    tensor<32x28x28x1xf32> to tensor<1x28x28x1xf32>
  %result_slice = call @cnn_forward(%batch_slice) : 
    (tensor<1x28x28x1xf32>) -> tensor<1x10xf32>
  // 存储结果...
}
```

## 5. 性能调优

### 5.1 性能分析工具
```bash
# 编译时优化分析
mlir-opt --pass-pipeline="builtin.module(func.func(linalg-fuse-elementwise-ops))" \
         --mlir-timing-display=tree input.mlir

# 运行时性能分析
perf record ./mlir_executable
perf report
```

### 5.2 内存优化
```mlir
// 原地操作减少内存分配
%result = linalg.generic {
  indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                   affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]
} ins(%input : tensor<?x?xf32>) outs(%input : tensor<?x?xf32>) {
^bb0(%in: f32, %out: f32):
  %relu = arith.maximumf %in, %c0 : f32
  linalg.yield %relu : f32
} -> tensor<?x?xf32>
```

## 6. 调试技巧

### 6.1 IR可视化
```bash
# 生成图形化IR表示
mlir-opt --mlir-print-ir-after-all input.mlir > debug_output.mlir

# 验证IR正确性
mlir-opt --verify-diagnostics input.mlir
```

### 6.2 类型检查
MLIR提供强类型检查，帮助发现形状不匹配等问题：

```mlir
// 错误示例：形状不匹配
%wrong = linalg.matmul ins(%a, %b : tensor<32x784xf32>, tensor<128x64xf32>) 
                       outs(%c : tensor<32x64xf32>) -> tensor<32x64xf32>
// 错误：%a的列数(784)与%b的行数(128)不匹配

// 正确示例
%correct = linalg.matmul ins(%a, %b : tensor<32x784xf32>, tensor<784x64xf32>) 
                         outs(%c : tensor<32x64xf32>) -> tensor<32x64xf32>
```

## 7. 实际项目集成

### 7.1 与PyTorch集成
```python
# Python端：导出MLIR
import torch
import torch_mlir

model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# 转换为MLIR
mlir_module = torch_mlir.compile(model, example_input)
```

### 7.2 与TensorFlow集成
```python
# TensorFlow -> MLIR
import tensorflow as tf
from tensorflow.compiler.mlir.tensorflow import tf_mlir_translate_cl

# 保存模型
model.save('model.pb')

# 转换为MLIR
mlir_output = tf_mlir_translate_cl.import_graphdef('model.pb')
```

## 8. 最佳实践

### 8.1 代码组织
- 将不同层类型分别定义在独立函数中
- 使用描述性的函数和变量名
- 添加详细的注释说明张量形状和操作语义

### 8.2 性能优化
- 优先使用高级Linalg操作而非低级循环
- 合理设计内存布局以提高缓存效率
- 利用MLIR的自动优化pass

### 8.3 调试策略
- 从简单模型开始，逐步增加复杂度
- 使用MLIR的验证工具检查IR正确性
- 对比不同优化级别的性能表现

## 9. 常见问题解答

### Q1: 如何处理动态形状？
A: 使用`?`表示动态维度，运行时进行形状推断：
```mlir
%dynamic : tensor<?x?xf32>  // 动态形状
%static : tensor<32x784xf32>  // 静态形状
```

### Q2: 如何实现自定义激活函数？
A: 使用`linalg.generic`操作定义逐元素计算：
```mlir
func.func @gelu(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.generic {
    // 索引映射和迭代器类型
  } ins(%input : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%x: f32, %out: f32):
    // GELU计算：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // 实现细节...
    linalg.yield %gelu_result : f32
  } -> tensor<?x?xf32>
  
  return %result : tensor<?x?xf32>
}
```

### Q3: 如何优化内存使用？
A: 
- 使用原地操作减少临时张量
- 合理规划内存池分配
- 利用梯度检查点技术

## 10. 进阶主题

### 10.1 自动微分
MLIR支持自动微分，可以自动生成反向传播代码。

### 10.2 量化支持
支持INT8/INT16等低精度计算，提高推理性能。

### 10.3 分布式训练
支持多GPU、多节点的分布式训练优化。

这个教程为您提供了使用MLIR进行深度学习开发的完整指南。通过实践这些示例，您将能够构建高效的深度学习编译器和优化器。
