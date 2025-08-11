// Linear Layer: layer0
func.func @layer0(%input: tensor<32x784xf32>, %weight: tensor<784x128xf32>, %bias: tensor<128xf32>) -> tensor<32x128xf32> {
  %t0 = linalg.matmul ins(%input, %weight : tensor<32x784xf32>, tensor<784x128xf32>) outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  %t1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias, %t0 : tensor<128xf32>, tensor<32x128xf32>) outs(%init : tensor<32x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<32x128xf32>
  return %t1 : tensor<32x128xf32>
}

// Linear Layer: layer1
func.func @layer1(%input: tensor<32x128xf32>, %weight: tensor<128x64xf32>, %bias: tensor<64xf32>) -> tensor<32x64xf32> {
  %t2 = linalg.matmul ins(%input, %weight : tensor<32x128xf32>, tensor<128x64xf32>) outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>
  %t3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias, %t2 : tensor<64xf32>, tensor<32x64xf32>) outs(%init : tensor<32x64xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<32x64xf32>
  return %t3 : tensor<32x64xf32>
}

// Linear Layer: layer2
func.func @layer2(%input: tensor<32x64xf32>, %weight: tensor<64x10xf32>, %bias: tensor<10xf32>) -> tensor<32x10xf32> {
  %t4 = linalg.matmul ins(%input, %weight : tensor<32x64xf32>, tensor<64x10xf32>) outs(%init : tensor<32x10xf32>) -> tensor<32x10xf32>
  %t5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias, %t4 : tensor<10xf32>, tensor<32x10xf32>) outs(%init : tensor<32x10xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<32x10xf32>
  return %t5 : tensor<32x10xf32>
}

// ReLU Activation Function
func.func @relu(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %t6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %max = arith.maximumf %arg0, %c0 : f32
    linalg.yield %max : f32
  } -> tensor<?x?xf32>
  return %t6 : tensor<?x?xf32>
}

// Softmax Activation Function
func.func @softmax(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %t7 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<?x?xf32>) outs(%init_max : tensor<?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %max = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %max : f32
  } -> tensor<?xf32>
  %t8 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %t7 : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sub = arith.subf %arg0, %arg1 : f32
    %exp = math.exp %sub : f32
    linalg.yield %exp : f32
  } -> tensor<?x?xf32>
  %t9 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%t8 : tensor<?x?xf32>) outs(%init_sum : tensor<?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  %t10 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%t8, %t9 : tensor<?x?xf32>, tensor<?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %div = arith.divf %arg0, %arg1 : f32
    linalg.yield %div : f32
  } -> tensor<?x?xf32>
  return %t10 : tensor<?x?xf32>
}

// Convolutional Layer (2D Convolution)
func.func @conv2d(%input: tensor<1x28x28x1xf32>, %kernel: tensor<3x3x1x32xf32>, %bias: tensor<32xf32>) -> tensor<1x26x26x32xf32> {
  %t11 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %kernel : tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>)
    outs(%init : tensor<1x26x26x32xf32>) -> tensor<1x26x26x32xf32>
  %t12 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bias, %t11 : tensor<32xf32>, tensor<1x26x26x32xf32>) outs(%init : tensor<1x26x26x32xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<1x26x26x32xf32>
  return %t12 : tensor<1x26x26x32xf32>
}

// Complete Neural Network: Input -> Hidden -> Output
func.func @neural_network(%input: tensor<32x784xf32>, %w1: tensor<784x128xf32>, %b1: tensor<128xf32>,
                         %w2: tensor<128x64xf32>, %b2: tensor<64xf32>,
                         %w3: tensor<64x10xf32>, %b3: tensor<10xf32>) -> tensor<32x10xf32> {
  %t13 = call @layer0(%input, %w1, %b1) : (tensor<32x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
  %t14 = call @relu(%t13) : (tensor<32x128xf32>) -> tensor<32x128xf32>
  %t15 = call @layer1(%t14, %w2, %b2) : (tensor<32x128xf32>, tensor<128x64xf32>, tensor<64xf32>) -> tensor<32x64xf32>
  %t16 = call @relu(%t15) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  %t17 = call @layer2(%t16, %w3, %b3) : (tensor<32x64xf32>, tensor<64x10xf32>, tensor<10xf32>) -> tensor<32x10xf32>
  %t18 = call @softmax(%t17) : (tensor<32x10xf32>) -> tensor<32x10xf32>
  return %t18 : tensor<32x10xf32>
}

// Optimization Passes Configuration
// These would be applied during compilation:
//
// 1. Tensor Fusion Pass
//    - Fuses consecutive operations to reduce memory overhead
//    - Example: conv2d + bias_add + relu -> fused_conv2d_bias_relu
//
// 2. Memory Layout Optimization
//    - Optimizes tensor layouts for target hardware
//    - NCHW vs NHWC format selection
//
// 3. Loop Tiling and Vectorization
//    - Tiles large tensor operations for cache efficiency
//    - Vectorizes operations for SIMD instructions
//
// 4. Constant Folding
//    - Pre-computes constant expressions at compile time
//
// 5. Dead Code Elimination
//    - Removes unused operations and tensors

