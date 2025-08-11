// Convolutional Neural Network Example in MLIR
// This demonstrates CNN operations for image classification

// 2D Convolution layer
func.func @conv2d(%input: tensor<1x28x28x1xf32>, 
                  %kernel: tensor<3x3x1x32xf32>, 
                  %bias: tensor<32xf32>) -> tensor<1x26x26x32xf32> {
  
  // Convolution operation
  %0 = linalg.conv_2d_nchw_fchw {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %kernel : tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>)
    outs(%init : tensor<1x26x26x32xf32>) -> tensor<1x26x26x32xf32>
  
  // Add bias
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, 
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, 
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%bias, %0 : tensor<32xf32>, tensor<1x26x26x32xf32>) 
    outs(%init : tensor<1x26x26x32xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<1x26x26x32xf32>
  
  return %1 : tensor<1x26x26x32xf32>
}

// Max pooling layer
func.func @max_pool2d(%input: tensor<1x26x26x32xf32>) -> tensor<1x13x13x32xf32> {
  %0 = linalg.pooling_nchw_max {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<2> : tensor<2xi64>
  } ins(%input, %kernel : tensor<1x26x26x32xf32>, tensor<2x2xf32>)
    outs(%init : tensor<1x13x13x32xf32>) -> tensor<1x13x13x32xf32>
  
  return %0 : tensor<1x13x13x32xf32>
}

// Batch normalization
func.func @batch_norm(%input: tensor<1x13x13x32xf32>,
                      %scale: tensor<32xf32>,
                      %offset: tensor<32xf32>,
                      %mean: tensor<32xf32>,
                      %variance: tensor<32xf32>) -> tensor<1x13x13x32xf32> {
  %eps = arith.constant 1e-5 : f32
  
  // Normalize: (x - mean) / sqrt(variance + eps)
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input, %mean, %variance, %scale, %offset : 
        tensor<1x13x13x32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>)
    outs(%init : tensor<1x13x13x32xf32>) {
  ^bb0(%x: f32, %m: f32, %v: f32, %s: f32, %o: f32, %out: f32):
    %sub = arith.subf %x, %m : f32
    %add_eps = arith.addf %v, %eps : f32
    %sqrt = math.sqrt %add_eps : f32
    %div = arith.divf %sub, %sqrt : f32
    %scale_mul = arith.mulf %div, %s : f32
    %result = arith.addf %scale_mul, %o : f32
    linalg.yield %result : f32
  } -> tensor<1x13x13x32xf32>
  
  return %0 : tensor<1x13x13x32xf32>
}

// Complete CNN model
func.func @cnn_model(%input: tensor<1x28x28x1xf32>) -> tensor<1x10xf32> {
  // Convolution block 1
  %conv1 = call @conv2d(%input, %kernel1, %bias1) : 
    (tensor<1x28x28x1xf32>, tensor<3x3x1x32xf32>, tensor<32xf32>) -> tensor<1x26x26x32xf32>
  %relu1 = call @relu(%conv1) : (tensor<1x26x26x32xf32>) -> tensor<1x26x26x32xf32>
  %pool1 = call @max_pool2d(%relu1) : (tensor<1x26x26x32xf32>) -> tensor<1x13x13x32xf32>
  
  // Convolution block 2
  %conv2 = call @conv2d(%pool1, %kernel2, %bias2) : 
    (tensor<1x13x13x32xf32>, tensor<3x3x32x64xf32>, tensor<64xf32>) -> tensor<1x11x11x64xf32>
  %relu2 = call @relu(%conv2) : (tensor<1x11x11x64xf32>) -> tensor<1x11x11x64xf32>
  %pool2 = call @max_pool2d(%relu2) : (tensor<1x11x11x64xf32>) -> tensor<1x5x5x64xf32>
  
  // Flatten
  %flatten = tensor.reshape %pool2 : tensor<1x5x5x64xf32> into tensor<1x1600xf32>
  
  // Fully connected layers
  %fc1 = call @linear_layer(%flatten, %fc_weight1, %fc_bias1) : 
    (tensor<1x1600xf32>, tensor<1600x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
  %fc1_relu = call @relu(%fc1) : (tensor<1x128xf32>) -> tensor<1x128xf32>
  
  %fc2 = call @linear_layer(%fc1_relu, %fc_weight2, %fc_bias2) : 
    (tensor<1x128xf32>, tensor<128x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
  
  %output = call @softmax(%fc2) : (tensor<1x10xf32>) -> tensor<1x10xf32>
  
  return %output : tensor<1x10xf32>
}
