// Neural Network Example in MLIR
// This demonstrates a complete feedforward neural network for MNIST classification

// Linear layer function (fully connected layer)
func.func @linear_layer(%input: tensor<32x784xf32>, %weight: tensor<784x128xf32>, %bias: tensor<128xf32>) -> tensor<32x128xf32> {
  // Matrix multiplication: input @ weight
  %0 = linalg.matmul ins(%input, %weight : tensor<32x784xf32>, tensor<784x128xf32>) 
                     outs(%init : tensor<32x128xf32>) -> tensor<32x128xf32>
  
  // Add bias
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%bias, %0 : tensor<128xf32>, tensor<32x128xf32>) 
    outs(%init : tensor<32x128xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<32x128xf32>
  
  return %1 : tensor<32x128xf32>
}

// ReLU activation function
func.func @relu(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %max = arith.maximumf %arg0, %c0 : f32
    linalg.yield %max : f32
  } -> tensor<?x?xf32>
  
  return %0 : tensor<?x?xf32>
}

// Softmax activation function
func.func @softmax(%input: tensor<32x10xf32>) -> tensor<32x10xf32> {
  // Step 1: Find max for numerical stability
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<32x10xf32>) outs(%init_max : tensor<32xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %max = arith.maximumf %arg0, %arg1 : f32
    linalg.yield %max : f32
  } -> tensor<32xf32>
  
  // Step 2: Subtract max and compute exp
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input, %0 : tensor<32x10xf32>, tensor<32xf32>) 
    outs(%init : tensor<32x10xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %sub = arith.subf %arg0, %arg1 : f32
    %exp = math.exp %sub : f32
    linalg.yield %exp : f32
  } -> tensor<32x10xf32>
  
  // Step 3: Compute sum of exponentials
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%1 : tensor<32x10xf32>) outs(%init_sum : tensor<32xf32>) {
  ^bb0(%arg0: f32, %arg1: f32):
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  } -> tensor<32xf32>
  
  // Step 4: Divide by sum
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, 
                     affine_map<(d0, d1) -> (d0)>, 
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%1, %2 : tensor<32x10xf32>, tensor<32xf32>) 
    outs(%init : tensor<32x10xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %div = arith.divf %arg0, %arg1 : f32
    linalg.yield %div : f32
  } -> tensor<32x10xf32>
  
  return %3 : tensor<32x10xf32>
}

// Complete neural network
func.func @neural_network(%input: tensor<32x784xf32>, 
                         %w1: tensor<784x128xf32>, %b1: tensor<128xf32>,
                         %w2: tensor<128x64xf32>, %b2: tensor<64xf32>,
                         %w3: tensor<64x10xf32>, %b3: tensor<10xf32>) -> tensor<32x10xf32> {
  
  // Layer 1: Input -> Hidden1 (784 -> 128)
  %layer1 = call @linear_layer(%input, %w1, %b1) : (tensor<32x784xf32>, tensor<784x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
  %relu1 = call @relu(%layer1) : (tensor<32x128xf32>) -> tensor<32x128xf32>
  
  // Layer 2: Hidden1 -> Hidden2 (128 -> 64)  
  %layer2 = call @linear_layer(%relu1, %w2, %b2) : (tensor<32x128xf32>, tensor<128x64xf32>, tensor<64xf32>) -> tensor<32x64xf32>
  %relu2 = call @relu(%layer2) : (tensor<32x64xf32>) -> tensor<32x64xf32>
  
  // Layer 3: Hidden2 -> Output (64 -> 10)
  %layer3 = call @linear_layer(%relu2, %w3, %b3) : (tensor<32x64xf32>, tensor<64x10xf32>, tensor<10xf32>) -> tensor<32x10xf32>
  %output = call @softmax(%layer3) : (tensor<32x10xf32>) -> tensor<32x10xf32>
  
  return %output : tensor<32x10xf32>
}
