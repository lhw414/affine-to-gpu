#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<6x12xf32>) -> tensor<6x12xf32> {
    %cst = arith.constant dense_resource<torch_tensor_12_torch.float32> : tensor<12xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-05 : f64
    %cst_2 = arith.constant 1.200000e+01 : f32
    %cst_3 = arith.constant dense_resource<torch_tensor_12_torch.float32_1> : tensor<12xf32>
    %0 = tensor.empty() : tensor<6x1xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<6x1xf32>) -> tensor<6x1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<6x12xf32>) outs(%1 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.addf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<6x1xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<6x1xf32>) outs(%0 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.divf %in, %cst_2 : f32
      linalg.yield %16 : f32
    } -> tensor<6x1xf32>
    %4 = tensor.empty() : tensor<6x12xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<6x1xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<6x12xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %5 : tensor<6x12xf32>, tensor<6x12xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.subf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<6x12xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %6 : tensor<6x12xf32>, tensor<6x12xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.mulf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<6x12xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<6x12xf32>) outs(%1 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.addf %in, %out : f32
      linalg.yield %16 : f32
    } -> tensor<6x1xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<6x1xf32>) outs(%0 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.divf %in, %cst_2 : f32
      linalg.yield %16 : f32
    } -> tensor<6x1xf32>
    %10 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<6x1xf32>) outs(%0 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.truncf %cst_1 : f64 to f32
      %17 = arith.addf %in, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<6x1xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<6x1xf32>) outs(%0 : tensor<6x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = math.rsqrt %in : f32
      linalg.yield %16 : f32
    } -> tensor<6x1xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<6x1xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<6x12xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%6, %12 : tensor<6x12xf32>, tensor<6x12xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.mulf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<6x12xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %cst : tensor<6x12xf32>, tensor<12xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.mulf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<6x12xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%14, %cst_3 : tensor<6x12xf32>, tensor<12xf32>) outs(%4 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %16 = arith.addf %in, %in_4 : f32
      linalg.yield %16 : f32
    } -> tensor<6x12xf32>
    return %15 : tensor<6x12xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_12_torch.float32: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_12_torch.float32_1: "0x04000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    }
  }
#-}
