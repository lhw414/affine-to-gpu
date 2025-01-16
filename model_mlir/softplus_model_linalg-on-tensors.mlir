#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<6x12xf32>) -> tensor<6x12xf32> {
    %cst = arith.constant 2.000000e+01 : f64
    %0 = tensor.empty() : tensor<6x12xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x12xf32>) outs(%0 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.exp %in : f32
      linalg.yield %6 : f32
    } -> tensor<6x12xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<6x12xf32>) outs(%0 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = math.log1p %in : f32
      linalg.yield %6 : f32
    } -> tensor<6x12xf32>
    %3 = tensor.empty() : tensor<6x12xi1>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x12xf32>) outs(%3 : tensor<6x12xi1>) {
    ^bb0(%in: f32, %out: i1):
      %6 = arith.extf %in : f32 to f64
      %7 = arith.cmpf ogt, %6, %cst : f64
      linalg.yield %7 : i1
    } -> tensor<6x12xi1>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg0, %2 : tensor<6x12xi1>, tensor<6x12xf32>, tensor<6x12xf32>) outs(%0 : tensor<6x12xf32>) {
    ^bb0(%in: i1, %in_0: f32, %in_1: f32, %out: f32):
      %6 = arith.select %in, %in_0, %in_1 : f32
      linalg.yield %6 : f32
    } -> tensor<6x12xf32>
    return %5 : tensor<6x12xf32>
  }
}
