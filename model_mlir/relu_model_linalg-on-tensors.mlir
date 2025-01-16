#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @main(%arg0: tensor<6x12xf32>) -> tensor<6x12xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<6x12xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<6x12xf32>) outs(%0 : tensor<6x12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ugt, %in, %cst : f32
      %3 = arith.select %2, %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<6x12xf32>
    return %1 : tensor<6x12xf32>
  }
}
