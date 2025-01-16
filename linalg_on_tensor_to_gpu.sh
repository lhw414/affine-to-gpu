/path/to/mlir-opt /path/to/input_mlir -o /path/to/output_mlir \
--mlir-print-ir-after-failure \
-pass-pipeline="builtin.module(
  canonicalize,
  one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},
  buffer-deallocation-pipeline,
  canonicalize,
  convert-linalg-to-affine-loops,
  func.func(convert-affine-for-to-gpu),
  lower-affine,
  convert-scf-to-cf,
  finalize-memref-to-llvm,
  convert-cf-to-llvm,
  convert-arith-to-llvm,
  convert-func-to-llvm,
  reconcile-unrealized-casts
)"