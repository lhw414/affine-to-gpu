## How to convert torch-mlir(linalg-on-tensors) to llvm-ir(for cpu) and gpu dialect(for gpu)

### 1. pytorch model to torch-mlir

In ./model_py folder, there are some torch models that contain simple layer like Relu, LayerNorm, sigmoid, tanh, etc.
Thery're already converted to torch-mlir format using [torch-mlir-framework](https://github.com/llvm/torch-mlir). You can see torch-mlir(linalg-on-tensors dialect) code in ./model_mlir folder.

### 2-1. torch-mlir to llvm-ir(for cpu)

Using ./linalg_on_tensor_to_cpu.sh, you can convert torch-mlir to llvm-ir.
The conversion works well, but the executable generated from the LLVM IR is not running properly due to a segmentation fault error. (This needs to be checked later)

### 2-2. torch-mlir to gpu dialect(for gpu)

Using ./linalg_on_tensor_to_gpu.sh, you can convert torch-mlir to gpu dialect.
In the convert-affine-for-to-gpu pass, an error such as "'affine.load' op index must be a valid dimension or symbol identifier" occurs, causing gpu conversion to fail.

### 3. LLVM MLIR Passes

The following are brief descriptions of the LLVM MLIR passes used in the conversion process:

- `canonicalize`: Simplifies and canonicalizes the IR by applying various transformations.
- `one-shot-bufferize`: Converts tensor operations to buffer operations, preparing for memory allocation.
- `buffer-deallocation-pipeline`: Manages the deallocation of buffers to avoid memory leaks.
- `convert-linalg-to-affine-loops`: Converts Linalg operations to affine loops.
- `convert-affine-for-to-gpu`: Converts affine loops to GPU-compatible operations.
- `lower-affine`: Lowers affine constructs to lower-level IR.
- `convert-scf-to-cf`: Converts structured control flow (SCF) operations to control flow (CF) operations.
- `finalize-memref-to-llvm`: Finalizes the conversion of memory references to LLVM IR.
- `convert-cf-to-llvm`: Converts control flow operations to LLVM IR.
- `convert-arith-to-llvm`: Converts arithmetic operations to LLVM IR.
- `convert-func-to-llvm`: Converts function operations to LLVM IR.
- `reconcile-unrealized-casts`: Reconciles any unrealized casts in the IR.

`--mlir-print-ir-after-failure`: For debugging purposes, prints the IR after a failure occurs.
