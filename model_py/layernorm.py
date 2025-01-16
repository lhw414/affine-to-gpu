import torch
from torch import nn
from torch_mlir import fx


class LayerNormModel(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNormModel, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.layer_norm(x)


model = LayerNormModel(12).eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/layer_norm_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("LayerNorm model exported to layer_norm_model_linalg-on-tensors.mlir")
