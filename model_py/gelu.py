import torch
from torch import nn
from torch_mlir import fx


class GELUModel(nn.Module):
    def __init__(self):
        super(GELUModel, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)


model = GELUModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/gelu_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("GELU model exported to gelu_model_linalg-on-tensors.mlir")
