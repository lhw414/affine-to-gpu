import torch
from torch import nn
from torch_mlir import fx


class TanhModel(nn.Module):
    def __init__(self):
        super(TanhModel, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x)


model = TanhModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/tanh_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("Tanh model exported to tanh_model_linalg-on-tensors.mlir")
