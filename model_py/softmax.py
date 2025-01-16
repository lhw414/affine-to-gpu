import torch
from torch import nn
from torch_mlir import fx


class SoftmaxModel(nn.Module):
    def __init__(self, dim):
        super(SoftmaxModel, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.softmax(x)


model = SoftmaxModel(dim=-1).eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/softmax_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("Softmax model exported to softmax_model_linalg-on-tensors.mlir")
