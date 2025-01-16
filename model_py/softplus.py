import torch
from torch import nn
from torch_mlir import fx


class SoftplusModel(nn.Module):
    def __init__(self):
        super(SoftplusModel, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(x)


model = SoftplusModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/softplus_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("Softplus model exported to softplus_model_linalg-on-tensors.mlir")
