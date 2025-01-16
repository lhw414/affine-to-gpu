import torch
from torch import nn
from torch_mlir import fx


class ELUModel(nn.Module):
    def __init__(self):
        super(ELUModel, self).__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x)


model = ELUModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/elu_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("ELU model exported to elu_model_linalg-on-tensors.mlir")
