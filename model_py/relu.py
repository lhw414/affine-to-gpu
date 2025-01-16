import torch
from torch import nn
from torch_mlir import fx


class ReLUModel(nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


model = ReLUModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/relu_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("ReLU model exported to relu_model_linalg-on-tensors.mlir")
