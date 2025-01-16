import torch
from torch import nn
from torch_mlir import fx


class DropoutModel(nn.Module):
    def __init__(self):
        super(DropoutModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        return self.dropout(x)


model = DropoutModel().train()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/dropout_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("Dropout model exported to dropout_model_linalg-on-tensors.mlir")
