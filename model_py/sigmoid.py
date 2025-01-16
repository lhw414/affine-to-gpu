import torch
from torch import nn
from torch_mlir import fx


class SigmoidModel(nn.Module):
    def __init__(self):
        super(SigmoidModel, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x)


model = SigmoidModel().eval()
example_input = torch.randn(6, 12)

mlir_module = fx.export_and_import(
    model, example_input, output_type="linalg-on-tensors"
)

with open("../model_mlir/sigmoid_model_linalg-on-tensors.mlir", "w") as f:
    f.write(str(mlir_module.operation))

print("Sigmoid model exported to sigmoid_model_linalg-on-tensors.mlir")
