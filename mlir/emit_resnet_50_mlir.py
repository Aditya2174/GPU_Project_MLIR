import torch
import torchvision.models as models
from torch_mlir import fx
import torch_mlir
from torch_mlir import torchscript

m = models.resnet50(weights="IMAGENET1K_V1").eval()
example = torch.randn(1, 3, 224, 224)

module = torchscript.compile(
    m, example, output_type="linalg-on-tensors"
)

text = str(module)

with open("resnet50_linalg.mlir", "w") as f:
    f.write(text)

