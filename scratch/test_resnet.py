import torch
import torchvision.models as models

resnet = models.resnet34()
x = torch.randn(1, 3, 768, 768)

# Stem
x0 = resnet.relu(resnet.bn1(resnet.conv1(x)))
print("x0:", x0.shape) # Expected 1/2, 64
pool = resnet.maxpool(x0)
x1 = resnet.layer1(pool)
print("x1:", x1.shape) # Expected 1/4, 64
x2 = resnet.layer2(x1)
print("x2:", x2.shape) # Expected 1/8, 128
x3 = resnet.layer3(x2)
print("x3:", x3.shape) # Expected 1/16, 256
x4 = resnet.layer4(x3)
print("x4:", x4.shape) # Expected 1/32, 512
