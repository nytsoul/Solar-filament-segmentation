import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x

class UnetPlusPlus(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights=None, in_channels=1, classes=1):
        super().__init__()
        
        # Load ResNet34 from torchvision (no internet download if weights=None)
        resnet = models.resnet34(weights=None) 
        
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1) # 64
        self.encoder2 = resnet.layer2 # 128
        self.encoder3 = resnet.layer3 # 256
        self.encoder4 = resnet.layer4 # 512
        
        # Level 3
        self.conv3_1 = DecoderBlock(256 + 512, 256)
        
        # Level 2
        self.conv2_1 = DecoderBlock(128 + 256, 128)
        self.conv2_2 = DecoderBlock(128 + 128 + 256, 128)
        
        # Level 1
        self.conv1_1 = DecoderBlock(64 + 128, 64)
        self.conv1_2 = DecoderBlock(64 + 64 + 128, 64)
        self.conv1_3 = DecoderBlock(64 + 64 + 64 + 128, 64)
        
        # Level 0
        self.conv0_1 = DecoderBlock(64 + 64, 32)
        self.conv0_2 = DecoderBlock(64 + 32 + 64, 32)
        self.conv0_3 = DecoderBlock(64 + 32*2 + 64, 32)
        self.conv0_4 = DecoderBlock(64 + 32*3 + 64, 32)
        
        self.final_up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.encoder0(x)
        x1_0 = self.encoder1(x0_0)
        x2_0 = self.encoder2(x1_0)
        x3_0 = self.encoder3(x2_0)
        x4_0 = self.encoder4(x3_0)
        
        up = lambda feat: F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
        
        x3_1 = self.conv3_1(torch.cat([x3_0, up(x4_0)], 1))
        
        x2_1 = self.conv2_1(torch.cat([x2_0, up(x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, up(x3_1)], 1))
        
        x1_1 = self.conv1_1(torch.cat([x1_0, up(x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up(x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, up(x2_2)], 1))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, up(x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up(x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up(x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up(x1_3)], 1))
        
        out = self.final_up(x0_4)
        out = self.final_conv(out)
        return out

def get_baseline_model(config):
    model_cfg = config.get('model', {})
    arch = model_cfg.get('architecture', 'UnetPlusPlus')
    encoder_name = model_cfg.get('encoder_name', 'resnet34')
    encoder_weights = model_cfg.get('encoder_weights', 'imagenet')
    in_channels = model_cfg.get('in_channels', 1)
    classes = model_cfg.get('classes', 1)
    
    # We ignore internet-dependent encoder_weights for Kaggle offline compliance
    
    if arch == 'UnetPlusPlus' and encoder_name == 'resnet34':
        model = UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Custom implementation only supports UnetPlusPlus with resnet34. Got {arch} and {encoder_name}")
        
    return model
