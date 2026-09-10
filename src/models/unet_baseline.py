import os
import sys

try:
    import segmentation_models_pytorch as smp
except ImportError:
    vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vendor'))
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    import segmentation_models_pytorch as smp


def get_baseline_model(config):
    model_cfg = config.get('model', {})
    arch = model_cfg.get('architecture', 'UnetPlusPlus')
    encoder_name = model_cfg.get('encoder_name', 'resnet34')
    # Use None for weights to ensure offline compliance on Kaggle
    encoder_weights = None
    in_channels = model_cfg.get('in_channels', 1)
    classes = model_cfg.get('classes', 1)
    
    if arch == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unsupported architecture. Got {arch}")
        
    return model
