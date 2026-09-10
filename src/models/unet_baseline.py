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
