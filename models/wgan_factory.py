"""
WGAN Model Factory with Registry Pattern

This module provides a simple and clean model factory for WGAN (Wasserstein GAN) models
using a registry pattern for dynamic model creation.

Usage Examples:
    
    # New backbone config interface (recommended)
    generator = create_model(
        cfg.models.backbone.name,
        cfg.models.backbone.generator
    ).to(device)
    
    discriminator = create_model(
        cfg.models.backbone.name,
        cfg.models.backbone.discriminator  
    ).to(device)
    
    # Legacy interface (still supported)
    generator = create_model("generator_default", channels=3, in_dim=100)
    discriminator = create_model("discriminator_default", channels=3)
    
    # Backbone config structure example:
    # cfg.models.backbone.generator = {
    #     "name": "generator_default",
    #     "channels": 3,
    #     "in_dim": 100
    # }
    # cfg.models.backbone.discriminator = {
    #     "name": "discriminator_spectral_norm",
    #     "channels": 3
    # }
    
    # Complete GAN training example with backbone configs
    import torch
    
    # Create models using backbone configuration
    G = create_model(cfg.models.backbone.name, cfg.models.backbone.generator)
    D = create_model(cfg.models.backbone.name, cfg.models.backbone.discriminator)
    
    # Move to device and use as normal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    
    # Generate fake images
    z = torch.randn(64, cfg.models.backbone.generator["in_dim"], 1, 1).to(device)
    fake_images = G(z)  # Shape: (64, 3, 32, 32)
    
    # Discriminate images
    real_images = torch.randn(64, 3, 32, 32).to(device)
    d_real = D(real_images)
    d_fake = D(fake_images)

Available Models:
    Generator Variants:
    - "generator_default" or "generator": Standard DCGAN Generator (12.14M params)
        Args: channels (int), in_dim (int)
    - "generator_resnet": ResNet-style Generator with skip connections (25.58M params)  
        Args: channels (int), in_dim (int)
    - "generator_depthwise": Lightweight Generator with depthwise separable convolutions (2.34M params)
        Args: channels (int), in_dim (int)
    - "generator_squeeze_excite": Generator with Squeeze-and-Excitation blocks (12.18M params)
        Args: channels (int), in_dim (int)
        
    Discriminator Variants:
    - "discriminator_default" or "discriminator": Standard DCGAN Discriminator (10.52M params)
        Args: channels (int), use_spectral_norm (bool, default=False), use_normalization (bool, default=True)
    - "discriminator_wgan_gp": Pure WGAN-GP Discriminator without normalization layers (9.18M params)
        Args: channels (int)
    - "discriminator_spectral_norm": Discriminator with spectral normalization - recommended for WGAN-GP (9.18M params)
        Args: channels (int)
    - "discriminator_resnet": ResNet-style Discriminator with skip connections (22.98M params)
        Args: channels (int), use_spectral_norm (bool, default=True), use_normalization (bool, default=False)
    - "discriminator_depthwise": Ultra-lightweight Discriminator with depthwise separable convolutions (0.70M params)
        Args: channels (int), use_spectral_norm (bool, default=True)
    - "discriminator_attention": Discriminator with self-attention mechanism (10.85M params)
        Args: channels (int), use_spectral_norm (bool, default=True)


Model Specifications:
    Generator:
        - Input: Noise vector (batch_size, in_dim, 1, 1)
        - Output: Generated images (batch_size, channels, 32, 32)
        - Architecture: 4-layer transposed convolution with batch normalization
        - Activation: ReLU for hidden layers, Tanh for output
        
    Discriminator:
        - Input: Images (batch_size, channels, 32, 32)
        - Output: Critic scores (batch_size, 1, 1, 1)
        - Architecture: 4-layer convolution with instance normalization
        - Activation: LeakyReLU throughout (no sigmoid at output for WGAN)
        - Additional: feature_extraction() method for intermediate features

Registry Pattern:
    - Use @register("model_name") to register new model classes
    - Use create_model("model_name", **kwargs) to instantiate models
    - Models are automatically registered when this module is imported
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from omegaconf import OmegaConf

# Model Registry
_MODELS = {}


def initialize_weights(module, init_type='dcgan'):
    """
    高效的权重初始化函数，针对WGAN-GP优化
    
    Args:
        module: 要初始化的模块
        init_type: 初始化类型 ('dcgan', 'xavier', 'orthogonal', 'kaiming')
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        # 卷积层初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1):
            if init_type == 'dcgan':
                # DCGAN paper推荐的初始化，对GAN效果最佳
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                # Xavier初始化，适合深层网络
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                # He初始化，适合ReLU激活
                nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                # 正交初始化，保持梯度稳定
                nn.init.orthogonal_(m.weight.data, gain=0.02)
            
            # 偏置项初始化
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        # BatchNorm和InstanceNorm初始化
        elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        # GroupNorm初始化
        elif classname.find('GroupNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        # Linear层初始化
        elif classname.find('Linear') != -1:
            if init_type == 'dcgan':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        
        # Embedding层初始化
        elif classname.find('Embedding') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    module.apply(init_func)


def spectral_norm_init(module):
    """
    为使用谱归一化的模块进行特殊初始化
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 谱归一化模块使用更小的初始化方差
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    module.apply(init_func)


def generator_specific_init(module):
    """
    Generator专用初始化，追求更好的生成质量
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if classname.find('ConvTranspose') != -1:
            # 转置卷积使用双线性插值权重初始化，减少棋盘效应
            if hasattr(m, 'weight'):
                # 使用较小的标准差避免梯度爆炸
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                
        elif classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
        # 偏置项统一初始化为0
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
        # BatchNorm初始化
        elif classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    module.apply(init_func)


def discriminator_specific_init(module, use_spectral_norm=False):
    """
    Discriminator专用初始化，追求训练稳定性
    """
    def init_func(m):
        classname = m.__class__.__name__
        
        if classname.find('Conv') != -1:
            if use_spectral_norm:
                # 谱归一化情况下使用更保守的初始化
                nn.init.normal_(m.weight.data, 0.0, 0.01)
            else:
                # 标准初始化
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                
        elif classname.find('Linear') != -1:
            if use_spectral_norm:
                nn.init.normal_(m.weight.data, 0.0, 0.01)
            else:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        
        # 偏置项初始化
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
        # InstanceNorm初始化（WGAN-GP推荐）
        elif classname.find('InstanceNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    module.apply(init_func)


def register(name):
    """
    Decorator to register model classes in the factory.

    Args:
        name (str): Name to register the model under

    Returns:
        Decorated class that is registered in the model factory

    Example:
        @register("my_model")
        class MyModel(nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                # model definition
    """

    def decorator(cls):
        _MODELS[name] = cls
        return cls

    return decorator


def create_model(name, config=None, **kwargs):
    """
    Factory function to create model instances.
    
    Supports two calling patterns:
    1. Legacy: create_model("model_name", param1=value1, param2=value2)
    2. New: create_model("backbone_name", {"name": "model_name", "param1": value1, ...})

    Args:
        name (str): Name of the registered model OR backbone name
        config (dict, optional): Configuration dict containing 'name' and parameters
        **kwargs: Arguments to pass to the model constructor (legacy mode)

    Returns:
        nn.Module: Instantiated model

    Raises:
        KeyError: If model name is not registered
        ValueError: If config format is invalid

    Examples:
        # Legacy mode
        model = create_model("generator_default", channels=3, in_dim=100)
        
        # New backbone config mode
        generator_config = {"name": "generator_default", "channels": 3, "z_dim": 100}
        model = create_model("wgan-gp", generator_config)
    """
    # Handle legacy mode first
    if config is None:
        # Legacy mode - direct model creation
        if name not in _MODELS:
            available = list(_MODELS.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")
        return _MODELS[name](**kwargs)
    
    # New config-based mode
    config = OmegaConf.to_container(config, resolve=True)
    # Remove checkpoint_path from config if present
    if isinstance(config, dict) and 'checkpoint_path' in config:
        config = {k: v for k, v in config.items() if k != 'checkpoint_path'}
    
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    
    if 'name' not in config:
        raise ValueError("Config must contain 'name' field")
    
    model_name = config['name']
    # Extract parameters from config, excluding 'name'
    model_params = {k: v for k, v in config.items() if k != 'name'}
    
    if model_name not in _MODELS:
        available = list(_MODELS.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available}")
    
    return _MODELS[model_name](**model_params)


@register("generator_default")
class Generator(nn.Module):

    def __init__(self, channels, in_dim):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(
                in_channels=in_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            # State (1024x4x4)
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()
        
        # 应用优化的Generator初始化
        generator_specific_init(self)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


@register("discriminator_default")
class Discriminator(nn.Module):

    def __init__(self, channels, use_spectral_norm=False, use_normalization=True):
        super().__init__()
        
        def conv_block(in_ch, out_ch, kernel_size, stride, padding, first_layer=False):
            """创建卷积块，可选择性添加谱归一化和实例归一化"""
            conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            
            # Apply spectral normalization if requested
            if use_spectral_norm:
                conv = spectral_norm(conv)
            
            layers = [conv]
            
            # Add normalization (except for first layer and if disabled)
            # WGAN-GP理论建议第一层不使用归一化
            if use_normalization and not first_layer:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx32x32)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            conv_block(channels, 256, 4, 2, 1, first_layer=True),
            # State (256x16x16)
            conv_block(256, 512, 4, 2, 1),
            # State (512x8x8)
            conv_block(512, 1024, 4, 2, 1),
        )
        # output of main module --> State (1024x4x4)

        # Output layer
        output_conv = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
        if use_spectral_norm:
            output_conv = spectral_norm(output_conv)
        
        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            output_conv
        )
        
        # 应用优化的Discriminator初始化
        discriminator_specific_init(self, use_spectral_norm=use_spectral_norm)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


# Additional Generator Variants with Different Backbones

@register("generator_resnet")
class ResNetGenerator(nn.Module):
    """ResNet-style Generator with skip connections for improved gradient flow"""
    
    def __init__(self, channels, in_dim):
        super().__init__()
        # Initial projection
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        
        # ResNet blocks with upsampling
        self.res_block1 = self._make_res_block(1024, 512, upsample=True)
        self.res_block2 = self._make_res_block(512, 256, upsample=True)
        
        # Final output layer
        self.final = nn.ConvTranspose2d(256, channels, 4, 2, 1)
        self.output = nn.Tanh()
        
        # ResNet特殊初始化：使用Xavier初始化提升梯度流
        self._init_weights()
    
    def _init_weights(self):
        """ResNet专用权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # ResNet使用Xavier初始化，保证残差连接稳定
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
    
    def _make_res_block(self, in_channels, out_channels, upsample=False):
        layers = []
        
        # Main path
        if upsample:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        
        layers.extend([
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        # Skip connection
        skip = nn.Sequential()
        if in_channels != out_channels:
            if upsample:
                skip = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
        
        return nn.ModuleDict({'main': nn.Sequential(*layers), 'skip': skip})
    
    def forward(self, x):
        x = self.initial(x)
        
        # ResNet block 1
        residual = self.res_block1['skip'](x)
        x = self.res_block1['main'](x) + residual
        x = torch.relu(x)
        
        # ResNet block 2
        residual = self.res_block2['skip'](x)
        x = self.res_block2['main'](x) + residual
        x = torch.relu(x)
        
        # Final output
        x = self.final(x)
        return self.output(x)


@register("generator_depthwise")
class DepthwiseGenerator(nn.Module):
    """Generator using depthwise separable convolutions for efficiency"""
    
    def __init__(self, channels, in_dim):
        super().__init__()
        self.main_module = nn.Sequential(
            # Initial layer
            nn.ConvTranspose2d(in_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            
            # Depthwise separable blocks
            self._depthwise_block(1024, 512, 4, 2, 1),
            self._depthwise_block(512, 256, 4, 2, 1),
            
            # Final layer
            nn.ConvTranspose2d(256, channels, 4, 2, 1),
        )
        self.output = nn.Tanh()
        
        # Depthwise卷积特殊初始化
        self._init_depthwise_weights()
    
    def _init_depthwise_weights(self):
        """Depthwise卷积专用初始化"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if m.groups > 1:  # Depthwise convolution
                    # Depthwise使用更小的方差防止梯度消失
                    nn.init.normal_(m.weight.data, 0.0, 0.01)
                else:  # Regular or pointwise convolution
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
    
    def _depthwise_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # Depthwise convolution
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding, 
                             groups=in_channels, bias=False),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


@register("generator_squeeze_excite")
class SqueezeExciteGenerator(nn.Module):
    """Generator with Squeeze-and-Excitation blocks for adaptive feature recalibration"""
    
    def __init__(self, channels, in_dim):
        super().__init__()
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 1024, 4, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        
        # SE blocks
        self.se_block1 = self._make_se_block(1024, 512, upsample=True)
        self.se_block2 = self._make_se_block(512, 256, upsample=True)
        
        self.final = nn.ConvTranspose2d(256, channels, 4, 2, 1)
        self.output = nn.Tanh()
        
        # SE模块特殊初始化
        self._init_se_weights()
    
    def _init_se_weights(self):
        """Squeeze-and-Excitation专用初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        
        # SE模块的最后一层使用更小的初始化，避免过度激活
        for name, module in self.named_modules():
            if 'se' in name and isinstance(module, nn.Conv2d):
                # SE的gate层使用更小的初始化
                if module.out_channels == module.in_channels:  # Final SE layer
                    nn.init.normal_(module.weight.data, 0.0, 0.01)
    
    def _make_se_block(self, in_channels, out_channels, upsample=False, reduction=16):
        # Main convolution
        if upsample:
            conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        
        # Squeeze-and-Excitation
        se_channels = max(out_channels // reduction, 8)
        se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, se_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(se_channels, out_channels, 1),
            nn.Sigmoid(),
        )
        
        return nn.ModuleDict({
            'conv': conv,
            'bn': nn.BatchNorm2d(out_channels),
            'se': se
        })
    
    def forward(self, x):
        x = self.initial(x)
        
        # SE block 1
        x = self.se_block1['conv'](x)
        x = self.se_block1['bn'](x)
        x = torch.relu(x)
        se_weight = self.se_block1['se'](x)
        x = x * se_weight
        
        # SE block 2
        x = self.se_block2['conv'](x)
        x = self.se_block2['bn'](x)
        x = torch.relu(x)
        se_weight = self.se_block2['se'](x)
        x = x * se_weight
        
        # Final output
        x = self.final(x)
        return self.output(x)


# Additional Discriminator Variants with Different Backbones

@register("discriminator_wgan_gp")  
class WGANGPDiscriminator(nn.Module):
    """WGAN-GP专用判别器 - 完全移除归一化层以符合理论要求"""
    
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # 完全移除归一化层，符合WGAN-GP理论
            nn.Conv2d(channels, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1024, 4, 2, 1),  
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.output = nn.Conv2d(1024, 1, 4, 1, 0)
        
        # WGAN-GP判别器特殊初始化
        initialize_weights(self, init_type='dcgan')

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


@register("discriminator_spectral_norm")
class SpectralNormDiscriminator(nn.Module):
    """使用谱归一化的判别器 - 推荐用于WGAN-GP训练"""
    
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # 使用谱归一化替代实例归一化，符合1-Lipschitz约束
            spectral_norm(nn.Conv2d(channels, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.output = spectral_norm(nn.Conv2d(1024, 1, 4, 1, 0))
        
        # 谱归一化专用初始化
        spectral_norm_init(self)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


@register("discriminator_resnet")
class ResNetDiscriminator(nn.Module):
    """ResNet-style Discriminator with skip connections and optional spectral normalization"""
    
    def __init__(self, channels, use_spectral_norm=True, use_normalization=False):
        super().__init__()
        
        def maybe_spectral_norm(layer):
            return spectral_norm(layer) if use_spectral_norm else layer
        
        def maybe_norm(channels):
            if use_normalization:
                return nn.InstanceNorm2d(channels, affine=True)
            else:
                return nn.Identity()
        
        self.initial = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(channels, 256, 4, 2, 1)),
            maybe_norm(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # ResNet blocks with downsampling
        self.res_block1 = self._make_res_block(256, 512, downsample=True, 
                                               use_spectral_norm=use_spectral_norm, 
                                               use_normalization=use_normalization)
        self.res_block2 = self._make_res_block(512, 1024, downsample=True,
                                               use_spectral_norm=use_spectral_norm, 
                                               use_normalization=use_normalization)
        
        output_conv = nn.Conv2d(1024, 1, 4, 1, 0)
        self.output = maybe_spectral_norm(output_conv)
        
        # ResNet判别器特殊初始化
        if use_spectral_norm:
            spectral_norm_init(self)
        else:
            discriminator_specific_init(self, use_spectral_norm=False)
    
    def _make_res_block(self, in_channels, out_channels, downsample=False, 
                        use_spectral_norm=True, use_normalization=False):
        
        def maybe_spectral_norm_local(layer):
            return spectral_norm(layer) if use_spectral_norm else layer
        
        def maybe_norm_local(channels):
            if use_normalization:
                return nn.InstanceNorm2d(channels, affine=True)
            else:
                return nn.Identity()
        
        layers = []
        
        # Main path
        if downsample:
            layers.append(maybe_spectral_norm_local(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)))
        else:
            layers.append(maybe_spectral_norm_local(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)))
        
        layers.extend([
            maybe_norm_local(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_spectral_norm_local(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)),
            maybe_norm_local(out_channels),
        ])
        
        # Skip connection
        skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            skip_layers = []
            if downsample:
                skip_layers.append(nn.AvgPool2d(2))
            skip_layers.extend([
                maybe_spectral_norm_local(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)),
                maybe_norm_local(out_channels)
            ])
            skip = nn.Sequential(*skip_layers)
        
        return nn.ModuleDict({'main': nn.Sequential(*layers), 'skip': skip})
    
    def forward(self, x):
        x = self.initial(x)
        
        # ResNet block 1
        residual = self.res_block1['skip'](x)
        x = self.res_block1['main'](x) + residual
        x = torch.relu(x)
        
        # ResNet block 2  
        residual = self.res_block2['skip'](x)
        x = self.res_block2['main'](x) + residual
        x = torch.relu(x)
        
        return self.output(x)
    
    def feature_extraction(self, x):
        x = self.initial(x)
        
        # ResNet blocks for feature extraction
        residual = self.res_block1['skip'](x)
        x = self.res_block1['main'](x) + residual
        x = torch.relu(x)
        
        residual = self.res_block2['skip'](x)
        x = self.res_block2['main'](x) + residual
        x = torch.relu(x)
        
        return x.view(-1, 1024 * 4 * 4)


@register("discriminator_depthwise")
class DepthwiseDiscriminator(nn.Module):
    """Discriminator using depthwise separable convolutions with optional spectral normalization"""
    
    def __init__(self, channels, use_spectral_norm=True):
        super().__init__()
        self.main_module = nn.Sequential(
            # Initial layer
            self._maybe_spectral_norm(nn.Conv2d(channels, 256, 4, 2, 1), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Depthwise separable blocks
            self._depthwise_block(256, 512, 4, 2, 1, use_spectral_norm),
            self._depthwise_block(512, 1024, 4, 2, 1, use_spectral_norm),
        )
        
        self.output = self._maybe_spectral_norm(nn.Conv2d(1024, 1, 4, 1, 0), use_spectral_norm)
        
        # Depthwise判别器特殊初始化
        if use_spectral_norm:
            spectral_norm_init(self)
        else:
            self._init_depthwise_discriminator()
    
    def _init_depthwise_discriminator(self):
        """Depthwise判别器专用初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups > 1:  # Depthwise convolution
                    nn.init.normal_(m.weight.data, 0.0, 0.01)
                else:  # Regular or pointwise convolution
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
    
    def _maybe_spectral_norm(self, layer, use_spectral_norm):
        return spectral_norm(layer) if use_spectral_norm else layer
    
    def _depthwise_block(self, in_channels, out_channels, kernel_size, stride, padding, use_spectral_norm=True):
        return nn.Sequential(
            # Depthwise convolution
            self._maybe_spectral_norm(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                         groups=in_channels, bias=False), use_spectral_norm),
            # Pointwise convolution
            self._maybe_spectral_norm(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
    
    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


# Convenience aliases for backward compatibility
@register("generator")
def create_default_generator(**kwargs):
    return Generator(**kwargs)


@register("discriminator")
def create_default_discriminator(**kwargs):
    return Discriminator(**kwargs)
