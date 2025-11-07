"""
WGAN Model Factory with Registry Pattern

This module provides a simple and clean model factory for WGAN (Wasserstein GAN) models
using a registry pattern for dynamic model creation.

Usage Examples:
    
    # Basic usage - create models directly
    generator = create_model("generator", channels=3, in_dim=100)
    discriminator = create_model("discriminator", channels=3)
    
    # Complete example for GAN training
    import torch
    
    # Create models
    G = create_model("generator", channels=3, in_dim=100)
    D = create_model("discriminator", channels=3)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    
    # Generate fake images
    z = torch.randn(64, 100, 1, 1).to(device)  # Batch of noise vectors
    fake_images = G(z)  # Shape: (64, 3, 32, 32)
    
    # Discriminate real/fake images
    real_images = torch.randn(64, 3, 32, 32).to(device)  # Real image batch
    d_real = D(real_images)  # Discriminator output for real images
    d_fake = D(fake_images)  # Discriminator output for fake images

Available Models:
    - "generator": WGAN Generator
        Args:
            channels (int): Number of output channels (e.g., 3 for RGB, 1 for grayscale)
            in_dim (int): Input noise dimension (typically 100)
        
    - "discriminator": WGAN Discriminator/Critic  
        Args:
            channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale)

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

# Model Registry
_MODELS = {}


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


def create_model(name, **kwargs):
    """
    Factory function to create model instances.

    Args:
        name (str): Name of the registered model
        **kwargs: Arguments to pass to the model constructor

    Returns:
        nn.Module: Instantiated model

    Raises:
        KeyError: If model name is not registered

    Example:
        model = create_model("generator", channels=3, in_dim=100)
    """
    if name not in _MODELS:
        available = list(_MODELS.keys())
        raise KeyError(f"Model '{name}' not found. Available models: {available}")
    return _MODELS[name](**kwargs)


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

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


@register("discriminator_default")
class Discriminator(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(
                in_channels=channels,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(
                in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0
            )
        )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)
