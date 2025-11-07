#!/usr/bin/env python3
"""
Test script for different WGAN backbone architectures.
Compares parameter counts and basic functionality across variants.
"""

import torch
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.wgan_factory import create_model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_param_count(count):
    """Format parameter count in a readable way."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.2f}K"
    else:
        return str(count)


def test_model_variants():
    """Test all generator and discriminator variants."""
    print("🧪 Testing WGAN Backbone Variants\n")
    
    # Test configurations
    channels = 3  # RGB images
    in_dim = 100  # Noise dimension
    batch_size = 4
    
    # Generator variants
    generator_variants = [
        "generator_default",
        "generator_resnet", 
        "generator_depthwise",
        "generator_squeeze_excite"
    ]
    
    # Discriminator variants
    discriminator_variants = [
        "discriminator_default",
        "discriminator_resnet",
        "discriminator_depthwise", 
        "discriminator_attention"
    ]
    
    print("📊 Generator Comparison")
    print("-" * 60)
    print(f"{'Variant':<20} {'Parameters':<12} {'Output Shape':<15} {'Status'}")
    print("-" * 60)
    
    generator_results = {}
    
    for variant in generator_variants:
        try:
            # Create model
            model = create_model(variant, channels=channels, in_dim=in_dim)
            param_count = count_parameters(model)
            
            # Test forward pass
            z = torch.randn(batch_size, in_dim, 1, 1)
            with torch.no_grad():
                output = model(z)
            
            output_shape = tuple(output.shape)
            status = "✅ OK"
            
            generator_results[variant] = {
                'params': param_count,
                'output_shape': output_shape,
                'model': model
            }
            
            print(f"{variant:<20} {format_param_count(param_count):<12} {str(output_shape):<15} {status}")
            
        except Exception as e:
            print(f"{variant:<20} {'ERROR':<12} {'N/A':<15} ❌ {str(e)[:30]}")
    
    print("\n📊 Discriminator Comparison")  
    print("-" * 70)
    print(f"{'Variant':<22} {'Parameters':<12} {'Output Shape':<15} {'Feature Shape':<15} {'Status'}")
    print("-" * 70)
    
    discriminator_results = {}
    
    for variant in discriminator_variants:
        try:
            # Create model
            model = create_model(variant, channels=channels)
            param_count = count_parameters(model)
            
            # Test forward pass
            x = torch.randn(batch_size, channels, 32, 32)
            with torch.no_grad():
                output = model(x)
                features = model.feature_extraction(x)
            
            output_shape = tuple(output.shape)
            feature_shape = tuple(features.shape)
            status = "✅ OK"
            
            discriminator_results[variant] = {
                'params': param_count,
                'output_shape': output_shape,
                'feature_shape': feature_shape,
                'model': model
            }
            
            print(f"{variant:<22} {format_param_count(param_count):<12} {str(output_shape):<15} {str(feature_shape):<15} {status}")
            
        except Exception as e:
            print(f"{variant:<22} {'ERROR':<12} {'N/A':<15} {'N/A':<15} ❌ {str(e)[:20]}")
    
    # Parameter comparison analysis
    print("\n📈 Parameter Analysis")
    print("-" * 50)
    
    if generator_results:
        default_gen_params = generator_results.get('generator_default', {}).get('params', 0)
        print(f"Generator baseline (default): {format_param_count(default_gen_params)}")
        
        for variant, results in generator_results.items():
            if variant != 'generator_default':
                ratio = results['params'] / default_gen_params if default_gen_params > 0 else 0
                diff = results['params'] - default_gen_params
                sign = "+" if diff > 0 else ""
                print(f"  {variant}: {sign}{diff:,} params ({ratio:.2f}x)")
    
    if discriminator_results:
        default_disc_params = discriminator_results.get('discriminator_default', {}).get('params', 0)
        print(f"\nDiscriminator baseline (default): {format_param_count(default_disc_params)}")
        
        for variant, results in discriminator_results.items():
            if variant != 'discriminator_default':
                ratio = results['params'] / default_disc_params if default_disc_params > 0 else 0
                diff = results['params'] - default_disc_params
                sign = "+" if diff > 0 else ""
                print(f"  {variant}: {sign}{diff:,} params ({ratio:.2f}x)")
    
    print("\n🎯 Usage Examples")
    print("-" * 30)
    print("# Create models using factory")
    print("from models.wgan_factory import create_model")
    print()
    print("# Default models")
    print("G = create_model('generator_default', channels=3, in_dim=100)")
    print("D = create_model('discriminator_default', channels=3)")
    print()
    print("# ResNet variants")
    print("G_resnet = create_model('generator_resnet', channels=3, in_dim=100)")
    print("D_resnet = create_model('discriminator_resnet', channels=3)")
    print()
    print("# Lightweight variants")
    print("G_dw = create_model('generator_depthwise', channels=3, in_dim=100)")
    print("D_dw = create_model('discriminator_depthwise', channels=3)")
    print()
    print("# Enhanced variants")
    print("G_se = create_model('generator_squeeze_excite', channels=3, in_dim=100)")
    print("D_attn = create_model('discriminator_attention', channels=3)")


if __name__ == "__main__":
    test_model_variants()