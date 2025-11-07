#!/usr/bin/env python3
"""
Test script specifically for WGAN-GP discriminator variants.
Tests different normalization strategies for WGAN-GP compliance.
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


def test_wgan_gp_variants():
    """Test WGAN-GP specific discriminator variants."""
    print("🔧 Testing WGAN-GP Discriminator Variants\n")
    
    channels = 3
    batch_size = 4
    
    # WGAN-GP specific discriminator variants
    wgan_gp_variants = [
        ("discriminator_default (default)", "discriminator_default", {"channels": channels}),
        ("discriminator_default (no norm)", "discriminator_default", {"channels": channels, "use_normalization": False}),
        ("discriminator_default (spectral)", "discriminator_default", {"channels": channels, "use_spectral_norm": True}),
        ("discriminator_default (spectral+no norm)", "discriminator_default", {"channels": channels, "use_spectral_norm": True, "use_normalization": False}),
        ("discriminator_wgan_gp", "discriminator_wgan_gp", {"channels": channels}),
        ("discriminator_spectral_norm", "discriminator_spectral_norm", {"channels": channels}),
        ("discriminator_resnet (spectral)", "discriminator_resnet", {"channels": channels}),
        ("discriminator_depthwise (spectral)", "discriminator_depthwise", {"channels": channels}),
        ("discriminator_attention (spectral)", "discriminator_attention", {"channels": channels}),
    ]
    
    print("📊 WGAN-GP Discriminator Comparison")
    print("-" * 80)
    print(f"{'Variant':<35} {'Parameters':<12} {'Output Shape':<15} {'Status'}")
    print("-" * 80)
    
    results = {}
    
    for name, model_type, kwargs in wgan_gp_variants:
        try:
            # Create model
            model = create_model(model_type, **kwargs)
            param_count = count_parameters(model)
            
            # Test forward pass
            x = torch.randn(batch_size, channels, 32, 32)
            with torch.no_grad():
                output = model(x)
                features = model.feature_extraction(x)
            
            output_shape = tuple(output.shape)
            feature_shape = tuple(features.shape)
            status = "✅ OK"
            
            results[name] = {
                'params': param_count,
                'output_shape': output_shape,
                'feature_shape': feature_shape,
                'model': model
            }
            
            print(f"{name:<35} {format_param_count(param_count):<12} {str(output_shape):<15} {status}")
            
        except Exception as e:
            print(f"{name:<35} {'ERROR':<12} {'N/A':<15} ❌ {str(e)[:30]}")
    
    # Analyze parameter differences due to normalization strategies
    print("\n📈 Parameter Analysis by Normalization Strategy")
    print("-" * 60)
    
    default_params = results.get("discriminator_default (default)", {}).get('params', 0)
    if default_params > 0:
        print(f"Baseline (InstanceNorm): {format_param_count(default_params)}")
        
        # Compare different normalization strategies
        comparisons = [
            ("No normalization", "discriminator_default (no norm)"),
            ("Spectral norm only", "discriminator_default (spectral+no norm)"),
            ("Pure WGAN-GP", "discriminator_wgan_gp"),
            ("Spectral norm (recommended)", "discriminator_spectral_norm"),
        ]
        
        for strategy, variant_name in comparisons:
            if variant_name in results:
                variant_params = results[variant_name]['params']
                diff = variant_params - default_params
                ratio = variant_params / default_params
                sign = "+" if diff > 0 else ""
                print(f"  {strategy:<25}: {sign}{diff:,} params ({ratio:.3f}x)")
    
    print("\n🎯 WGAN-GP Usage Recommendations")
    print("-" * 50)
    print("✅ Recommended for WGAN-GP:")
    print("  1. discriminator_spectral_norm (best theoretical compliance)")
    print("  2. discriminator_wgan_gp (pure WGAN-GP, no normalization)")
    print("  3. discriminator_default with use_spectral_norm=True, use_normalization=False")
    print()
    print("⚠️  Avoid for WGAN-GP:")
    print("  1. discriminator_default with default settings (uses InstanceNorm)")
    print("  2. Any variant with use_normalization=True")
    print()
    print("💡 Usage Examples:")
    print("# Best for WGAN-GP")
    print("D = create_model('discriminator_spectral_norm', channels=3)")
    print()
    print("# Alternative pure WGAN-GP")
    print("D = create_model('discriminator_wgan_gp', channels=3)")
    print()
    print("# Configurable default with WGAN-GP settings")
    print("D = create_model('discriminator_default', channels=3, ")
    print("                 use_spectral_norm=True, use_normalization=False)")


if __name__ == "__main__":
    test_wgan_gp_variants()