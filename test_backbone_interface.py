#!/usr/bin/env python3
"""
测试新的backbone配置接口
验证create_model函数的两种调用方式
"""

import torch
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.wgan_factory import create_model


def test_new_interface():
    """测试新的backbone配置接口"""
    print("🧪 Testing New Backbone Config Interface\n")
    
    # 模拟配置结构
    mock_configs = {
        "wgan-gp": {
            "generator": {
                "name": "generator_default",
                "channels": 3,
                "in_dim": 100
            },
            "discriminator": {
                "name": "discriminator_wgan_gp", 
                "channels": 3
            }
        },
        "wgan-gp-spectral": {
            "generator": {
                "name": "generator_default",
                "channels": 3,
                "in_dim": 100
            },
            "discriminator": {
                "name": "discriminator_spectral_norm",
                "channels": 3
            }
        },
        "resnet-variant": {
            "generator": {
                "name": "generator_resnet",
                "channels": 3,
                "in_dim": 100
            },
            "discriminator": {
                "name": "discriminator_resnet", 
                "channels": 3,
                "use_spectral_norm": True,
                "use_normalization": False
            }
        }
    }
    
    print("📊 Testing New Interface")
    print("-" * 60)
    print(f"{'Backbone':<20} {'Model Type':<15} {'Status'}")
    print("-" * 60)
    
    device = torch.device('cpu')  # 使用CPU进行测试
    
    for backbone_name, config in mock_configs.items():
        try:
            # 测试生成器创建
            generator = create_model(
                backbone_name,
                config["generator"]
            ).to(device)
            
            # 测试判别器创建
            discriminator = create_model(
                backbone_name,
                config["discriminator"] 
            ).to(device)
            
            # 测试前向传播
            with torch.no_grad():
                z = torch.randn(2, config["generator"]["in_dim"], 1, 1)
                fake_images = generator(z)
                d_fake = discriminator(fake_images)
            
            print(f"{backbone_name:<20} {'Generator':<15} ✅ OK")
            print(f"{'':<20} {'Discriminator':<15} ✅ OK")
            
        except Exception as e:
            print(f"{backbone_name:<20} {'Error':<15} ❌ {str(e)[:30]}")
    
    print("\n📊 Testing Legacy Interface")
    print("-" * 60)
    print(f"{'Model Name':<25} {'Status'}")
    print("-" * 60)
    
    # 测试传统接口仍然工作
    legacy_tests = [
        ("generator_default", {"channels": 3, "in_dim": 100}),
        ("discriminator_default", {"channels": 3}),
        ("discriminator_spectral_norm", {"channels": 3}),
    ]
    
    for model_name, params in legacy_tests:
        try:
            model = create_model(model_name, **params)
            print(f"{model_name:<25} ✅ OK")
        except Exception as e:
            print(f"{model_name:<25} ❌ {str(e)[:30]}")
    
    print("\n🎯 Usage Examples")
    print("-" * 40)
    print("# New backbone config interface:")
    print("from models.wgan_factory import create_model")
    print()
    print("# Create generator using backbone config")
    print("generator = create_model(")
    print("    cfg.models.backbone.name,")
    print("    cfg.models.backbone.generator")
    print(").to(device)")
    print()
    print("# Create discriminator using backbone config") 
    print("discriminator = create_model(")
    print("    cfg.models.backbone.name,")
    print("    cfg.models.backbone.discriminator")
    print(").to(device)")
    print()
    print("# Legacy interface still works:")
    print("model = create_model('generator_default', channels=3, in_dim=100)")


def test_error_handling():
    """测试错误处理"""
    print("\n\n🛡️  Testing Error Handling")
    print("-" * 40)
    
    # 测试无效配置
    test_cases = [
        ("Invalid config type", "backbone", "not_a_dict"),
        ("Missing name field", "backbone", {}),
        ("Invalid model name", "backbone", {"name": "invalid_model", "channels": 3}),
    ]
    
    for description, backbone, config in test_cases:
        try:
            create_model(backbone, config)
            print(f"❌ {description}: Should have raised an error")
        except (ValueError, KeyError) as e:
            print(f"✅ {description}: Correctly raised {type(e).__name__}")
        except Exception as e:
            print(f"⚠️  {description}: Unexpected error {type(e).__name__}")


if __name__ == "__main__":
    test_new_interface()
    test_error_handling()