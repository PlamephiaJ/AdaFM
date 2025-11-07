#!/usr/bin/env python3
"""
测试脚本：验证模型工厂中的权重初始化是否正常工作
"""

import torch
import torch.nn as nn
from models.wgan_factory import create_model
import numpy as np

def analyze_weights(model, model_name):
    """分析模型权重统计信息"""
    print(f"\n=== {model_name} 权重分析 ===")
    
    conv_weights = []
    norm_weights = []
    conv_biases = []
    norm_biases = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight_data = module.weight.data.cpu().numpy().flatten()
                conv_weights.extend(weight_data)
                print(f"  {name}: 权重 mean={np.mean(weight_data):.6f}, std={np.std(weight_data):.6f}, min={np.min(weight_data):.6f}, max={np.max(weight_data):.6f}")
            
            if hasattr(module, 'bias') and module.bias is not None:
                bias_data = module.bias.data.cpu().numpy().flatten()
                conv_biases.extend(bias_data)
                print(f"  {name}: 偏置 mean={np.mean(bias_data):.6f}, std={np.std(bias_data):.6f}")
        
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight_data = module.weight.data.cpu().numpy().flatten()
                norm_weights.extend(weight_data)
            if hasattr(module, 'bias') and module.bias is not None:
                bias_data = module.bias.data.cpu().numpy().flatten()
                norm_biases.extend(bias_data)
    
    if conv_weights:
        print(f"  总体卷积权重: mean={np.mean(conv_weights):.6f}, std={np.std(conv_weights):.6f}")
    if conv_biases:
        print(f"  总体卷积偏置: mean={np.mean(conv_biases):.6f}, std={np.std(conv_biases):.6f}")
    if norm_weights:
        print(f"  总体归一化权重: mean={np.mean(norm_weights):.6f}, std={np.std(norm_weights):.6f}")
    if norm_biases:
        print(f"  总体归一化偏置: mean={np.mean(norm_biases):.6f}, std={np.std(norm_biases):.6f}")


def test_model_initialization():
    """测试所有模型的初始化"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 测试配置
    test_configs = [
        # Generators
        ("generator_default", {"channels": 3, "in_dim": 100}),
        ("generator_resnet", {"channels": 3, "in_dim": 100}),
        ("generator_depthwise", {"channels": 3, "in_dim": 100}),
        ("generator_squeeze_excite", {"channels": 3, "in_dim": 100}),
        
        # Discriminators
        ("discriminator_default", {"channels": 3}),
        ("discriminator_wgan_gp", {"channels": 3}),
        ("discriminator_spectral_norm", {"channels": 3}),
        ("discriminator_resnet", {"channels": 3}),
        ("discriminator_depthwise", {"channels": 3}),
    ]
    
    for model_name, config in test_configs:
        try:
            print(f"\n{'='*60}")
            print(f"测试模型: {model_name}")
            print(f"配置: {config}")
            
            # 创建模型
            model = create_model(model_name, **config).to(device)
            
            # 分析权重
            analyze_weights(model, model_name)
            
            # 测试前向传播
            if "generator" in model_name:
                test_input = torch.randn(4, config["in_dim"], 1, 1).to(device)
                expected_output_shape = (4, config["channels"], 32, 32)
            else:
                test_input = torch.randn(4, config["channels"], 32, 32).to(device)
                expected_output_shape = (4, 1, 1, 1)
            
            with torch.no_grad():
                output = model(test_input)
                print(f"  输入形状: {test_input.shape}")
                print(f"  输出形状: {output.shape}")
                print(f"  期望形状: {expected_output_shape}")
                
                if output.shape == expected_output_shape:
                    print("  ✅ 形状测试通过")
                else:
                    print("  ❌ 形状测试失败")
                
                # 检查输出范围
                output_min = output.min().item()
                output_max = output.max().item()
                output_mean = output.mean().item()
                output_std = output.std().item()
                
                print(f"  输出统计: min={output_min:.6f}, max={output_max:.6f}, mean={output_mean:.6f}, std={output_std:.6f}")
                
                # 检查梯度
                if test_input.requires_grad:
                    test_input.requires_grad_(True)
                    output = model(test_input)
                    loss = output.mean()
                    loss.backward()
                    
                    # 检查模型参数是否有梯度
                    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                    print(f"  梯度检查: {'✅ 有梯度' if has_grad else '❌ 无梯度'}")
            
            print(f"  ✅ {model_name} 测试完成")
            
        except Exception as e:
            print(f"  ❌ {model_name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()


def test_initialization_effects():
    """测试不同初始化方法的效果"""
    print(f"\n{'='*60}")
    print("测试初始化效果对比")
    
    # 创建两个相同的模型，一个有初始化，一个没有
    model_with_init = create_model("generator_default", channels=3, in_dim=100)
    
    # 创建一个没有初始化的模型进行对比
    from models.wgan_factory import Generator
    model_without_init = Generator(channels=3, in_dim=100)
    
    # 手动重置权重为PyTorch默认初始化
    def reset_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # PyTorch默认初始化
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    model_without_init.apply(reset_weights)
    
    print("\\n对比分析:")
    analyze_weights(model_with_init, "带优化初始化的Generator")
    analyze_weights(model_without_init, "PyTorch默认初始化的Generator")


if __name__ == "__main__":
    print("开始测试模型初始化...")
    test_model_initialization()
    test_initialization_effects()
    print("\\n所有测试完成!")