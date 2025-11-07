# WGAN Backbone Variants Documentation

本文档描述了 WGAN 模型工厂中可用的不同 backbone 架构变体，特别关注**WGAN-GP理论兼容**的判别器归一化策略。

## 🚨 重要更新：WGAN-GP归一化策略

### 理论背景
WGAN-GP要求判别器满足**1-Lipschitz约束**，传统的BatchNorm/InstanceNorm可能与梯度惩罚机制产生冲突：
- **问题**：InstanceNorm会改变梯度的尺度和方向，干扰梯度惩罚的有效性
- **解决**：使用谱归一化(Spectral Normalization)或完全移除归一化层

### 推荐配置

#### ✅ WGAN-GP推荐使用
1. **`discriminator_spectral_norm`** - 使用谱归一化（推荐）
2. **`discriminator_wgan_gp`** - 纯WGAN-GP，无归一化
3. **`discriminator_default`** 配置 `use_spectral_norm=True, use_normalization=False`

#### ⚠️ WGAN-GP不推荐
1. **`discriminator_default`** 默认设置（使用InstanceNorm）
2. 任何 `use_normalization=True` 的变体

## 📋 概览

WGAN 模型工厂现在支持多种 backbone 架构，每种都有不同的归纳偏置和参数效率特点：

### Generator 变体

| 模型名称 | 参数量 | 特点 | 适用场景 |
|---------|--------|------|---------|
| `generator_default` | 12.14M | 标准 DCGAN 架构 | 基准模型，稳定训练 |
| `generator_resnet` | 25.58M (2.11x) | ResNet 跳跃连接 | 需要更好梯度流，深度网络 |
| `generator_depthwise` | 2.34M (0.19x) | 深度可分离卷积 | 移动端部署，内存受限 |
| `generator_squeeze_excite` | 12.18M (1.00x) | SE 注意力机制 | 特征重标定，质量提升 |

### Discriminator 变体

| 模型名称 | 参数量 | 归一化策略 | WGAN-GP兼容 | 适用场景 |
|---------|--------|------------|-------------|---------|
| `discriminator_default` | 10.52M | InstanceNorm (可配置) | ⚠️ 需配置 | 通用基准，可配置 |
| `discriminator_spectral_norm` | 10.52M | 仅谱归一化 | ✅ 推荐 | **WGAN-GP训练首选** |
| `discriminator_wgan_gp` | 10.52M | 无归一化 | ✅ 理论最佳 | 纯WGAN-GP理论实现 |
| `discriminator_resnet` | 22.97M | 谱归一化+可选IN | ✅ 默认兼容 | 需要更强判别能力 |
| `discriminator_depthwise` | 0.70M | 谱归一化 | ✅ 默认兼容 | 极致轻量化 |
| `discriminator_attention` | 10.84M | 谱归一化 | ✅ 默认兼容 | 长程依赖，细节捕捉 |

## 🚀 使用方法

### WGAN-GP推荐用法

```python
from models.wgan_factory import create_model

# 推荐：谱归一化判别器（最佳平衡）
G = create_model('generator_default', channels=3, in_dim=100)
D = create_model('discriminator_spectral_norm', channels=3)

# 替代：纯WGAN-GP（理论最佳）
D_pure = create_model('discriminator_wgan_gp', channels=3)

# 可配置：使用默认模型但关闭InstanceNorm
D_configured = create_model('discriminator_default', channels=3,
                           use_spectral_norm=True, use_normalization=False)
```

### 传统GAN用法

```python
# 传统DCGAN风格（不推荐用于WGAN-GP）
G = create_model('generator_default', channels=3, in_dim=100)
D = create_model('discriminator_default', channels=3)  # 使用InstanceNorm
```

### 轻量化变体（适合资源受限环境）

```python
# 极致轻量化组合 - 总参数约 3M
G_light = create_model('generator_depthwise', channels=3, in_dim=100)    # 2.34M
D_light = create_model('discriminator_depthwise', channels=3)           # 0.70M

# 平衡轻量化组合 - 总参数约 13M  
G_balanced = create_model('generator_squeeze_excite', channels=3, in_dim=100)  # 12.18M
D_balanced = create_model('discriminator_attention', channels=3)               # 10.85M
```

### 高性能变体（适合性能优先场景）

```python
# 高性能组合 - 总参数约 48M
G_powerful = create_model('generator_resnet', channels=3, in_dim=100)    # 25.58M
D_powerful = create_model('discriminator_resnet', channels=3)           # 22.98M
```

## 🔧 架构特点详解

### Generator 架构

#### ResNet Generator
- **特点**: 使用跳跃连接改善梯度流
- **优势**: 更稳定的训练，可以训练更深的网络
- **劣势**: 参数量较大 (2.11x)
- **适用**: 当训练不稳定或需要更高质量生成时

#### Depthwise Generator  
- **特点**: 深度可分离卷积显著减少参数
- **优势**: 极低参数量 (0.19x)，快速推理
- **劣势**: 可能表达能力有限
- **适用**: 移动端部署，资源受限环境

#### Squeeze-Excite Generator
- **特点**: SE 模块进行通道注意力
- **优势**: 参数量几乎不变，质量可能提升
- **劣势**: 略增加计算复杂度
- **适用**: 想要改善质量但不增加太多参数

### Discriminator 架构

#### ResNet Discriminator
- **特点**: 跳跃连接增强判别能力
- **优势**: 更强的特征提取能力
- **劣势**: 参数量较大 (2.18x)
- **适用**: 生成器很强时需要更强判别器

#### Depthwise Discriminator
- **特点**: 深度可分离卷积极致轻量
- **优势**: 仅 0.07x 参数量
- **劣势**: 判别能力可能不足
- **适用**: 实验快速原型，资源极限环境

#### Attention Discriminator  
- **特点**: 自注意力捕捉长程依赖
- **优势**: 参数量几乎不变，更好的全局建模
- **劣势**: 略增加计算复杂度
- **适用**: 需要捕捉图像全局结构关系

## 📊 性能建议

### 训练阶段建议

1. **快速实验**: 使用 depthwise 变体快速验证想法
2. **质量优先**: 使用 ResNet 变体获得最佳效果  
3. **平衡选择**: 使用 SE/attention 变体在参数和性能间平衡

### 部署阶段建议

1. **移动端**: depthwise 变体，最小内存占用
2. **云端**: ResNet 变体，最佳效果
3. **边缘计算**: SE/attention 变体，平衡性能

## 🧪 实验结果验证

所有模型变体都已通过测试：
- ✅ 模型创建成功
- ✅ 前向传播正常  
- ✅ 输出尺寸正确
- ✅ 特征提取功能正常

可运行 `python3 test_backbones.py` 进行完整验证。

## 🔄 后向兼容性

原始的模型名称仍然支持：
- `"generator"` → `generator_default`  
- `"discriminator"` → `discriminator_default`

这确保了现有代码的兼容性。