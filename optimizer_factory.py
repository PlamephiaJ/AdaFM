import torch
from typing import Dict, Callable, Tuple, Any
from omegaconf import DictConfig
from pathlib import Path

class OptimizerRegistry:
    """优化器注册器，用于管理不同类型的优化器创建逻辑"""
    
    def __init__(self):
        self._optimizers: Dict[str, Callable] = {}
    
    def register(self, name: str):
        """注册优化器创建函数的装饰器"""
        def decorator(func: Callable):
            self._optimizers[name] = func
            return func
        return decorator
    
    def create_optimizers(self, name: str, generator, discriminator, cfg: DictConfig, results_folder: Path) -> Tuple[Any, Any]:
        """创建生成器和判别器的优化器"""
        if name not in self._optimizers:
            raise NotImplementedError(f"Optimizer {name} is not implemented.")
        
        return self._optimizers[name](generator, discriminator, cfg, results_folder)
    
    def list_available_optimizers(self) -> list:
        """列出所有可用的优化器"""
        return list(self._optimizers.keys())

# 创建全局注册器实例
optimizer_registry = OptimizerRegistry()

@optimizer_registry.register("adam")
def create_adam_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建Adam优化器"""
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.optimizers.lr,
        betas=(cfg.optimizers.b1, cfg.optimizers.b2),
    )
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=cfg.optimizers.lr,
        betas=(cfg.optimizers.b1, cfg.optimizers.b2),
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("adafm")
def create_adafm_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建AdaFM优化器"""
    from optimizers.AdaFM import AdaFM

    d_optimizer = AdaFM(
        discriminator.parameters(),
        lr=cfg.optimizers.lr_y,
        beta=cfg.optimizers.beta_for_VRAda,
        results_folder=results_folder,
    )
    g_optimizer = AdaFM(
        generator.parameters(),
        lr=cfg.optimizers.lr_x,
        opponent_optim=d_optimizer,
        beta=cfg.optimizers.beta_for_VRAda,
        results_folder=results_folder,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("tiada")
def create_tiada_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建TiAda优化器"""
    from optimizers.TiAda import TiAda

    d_optimizer = TiAda(
        discriminator.parameters(),
        beta=cfg.optimizers.beta,
        lr=cfg.optimizers.lr_y,
        results_folder=results_folder,
    )
    g_optimizer = TiAda(
        generator.parameters(),
        beta=cfg.optimizers.beta,
        opponent_optim=d_optimizer,
        lr=cfg.optimizers.lr_x,
        results_folder=results_folder,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("msgda")
def create_msgda_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建MSGDA优化器"""
    from optimizers.msgda import MSGDA

    d_optimizer = MSGDA(
        discriminator.parameters(),
        lr=cfg.optimizers.lr_discriminator,
        beta=cfg.optimizers.beta_discriminator,
        results_folder=results_folder,
    )
    g_optimizer = MSGDA(
        generator.parameters(),
        lr=cfg.optimizers.lr_generator,
        opponent_optim=d_optimizer,
        beta=cfg.optimizers.beta_generator,
        results_folder=results_folder,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("pesg")
def create_pesg_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建PESG优化器"""
    from optimizers.pesg import PESG

    d_optimizer = PESG(
        discriminator.parameters(),
        total_iter=cfg.models.generator_iters * cfg.optimizers.critic_iters,
        lr=cfg.optimizers.lr,
        clip_value=cfg.optimizers.clip_value,
        weight_decay=cfg.optimizers.weight_decay,
        epoch_decay=cfg.optimizers.epoch_decay,
        momentum=cfg.optimizers.momentum,
        decay_iters=cfg.optimizers.decay_iters,
        decay_factor=cfg.optimizers.decay_factor,
        results_folder=results_folder,
    )
    g_optimizer = PESG(
        generator.parameters(),
        total_iter=cfg.models.generator_iters,
        lr=cfg.optimizers.lr,
        clip_value=cfg.optimizers.clip_value,
        weight_decay=cfg.optimizers.weight_decay,
        epoch_decay=cfg.optimizers.epoch_decay,
        momentum=cfg.optimizers.momentum,
        decay_iters=cfg.optimizers.decay_iters,
        decay_factor=cfg.optimizers.decay_factor,
        opponent_optim=d_optimizer,
        results_folder=results_folder,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("adagrad")
def create_adagrad_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建Adagrad优化器"""
    d_optimizer = torch.optim.Adagrad(
        discriminator.parameters(),
        lr=cfg.optimizers.lr,
        initial_accumulator_value=cfg.optimizers.initial_accumulator_value,
    )
    g_optimizer = torch.optim.Adagrad(
        generator.parameters(),
        lr=cfg.optimizers.lr,
        initial_accumulator_value=cfg.optimizers.initial_accumulator_value,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("sgd")
def create_sgd_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建SGD优化器"""
    d_optimizer = torch.optim.SGD(
        discriminator.parameters(),
        lr=cfg.optimizers.lr,
        momentum=cfg.optimizers.momentum,
        weight_decay=cfg.optimizers.weight_decay,
    )
    g_optimizer = torch.optim.SGD(
        generator.parameters(),
        lr=cfg.optimizers.lr,
        momentum=cfg.optimizers.momentum,
        weight_decay=cfg.optimizers.weight_decay,
    )
    return g_optimizer, d_optimizer

@optimizer_registry.register("rmsprop")
def create_rmsprop_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """创建RMSprop优化器"""
    d_optimizer = torch.optim.RMSprop(
        discriminator.parameters(),
        lr=cfg.optimizers.lr,
        alpha=cfg.optimizers.alpha,
        eps=cfg.optimizers.eps,
    )
    g_optimizer = torch.optim.RMSprop(
        generator.parameters(),
        lr=cfg.optimizers.lr,
        alpha=cfg.optimizers.alpha,
        eps=cfg.optimizers.eps,
    )
    return g_optimizer, d_optimizer

# 注释掉的优化器可以轻松恢复
# @optimizer_registry.register("tiada-adam")
# def create_tiada_adam_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
#     """创建TiAda-Adam优化器"""
#     from optimizers.TiAda import TiAda_Adam
# 
#     d_optimizer = TiAda_Adam(
#         discriminator.parameters(),
#         lr=cfg.optimizers.lr,
#         alpha=cfg.optimizers.beta,
#         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
#     )
#     g_optimizer = TiAda_Adam(
#         generator.parameters(),
#         lr=cfg.optimizers.lr,
#         alpha=cfg.optimizers.beta,
#         opponent_optim=d_optimizer,
#         betas=(cfg.optimizers.b1, cfg.optimizers.b2),
#     )
#     return g_optimizer, d_optimizer

# @optimizer_registry.register("rsgda")
# def create_rsgda_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
#     """创建RSGDA优化器"""
#     from optimizers.RSGDA import RSGDA
# 
#     d_optimizer = RSGDA(
#         discriminator.parameters(),
#         beta_y=cfg.optimizers.beta_y,
#         lr_y=cfg.optimizers.lr_y,
#     )
#     g_optimizer = RSGDA(
#         generator.parameters(),
#         beta_x=cfg.optimizers.beta_x,
#         opponent_optim=d_optimizer,
#         lr_x=cfg.optimizers.lr_x,
#     )
#     return g_optimizer, d_optimizer

# @optimizer_registry.register("vradagda")
# def create_vradagda_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
#     """创建VRAdaGDA优化器"""
#     from optimizers.VRAdaGDA import VRAdaGDA
# 
#     d_optimizer = VRAdaGDA(
#         discriminator.parameters(),
#         beta_y=cfg.optimizers.beta_y,
#         lr_y=cfg.optimizers.lr_y,
#     )
#     g_optimizer = VRAdaGDA(
#         generator.parameters(),
#         beta_x=cfg.optimizers.beta_x,
#         opponent_optim=d_optimizer,
#         lr_x=cfg.optimizers.lr_x,
#     )
#     return g_optimizer, d_optimizer

def create_optimizers(generator, discriminator, cfg: DictConfig, results_folder: Path):
    """主要的优化器创建函数，保持向后兼容性"""
    return optimizer_registry.create_optimizers(
        cfg.optimizers.name, 
        generator, 
        discriminator, 
        cfg, 
        results_folder
    )

def get_available_optimizers():
    """获取所有可用的优化器列表"""
    return optimizer_registry.list_available_optimizers()
