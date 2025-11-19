import torch
import torch_fidelity
from models.wgan_factory import create_model
from omegaconf import DictConfig
import hydra

class FidelityGeneratorWrapper(torch.nn.Module):
    def __init__(self, generator, z_dim):
        super().__init__()
        self.generator = generator
        self.z_dim = z_dim

    def forward(self, z):
        # z: [N, z_dim]
        # 变成 ConvTranspose2d 需要的 4D 形状
        z = z.view(z.size(0), self.z_dim, 1, 1)
        x = self.generator(z)

        # The output of the original generator is in the range [-1, 1] (Tanh).
        x = (x.clamp(-1, 1) + 1) * 127.5
        x = x.clamp(0, 255).to(torch.uint8)
        return x

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    device = torch.device("cuda:0")
    
    z_dim = cfg.models.backbone.generator.in_dim


    generator = create_model(
        cfg.models.backbone.name, cfg.models.backbone.generator
    ).to(device)

    gen_for_fid = FidelityGeneratorWrapper(generator, z_dim).to(device)

    wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(gen_for_fid, cfg.models.backbone.generator.in_dim, 'normal', 0)

    fid_score = torch_fidelity.calculate_metrics(
        input1=wrapped_generator,
        input1_model_num_samples=50000,
        input2='cifar10-train',
        cuda=True if torch.cuda.is_available() else False,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        verbose=False
    )

    print(f"Frechet Inception Distance (FID): {fid_score}")
    
if __name__ == "__main__":
    main()