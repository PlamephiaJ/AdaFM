import torch
import torch_fidelity

import numpy as np
import torch.serialization as ts

ts.add_safe_globals([np._core.multiarray._reconstruct])

class FidelityGeneratorWrapper(torch.nn.Module):
    def __init__(self, generator, z_dim):
        super().__init__()
        self.generator = generator
        self.z_dim = z_dim

    def forward(self, z):
        # z: [N, z_dim]
        # The generator expects input of shape [N, z_dim, 1, 1]
        z = z.view(z.size(0), self.z_dim, 1, 1)
        x = self.generator(z)

        # The output of the original generator is in the range [-1, 1] (Tanh).
        x = (x.clamp(-1, 1) + 1) * 127.5
        x = x.clamp(0, 255).to(torch.uint8)
        return x


def get_fid_score(generator: torch.nn.Module, z_dim: int, device: torch.device, num_samples: int = 50000, dataset_name: str = 'cifar10') -> float:
    if dataset_name == 'cifar10':
        input2 = 'cifar10-train'
    elif dataset_name == 'cifar100':
        input2 = 'cifar100-train'
    else:
        raise ValueError(f"Unsupported dataset for FID calculation: {dataset_name}")

    gen_for_fid = FidelityGeneratorWrapper(generator, z_dim).to(device)

    wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(gen_for_fid, z_dim, 'normal', 0)

    with torch.no_grad():
        fid_score = torch_fidelity.calculate_metrics(
            input1=wrapped_generator,
            input1_model_num_samples=num_samples,
            input2=input2,
            cuda=True,
            fid=True,
            isc=False,
            kid=False,
            prc=False,
            verbose=False
        )

    return fid_score['frechet_inception_distance']