import torch
import torch.nn as nn


def perform_stratified_sampling(
    ray_origin: torch.Tensor,
    ray_direc: torch.Tensor,
    t_near: float,
    t_far: float,
    num_samples: int = 64,
):
    """
    "We use a stratified sampling approach where we partition [tn; tf ] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin."
    """
    # num_samples = 4
    # t_near=1.
    # t_far=5.
    bin_width = (t_far - t_near) / num_samples
    t_vals = torch.linspace(
        t_near, t_far - bin_width, num_samples, device=ray_origin.device,
    ) + bin_width * torch.rand(
        (ray_origin.size(0), num_samples), device=ray_origin.device,
    )
    return ray_origin[:, None, :] + t_vals[..., None] * ray_direc[:, None, :]
    # (batch_size, `num_samples`, 3)


def perform_hierarchical_volume_sampling(
    model: nn.Module,
    pe: nn.Module,
    ray_origin: torch.Tensor,
    ray_direc: torch.Tensor,
    t_near: float,
    t_far: float,
    num_coarse_samples: int = 64,
    num_fine_samples: int = 128,
):
    """
    # Coarse Sampling:
    # Uniformly samples points along each ray in the scene.
    # Provides an initial, approximate representation of the ray's interaction with the volume.
    # Fine Sampling (Hierarchical Sampling):
    # Uses the results of the coarse sampling to determine where to allocate additional samples.
    # Focuses more samples in regions where the coarse sampling indicates higher density or greater importance, such as edges or surfaces.
    """
    coarse_samples = perform_stratified_sampling(
        ray_origin=ray_origin,
        ray_direc=ray_direc,
        t_near=t_near,
        t_far=t_far,
        num_samples=num_coarse_samples,
    )  # (batch_size, `num_coarse_samples`, 3)
    model_out = model(pe(coarse_samples.view(-1, 3)))
    # (batch_size * `num_coarse_samples`, 3)
    pred_density = model_out[:, : 1]


if __name__ == "__main__":
    batch_size = 1
    t_near = 0.1
    t_far = 4.0
    num_samples = 64

    # Random example rays (origin and direction)
    ray_origin = torch.zeros(batch_size, 3)  # Rays start at the origin
    ray_direc = torch.randn(batch_size, 3)  # Random ray directions
    ray_direc = ray_direc / torch.norm(ray_direc, dim=-1, keepdim=True)  # Normalize directions
    # ray_direc.shape

    sampled_points = perform_stratified_sampling(ray_origin, ray_direc, t_near, t_far, num_samples)
    print(sampled_points.shape)
    print("sampled_points shape:", sampled_points.shape)  # Should be [batch_size, num_samples, 3]
