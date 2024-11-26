import torch


def stratified_sampling(ray_origins, ray_direcs, near, far, num_samples=64):
    """
    "We use a stratified sampling approach where we partition [tn; tf ] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin:"

    Args:
        ray_origins (torch.Tensor): Origin of the rays, shape [batch_size, 3].
        ray_origins (torch.Tensor): Direction of the rays, shape [batch_size, 3].
        near (float): Near bound of the ray.
        far (float): Far bound of the ray.
        num_samples (int): Number of samples to draw along each ray.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        z_vals (torch.Tensor): Sampled depth values along the rays, shape [batch_size, num_samples].
        sampled_points (torch.Tensor): Sampled points in 3D space, shape [batch_size, num_samples, 3].
    """
    # Step 1: Compute the bin edges
    batch_size = ray_origins.shape[0]
    bins = torch.linspace(near, far, num_samples + 1, device=ray_origins.device)  # [num_samples + 1]
    
    # Step 2: Uniformly sample within each bin
    bin_width = bins[1:] - bins[:-1]  # Bin widths [num_samples]
    bin_samples = bins[:-1] + bin_width * torch.rand((batch_size, num_samples), device=ray_origins.device)  # Random samples in bins

    # Step 3: Compute 3D points along the ray
    return ray_origins[:, None, :] + bin_samples[..., None] * ray_direcs[:, None, :]  # [batch_size, num_samples, 3]


if __name__ == "__main__":
    batch_size = 16
    near = 0.1
    far = 4.0
    num_samples = 64

    # Random example rays (origin and direction)
    ray_origins = torch.zeros(batch_size, 3)  # Rays start at the origin
    ray_direcs = torch.randn(batch_size, 3)  # Random ray directions
    ray_direcs = ray_direcs / torch.norm(ray_direcs, dim=-1, keepdim=True)  # Normalize directions
    # ray_direcs.shape

    sampled_points = stratified_sampling(ray_origins, ray_direcs, near, far, num_samples)
    # print("z_vals shape:", t_vals.shape)  # Should be [batch_size, num_samples]
    print("sampled_points shape:", sampled_points.shape)  # Should be [batch_size, num_samples, 3]
