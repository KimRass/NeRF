import torch
import torch.nn as nn
import torch.nn.functional as F

# Coarse Sampling:

# Uniformly samples points along each ray in the scene.
# Provides an initial, approximate representation of the ray's interaction with the volume.
# Fine Sampling (Hierarchical Sampling):

# Uses the results of the coarse sampling to determine where to allocate additional samples.
# Focuses more samples in regions where the coarse sampling indicates higher density or greater importance, such as edges or surfaces.


def render_rays(rays, model, fn_posenc, fn_posenc_d, num_fine_samples):
    # 1. pre process : make (pts and dirs) (embedded)
    embedded, z_vals, rays_d = pre_process(rays, fn_posenc, fn_posenc_d, opts)

    # 2. run model by net_chunk
    net_chunk = opts.net_chunk
    outputs_flat = torch.cat([model(embedded[i:i+net_chunk]) for i in range(0, embedded.shape[0], net_chunk)], 0)  # [n_pts, 4]
    size = [z_vals.size(0), z_vals.size(1), 4]      # [bs, n_samples, 4]
    outputs = outputs_flat.reshape(size)            # [bs, n_samples, 4]

    # 3. post process : render each pixel color by formula (3) in nerf paper
    rgb_map, disp_map, acc_map, weights, depth_map = post_process(outputs, z_vals, rays_d)

    # if num_fine_samples > 0:
    #     # 4. pre precess
    #     rays = rays.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
    #     embedded_fine, z_vals_fine, rays_d = pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts)

    #     # 5. run model by net_chunk
    #     outputs_fine_flat = torch.cat([model(embedded_fine[i:i + net_chunk], is_fine=True) for i in range(0, embedded_fine.shape[0], net_chunk)], 0)
    #     size_fine = [z_vals_fine.size(0), z_vals_fine.size(1), 4]  # [4096, 64 + 128, 4]
    #     outputs_fine = outputs_fine_flat.reshape(size_fine)

    #     # 6. post process : render each pixel color by formula (3) in nerf paper
    #     rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = post_process(outputs_fine, z_vals_fine, rays_d)

    #     return {'coarse': rgb_map, 'disp_coarse': disp_map, 'fine': rgb_map_fine, 'disp_fine': disp_map_fine}
    return {'coarse': rgb_map, 'disp_coarse': disp_map}


def render_by_chunk(ray_origins, ray_direcs, model, fn_posenc, fn_posenc_d, H, W, K, chunk, num_fine_samples):
    """
    `chunk`: The maximum number of rays to process at a time.
    `num_fine_samples`: The number of fine samples to draw during hierarchical sampling.
    """
    flat_ray_o, flat_ray_d = ray_origins.view(-1, 3), ray_direcs.view(-1, 3)
    # train : [batch_size, 3] / test :  [640000, 3]

    num_whole_rays = flat_ray_o.size(0)
    rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)
    # Combines the origins and directions into a single tensor for easier slicing and processing.

    ret_coarse = []
    ret_coarse_disp = []
    ret_fine = []
    ret_fine_disp = []

    for i in range(0, num_whole_rays, chunk):
        rgb_dict = render_rays(
            rays=rays[i: i + chunk],
            model=model,
            fn_posenc=fn_posenc,
            fn_posenc_d=fn_posenc_d,
            num_fine_samples=num_fine_samples,
        )
        # For each chunk, `render_rays` predicts the RGB values and disparities for the given rays using the NeRF model.

        if num_fine_samples > 0:                    # use fine rays
            ret_coarse.append(rgb_dict['coarse'])
            ret_coarse_disp.append(rgb_dict['disp_coarse'])
            ret_fine.append(rgb_dict['fine'])
            ret_fine_disp.append(rgb_dict['disp_fine'])
        else:                                        # use only coarse rays
            ret_coarse.append(rgb_dict['coarse'])
            ret_coarse_disp.append(rgb_dict['disp_coarse'])

    if num_fine_samples > 0:
        # RGB, DIST / RGB, DIST
        return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), torch.cat(ret_fine, dim=0), torch.cat(ret_fine_disp, dim=0)
    else:
        return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), None, None


def volume_rendering(c, sigma, rays_o, t):
    sigma = F.relu(sigma)[...,0]
    c = torch.sigmoid(c)
    delta = t[..., 1:] - t[..., :-1]
    delta = torch.cat([delta, torch.tensor([1e10], dtype=rays_o.dtype, device=rays_o.device).expand(t[...,:1].shape)], dim=-1)

    alpha = 1. - torch.exp(-sigma * delta)
    T = torch.cumprod(1. - alpha + 1e-10, -1)
    T = torch.roll(T, 1, -1)
    T[..., 0] = 1.

    w = T * alpha

    rgb = (w[..., None] * c).sum(dim=-2)
    return rgb


if __name__ == "__main__":
    from model import PositionalEncoding

    fn_posenc = PositionalEncoding(l=10)
    fn_posenc_d = PositionalEncoding(l=4)