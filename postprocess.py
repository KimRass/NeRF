import torch
import torch.nn as nn
import torch.nn.functional as F


# def render_rays(rays, model, fn_posenc, fn_posenc_d, num_fine_samples):
#     # 1. pre process : make (pts and dirs) (embedded)
#     embedded, z_vals, rays_d = pre_process(rays, fn_posenc, fn_posenc_d, opts)

#     # 2. run model by net_chunk
#     net_chunk = opts.net_chunk
#     outputs_flat = torch.cat([model(embedded[i:i+net_chunk]) for i in range(0, embedded.shape[0], net_chunk)], 0)  # [n_pts, 4]
#     size = [z_vals.size(0), z_vals.size(1), 4]      # [bs, n_samples, 4]
#     outputs = outputs_flat.reshape(size)            # [bs, n_samples, 4]

#     # 3. post process : render each pixel color by formula (3) in nerf paper
#     rgb_map, disp_map, acc_map, weights, depth_map = post_process(outputs, z_vals, rays_d)

#     # if num_fine_samples > 0:
#     #     # 4. pre precess
#     #     rays = rays.to('cuda:{}'.format(opts.gpu_ids[opts.rank]))
#     #     embedded_fine, z_vals_fine, rays_d = pre_process_for_hierarchical(rays, z_vals, weights, fn_posenc, fn_posenc_d, opts)

#     #     # 5. run model by net_chunk
#     #     outputs_fine_flat = torch.cat([model(embedded_fine[i:i + net_chunk], is_fine=True) for i in range(0, embedded_fine.shape[0], net_chunk)], 0)
#     #     size_fine = [z_vals_fine.size(0), z_vals_fine.size(1), 4]  # [4096, 64 + 128, 4]
#     #     outputs_fine = outputs_fine_flat.reshape(size_fine)

#     #     # 6. post process : render each pixel color by formula (3) in nerf paper
#     #     rgb_map_fine, disp_map_fine, acc_map_fine, weights_fine, depth_map_fine = post_process(outputs_fine, z_vals_fine, rays_d)

#     #     return {'coarse': rgb_map, 'disp_coarse': disp_map, 'fine': rgb_map_fine, 'disp_fine': disp_map_fine}
#     return {'coarse': rgb_map, 'disp_coarse': disp_map}


# def render_by_chunk(ray_origins, ray_direcs, model, fn_posenc, fn_posenc_d, H, W, K, chunk, num_fine_samples):
#     """
#     `chunk`: The maximum number of rays to process at a time.
#     `num_fine_samples`: The number of fine samples to draw during hierarchical sampling.
#     """
#     flat_ray_o, flat_ray_d = ray_origins.view(-1, 3), ray_direcs.view(-1, 3)
#     # train : [batch_size, 3] / test :  [640000, 3]

#     num_whole_rays = flat_ray_o.size(0)
#     rays = torch.cat((flat_ray_o, flat_ray_d), dim=-1)
#     # Combines the origins and directions into a single tensor for easier slicing and processing.

#     ret_coarse = []
#     ret_coarse_disp = []
#     ret_fine = []
#     ret_fine_disp = []

#     for i in range(0, num_whole_rays, chunk):
#         rgb_dict = render_rays(
#             rays=rays[i: i + chunk],
#             model=model,
#             fn_posenc=fn_posenc,
#             fn_posenc_d=fn_posenc_d,
#             num_fine_samples=num_fine_samples,
#         )
#         # For each chunk, `render_rays` predicts the RGB values and disparities for the given rays using the NeRF model.

#         if num_fine_samples > 0:                    # use fine rays
#             ret_coarse.append(rgb_dict['coarse'])
#             ret_coarse_disp.append(rgb_dict['disp_coarse'])
#             ret_fine.append(rgb_dict['fine'])
#             ret_fine_disp.append(rgb_dict['disp_fine'])
#         else:                                        # use only coarse rays
#             ret_coarse.append(rgb_dict['coarse'])
#             ret_coarse_disp.append(rgb_dict['disp_coarse'])

#     if num_fine_samples > 0:
#         # RGB, DIST / RGB, DIST
#         return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), torch.cat(ret_fine, dim=0), torch.cat(ret_fine_disp, dim=0)
#     else:
#         return torch.cat(ret_coarse, dim=0), torch.cat(ret_coarse_disp, dim=0), None, None


def volume_rendering(colors, densities, ray_origin, t):
    """
    Perform volume rendering to compute the final color of a ray.

    Args:
        sigmas (Tensor): Density values at sampled points along the ray [num_samples].
        colors (Tensor): RGB colors at sampled points [num_samples, 3].
        deltas (Tensor): Distances between consecutive sampled points [num_samples - 1].
        white_background (bool): If True, assumes a white background.

    Returns:
        final_color (Tensor): Rendered RGB color of the ray [3].
        accumulated_opacity (Tensor): Total opacity of the ray (alpha channel) [1].
    """
    batch_size = 2
    num_samples = 5
    densities = torch.randn(batch_size, num_samples)
    deltas = torch.randn(batch_size, num_samples - 1)
    densities_deltas = densities[:, : -1] * deltas
    t_i = torch.exp(-torch.cumsum(densities_deltas, dim=-1))  # $T_{i}$.
    pixel_colors = torch.cumsum(t_i * (1 - torch.exp(-densities_deltas)), dim=-1)
    pixel_colors.shape
    # * colors

    
    densities = F.relu(densities)[...,0]
    colors = torch.sigmoid(colors)
    deltas = t[..., 1:] - t[..., :-1]
    deltas = torch.cat(
        [
            deltas,
            torch.tensor(
                [1e10],
                dtype=ray_origin.dtype,
                device=ray_origin.device,
            ).expand(t[..., : 1].shape),
        ],
        dim=-1,
    )
    
    
    alpha = 1. - torch.exp(-densities * deltas)
    T = torch.cumprod(1. - alpha + 1e-10, -1)
    T = torch.roll(T, 1, -1)
    T[..., 0] = 1.

    w = T * alpha

    rgb = (w[..., None] * colors).sum(dim=-2)
    return rgb


if __name__ == "__main__":
    from model import PositionalEncoding

    fn_posenc = PositionalEncoding(l=10)
    fn_posenc_d = PositionalEncoding(l=4)