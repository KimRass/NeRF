import torch
import numpy as np

from data import load_blender

# "We synthesize images by sampling 5D coordinates (location and viewing direction) along camera rays (a), feeding those locations into an MLP to produce a color and volume density (b), and using volume ren- dering techniques to composite these values into an image (c). This rendering function is di erentiable, so we can optimize our scene representation by mini- mizing the residual between synthesized and ground truth observed images (d)."
# For each image, sample a set of rays corresponding to the pixels of the image.
# For each ray, compute the 3D points along the ray’s path.


def get_ray_origins_and_direcs(
    h: int, w: int, intrinsic: torch.Tensor, pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    이미지 1장에 대해서 작동.
    `intrinsic`: Intrinsic camera matrix $K$ of shape (3, 3).
    `pose`: Extrinsic matrix $[R|t]$ of shape (3, 4).
    Returns:
        `ray_origins`: `The starting point of the ray (camera position)
        `ray_direcs`: The direction of the ray in 3D space.
    """
    j, i = torch.meshgrid(
        torch.linspace(0, h - 1, h), torch.linspace(0, w - 1, w),
    )  # Image plane coordinates.

    dirs = torch.stack(
        [
            (i - intrinsic[0][2]) / intrinsic[0][0],  # $(`i` - c_{x}) / f_{x}$.
            -(j - intrinsic[1][2]) / intrinsic[1][1],  # $(`j` - c_{y}) / f_{y}$.
            # The $x$ and $y$ components are normalized by the focal lengths
            # ($f_{x}$ and $f_{y}$) to account for the camera's perspective projection.
            -torch.ones_like(i),
            # The $z$ component is $−1$, representing a ray moving "forward" in the camera's local space.
        ],
        dim=-1,
    )  # Rotation matrix $R$ of shape (`h`, `w`, 3).

    trans_vec = pose[:, 3]
    ray_origins = torch.tile(trans_vec, dims=(h, w, 1))  # Shape: (`h`, `w`, 3)
    # Each entry is the direction vector of a ray in 3D space.
    # The direction of each ray in world coordinates.
    # Repeats this origin for every ray in the image.
    # Since all rays originate from the camera center, `rays_o` is constant for a given camera pose.

    rot_mat = pose[:, : 3].T
    # The translation vector (last column of the pose matrix), representing the camera's origin in world space.
    ray_direcs = dirs @ rot_mat  # Shape: (`h`, `w`, 3)
    # Each entry is the camera's origin (usually the same for all rays).
    # The origin of each ray (the camera position in world coordinates).
    # This applies the rotation part of the pose matrix to the direction vectors to transform them from the camera's local coordinate system to the world coordinate system.
    return ray_origins, ray_direcs


def sample_rays_and_pixels(ray_origins, ray_direcs, img, batch_size=1024):
    h, w, *_ = img.shape
    coords = torch.stack(
        torch.meshgrid(torch.arange(h), torch.arange(w)),
        dim=-1,
    )  # (`h`, `w`, 2)
    coords = torch.reshape(coords, [-1, 2])  # (`hw`, 2)
    selected_idx = np.random.choice(
        coords.size(0), size=batch_size, replace=False,
    )
    selected_coords = coords[selected_idx]  # (`batch_size`, 2)

    ### Sample rays.
    selected_ray_origins = ray_origins[selected_coords[:, 0], selected_coords[:, 1]]
    # (`batch_size`, 3)
    selected_ray_direcs = ray_direcs[selected_coords[:, 0], selected_coords[:, 1]]
    # (`batch_size`, 3)

    ### Sample pixels.
    selected_img = img[selected_coords[:, 0], selected_coords[:, 1]]
    # (`batch_size`, 4)
    return selected_ray_origins, selected_ray_direcs, selected_img
