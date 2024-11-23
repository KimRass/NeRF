import numpy as np
import json
from pathlib import Path
import imageio.v2 as imageio
import torch


def load_blender(data_root, skip=1):
    splits = ["train", "val", "test"]

    metas = {}
    all_imgs = []
    all_poses = []
    counts = [0]
    for split in splits:
        json_path = data_root/f"transforms_{split}.json"
        with open(json_path, mode="r") as f:
            metas[split] = json.load(f)

        imgs = []
        poses = []
        for frame in metas[split]["frames"][:: skip]:
            imgs.append(
                imageio.imread(
                    (data_root/frame["file_path"]).with_suffix(".png")
                )  # RGBA image.
            )
            poses.append(np.array(frame["transform_matrix"]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)  # Extrinsic matrix $[R|t]$.
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

    _, h, w, _ = imgs.shape
    camera_angle_x = metas[split]["camera_angle_x"]
    focal = w / (2 * np.tan(camera_angle_x / 2))
    k = np.array(
        [
            [focal, 0, w / 2],  # `w / 2`: Center of image in pixel coordinates.
            [0, focal, h / 2],  # `h / 2`: Center of image in pixel coordinates.
            [0, 0, 1],
        ]
    )  # Intrinsic matrix $K$.
    # print(counts)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    return (
        np.concatenate(all_imgs, axis=0),
        np.concatenate(all_poses, axis=0),
        [h, w, k],
        i_split,
        None,
    )


def make_o_d(
    h: int, w: int, k: np.array, pose: np.array,
) -> tuple[np.array, np.array]:
    """
    `k`: Intrinsic matrix $K$.
    `pose`: Extrinsic matrix $[R|t]$.
    """
    # make catesian (x. y)
    i, j = torch.meshgrid(
        torch.linspace(0, w - 1, w),
        torch.linspace(0, h - 1, h),
    )
    i = i.t()
    j = j.t()

    dirs = torch.stack(
        [
            (i - k[0][2]) / k[0][0],
            -(j - k[1][2]) / k[1][1],
            # The $x$ and $y$ components are normalized by the focal lengths ($f_{x}$ and $f_{y}$) to account for the camera's perspective projection.
            -torch.ones_like(i),
            # The $z$ component is $−1$, representing a ray moving "forward" in the camera's local space.
        ],
        dim=-1,
    )  # Rotation matrix $R$.

    rays_d = dirs @ pose[: 3, : 3].T  # Shape: (`w`, `h`, 3)
    # Each entry is the camera's origin (usually the same for all rays).
    # The origin of each ray (the camera position in world coordinates).
    # This applies the rotation part of the pose matrix to the direction vectors to transform them from the camera's local coordinate system to the world coordinate system.
    rays_o = pose[: 3, -1].expand(rays_d.shape)  # Shape: (`w`, `h`, 3)
    # Each entry is the direction vector of a ray in 3D space.
    # The direction of each ray in world coordinates.
    # `pose[:3, -1]`: The translation vector (last column of the pose matrix), representing the camera's origin in world space.
    # Repeats this origin for every ray in the image.
    # Since all rays originate from the camera center, `rays_o` is constant for a given camera pose.
    return rays_o, rays_d


if __name__ == "__main__":
    data_root = Path("/Users/jongbeomkim/Documents/datasets/nerf_synthetic/ship")
    imgs, poses, [h, w, k], i_split, _ = load_blender(data_root)
    # print(imgs.shape, poses.shape)
    img = imgs[0]
    pose = poses[0]


i_split
np.random.choice(i_split[0])
poses[0].shape
poses[0][: 3, : 3]
poses[0][: 3, : 3].T
k
