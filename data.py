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
    fov = metas[split]["camera_angle_x"]
    # Camera's horizontal and vertical field of view (FoV).
    focal_len = w / (2 * np.tan(fov / 2))  # $f_{x} = f_{y}$.
    intrinsic = np.array(
        [
            [focal_len, 0, w / 2],  # `w / 2`: Center of image in pixel coordinates.
            [0, focal_len, h / 2],  # `h / 2`: Center of image in pixel coordinates.
            [0, 0, 1],
        ]
    )  # Intrinsic camera matrix $K = \begin{bmatrix} f{_x} & 0 & c{_x} \\ 0 & f{_y} & c{_y} \\ 0 & 0 & 1 \end{bmatrix}$ of shape (3, 3)..
    # $f_{x}$, $f_{y}$: Focal lengths (in pixels) along the $x$ and $y$ axes.
    # $c_{x}$, $c_{y}$: Principal point (the pixel corresponding to the optical center).
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    return (
        np.concatenate(all_imgs, axis=0),
        np.concatenate(all_poses, axis=0),
        intrinsic,
        i_split,
        None,
    )


if __name__ == "__main__":
    # Components of a Camera Pose
    # The camera pose is typically represented using extrinsic parameters, which describe:
        # Position: Where the camera is located in the 3D world (translation    vector).
        # Orientation: The direction the camera is facing in the 3D world (rotation matrix or equivalent representation).
    data_root = Path("/Users/jongbeomkim/Documents/datasets/nerf_synthetic/ship")
    imgs, poses, intrinsic, i_split, _ = load_blender(data_root)
    # print(imgs.shape, poses.shape, intrinsic.shape)
    img = imgs[0]
    pose = poses[0]
    print(img.shape, pose.shape)


    pose = torch.from_numpy(poses[0, : 3, :])