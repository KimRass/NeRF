# Reference: https://csm-kr.tistory.com/64

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, l, include_input: bool = True):
        super().__init__()

        self.include_input = include_input

        freq_bands = 2. ** torch.arange(0, l)
        self.fn = lambda x: torch.cat(
            [
                fn(freq * math.pi * x) for freq in freq_bands for fn in [torch.sin, torch.cos]
            ],
            dim=-1,
        )

    def forward(self, x):
        ori_x = x
        x = self.fn(x)
        if self.include_input:
            x = torch.cat([ori_x, x], dim=-1)
        return x


class NeRF(nn.Module):
    def __init__(
        self, width: int = 256, coord_channels: int = 60, direc_channels: int = 24,
    ):
        super().__init__()

        self.coord_channels = coord_channels
        self.direc_channels = direc_channels

        self.lin1 = nn.Sequential(
            nn.Linear(coord_channels, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
        )  # 4 layers.
        self.lin2 = nn.Sequential(
            nn.Linear(width + coord_channels, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
        )  # 4 layers

        ### Density.
        self.lin_density = nn.Linear(width, 1)  # `1`: Density.

        ### Color.
        self.lin3 = nn.Linear(width, width)
        self.lin_color = nn.Sequential(
            nn.Linear(direc_channels + width, width//2),
            # "... Using a ReLU activation and 128 channels."
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 3),  # `3`: RGB.
            nn.Sigmoid(),
        )
    def forward(self, x):
        coord, direc = torch.split(
            x, [self.coord_channels, self.direc_channels], dim=-1,
        )
        # `coord`: $\gamma{\mathbf{x}} = \gamma{(x, y, z)}$.
        # `direc`: $\gamma{\mathbf{d}} = \gamma{(\theta, \phi)}$.
        # "We represent a continuous scene as a 5D vector-valued function whose input is a 3D location $\mathbf{x} = (x, y, z)$ and 2D viewing direction $(\theta, \phi)$, and whose output is an emitted color $\mathbf{c} = (r, g, b)$ and volume density $\sigma$."

        x = self.lin1(coord)
        x = torch.cat([coord, x], dim=-1)
        # $`input_x`: \gamma(x)$
        # "The MLP $F_{\theta}$ first processes the input 3D coordinate $x$ with 8 fully-connected layers (using ReLU activations and 256 channels per layer), and outputs and a 256-dimensional feature vector."
        x = self.lin2(x)

        ### Volume density.
        density_out = self.lin_density(x)  # (batch_size, 1)
        # "... by restricting the network to predict the volume density $\sigma$ as a function of only the location $x$."
        # $\sigma(\mathbf{r}(t))$.

        ### Directional emitted color.
        x = self.lin3(x)
        x = torch.cat([x, direc], dim=-1)
        # "... allowing the RGB color $c$ to be predicted as a function of both location and viewing direction."
        rgb_out = self.lin_color(x)  # (batch_size, 3)
        # "This feature vector is then concatenated with the camera ray's viewing direction and passed to one additional fully-connected layer that output the view-dependent RGB color."
        # $\mathbf{c}(\mathbf{r}(t), d) = (r, g, b)$.
        return torch.cat([density_out, rgb_out], dim=-1)  # (batch_size, 4)


if __name__ == "__main__":
    batch_size = 4
    channels = 60
    direc_channels = 24
    device = torch.device("mps")
    nerf = NeRF(coord_channels=channels, direc_channels=direc_channels).to(device)
    x = torch.randn((batch_size, channels + direc_channels), device=device)
    out = nerf(x)
    print(out.shape)

    pe = PositionalEncoding(l=10)
    x = torch.randn((batch_size, 3), device=device)
    out = pe(x)
    print(out.shape)
