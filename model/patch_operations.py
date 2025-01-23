import torch.nn as nn
from .helpers import Norm

class PatchMerging3D(nn.Module):
    def __init__(self, in_dim, out_dim, downscaling_factor):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=(downscaling_factor, downscaling_factor, 1),
                stride=(downscaling_factor, downscaling_factor, 1)
            ),
            Norm(dim=out_dim),
        )

    def forward(self, x):
        # x: B, C, H, W, D
        x = self.net(x)
        return x  # B,  H //down_scaling, W//down_scaling, D, out_dim

class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim, upscaling_factor: int):
        super(PatchExpanding3D, self).__init__()

        stride = (upscaling_factor, upscaling_factor, 1)
        kernel_size = (upscaling_factor, upscaling_factor, 1)
        padding = ((kernel_size[0] - stride[0]) // 2, (kernel_size[1] - stride[1]) // 2, 0)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            Norm(out_dim),
        )

    def forward(self, x):
        '''X: B,C,X,Y,Z'''
        x = self.net(x)
        return x
