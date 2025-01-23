import torch.nn as nn
from einops import rearrange, einsum
import pdb
from typing import Union, List
from .helpers import CyclicShift3D, create_mask3D
import numpy as np

class WindowAttention3D(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            head_dim: int,
            shifted: bool,
            window_size: Union[int, List[int]],
            relative_pos_embedding: bool = True):
        super().__init__()

        # Window Size
        if type(window_size) is int:
            self.window_size = np.array([window_size, window_size, window_size])
        else:
            self.window_size = np.array(window_size)

        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.shifted = shifted

        if self.shifted is True:
            displacement = self.window_size // 2
            self.cyclic_shift = CyclicShift3D(
                    displacement=-displacement
            )
            self.cyclic_back_shift = CyclicShift3D(
                    displacement=displacement
            )
            self.x_mask = nn.Parameter(
                create_mask3D(
                    window_size=self.window_size,
                    displacement=displacement,
                    x_shift=True)
            )
            self.y_mask = nn.Parameter(
                create_mask3D(
                    window_size=self.window_size,
                    displacement=displacement,
                    y_shift=True
                )
            )
            self.z_mask = nn.Parameter(
                create_mask3D(
                    window_size=self.window_size,
                    displacement=displacement,
                    z_shift=True)
            )
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted is True:
            x = self.cyclic_shift(x)
        b, n_x, n_y, n_z, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_x = n_x // self.window_size[0]
        nw_y = n_y // self.window_size[1]
        nw_z = n_z // self.window_size[2]
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
                                h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.shifted is True:
            # Move the windows along the x-axis to the end to align with the x-axis mask (similar for other axes)
            dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
                             n_x=nw_x, n_y=nw_y)
            #   b   h n_y n_z n_x i j
            dots[:, :, :, :, -1] += self.x_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
            dots[:, :, :, :, -1] += self.y_mask

            dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
            dots[:, :, :, :, -1] += self.z_mask

            dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')
        attn = self.softmax(dots)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
                        h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2],
                        nw_x=nw_x, nw_y=nw_y, nw_z=nw_z)
        out = self.to_out(out)
        if self.shifted is True:
            out = self.cyclic_back_shift(out)
        return out






