import torch.nn as nn
from typing import Union, List
from einops.layers.torch import Rearrange
from .patch_operations import PatchMerging3D
from .blocks import ConvBlock, SwinBlock3D

class Encoder(nn.Module):
    def __init__(self,
                 in_dims: int,
                 hidden_dims: int,
                 downscaling_factor: int,
                 layers: int,
                 num_heads: int,
                 head_dim: int,
                 window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True,
                 dropout: float = 0.0):
        super().__init__()

        # Patch Merge
        self.patch_partition = PatchMerging3D(
                in_dim=in_dims,
                out_dim=hidden_dims,
                downscaling_factor=downscaling_factor
        )

        # Rearrange
        self.re1 = Rearrange('b c h w d -> b h w d c')
        self.re2 = Rearrange('b  h w d c -> b c h w d')

        # Blocks
        self.conv_block = ConvBlock(
                in_ch=hidden_dims,
                out_ch=hidden_dims
        )
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dims, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dims * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
                SwinBlock3D(dim=hidden_dims, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dims * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            dropout=dropout),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        x2 = self.conv_block(x)  # CNN for short length dependencies
        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:
            # Swin Blocks for long length dependencies
            x = regular_block(x)
            x = shifted_block(x)
        x = self.re2(x)
        x += x2  # Fuse
        return x
