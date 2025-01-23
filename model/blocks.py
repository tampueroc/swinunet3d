import torch.nn as nn
from typing import List, Union
from helpers import Norm, Residual3D, PreNorm3D
from attention import WindowAttention3D
from feed_forward import FeedForward3D

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        groups = min(in_ch, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x2 = x.clone()
        x = self.net(x) * x2
        return x

class SwinBlock3D(nn.Module):
    def __init__(self,
                 dim: int,
                 heads: int,
                 head_dim: int,
                 mlp_dim: int,
                 shifted: bool,
                 window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True,
                 droput: float = 0.0
                 ):
        super().__init__()
        self.attention_block = Residual3D(
            fn=PreNorm3D(
                dim=dim,
                fn=WindowAttention3D(
                        dim=dim,
                        heads=heads,
                        head_dim=head_dim,
                        shifted=shifted,
                        window_size=window_size,
                        relative_pos_embedding=relative_pos_embedding
                )
            )
        )
        self.mlp_block = Residual3D(
            fn=PreNorm3D(
                dim=dim,
                fn=FeedForward3D(
                        dim=dim,
                        hidden_dim=mlp_dim,
                        dropout=droput
                )
            )
        )

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class Converge(nn.Module):
    def __init__(self, dim: int):
        '''
        stack: The fusion method is implemented using stacking + linear transformation.
        add: The skip connection is implemented by directly adding.
        '''
        super(Converge, self).__init__()
        self.norm = Norm(dim=dim)

    def forward(self, x, enc_x):
        '''
        x: B, C, X, Y, Z
        enc_x: B, C, X, Y, Z
        '''
        assert x.shape == enc_x.shape
        x = x + enc_x
        x = self.norm(x)
        return x

