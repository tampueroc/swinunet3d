import pytorch_lightning as pl
from .encoder import Encoder
from .decoder import Decoder
from .blocks import Converge
import torch.nn as nn

class SwinUNet3D(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 layers: list,
                 downscaling_factors: list,
                 heads: list,
                 head_dim: int,
                 window_size: list,
                 dropout: float = 0.0,
                 relative_pos_embedding: bool = True,
                 num_classes: int = 1
                 ):
        super().__init__()
        self.enc12 = Encoder(
            in_dims=in_channels,
            hidden_dims=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.enc3 = Encoder(
            in_dims=hidden_dim,
            hidden_dims=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.enc4 = Encoder(
            in_dims=hidden_dim * 2,
            hidden_dims=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.enc5 = Encoder(
            in_dims=hidden_dim * 4,
            hidden_dims=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.dec4 = Decoder(
            in_dims=hidden_dim * 8,
            out_dims=hidden_dim * 4,
            layers=layers[2],
            upscaling_factor=downscaling_factors[3],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.dec3 = Decoder(
            in_dims=hidden_dim * 4,
            out_dims=hidden_dim * 2,
            layers=layers[1],
            upscaling_factor=downscaling_factors[2],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )
        self.dec12 = Decoder(
            in_dims=hidden_dim * 2,
            out_dims=hidden_dim,
            layers=layers[0],
            upscaling_factor=downscaling_factors[1],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            dropout=dropout,
            relative_pos_embedding=relative_pos_embedding
        )

        self.converge4 = Converge(
            dim=hidden_dim * 4
        )
        self.converge3 = Converge(
            dim=hidden_dim * 2
        )
        self.converge12 = Converge(
            dim=hidden_dim
        )

        self.final = nn.Sequential(
            nn.Conv3d(hidden_dim, num_classes, kernel_size=(1, 1, 1)),
            nn.Conv3d(num_classes, num_classes, kernel_size=(1, 1, 1))
        )

    def forward(self, x):

        down12 = self.enc12(x)
        down3 = self.enc3(down12)
        down4 = self.enc4(down3)

        features = self.enc5(down4)

        up4 = self.dec4(features)
        up4 = self.converge4(up4, down4)

        up3 = self.dec3(up4)
        up3 = self.converge3(up3, down3)

        up12 = self.dec12(up3)
        up12 = self.converge12(up12, down12)

        out = self.final(up12)

        return out

