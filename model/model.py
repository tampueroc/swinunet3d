import pytorch_lightning as pl
import torchmetrics
import torch.optim as optim
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder
from .blocks import Converge, FinalExpand3D
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
                 num_classes: int = 1,
                 learning_rate: float = 3e-4,
                 stl_channels: int = 32
                 ):
        super().__init__()
        self.learning_rate = learning_rate
         # Metrics for training
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        # Metrics for training
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        # Loss
        self.loss_fn = F.binary_cross_entropy_with_logits

        # Encoders
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

        # Decoder
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

        self.final = FinalExpand3D(
                in_dim=hidden_dim,
                out_dim=stl_channels,
                upscaling_factor=downscaling_factors[0]
        )
        self.out = nn.Sequential(
            nn.Conv3d(stl_channels, num_classes, kernel_size=1)
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
        out = self.out(out)

        return out

    def training_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq)
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("train_loss", loss)

        # Update metrics
        self.train_accuracy(pred, isochrone_mask)
        self.train_precision(pred, isochrone_mask)
        self.train_recall(pred, isochrone_mask)
        self.train_f1(pred, isochrone_mask)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=False)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=False)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=False)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def validation_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq)
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("val_loss", loss)

        # Update metrics
        self.val_accuracy(pred, isochrone_mask)
        self.val_precision(pred, isochrone_mask)
        self.val_recall(pred, isochrone_mask)
        self.val_f1(pred, isochrone_mask)

        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

