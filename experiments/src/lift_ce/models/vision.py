from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=0.15),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(
        self, images: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embedding(self.features(images))
        logits = self.classifier(hidden).squeeze(-1)
        if return_embedding:
            return logits, hidden
        return logits

    @torch.no_grad()
    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(images))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32 * 7 * 7, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder_conv(images).flatten(1)
        return self.encoder_fc(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_fc(latent).view(-1, 32, 7, 7)
        return self.decoder_conv(hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class ConvVAE(nn.Module):
    """Convolutional VAE for the gated MNIST latent-space experiment."""
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.encoder_hidden = nn.Sequential(nn.Flatten(), nn.Linear(128*7*7, 256), nn.ReLU(inplace=True))
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 128*7*7), nn.ReLU(inplace=True))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def encode_stats(self, images: torch.Tensor):
        h = self.encoder_hidden(self.encoder_conv(images))
        return self.mu(h), self.logvar(h).clamp(-12, 12)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_stats(images)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder_conv(self.decoder_fc(latent).view(-1,128,7,7))

    def forward(self, images: torch.Tensor):
        mu, logvar = self.encode_stats(images)
        z = mu + torch.randn_like(mu) * torch.exp(.5*logvar) if self.training else mu
        return self.decode(z), mu, logvar


class MNISTReferenceCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activation(images + self.block(images))


class StrongConvAutoencoder(nn.Module):
    """A stronger denoising realization model for MNIST counterfactual images.

    It keeps the same ``encode``/``decode`` interface as ``ConvAutoencoder`` but
    uses wider convolutional features and residual refinement blocks. This is
    still lightweight enough for the small MNIST experiments while producing
    much sharper decoded digits than the original two-layer autoencoder.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder_fc(self.encoder_conv(images))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_fc(latent).view(-1, 128, 7, 7)
        return self.decoder_conv(hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


def make_autoencoder(kind: str, latent_dim: int) -> nn.Module:
    if kind == "conv":
        return ConvAutoencoder(latent_dim)
    if kind == "strong":
        return StrongConvAutoencoder(latent_dim)
    raise ValueError(f"Unsupported autoencoder_type: {kind}")
