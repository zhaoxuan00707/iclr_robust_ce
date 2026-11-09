from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils import seed_everything


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, 1)

    def forward(
        self, values: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(values)
        logits = self.classifier(embedding).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits


def train_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    hidden_dims: list[int],
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[TabularMLP, float]:
    seed_everything(seed)
    model = TabularMLP(train_x.shape[1], hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(test_x)) >= 0.5).float()
        accuracy = float((prediction == test_y).float().mean())
    return model, accuracy


def validity(model: nn.Module, samples: torch.Tensor) -> float:
    with torch.no_grad():
        return float((torch.sigmoid(model(samples)) >= 0.5).float().mean())
