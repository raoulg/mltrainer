import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mltrainer import ReportTypes, TrainerSettings
from mltrainer.metrics import MAE


@pytest.fixture
def sample_settings(temp_logdir):
    return TrainerSettings(
        epochs=10,
        metrics=[MAE()],
        logdir=temp_logdir,
        train_steps=100,
        valid_steps=10,
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 0.001},
    )


class DummyDataset(Dataset):
    """Simple dataset that returns identity mappings"""

    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)  # 10-dimensional data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x  # Identity mapping for simple testing


class IdentityModel(nn.Module):
    """Model that tries to learn the identity function"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


@pytest.fixture
def train_dataloader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=16, shuffle=True)


@pytest.fixture
def valid_dataloader(dummy_dataset):
    return DataLoader(dummy_dataset, batch_size=16, shuffle=False)


@pytest.fixture
def model():
    return IdentityModel()


@pytest.fixture
def loss_fn():
    return nn.MSELoss()


@pytest.fixture
def temp_logdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)
