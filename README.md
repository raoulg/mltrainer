# 🚀 ML Trainer Package
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPi version](https://badgen.net/pypi/v/mltrainer/)](https://pypi.org/project/mltrainer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible and powerful PyTorch training framework with built-in logging, metrics tracking, and early stopping capabilities! 

## 📦 Key Components

- **Trainer**: Main training loop with validation and reporting
- **TrainerSettings**: Configuration management for training parameters
- **Models**: Collection of CNN and RNN architectures
- **Metrics**: Customizable evaluation metrics
- **Preprocessors**: Data preparation utilities

## 🛠️ Installation
Use uv, or if you want to use the 10-100x slower pip, i wont stop you.
```bash
uv add mltrainer # recommended
pip install mltrainer # i cant stop you
```

## 🎯 Quick Start

Here's a simple example using a CNN model with MNIST:

```python
from trainer import Trainer, TrainerSettings
from imagemodels import CNN
from metrics import Accuracy
from preprocessors import BasePreprocessor
from settings import ReportTypes
from pathlib import Path

# Define training settings
settings = TrainerSettings(
    epochs=10,
    metrics=[Accuracy()],
    logdir=Path("./logs"),
    train_steps=100,
    valid_steps=20,
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    optimizer_kwargs={"lr": 0.001},
    scheduler_kwargs={"factor": 0.1, "patience": 5},
    earlystop_kwargs={"patience": 7, "save": True}
)

# Initialize model and trainer
model = CNN(num_classes=10, kernel_size=3, filter1=32, filter2=64)
trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    traindataloader=train_loader,  # Your DataLoader
    validdataloader=valid_loader,  # Your DataLoader
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Start training
trainer.loop()
```

## 📊 Report Types

The package supports multiple reporting backends:

- 📈 **TENSORBOARD**: Real-time training visualization
- 📝 **TOML**: Configuration and model architecture serialization
- 📊 **MLFLOW**: Experiment tracking and model management
- 🔄 **RAY**: Distributed training support

Configure them in TrainerSettings:
```python
settings = TrainerSettings(
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
    # ... other settings
)
```

## 🔍 Metrics

Built-in metrics include:

- **Accuracy**: Classification accuracy
- **MAE**: Mean Absolute Error
- **MASE**: Mean Absolute Scaled Error (for time series)

Metrics are PyTorch-native and handle device placement automatically:

```python
from metrics import Accuracy, MAE

settings = TrainerSettings(
    metrics=[Accuracy(), MAE()],
    # ... other settings
)
```

## 🔄 Preprocessors

Two main preprocessors are available:

1. **BasePreprocessor**: Standard batch processing for fixed-size inputs
   ```python
   preprocessor = BasePreprocessor()
   batch_x, batch_y = preprocessor(batch)
   ```

2. **PaddedPreprocessor**: Handles variable-length sequences with padding
   ```python
   preprocessor = PaddedPreprocessor()
   padded_x, batch_y = preprocessor(sequence_batch)
   ```

## 🧠 Available Models

The package includes several model architectures:

### Image Models
- CNN with configurable filters
- Neural Network with customizable layers

### RNN Models
- Base RNN
- GRU with optional attention
- NLP models with embedding support

Example using AttentionGRU:
```python
config = {
    "input_size": 10,
    "hidden_size": 64,
    "output_size": 1,
    "num_layers": 2,
    "dropout": 0.1
}
model = AttentionGRU(config)
```

## ⚙️ Advanced Configuration

TrainerSettings supports comprehensive training configuration:

```python
settings = TrainerSettings(
    epochs=100,
    metrics=[Accuracy()],
    logdir=Path("./experiments"),
    train_steps=500,
    valid_steps=50,
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
    optimizer_kwargs={
        "lr": 1e-3,
        "weight_decay": 1e-5
    },
    scheduler_kwargs={
        "factor": 0.1,
        "patience": 10
    },
    earlystop_kwargs={
        "save": True,
        "verbose": True,
        "patience": 10
    }
)
```

## 🔔 Early Stopping

The trainer includes built-in early stopping with model checkpointing:

```python
settings = TrainerSettings(
    earlystop_kwargs={
        "patience": 7,      # Episodes to wait before stopping
        "save": True,       # Save best model
        "verbose": True,    # Print progress
        "delta": 0.001     # Minimum improvement threshold
    },
    # ... other settings
)
```

## 📝 Logging

The package uses loguru for comprehensive logging. All training progress, early stopping events, and potential issues are automatically logged:

```python
from loguru import logger

# Logs are automatically created in your logdir
# Example log message:
# [2024-02-13 14:30:22] INFO: Epoch 5 train 0.3421 test 0.2891 metric [0.8934]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
