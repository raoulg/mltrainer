from dataclasses import dataclass

from torch import Tensor, nn


class BaseModel(nn.Module):
    def __init__(self, observations: int, horizon: int) -> None:
        super().__init__()
        self.observations = observations
        self.horizon = horizon
        self.flatten = nn.Flatten()  # we have 3d data, the linear model wants 2D
        self.linear = nn.Linear(observations, horizon)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        return x


@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout: float = 0.0


class BaseRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, horizon: int
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(hidden_size, horizon)
        self.horizon = horizon

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class GRUmodel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            num_layers=config.num_layers,
        )
        self.linear = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionGRU(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config.__dict__
        self.rnn = nn.GRU(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"],
            batch_first=True,
            num_layers=self.config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config["hidden_size"],
            num_heads=4,
            dropout=self.config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class NLPmodel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config.__dict__
        self.emb = nn.Embedding(self.config["input_size"], self.config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=self.config["hidden_size"],
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"],
            batch_first=True,
            num_layers=self.config["num_layers"],
        )
        self.linear = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat


class AttentionNLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config.__dict__
        self.emb = nn.Embedding(self.config["input_size"], self.config["hidden_size"])
        self.rnn = nn.GRU(
            input_size=self.config["hidden_size"],
            hidden_size=self.config["hidden_size"],
            dropout=self.config["dropout"],
            batch_first=True,
            num_layers=self.config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.config["hidden_size"],
            num_heads=4,
            dropout=self.config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(self.config["hidden_size"], self.config["output_size"])

    def forward(self, x: Tensor) -> Tensor:
        x = self.emb(x)
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat
