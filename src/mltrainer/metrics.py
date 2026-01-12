from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Union

import numpy as np
import torch

NumericType = Union[torch.Tensor, np.ndarray]


class Metric(ABC):
    """Base class for all metrics with proper device and type handling"""

    @abstractmethod
    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """Actual metric computation - implemented by subclasses"""
        pass

    def __call__(self, y: NumericType, yhat: NumericType) -> float:
        """Handle device/type conversion and return final metric"""
        # Convert to tensors if needed
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(yhat, np.ndarray):
            yhat = torch.from_numpy(yhat)

        # Ensure we're working with the same device
        device = y.device
        y = y.to(device)
        yhat = yhat.to(device)

        # Compute metric
        result = self._compute(y, yhat)

        # Return as float for consistency
        return float(result.cpu().detach())

    @abstractmethod
    def __repr__(self) -> str:
        pass


class MAE(Metric):
    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(y - yhat))

    def __repr__(self) -> str:
        return "MAE"


class MASE(Metric):
    def __init__(self, train: Iterator, horizon: int) -> None:
        self.horizon = horizon
        with torch.no_grad():
            self.scale = self._calculate_scale(train)

    def _calculate_scale(self, train: Iterator) -> torch.Tensor:
        elist = []
        streamer = train.stream()  # type: ignore
        for _ in range(len(train)):  # type: ignore
            x, y = next(iter(streamer))
            yhat = self._naive_predict(x)
            e = torch.mean(torch.abs(y - yhat))
            elist.append(e)
        return torch.mean(torch.stack(elist))

    def _naive_predict(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., -self.horizon :, :].squeeze(-1)

    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        mae = torch.mean(torch.abs(y - yhat))
        return mae / self.scale

    def __repr__(self) -> str:
        return f"MASE(scale={self.scale:.3f})"


class Accuracy(Metric):
    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        predictions = yhat.argmax(dim=1)
        return (predictions == y).float().mean()

    def __repr__(self) -> str:
        return "Accuracy"


class F1Score(Metric):
    def __init__(self, average="micro", threshold=0.5, epsilon=1e-7):
        self.average = average
        self.threshold = threshold
        self.epsilon = epsilon

    def _compute(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        y_prob = torch.sigmoid(yhat)
        y_pred = (y_prob > self.threshold).float()

        tp = y_pred * y
        fp = y_pred * (1 - y)
        fn = (1 - y_pred) * y

        if self.average == "micro":
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            f1 = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + self.epsilon)

        elif self.average == "macro":
            tp_sum = tp.sum(dim=0)
            fp_sum = fp.sum(dim=0)
            fn_sum = fn.sum(dim=0)
            f1_per_class = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + self.epsilon)
            f1 = f1_per_class.mean()

        else:
            raise ValueError("Average must be 'micro' or 'macro'")

        return f1

    def __repr__(self) -> str:
        return f"F1({self.average})"
