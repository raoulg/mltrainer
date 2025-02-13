import numpy as np
import torch


def dummy_metric1(x):
    return x


def dummy_metric2(x):
    return x * 2


class TestMSE:
    """Simple MSE metric for testing that handles both numpy arrays and tensors"""

    def __repr__(self) -> str:
        return "TestMSE"

    def __call__(
        self, y: torch.Tensor | np.ndarray, yhat: torch.Tensor | np.ndarray
    ) -> float:
        # For testing purposes, we'll just return MSE as a float
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(yhat, torch.Tensor):
            yhat = yhat.detach().cpu().numpy()

        return float(((y - yhat) ** 2).mean())
