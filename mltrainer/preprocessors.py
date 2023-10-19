# from collections import Counter, OrderedDict

from torch.nn.utils.rnn import pad_sequence
from typing import Protocol
import torch


class PreprocessorProtocol(Protocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class BasePreprocessor(PreprocessorProtocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        return torch.stack(X), torch.stack(y)


class PaddedPreprocessor(PreprocessorProtocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
        return X_, torch.tensor(y)

