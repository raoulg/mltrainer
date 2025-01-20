# from collections import Counter, OrderedDict

from typing import Protocol

import torch
from torch.nn.utils.rnn import pad_sequence


class PreprocessorProtocol(Protocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]: ...


class BasePreprocessor(PreprocessorProtocol):
    def __call__(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        return torch.stack(X), torch.stack(y)


class PaddedPreprocessor(PreprocessorProtocol):
    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = zip(*batch)
        X_ = pad_sequence(X, batch_first=True, padding_value=0)  # type: ignore
        return X_, torch.tensor(y)
