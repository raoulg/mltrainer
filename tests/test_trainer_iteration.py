"""Regression tests for the per-step iterator (see mltrainer P0.6).

`next(iter(loader))` inside the step loop is correct for an *iterator* but not
for an *iterable* like a torch DataLoader: it rebuilds the iterator every step,
so training replays a reshuffled first batch and validation scores the same
first batch `valid_steps` times.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from mltrainer import Trainer, TrainerSettings
from tests.utils.dummy_metrics import TestMSE


class CountingDataset(Dataset):
    """Each row is its own index, so a batch is identifiable by its contents."""

    def __init__(self, size: int = 64) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        x = torch.full((10,), float(idx))
        return x, x


def build_trainer(model, loss_fn, temp_logdir, train_loader, valid_loader, **kwargs):
    settings = TrainerSettings(
        epochs=1,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
        **kwargs,
    )
    return Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,  # type: ignore
        traindataloader=train_loader,
        validdataloader=valid_loader,
        scheduler=None,
        device="cpu",
    )


def test_valid_loader_walks_the_whole_split(model, loss_fn, temp_logdir):
    """A shuffle=False DataLoader must yield a *different* batch per step."""
    dataset = CountingDataset(size=64)
    valid_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    trainer = build_trainer(
        model,
        loss_fn,
        temp_logdir,
        DataLoader(dataset, batch_size=16, shuffle=True),
        valid_loader,
        train_steps=1,
        valid_steps=4,
    )

    seen = [
        trainer._next_batch(valid_loader, "_valid_iter")[0][0, 0].item()
        for _ in range(4)
    ]
    assert seen == [0.0, 16.0, 32.0, 48.0]


def test_iterator_wraps_around_at_the_end(model, loss_fn, temp_logdir):
    """Asking for more steps than the loader holds restarts it, not crashes."""
    dataset = CountingDataset(size=32)
    valid_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    trainer = build_trainer(
        model,
        loss_fn,
        temp_logdir,
        DataLoader(dataset, batch_size=16, shuffle=True),
        valid_loader,
        train_steps=1,
        valid_steps=3,
    )

    seen = [
        trainer._next_batch(valid_loader, "_valid_iter")[0][0, 0].item()
        for _ in range(3)
    ]
    assert seen == [0.0, 16.0, 0.0]


def test_plain_iterator_still_advances(model, loss_fn, temp_logdir):
    """The old streamer path: an infinite generator, not a DataLoader."""

    def stream():
        i = 0
        while True:
            yield torch.full((4, 10), float(i)), torch.full((4, 10), float(i))
            i += 1

    streamer = stream()
    trainer = build_trainer(
        model,
        loss_fn,
        temp_logdir,
        streamer,
        streamer,
        train_steps=1,
        valid_steps=3,
    )

    seen = [
        trainer._next_batch(streamer, "_valid_iter")[0][0, 0].item() for _ in range(3)
    ]
    assert seen == [0.0, 1.0, 2.0]


def test_validation_metrics_cover_more_than_one_batch(model, loss_fn, temp_logdir):
    """End to end: evalbatches over a loader whose batches differ wildly must
    not report the metric of a single batch."""
    dataset = CountingDataset(size=64)
    valid_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    trainer = build_trainer(
        model,
        loss_fn,
        temp_logdir,
        DataLoader(dataset, batch_size=16, shuffle=True),
        valid_loader,
        train_steps=1,
        valid_steps=4,
    )

    metric_dict, _ = trainer.evalbatches()

    # what the bug reported: the same model scored on batch 0, four times over
    x, y = next(iter(valid_loader))
    with torch.no_grad():
        first_only = float(TestMSE()(y, trainer.model(x)))

    assert abs(metric_dict[str(TestMSE())] - first_only) > 1e-3
