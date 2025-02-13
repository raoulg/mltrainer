from pathlib import Path

import pytest
import torch
import torch.optim as optim

from mltrainer import ReportTypes, Trainer, TrainerSettings
from mltrainer.metrics import MAE
from mltrainer.trainer import dir_add_timestamp
from tests.utils.dummy_metrics import TestMSE


def test_trainer_init(model, train_dataloader, valid_dataloader, loss_fn, temp_logdir):
    settings = TrainerSettings(
        epochs=2,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        train_steps=5,
        valid_steps=2,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,  # type: ignore
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=None,
        device="cpu",
    )

    assert trainer is not None
    assert trainer.model is not None
    assert trainer.settings.epochs == 2
    assert trainer.settings.train_steps == 5
    assert trainer.settings.valid_steps == 2
    assert trainer.settings.optimizer_kwargs == {"lr": 0.01}


def test_trainer_single_epoch(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir
):
    settings = TrainerSettings(
        epochs=1,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,  # type: ignore
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=None,
        device="cpu",
    )

    trainer.loop()


@pytest.mark.tensorboard
def test_trainer_with_tensorboard(
    model, train_dataloader, valid_dataloader, loss_fn, tmp_path
):
    """Separate test for TensorBoard functionality using pytest's tmp_path"""
    settings = TrainerSettings(
        epochs=1,
        metrics=[TestMSE()],
        logdir=tmp_path,  # Use pytest's tmp_path instead of our temp_logdir
        train_steps=2,
        valid_steps=1,
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 0.01},
        earlystop_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,  # type: ignore
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=None,
        device="cpu",
    )

    trainer.loop()

    if hasattr(trainer, "writer"):
        assert trainer.writer is not None
        trainer.writer.close()


# tests/test_trainer.py


def test_dir_add_timestamp_default():
    """Test dir_add_timestamp with default None argument"""
    result = dir_add_timestamp()
    assert result.parent == Path(".")
    assert len(result.name) == 15  # YYYYMMDD-HHMMSS format


def test_scheduler_missing_kwargs(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir
):
    """Test that trainer raises error when scheduler is provided without kwargs"""
    settings = TrainerSettings(
        epochs=1,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
        scheduler_kwargs=None,  # This should trigger the error
    )

    with pytest.raises(
        ValueError, match="Missing 'scheduler_kwargs' in TrainerSettings"
    ):
        Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optim.Adam,
            traindataloader=train_dataloader,
            validdataloader=valid_dataloader,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device="cpu",
        )


def test_trainer_with_reduce_lr_scheduler(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir
):
    """Test trainer with ReduceLROnPlateau scheduler"""
    settings = TrainerSettings(
        epochs=2,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.1},
        scheduler_kwargs={"factor": 0.1, "patience": 1},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
        device="cpu",
    )

    initial_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    trainer.loop()
    final_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    assert final_lr <= initial_lr  # LR should have been reduced


def test_early_stopping_save(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir
):
    """Test early stopping with model saving"""
    settings = TrainerSettings(
        epochs=3,
        metrics=[TestMSE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
        earlystop_kwargs={"patience": 1, "save": True, "verbose": True},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=None,
        device="cpu",
    )

    # Run training
    trainer.loop()

    # Check that checkpoint file was created
    checkpoint_path = temp_logdir / trainer.log_dir.name / "checkpoint.pt"
    assert checkpoint_path.exists()

    # Load and verify the saved model
    loaded_model = torch.load(checkpoint_path, weights_only=False)
    assert isinstance(loaded_model, type(model))


def test_scheduler_step_without_plateau(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir
):
    """Test scheduler step for non-ReduceLROnPlateau scheduler (line 166)"""
    settings = TrainerSettings(
        epochs=1,
        metrics=[MAE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[],
        optimizer_kwargs={"lr": 0.01},
        scheduler_kwargs={"step_size": 1, "gamma": 0.1},
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=torch.optim.lr_scheduler.StepLR,
        device="cpu",
    )

    # Get initial learning rate
    initial_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore

    # Run one epoch
    trainer.loop()

    # Verify learning rate was updated
    final_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    assert final_lr == initial_lr * 0.1  # StepLR with gamma=0.1
