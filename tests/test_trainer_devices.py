import pytest
import torch
import torch.optim as optim

from mltrainer import ReportTypes, Trainer, TrainerSettings
from mltrainer.metrics import MAE


def get_available_devices():
    """Get list of available devices for testing"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", get_available_devices())
def test_trainer_device_compatibility(
    model, train_dataloader, valid_dataloader, loss_fn, temp_logdir, device
):
    """Test trainer works with different devices"""
    settings = TrainerSettings(
        epochs=1,
        metrics=[MAE()],
        logdir=temp_logdir,
        train_steps=2,
        valid_steps=1,
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 0.01},
        earlystop_kwargs=None,
    )

    # Skip test if device is not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        pytest.skip("MPS device not available")

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=loss_fn,
        optimizer=optim.Adam,  # type: ignore
        traindataloader=train_dataloader,
        validdataloader=valid_dataloader,
        scheduler=None,
        device=device,
    )

    trainer.loop()  # Should complete without errors
