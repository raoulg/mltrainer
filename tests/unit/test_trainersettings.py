from mltrainer import ReportTypes, TrainerSettings
from mltrainer.metrics import MAE


def test_basic_initialization(sample_settings):
    assert sample_settings.epochs == 10
    assert len(sample_settings.metrics) == 1
    assert isinstance(sample_settings.metrics[0], MAE)
    assert sample_settings.train_steps == 100
    assert sample_settings.valid_steps == 10
    assert sample_settings.reporttypes == [ReportTypes.TENSORBOARD]
    assert sample_settings.optimizer_kwargs == {"lr": 0.001}


def test_default_values(sample_settings):
    """Test that default values are set correctly"""
    assert sample_settings.scheduler_kwargs == {"factor": 0.1, "patience": 10}
    assert sample_settings.earlystop_kwargs == {
        "save": False,
        "verbose": True,
        "patience": 10,
    }


def test_logdir_creation(temp_logdir):
    """Test that logdir is created if it doesn't exist"""
    new_dir = temp_logdir / "new_test_dir"
    _ = TrainerSettings(
        epochs=1,
        metrics=[MAE()],
        logdir=new_dir,
        train_steps=1,
        valid_steps=1,
        reporttypes=[ReportTypes.TENSORBOARD],
    )
    assert new_dir.exists()
