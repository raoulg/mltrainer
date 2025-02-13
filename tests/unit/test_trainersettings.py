from mltrainer.settings import ReportTypes, TrainerSettings
from tests.utils.dummy_metrics import dummy_metric1


def test_basic_initialization(sample_settings):
    """Test that TrainerSettings can be initialized with basic parameters."""
    assert sample_settings.epochs == 10
    assert len(sample_settings.metrics) == 2
    assert sample_settings.train_steps == 100
    assert sample_settings.valid_steps == 10
    assert len(sample_settings.reporttypes) == 2


def test_default_values(sample_settings):
    """Test that default values are set correctly."""
    assert sample_settings.optimizer_kwargs["lr"] == 1e-3
    assert sample_settings.optimizer_kwargs["weight_decay"] == 1e-5
    assert sample_settings.scheduler_kwargs["factor"] == 0.1
    assert sample_settings.earlystop_kwargs["patience"] == 10


def test_logdir_creation(temp_dir):
    """Test that logdir is created if it doesn't exist."""
    log_path = temp_dir / "new_logs"
    assert not log_path.exists()

    _ = TrainerSettings(
        epochs=5,
        metrics=[dummy_metric1],
        logdir=log_path,
        train_steps=50,
        valid_steps=5,
        reporttypes=[ReportTypes.TOML],
    )

    assert log_path.exists()
    assert log_path.is_dir()
