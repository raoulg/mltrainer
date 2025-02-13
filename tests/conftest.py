import shutil
import tempfile
from pathlib import Path

import pytest

from mltrainer.settings import ReportTypes, TrainerSettings
from tests.utils.dummy_metrics import dummy_metric1, dummy_metric2


@pytest.fixture
def test_debug_dir(request):
    """Provide a persistent directory for debugging test failures.
    The directory is cleaned only if tests pass."""
    debug_dir = Path(__file__).parent / "test_debug"
    debug_dir.mkdir(exist_ok=True)
    yield debug_dir

    # Only clean up if all tests in the session passed
    if not request.session.testsfailed:
        shutil.rmtree(debug_dir)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's always cleaned up."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_settings(test_debug_dir):
    """Create a sample TrainerSettings instance."""
    return TrainerSettings(
        epochs=10,
        metrics=[dummy_metric1, dummy_metric2],
        logdir=test_debug_dir / "logs",
        train_steps=100,
        valid_steps=10,
        reporttypes=[ReportTypes.TOML, ReportTypes.TENSORBOARD],
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-5},
        scheduler_kwargs={"factor": 0.1, "patience": 10},
        earlystop_kwargs={
            "save": False,
            "verbose": True,
            "patience": 10,
        },
    )
