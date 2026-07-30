from typing import Iterator, Tuple

import numpy as np
import pytest
import torch

from mltrainer.metrics import MAE, MASE, Accuracy


def get_available_devices():
    """Get list of available devices for testing"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=get_available_devices())
def device(request):
    """Parameterized fixture for available devices"""
    return request.param


@pytest.fixture
def tensor_pairs(device):
    """Generate test tensor pairs on the specified device"""
    y = torch.randn(10, 5).to(device)
    yhat = torch.randn(10, 5).to(device)
    return y, yhat


@pytest.fixture
def numpy_pairs():
    """Generate test numpy array pairs"""
    y = np.random.randn(10, 5)
    yhat = np.random.randn(10, 5)
    return y, yhat


@pytest.mark.parametrize("metric_class", [MAE])
def test_metric_tensor_inputs(metric_class, tensor_pairs):
    """Test metrics with tensor inputs"""
    metric = metric_class()
    y, yhat = tensor_pairs
    result = metric(y, yhat)
    assert isinstance(result, float)


@pytest.mark.parametrize("metric_class", [MAE])
def test_metric_numpy_inputs(metric_class, numpy_pairs):
    """Test metrics with numpy inputs"""
    metric = metric_class()
    y, yhat = numpy_pairs
    result = metric(y, yhat)
    assert isinstance(result, float)


def test_accuracy_metric(device):
    """Specific test for accuracy metric"""
    metric = Accuracy()
    y = torch.randint(0, 3, (10,)).to(device)
    yhat = torch.zeros(10, 3).to(device)
    # Set perfect predictions for testing
    for i in range(10):
        yhat[i, y[i]] = 1.0
    result = metric(y, yhat)
    assert result == 1.0  # Should be perfect accuracy


class MockTrainIterator(Iterator[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data):
        self.data = data
        self._index = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._index >= len(self.data):
            self._index = 0  # Reset for reuse
            raise StopIteration
        item = self.data[self._index]
        self._index += 1
        return item

    def stream(self):
        return iter(self)


@pytest.fixture
def mock_train_data():
    """Create mock training data with known patterns"""
    batch_size = 3
    seq_length = 5
    feature_dim = 1

    # Create data with known pattern for reliable scale calculation
    x = torch.ones(batch_size, seq_length, feature_dim)
    y = torch.ones(batch_size, 2) * 2  # Target values are 2
    samples = [(x, y) for _ in range(4)]

    return MockTrainIterator(samples)


@pytest.fixture
def basic_mase():
    """Create a simple MASE instance with known non-zero scale"""
    x = torch.ones(2, 5, 1)
    y = torch.ones(2, 2) * 2  # Make targets different from inputs for non-zero scale
    train_data = MockTrainIterator([(x, y)])
    return MASE(train_data, horizon=2)


def test_mase_initialization(mock_train_data):
    """Test MASE initialization and scale calculation"""
    mase = MASE(mock_train_data, horizon=2)
    assert isinstance(mase.scale, torch.Tensor)
    assert mase.scale.ndim == 0  # Should be a scalar
    assert mase.horizon == 2
    assert not torch.isnan(mase.scale)
    assert mase.scale > 0  # Scale should be positive


def test_mase_naive_predict():
    """Test the naive prediction method"""
    # Create valid training data for initialization
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    y = torch.tensor(
        [[6.0, 7.0]]
    )  # Different from last values to ensure non-zero scale
    train_data = MockTrainIterator([(x, y)])

    mase = MASE(train_data, horizon=2)

    prediction = mase._naive_predict(x)
    expected = torch.tensor([[4.0, 5.0]])  # Last two values for horizon=2

    assert torch.allclose(prediction, expected)
    assert prediction.shape == (1, 2)  # Batch size 1, horizon 2


def test_mase_compute():
    """Test the actual MASE computation"""
    # Create training data that will give a known non-zero scale
    x = torch.ones(1, 5, 1)
    y = torch.ones(1, 2) * 2  # Target is different from naive forecast
    train_data = MockTrainIterator([(x, y)])
    mase = MASE(train_data, horizon=2)

    # Verify scale is non-zero
    assert mase.scale > 0
    assert not torch.isnan(mase.scale)

    # Test with predictions matching target
    y_test = torch.tensor([[2.0, 2.0]])
    yhat_test = torch.tensor([[2.0, 2.0]])
    result = mase._compute(y_test, yhat_test)
    assert not torch.isnan(result)
    assert float(result) == 0.0  # Perfect predictions should give MASE = 0

    # Test with known error
    yhat_test = torch.tensor([[3.0, 3.0]])  # Off by 1.0
    result = mase._compute(y_test, yhat_test)
    assert not torch.isnan(result)
    assert float(result) > 0.0  # Error should give positive MASE


def test_mase_full_pipeline(device):
    """Test the complete MASE pipeline with different devices"""
    x = torch.ones(2, 5, 1).to(device)
    y = torch.ones(2, 2).to(device) * 2  # Different from naive forecast
    train_data = MockTrainIterator([(x, y)])

    mase = MASE(train_data, horizon=2)

    test_y = torch.ones(2, 2).to(device) * 2
    test_yhat = torch.ones(2, 2).to(device) * 2

    result = mase(test_y, test_yhat)
    assert isinstance(result, float)
    assert not torch.isnan(torch.tensor(result))


def test_mase_repr():
    """Test the string representation of MASE"""
    x = torch.ones(1, 5, 1)
    y = torch.ones(1, 2) * 2
    train_data = MockTrainIterator([(x, y)])
    mase = MASE(train_data, horizon=2)

    repr_str = repr(mase)
    assert isinstance(repr_str, str)
    assert "MASE" in repr_str
    assert "scale=" in repr_str
    assert "nan" not in repr_str.lower()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [3, 5, 7])
def test_mase_different_dimensions(batch_size, seq_length):
    """Test MASE with different input dimensions"""
    x = torch.ones(batch_size, seq_length, 1)
    y = torch.ones(batch_size, 2) * 2  # Different from naive forecast
    train_data = MockTrainIterator([(x, y)])

    mase = MASE(train_data, horizon=2)

    test_y = torch.ones(batch_size, 2) * 2
    test_yhat = torch.ones(batch_size, 2) * 2

    result = mase(test_y, test_yhat)
    assert isinstance(result, float)
    assert not torch.isnan(torch.tensor(result))


def test_mase_with_numpy_inputs(basic_mase):
    """Test MASE with numpy array inputs"""
    import numpy as np

    y = np.ones((2, 2)) * 2
    yhat = np.ones((2, 2)) * 2

    result = basic_mase(y, yhat)
    assert isinstance(result, float)
    assert not torch.isnan(torch.tensor(result))


def test_mase_error_cases():
    """Test MASE with invalid inputs"""
    x = torch.ones(1, 5, 1)
    y = torch.ones(1, 2) * 2
    train_data = MockTrainIterator([(x, y)])
    mase = MASE(train_data, horizon=2)

    with pytest.raises(RuntimeError):
        # Test with mismatched dimensions
        y_test = torch.ones(2, 2)
        yhat_test = torch.ones(3, 2)
        mase(y_test, yhat_test)


def test_mase_minimum_data():
    """Test MASE with minimum valid data"""
    x = torch.ones(1, 3, 1)  # Minimum sequence length for horizon=2
    y = torch.ones(1, 2) * 2
    train_data = MockTrainIterator([(x, y)])

    mase = MASE(train_data, horizon=2)
    assert not torch.isnan(mase.scale)
    assert mase.scale > 0


# ---------------------------------------------------------------------------
# MASE over things that are not a mads_datasets streamer.
#
# `_calculate_scale` used to be written against BaseDatastreamer: it called
# `train.stream()`, needed `len(train)`, and pulled batches with
# `next(iter(streamer))`. None of that works on a torch DataLoader, and the
# last one is the per-step iterator-rebuild bug fixed in Trainer._next_batch.
# ---------------------------------------------------------------------------


class SeriesDataset(torch.utils.data.Dataset):
    """Windowed series: x is (seq_length, 1), y is the next `horizon` steps.

    Each item is built from its own index, so no two batches are alike.
    """

    def __init__(self, n: int = 8, seq_length: int = 5, horizon: int = 2):
        self.n, self.seq_length, self.horizon = n, seq_length, horizon

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.full((self.seq_length, 1), float(idx))
        y = torch.full((self.horizon,), float(idx) + 1.0)
        return x, y


def endless_batches():
    while True:
        yield torch.zeros(2, 3, 1), torch.ones(2, 2)


def test_mase_accepts_a_dataloader():
    loader = torch.utils.data.DataLoader(SeriesDataset(n=8), batch_size=2)

    mase = MASE(loader, horizon=2)

    # the naive forecast repeats x's last `horizon` steps, i.e. idx, while y is
    # idx + 1 -- so every batch has an absolute error of exactly 1.0
    assert float(mase.scale) == pytest.approx(1.0)


def test_mase_walks_the_whole_loader_not_just_the_first_batch():
    """The bug this guards: `next(iter(loader))` per step re-reads batch 0."""

    class GrowingError(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            # error for this item is idx: 0, 1, 2, 3 -> mean 1.5
            return torch.zeros(3, 1), torch.full((2,), float(idx))

    loader = torch.utils.data.DataLoader(GrowingError(), batch_size=1)

    mase = MASE(loader, horizon=2)

    # the first batch alone gives 0.0, which is what the old code computed
    assert float(mase.scale) == pytest.approx(1.5)


def test_mase_accepts_a_plain_generator_with_n_batches():
    mase = MASE(endless_batches(), horizon=2, n_batches=3)
    assert float(mase.scale) == pytest.approx(1.0)


def test_mase_refuses_an_unbounded_iterable_without_n_batches():
    with pytest.raises(ValueError, match="n_batches"):
        MASE(endless_batches(), horizon=2)


def test_mase_n_batches_bounds_a_legacy_streamer():
    x = torch.ones(2, 5, 1)
    y = torch.ones(2, 2) * 2
    train_data = MockTrainIterator([(x, y)] * 4)

    mase = MASE(train_data, horizon=2, n_batches=2)
    assert not torch.isnan(mase.scale)


def test_mase_rejects_an_empty_iterable():
    with pytest.raises(ValueError, match="no batches"):
        MASE([], horizon=2)
