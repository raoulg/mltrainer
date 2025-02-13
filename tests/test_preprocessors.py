# tests/test_preprocessors.py
import pytest
import torch

from mltrainer.preprocessors import BasePreprocessor, PaddedPreprocessor


@pytest.fixture
def fixed_batch():
    """Batch with fixed-size tensors"""
    batch = [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0])),
        (torch.tensor([4.0, 5.0, 6.0]), torch.tensor([0.0])),
        (torch.tensor([7.0, 8.0, 9.0]), torch.tensor([1.0])),
    ]
    return batch


@pytest.fixture
def variable_batch():
    """Batch with variable-length tensors"""
    batch = [
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0])),
        (torch.tensor([4.0, 5.0]), torch.tensor([0.0])),
        (torch.tensor([7.0]), torch.tensor([1.0])),
    ]
    return batch


def test_base_preprocessor_fixed_size(fixed_batch):
    """Test BasePreprocessor with fixed-size tensors"""
    preprocessor = BasePreprocessor()
    X, y = preprocessor(fixed_batch)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape == (3, 3)  # 3 samples, 3 features
    assert y.shape == (3, 1)  # 3 samples, 1 target
    assert torch.allclose(X[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(y, torch.tensor([[1.0], [0.0], [1.0]]))


def test_base_preprocessor_fails_variable_size(variable_batch):
    """Test that BasePreprocessor raises error with variable-length tensors"""
    preprocessor = BasePreprocessor()
    with pytest.raises(RuntimeError):
        X, y = preprocessor(variable_batch)


def test_padded_preprocessor_fixed_size(fixed_batch):
    """Test PaddedPreprocessor with fixed-size tensors"""
    preprocessor = PaddedPreprocessor()
    X, y = preprocessor(fixed_batch)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape == (3, 3)  # Should maintain original shape
    assert y.shape == (3,)  # Target shape should be unchanged
    assert torch.allclose(X[0], torch.tensor([1.0, 2.0, 3.0]))


def test_padded_preprocessor_variable_size(variable_batch):
    """Test PaddedPreprocessor with variable-length tensors"""
    preprocessor = PaddedPreprocessor()
    X, y = preprocessor(variable_batch)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape == (3, 3)  # Should pad to longest sequence (3)
    assert y.shape == (3,)  # Target shape should be unchanged

    # Check padding
    assert torch.allclose(X[0], torch.tensor([1.0, 2.0, 3.0]))  # Original
    assert torch.allclose(X[1], torch.tensor([4.0, 5.0, 0.0]))  # Padded
    assert torch.allclose(X[2], torch.tensor([7.0, 0.0, 0.0]))  # Padded


def test_padded_preprocessor_empty_sequence():
    """Test PaddedPreprocessor with an empty sequence in batch"""
    batch = [
        (torch.tensor([1.0, 2.0]), torch.tensor([1.0])),
        (torch.tensor([]), torch.tensor([0.0])),
        (torch.tensor([7.0]), torch.tensor([1.0])),
    ]
    preprocessor = PaddedPreprocessor()
    X, y = preprocessor(batch)

    assert X.shape == (3, 2)  # Should pad to longest sequence (2)
    assert torch.allclose(X[1], torch.tensor([0.0, 0.0]))  # Empty sequence padded


def test_padded_preprocessor_single_item():
    """Test PaddedPreprocessor with a single item"""
    batch = [(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0]))]
    preprocessor = PaddedPreprocessor()
    X, y = preprocessor(batch)

    assert X.shape == (1, 3)
    assert y.shape == (1,)
    assert torch.allclose(X[0], torch.tensor([1.0, 2.0, 3.0]))


def test_preprocessor_empty_batch():
    """Test both preprocessors with empty batch"""
    batch = []
    base_preprocessor = BasePreprocessor()
    padded_preprocessor = PaddedPreprocessor()

    with pytest.raises(ValueError):
        base_preprocessor(batch)

    with pytest.raises(ValueError):
        padded_preprocessor(batch)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
def test_different_batch_sizes(batch_size):
    """Test preprocessors with different batch sizes"""
    batch = [(torch.randn(3), torch.tensor([1.0])) for _ in range(batch_size)]

    base_preprocessor = BasePreprocessor()
    X, y = base_preprocessor(batch)
    assert X.shape == (batch_size, 3)
    assert y.shape == (batch_size, 1)

    padded_preprocessor = PaddedPreprocessor()
    X, y = padded_preprocessor(batch)
    assert X.shape == (batch_size, 3)
    assert y.shape == (batch_size,)
