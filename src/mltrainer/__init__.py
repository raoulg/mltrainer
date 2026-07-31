from importlib.metadata import PackageNotFoundError, version

from mltrainer.settings import ReportTypes, TrainerSettings
from mltrainer.trainer import Trainer

__all__ = ["Trainer", "TrainerSettings", "ReportTypes"]

try:
    __version__ = version("mltrainer")
except PackageNotFoundError:
    __version__ = "unknown"
