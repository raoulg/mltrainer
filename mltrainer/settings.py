from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, root_validator


class FileTypes(Enum):
    JPG = ".jpg"
    PNG = ".png"
    TXT = ".txt"
    ZIP = ".zip"
    TGZ = ".tgz"
    TAR = ".tar.gz"
    GZ = ".gz"


class ReportTypes(Enum):
    GIN = 1
    TENSORBOARD = 2
    MLFLOW = 3
    RAY = 4


class FormattedBase(BaseModel):
    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


class TrainerSettings(FormattedBase):
    epochs: int
    metrics: List[Callable]
    logdir: Path
    train_steps: int
    valid_steps: int
    reporttypes: List[ReportTypes]
    optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-5}
    scheduler_kwargs: Optional[Dict[str, Any]] = {"factor": 0.1, "patience": 10}
    earlystop_kwargs: Optional[Dict[str, Any]] = {
        "save": False,
        "verbose": True,
        "patience": 10,
    }

    class Config:
        arbitrary_types_allowed = True

    @root_validator
    def check_path(cls, values: Dict) -> Dict:  # noqa: N805
        datadir = values.get("logdir").resolve()
        if not datadir.exists():  # type: ignore
            raise FileNotFoundError(
                f"Make sure the datadir exists.\n Found {datadir} to be non-existing."
            )
        return values
