from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator


class FileTypes(Enum):
    JPG = ".jpg"
    PNG = ".png"
    TXT = ".txt"
    ZIP = ".zip"
    TGZ = ".tgz"
    TAR = ".tar.gz"
    GZ = ".gz"


class ReportTypes(Enum):
    TOML = "TOML"
    TENSORBOARD = "TENSORBOARD"
    MLFLOW = "MLFLOW"
    RAY = "RAY"


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("logdir")
    @classmethod
    def check_path(cls, logdir: Path) -> Path:  # noqa: N805
        if isinstance(logdir, str):
            logdir = Path(logdir)
        if not logdir.resolve().exists():
            # exist_ok: ray tune runs trials in parallel processes, and two of them
            # constructing settings with the same fresh logdir will race between the
            # check above and this call -- the loser died with FileExistsError.
            logdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created logdir {logdir.absolute()}")
        return logdir
