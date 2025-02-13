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
        if isinstance(logdir, str):  # type: ignore
            logdir = Path(logdir)  # type: ignore
        if not logdir.resolve().exists():  # type: ignore
            logdir.mkdir(parents=True)
            logger.info(f"Created logdir {logdir.absolute()}")
        return logdir

    # def to_toml_dict(self) -> dict:
    #     """Convert the model to a TOML-compatible dictionary."""
    #     data = self.model_dump(exclude_none=True)
    #
    #     # Handle special cases
    #     if "metrics" in data:
    #         data["metrics"] = [
    #             f"{metric.__module__}.{metric.__name__}"
    #             if isinstance(metric, Callable) and hasattr(metric, "__name__")
    #             else f"{metric.__class__.__module__}.{metric.__class__.__name__}"
    #             for metric in data["metrics"]
    #         ]
    #
    #     # Convert Path objects to strings
    #     for key, value in data.items():
    #         if isinstance(value, Path):
    #             data[key] = str(value)
    #
    #     return data
    #
    # def save_toml_example(self, file_path: str | Path) -> None:
    #     """Save an example TOML configuration file with comments."""
    #     if isinstance(file_path, str):
    #         file_path = Path(file_path)
    #
    #     data = self.to_toml_dict()
    #     lines = []
    #
    #     # Start with just the simple scalar values we know work
    #     lines.append("epochs = {}\n".format(data["epochs"]))
    #     metric_paths = [
    #         f"{metric.__module__}.{metric.__name__}" for metric in self.metrics
    #     ]
    #     lines.append("metrics = {}\n".format(str(metric_paths).replace("'", '"')))
    #     lines.append('logdir = "{}"\n'.format(str(data["logdir"])))
    #     lines.append("train_steps = {}\n".format(data["train_steps"]))
    #     lines.append("valid_steps = {}\n".format(data["valid_steps"]))
    #
    #     # Add reporttypes as list of strings (enum names)
    #     report_types = [rt.value for rt in data["reporttypes"]]
    #     lines.append("reporttypes = {}\n".format(str(report_types).replace("'", '"')))
    #
    #     if "optimizer_kwargs" in data:
    #         lines.append(
    #             self.dict_to_toml("optimizer_kwargs", data["optimizer_kwargs"])
    #         )
    #
    #     if "scheduler_kwargs" in data:
    #         lines.append(
    #             self.dict_to_toml("scheduler_kwargs", data["scheduler_kwargs"])
    #         )
    #
    #     if "earlystop_kwargs" in data:
    #         lines.append(
    #             self.dict_to_toml("earlystop_kwargs", data["earlystop_kwargs"])
    #         )
    #
    #     with open(file_path, "w", encoding="utf-8") as f:
    #         f.write("".join(lines))
    #
    # @staticmethod
    # def dict_to_toml(name: str, data: dict) -> str:
    #     """
    #     Convert a dictionary into a TOML table format.
    #
    #     Args:
    #         name (str): Name of the TOML section.
    #         data (dict): Dictionary to convert.
    #
    #     Returns:
    #         str: TOML formatted table string.
    #     """
    #     toml_lines = [f"[{name}]"]
    #     for k, v in data.items():
    #         # Convert booleans to lowercase ('true'/'false')
    #         if isinstance(v, bool):
    #             v = str(v).lower()
    #         toml_lines.append(f"{k} = {v}")
    #     return "\n".join(toml_lines) + "\n"
    #
    # @classmethod
    # def from_toml(cls, file_path: str | Path) -> "TrainerSettings":
    #     """Load the model from a TOML file."""
    #     if isinstance(file_path, str):
    #         file_path = Path(file_path)
    #
    #     with open(file_path, "rb") as f:
    #         data = tomllib.load(f)
    #
    #     print(data)
    #
    #     # Convert string paths to Path objects
    #     if "logdir" in data and isinstance(data["logdir"], str):
    #         data["logdir"] = Path(data["logdir"])
    #         # Convert reporttype strings back to enums
    #     if "reporttypes" in data:
    #         data["reporttypes"] = [ReportTypes(rt) for rt in data["reporttypes"]]
    #
    #     # Import metrics from their module paths
    #     if "metrics" in data:
    #         metrics = []
    #         for metric_path in data["metrics"]:
    #             try:
    #                 module_path, func_name = metric_path.rsplit(".", 1)
    #                 module = __import__(module_path, fromlist=[func_name])
    #                 metric = getattr(module, func_name)
    #                 metrics.append(metric)
    #             except (ImportError, AttributeError) as e:
    #                 logger.warning(
    #                     f"Could not load metric {metric_path}. "
    #                     f"Error: {e}. Using placeholder."
    #                 )
    #         data["metrics"] = metrics
    #
    #     return cls(**data)
