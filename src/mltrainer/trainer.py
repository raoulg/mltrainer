from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Type, TypeVar

import mlflow
import ray
import torch
from loguru import logger
from tomlserializer import TOMLSerializer
from torch.optim import Optimizer
from tqdm import tqdm

from mltrainer import ReportTypes, TrainerSettings

OptimizerType = TypeVar("OptimizerType", bound=torch.optim.Optimizer)


def dir_add_timestamp(log_dir: Optional[Path] = None) -> Path:
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir / timestamp
    logger.info(f"Logging to {log_dir}")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    return log_dir


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        settings: TrainerSettings,
        loss_fn: Callable,
        optimizer: Type[Optimizer],
        traindataloader: Iterator,
        validdataloader: Iterator,
        scheduler: Optional[Callable],
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.log_dir = dir_add_timestamp(settings.logdir)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.device = device
        self.writer = None

        if self.device:
            self.model.to(self.device)

        self.optimizer = optimizer(  # type: ignore
            model.parameters(), **settings.optimizer_kwargs
        )
        self.last_epoch = 0

        if scheduler:
            if settings.scheduler_kwargs is None:
                raise ValueError("Missing 'scheduler_kwargs' in TrainerSettings.")
            self.scheduler = scheduler(self.optimizer, **settings.scheduler_kwargs)
        else:
            self.scheduler = None

        if settings.earlystop_kwargs is not None:
            logger.info(
                "Found earlystop_kwargs in settings."
                "Set to None if you dont want earlystopping."
            )
            self.early_stopping: Optional[EarlyStopping] = EarlyStopping(
                self.log_dir, **settings.earlystop_kwargs
            )
        else:
            self.early_stopping = None

        if ReportTypes.TENSORBOARD in self.settings.reporttypes:
            from torch.utils.tensorboard.writer import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir)

        if ReportTypes.TOML in self.settings.reporttypes:
            TOMLSerializer.save(model, self.log_dir / "model.toml")
            TOMLSerializer.save(settings, self.log_dir / "settings.toml")

    def __del__(self):
        """Cleanup method to ensure proper resource handling"""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()

    def loop(self) -> None:
        epoch = 0
        for epoch in tqdm(range(self.settings.epochs), colour="#1e4706"):
            train_loss = self.trainbatches()
            metric_dict, test_loss = self.evalbatches()
            self.report(epoch, train_loss, test_loss, metric_dict)

            if self.early_stopping:
                self.early_stopping(test_loss, self.model)  # type: ignore

            if self.early_stopping is not None and self.early_stopping.early_stop:
                logger.info("Interrupting loop due to early stopping patience.")
                self.last_epoch = epoch
                if self.early_stopping.save:
                    self.model = self.early_stopping.get_best()  # type: ignore
                else:
                    logger.info(
                        "early_stopping_save was false, using latest model."
                        "Set to true to retrieve best model."
                    )
                break
        self.last_epoch += epoch

    def trainbatches(self) -> float:
        self.model.train()
        train_loss: float = 0.0
        train_steps = self.settings.train_steps
        for _ in tqdm(range(train_steps), colour="#1e4706"):
            x, y = next(iter(self.traindataloader))
            if self.device:
                x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()  # type: ignore
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()  # type: ignore
            train_loss += loss.cpu().detach().numpy()
        train_loss /= train_steps

        return train_loss

    def evalbatches(self) -> Tuple[Dict[str, float], float]:
        """Evaluate model on validation data with proper device handling"""
        self.model.eval()
        valid_steps = self.settings.valid_steps
        test_loss: float = 0.0
        metric_dict: Dict[str, float] = {}

        with torch.no_grad():  # Prevent gradient computation during evaluation
            for _ in range(valid_steps):
                x, y = next(iter(self.validdataloader))
                if self.device:
                    x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                yhat = self.model(x)

                # Calculate loss (already handled by loss_fn)
                test_loss += float(self.loss_fn(yhat, y).cpu())

                # Calculate metrics (now handled by metric classes)
                for m in self.settings.metrics:
                    metric_dict[str(m)] = metric_dict.get(str(m), 0.0) + m(y, yhat)

        # Average the results
        test_loss /= valid_steps
        for key in metric_dict:
            metric_dict[str(key)] = metric_dict[str(key)] / valid_steps

        # Handle scheduler
        if self.scheduler:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(test_loss)
            else:
                self.scheduler.step()

        return metric_dict, test_loss

    def report(
        self, epoch: int, train_loss: float, test_loss: float, metric_dict: Dict
    ) -> None:
        if (self.last_epoch != 0) and (epoch == 0):
            self.last_epoch += 1
            logger.info(f"Resuming epochs from previous training at {self.last_epoch}")
        if self.last_epoch != 0:
            epoch += self.last_epoch
        reporttypes = self.settings.reporttypes
        self.test_loss = test_loss

        if ReportTypes.RAY in reporttypes:
            ray.train.report(  # type: ignore
                {
                    "iterations": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    **metric_dict,
                }
            )

        if ReportTypes.MLFLOW in reporttypes:
            mlflow.log_metric("Loss/train", train_loss, step=epoch)
            mlflow.log_metric("Loss/test", test_loss, step=epoch)
            for m in metric_dict:
                mlflow.log_metric(f"metric/{m}", metric_dict[m], step=epoch)
            lr = [group["lr"] for group in self.optimizer.param_groups][0]  # type: ignore
            mlflow.log_metric("learning_rate", lr, step=epoch)

        if ReportTypes.TENSORBOARD in reporttypes:
            assert self.writer is not None
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            for m in metric_dict:
                self.writer.add_scalar(f"metric/{m}", metric_dict[m], epoch)
            lr = [group["lr"] for group in self.optimizer.param_groups][0]  # type: ignore
            self.writer.add_scalar("learning_rate", lr, epoch)

        metric_scores = [f"{v:.4f}" for v in metric_dict.values()]
        logger.info(
            f"Epoch {epoch} train {train_loss:.4f} test {test_loss:.4f} metric {metric_scores}"  # noqa E501
        )


class EarlyStopping:
    def __init__(
        self,
        log_dir: Path,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        save: bool = False,
    ) -> None:
        """
        Args:
            log_dir (Path): location to save checkpoint to.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as
            an improvement. Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = Path(log_dir) / "checkpoint.pt"
        self.save = save

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        # first epoch best_loss is still None
        if self.best_loss is None:
            self.best_loss = val_loss  # type: ignore
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif val_loss + self.delta >= self.best_loss:  # type: ignore
            # we minimize loss. If current loss did not improve
            # the previous best (with a delta) it is considered not to improve.
            self.counter += 1
            logger.info(
                f"best loss: {self.best_loss:.4f}, current loss {val_loss:.4f}."
                f"Counter {self.counter}/{self.patience}."
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # if not the first run, and val_loss is smaller, we improved.
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: torch.nn.Module) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            logger.info(
                f"Validation loss ({self.best_loss:.4f} --> {val_loss:.4f})."
                f"Saving {self.path} ..."
            )
        torch.save(model, self.path)
        self.val_loss_min = val_loss

    def get_best(self) -> torch.nn.Module:
        if self.verbose:
            logger.info(f"retrieving best model from {self.path}")
        return torch.load(self.path, weights_only=False)
