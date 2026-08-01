"""Regression tests for reporting to Ray Tune.

`RayReporter` used to call `ray.train.report`. `ray.train` is the distributed
*training* library; `ray.tune` is the hypertuner. Since ray 2.43 (Train V2)
calling the train one from inside a tune trainable raises a DeprecationWarning
instead of reporting, so `ReportTypes.RAY` was broken for every ray version this
package supports (the floor is 2.45).

It failed invisibly: the raise happens inside the trial process, which then
segfaults during teardown with torch loaded, so ray reports
"Worker exit type: SYSTEM_ERROR ... the worker crashed unexpectedly" and the
actual message never reaches the driver. Hence the marked integration test —
the unit test alone would not have caught the *reason* this went unnoticed.
"""

import pytest

ray = pytest.importorskip("ray", reason="ray is an optional dependency (mltrainer[tune])")

from mltrainer.trainer import RayReporter  # noqa: E402


def test_reporter_uses_the_tune_api_not_the_train_api(monkeypatch):
    """`ray.train.report` raises inside a tune session; `ray.tune.report` is correct."""
    from ray import tune

    calls = []
    monkeypatch.setattr(tune, "report", lambda metrics, **kw: calls.append(metrics))

    def explode(*args, **kwargs):  # pragma: no cover - fails the test if reached
        raise AssertionError("RayReporter must not call ray.train.report")

    monkeypatch.setattr(ray.train, "report", explode)

    RayReporter().report(
        epoch=2, train_loss=0.5, test_loss=0.4, metric_dict={"Accuracy": 0.9}
    )

    assert calls == [
        {
            "iterations": 2,
            "train_loss": 0.5,
            "test_loss": 0.4,
            "Accuracy": 0.9,
        }
    ]


@pytest.mark.ray
def test_trial_using_reporttypes_ray_completes(tmp_path):
    """End to end: a real tune trial reporting through the Trainer must not die.

    This is the test that would have caught the breakage. It starts a real (tiny,
    single-cpu) ray cluster, so it is marked and excluded from the default run.
    """
    import torch
    from ray import tune
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    logdir = tmp_path / "logs"
    logdir.mkdir()

    def trial(config):
        from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics

        torch.set_num_threads(1)
        x, y = torch.randn(64, 10), torch.randint(0, 3, (64,))
        loader = DataLoader(TensorDataset(x, y), batch_size=32)
        settings = TrainerSettings(
            epochs=2,
            metrics=[metrics.Accuracy()],
            logdir=config["logdir"],
            train_steps=len(loader),
            valid_steps=len(loader),
            reporttypes=[ReportTypes.RAY],
            earlystop_kwargs=None,
        )
        Trainer(
            model=nn.Linear(10, 3),
            settings=settings,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=optim.Adam,  # type: ignore
            traindataloader=loader,
            validdataloader=loader,
            scheduler=None,
            device="cpu",
        ).loop()

    ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)
    try:
        results = tune.Tuner(
            tune.with_parameters(trial),
            param_space={"logdir": str(logdir)},
            tune_config=tune.TuneConfig(metric="test_loss", mode="min", num_samples=1),
            run_config=tune.RunConfig(storage_path=str(tmp_path / "ray"), verbose=0),
        ).fit()
        result = results[0]
        assert result.error is None, f"trial failed: {result.error}"
        # the Trainer reported through RayReporter, so its keys are in the result
        assert "test_loss" in result.metrics
        assert "Accuracy" in result.metrics
    finally:
        ray.shutdown()
