from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def disable_ultralytics_builtin_wandb() -> None:
    """Disable Ultralytics' implicit W&B callbacks for this process.

    The project uses explicit opt-in callbacks below so installing W&B cannot silently change a non-W&B run.
    """
    try:
        module = importlib.import_module("ultralytics.utils.callbacks.wb")
        callbacks = getattr(module, "callbacks", None)
        if isinstance(callbacks, dict):
            callbacks.clear()
    except Exception:
        pass


class OptionalWandbLogger:
    """Fail-open, lazy-import W&B logger used only by training/evaluation utilities."""

    def __init__(self, *, enabled: bool, entity: str, project: str, group: str,
                 run_name: str, config: dict[str, Any], job_type: str = "train") -> None:
        self.requested = enabled
        self.active = False
        self.status = "DISABLED"
        self.error: str | None = None
        self.run_id: str | None = None
        self._wandb: Any = None
        self._run: Any = None
        if not enabled:
            return
        try:
            self._wandb = importlib.import_module("wandb")
            self._run = self._wandb.init(
                entity=entity, project=project, group=group, name=run_name,
                job_type=job_type, config=config, reinit=True,
            )
            self.run_id = str(self._run.id)
            self.active = True
            self.status = "ACTIVE"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "FAILED_OPEN"
            self.active = False
            print(f"W&B optional logging disabled after initialization failure: {self.error}")

    def on_fit_epoch_end(self, trainer: Any) -> None:
        if not self.active:
            return
        try:
            payload: dict[str, float] = {}
            payload.update({str(k): float(v) for k, v in dict(getattr(trainer, "metrics", {})).items()})
            if getattr(trainer, "tloss", None) is not None:
                payload.update({str(k): float(v) for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()})
            payload.update({str(k): float(v) for k, v in dict(getattr(trainer, "lr", {})).items()})
            payload["epoch"] = int(trainer.epoch) + 1
            self._run.log(payload, step=int(trainer.epoch) + 1, commit=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "FAILED_OPEN"
            self.active = False
            print(f"W&B optional epoch logging disabled after failure: {self.error}")

    def log_artifact(self, *, name: str, artifact_type: str, files: list[Path]) -> None:
        if not self.active:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=artifact_type)
            added = 0
            for path in files:
                if path.is_file():
                    artifact.add_file(str(path.resolve()), name=path.name)
                    added += 1
            if added:
                self._run.log_artifact(artifact)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "FAILED_OPEN"
            print(f"W&B optional artifact logging failed without affecting training: {self.error}")

    def finish(self, *, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            self._run.finish(exit_code=exit_code)
            if self.status == "ACTIVE":
                self.status = "COMPLETE"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "FAILED_OPEN"
            print(f"W&B optional finish failed without affecting the experiment: {self.error}")
