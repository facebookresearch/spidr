# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Checkpoint management for training runs."""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import GradScaler, nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from spidr.tools import MetricsQueue

logger = logging.getLogger()


def remove_param_group(optim: Optimizer, index: int) -> None:
    for param in optim.param_groups[index]["params"]:
        if param in optim.state:
            del optim.state[param]
    del optim.param_groups[index]


def find_checkpoints(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("step_*.pt"), key=lambda x: int(x.stem.removeprefix("step_")))
    if (folder / "final.pt").is_file():
        paths.append(folder / "final.pt")
    return paths


@dataclass
class Checkpoint:
    model: nn.Module
    optimizer: Optimizer
    scheduler: LRScheduler
    scaler: GradScaler
    epoch: torch.Tensor
    step: torch.Tensor


class Checkpointer:
    def __init__(
        self,
        folder: Path,
        interval: int,
        keep_latest: int = -1,
        *,
        disable: bool = False,
        start: int = 0,
    ) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.keep_latest = keep_latest
        self.disable = disable or interval <= 0
        self._path: Path | None = None
        self._state: Checkpoint
        self._metrics_queue = MetricsQueue(self.folder / "metrics.jsonl", interval, start=start)

    @property
    def step(self) -> torch.Tensor:
        return self._state.step

    @property
    def epoch(self) -> torch.Tensor:
        return self._state.epoch

    @property
    def last(self) -> Path | None:
        if self._path is not None:
            return self._path
        if len(checkpoints := find_checkpoints(self.folder)) > 0:
            self._path = checkpoints[-1]
            return self._path
        return None

    @property
    def metrics(self) -> Path:
        return self.folder / "metrics.jsonl"

    def init_state(self, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, scaler: GradScaler) -> None:
        epoch, step = torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)
        ckpt = Checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, epoch=epoch, step=step)
        self._state = ckpt

    def _purge_old_checkpoints(self) -> None:
        if self.keep_latest <= 0:
            return
        if len(checkpoints := find_checkpoints(self.folder)) <= self.keep_latest:
            return
        to_delete = checkpoints[: -self.keep_latest]
        for path in to_delete:
            logger.info("Deleting old checkpoint %s", path)
            path.unlink()

    def _save_to_path(self, path: Path, step: int, epoch: int) -> bool:
        if self.disable:
            return False
        self._state.step.fill_(step)
        self._state.epoch.fill_(epoch)
        torch.save(
            {
                "model": self._state.model.state_dict(),
                "optimizer": self._state.optimizer.state_dict(),
                "scheduler": self._state.scheduler.state_dict(),
                "scaler": self._state.scaler.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
            },
            path,
        )
        self._path = path
        self._purge_old_checkpoints()
        return True

    def save(self, step: int, epoch: int, *, force: bool = False) -> bool:
        should_save = (not self.disable) and (force or step % self.interval == 0)
        if not should_save:
            return False
        return self._save_to_path(self.folder / f"step_{step}.pt", step, epoch)

    def save_final(self, step: int, epoch: int) -> bool:
        return self._save_to_path(self.folder / "final.pt", step, epoch)

    def _load_from_path(self, path: Path) -> bool:
        if self.disable or not path.is_file():
            return False
        if path.name == "final.pt":
            logger.warning("Loading final checkpoint")
        ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        self._state.step = ckpt["step"]
        self._state.epoch = ckpt["epoch"]
        self._state.scheduler.load_state_dict(ckpt["scheduler"])
        self._state.scaler.load_state_dict(ckpt["scaler"])

        consume_prefix_in_state_dict_if_present(ckpt["model"], "module.")
        self._state.model.load_state_dict(ckpt["model"])
        if (
            hasattr(self._state.model, "current_step")
            and hasattr(self._state.model, "freeze_step")
            and self._state.model.current_step >= self._state.model.freeze_step
        ):
            self._state.model.freeze_extractor()
            if len(self._state.optimizer.param_groups) > 1:
                remove_param_group(self._state.optimizer, 1)
        if len(self._state.optimizer.param_groups) != len(ckpt["optimizer"]["param_groups"]):
            raise ValueError("Inconsistency across current and checkpoint's param_groups")
        self._state.optimizer.load_state_dict(ckpt["optimizer"])
        return True

    def load_existing_run(self) -> bool:
        if (path := self.last) is None:
            return False
        return self._load_from_path(path)

    def load(self, step: int) -> bool:
        return self._load_from_path(self.folder / f"step_{step}.pt")

    def find_new_metrics(self) -> Generator[dict[str, Any], None, None]:
        yield from self._metrics_queue()
