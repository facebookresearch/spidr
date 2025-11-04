# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Keep track of metrics."""

import torch


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self) -> None:
        self.avg = torch.zeros((), device=self.device, dtype=self.dtype)
        self.sum = torch.zeros((), device=self.device, dtype=self.dtype)
        self.count = torch.zeros((), device=self.device, dtype=torch.long)

    def update(self, val: torch.Tensor, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def pop(self) -> float:
        avg = self.avg
        self.reset()
        return avg.item()


class AverageMeters:
    """Record average meters for multiple values."""

    def __init__(self, names: list[str], device: torch.device, *, dtype: torch.dtype = torch.float32) -> None:
        self.names = names
        self.meters = {name: AverageMeter(device, dtype) for name in names}

    def __getitem__(self, name: str) -> AverageMeter:
        return self.meters[name]

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def update(self, **kwargs: torch.Tensor) -> None:
        for name, value in kwargs.items():
            self.meters[name].update(value)

    def pop(self) -> dict[str, float]:
        return {name: meter.pop() for name, meter in self.meters.items()}
