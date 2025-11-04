# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Watch for metrics added to a JSONL file and yield them in order.

This whole snippet is a workaround for wandb's limitations.
We want to train the model in a distributed setting, and launch validation
jobs in separate processes. The outcome of these validation jobs should be
available in the same wandb run as the training job. However, wandb does not
support multiple processes opened that can write to the same run.
It also only supports having a monotonic increasing step number.
Therefore, we write the metrics from the validation jobs in a JSONL file like this:
```
{"step": 100, "group": "dev-clean", "metric1": 0.1, "metric2": 0.2}
{"step": 100, "group": "dev-other", "metric3": 0.2, "metric4": 0.3}
{"step": 200, "group": "dev-clean", "metric1": 0.3, "metric2": 0.4}
```
The `MetricsQueue` class reads this file, and yields the metrics in order of
increasing step. It also waits for the next step to be available before
yielding the metrics for the following steps: this is useful if a job for
step N+1 has ended before the job for step N.
This approach has limitations:
- If some lines are removed or modified in the JSONL file, the `MetricsQueue`
  will raise a ValueError.
- If a job fails, the metrics for the following steps will be available in the
  JSONL but will no be logged.
  You can run `python -m spidr.tools.metrics_queue /path/to/jsonl --runid <runid> --entity <entity>`
  to push the unlogged metrics to wandb when training has ended.
"""

import difflib
import json
from collections import defaultdict
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any


def pull_inserted_lines(a: Sequence[str], b: Sequence[str]) -> Generator[str, None, None]:
    """Adapted from difflib.Differ.compare.

    a and b are lines from two files. a is the old version of the file, b is the new one.
    This function will yield the lines that were inserted in b compared to a.
    Any line that is either removed or modified will raise a ValueError.
    """
    cruncher = difflib.SequenceMatcher(None, a, b)
    for tag, _, _, blo, bhi in cruncher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            for i in range(blo, bhi):
                yield b[i]
        else:
            raise ValueError(
                f"Invalid tag {tag}, only 'equal' and 'insert' are supported."
                f"This mean that some lines were either removed or modified."
            )


class MetricsQueue:
    """Wait and yield metrics from a JSONL file."""

    def __init__(self, path: Path, interval: int, start: int) -> None:
        self.interval = interval
        self.path = Path(path)
        self._queues: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._next_step: defaultdict[str, int] = defaultdict(lambda: start)
        self._previous: list[str] = []
        if self.path.is_file():
            self.reset()

    def reset(self) -> None:
        """Reset the expected next step."""
        with self.path.open() as f:
            self._previous = f.read().splitlines()

        for line in self.path.read_text().splitlines():
            metric = json.loads(line)
            step = metric.pop("step")
            group = metric.pop("group")
            self._next_step[group] = max(self._next_step[group], step + self.interval)

    def pull(self) -> Generator[str, None, None]:
        """Pull the new lines from the JSONL file and yield them."""
        if not self.path.is_file():
            return
        with self.path.open() as f:
            lines = f.read().splitlines()
        yield from pull_inserted_lines(self._previous, lines)
        self._previous = lines

    def populate_queues(self) -> None:
        """Add the new metrics to the queues for each group of metrics."""
        for line in self.pull():
            metric: dict[str, Any] = json.loads(line)
            step: int = metric.pop("step")
            group: str = metric.pop("group")
            if group not in self._queues:
                self._queues[group] = []
            self._queues[group].append((step, metric))

    def __call__(self) -> Generator[dict[str, Any], None, None]:
        """Yield the metrics to log in order of increasing step for each group."""
        self.populate_queues()
        new_queues = {}
        for group, queue in self._queues.items():
            idx = 0
            sorted_queue = sorted(queue, key=lambda x: x[0])
            for step, metric in sorted_queue:
                if step < self._next_step[group]:
                    raise ValueError(
                        f"This should not be happening: a new entry from an existing group ({group}) has been added, "
                        f"and its step ({step}) is lower than the next expected step ({self._next_step[group]}). "
                        f"The wrong entry is '{metric}'."
                    )
                if step == self._next_step[group]:
                    self._next_step[group] += self.interval
                    idx += 1
                    yield {f"{group}/{name}": value for name, value in metric.items()} | {f"{group}/step": step}
                else:
                    break
            new_queues[group] = sorted_queue[idx:]
        self._queues = new_queues


def push_later_to_wandb(path: Path, runid: str | None, entity: str) -> None:
    """Push unlogged metrics from a finished run to wandb."""
    import polars as pl  # noqa: PLC0415
    import wandb  # noqa: PLC0415

    metrics = pl.read_ndjson(path)
    api = wandb.Api()
    with (path.parent / "config.json").open() as f:
        config = json.load(f)
    project, name = config["run"]["wandb_project"], config["run"]["wandb_name"]

    if runid is None:
        candidates = [run for run in api.runs(f"{entity}/{project}") if run.name == name and run.state != "running"]
        if len(candidates) > 1:
            raise ValueError(f"Found {len(candidates)} candidate runs. Please specify the run ID.")
        if not candidates:
            raise ValueError("No candidate run found: wrong name or still running.")
        runid = candidates[0].id
    history = api.run(f"{entity}/{project}/{runid}").history()
    max_steps = {group: history[f"{group}/step"].max().astype(int) for group in metrics["group"].unique()}
    metrics = metrics.filter(pl.col("step") > pl.col("group").replace_strict(max_steps))
    if len(metrics) == 0:
        return
    wandb.init(id=runid, project=project, entity=entity, resume="must", dir=Path(config["run"]["workdir"]).resolve())
    for _, row in metrics.sort(by=["step", "group"]).iter_rows():
        wandb.log({f"{row['group']}/{name}": value for name, value in row.items() if name != "group"})
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Push metrics that were not logged to wandb.")
    parser.add_argument("path", type=Path, help="Path to the JSONL file.")
    parser.add_argument("--entity", type=str, required=True, help="Wandb account.")
    parser.add_argument("--runid", type=str, help="Run ID in wandb.")
    args = parser.parse_args()
    push_later_to_wandb(args.path, args.runid, args.entity)
