# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""PyTorch profiler."""

import contextlib
import importlib.resources
import string
import tempfile
from collections.abc import Generator
from functools import partial
from pathlib import Path

from torch.profiler import ProfilerAction, profile


def html_trace_handler(p: profile, path: str | Path) -> None:
    """Adapted from https://github.com/facebookresearch/lingua/blob/main/lingua/profiling.py."""
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json") as fp:
        p.export_chrome_trace(fp.name)
        content = Path(fp.name).read_text(encoding="utf-8")
    viztracer = importlib.resources.files("viztracer")
    template = (viztracer / "html/trace_viewer_embedder.html").read_text(encoding="utf-8")
    sub = {"trace_viewer_full": (viztracer / "html/trace_viewer_full.html").read_text(encoding="utf-8")}
    sub["json_data"] = content.replace("</script>", "<\\/script>")
    Path(path).write_text(string.Template(template).substitute(sub), encoding="utf-8")


def scheduler_fn(step: int, skip_first: int, warmup: int, active: int) -> ProfilerAction:
    if step < skip_first or step >= skip_first + warmup + active:
        return ProfilerAction.NONE
    if step < skip_first + warmup:
        return ProfilerAction.WARMUP
    if step < skip_first + warmup + active - 1:
        return ProfilerAction.RECORD
    return ProfilerAction.RECORD_AND_SAVE


class NullProfiler:
    """A no-op profiler that does nothing."""

    def step(self) -> None:  # noqa: PLR6301
        return


@contextlib.contextmanager
def profiler_context(
    path: str | Path | None,
    skip_first: int = 1_000,
    warmup: int = 5,
    active: int = 2,
) -> Generator[profile | NullProfiler]:
    if path is not None:
        with profile(
            schedule=partial(scheduler_fn, skip_first=skip_first, warmup=warmup, active=active),
            on_trace_ready=partial(html_trace_handler, path=path),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        ) as profiler:
            yield profiler
    else:
        yield NullProfiler()
