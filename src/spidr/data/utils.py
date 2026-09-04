# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Data utilities."""

import mmap
import os
import resource
from functools import lru_cache
from pathlib import Path

import polars as pl
from torchcodec.decoders import AudioDecoder


def num_samples(source: str | Path | bytes, *, verify: bool = False) -> int:
    metadata = AudioDecoder(source).metadata
    duration, sample_rate = metadata.duration_seconds_from_header, metadata.sample_rate
    if duration is None or sample_rate is None:
        exception = "Could not determine duration or sample rate from header"
        raise ValueError(exception + (f" for {source}" if isinstance(source, str | Path) else ""))
    samples = duration * sample_rate
    if verify and not samples.is_integer():
        raise ValueError(
            f"Number of samples {samples} is not an integer"
            + (f" in {source}" if isinstance(source, (str, Path)) else "")
        )
    return int(samples)


def _archive_cache_maxsize() -> int:
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return max(64, soft_limit // 2)


@lru_cache(maxsize=_archive_cache_maxsize())
def _archive_mmap(archive: str) -> mmap.mmap:
    # mmap dups the file descriptor internally, so the file handle doesn't need to stay open.
    # Cached for the lifetime of the (worker) process: reused across every sample from this archive.
    with Path(archive).open("rb") as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


def bytes_from_archive(archive: Path | str, offset: int, file_size: int) -> bytes:
    mmap_o = _archive_mmap(str(archive))
    return mmap_o[offset : offset + file_size]


def read_manifest(path: Path | str) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix == ".jsonl":
        return pl.read_ndjson(path)
    if path.suffix != ".tsv":
        raise ValueError("Only .csv, .jsonl and .tsv files are supported")
    with path.open("r") as file:
        root = Path(file.readline().strip())
    if not root.is_dir():
        raise ValueError("First line must be the root directory of the dataset")
    return (
        pl.scan_csv(path, separator="\t", skip_rows=1, has_header=False, new_columns=["fileid", "num_samples"])
        .with_columns((f"{root}{os.sep}" + pl.col("fileid")).alias("path"))
        .select("fileid", "path", "num_samples")
        .collect()
    )
