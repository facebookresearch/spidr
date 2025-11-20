# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Data utilities."""

import mmap
import os
from pathlib import Path

import polars as pl
from torchcodec.decoders import AudioDecoder


def num_samples(source: str | Path | bytes, *, verify: bool = False) -> int:
    metadata = AudioDecoder(source).metadata
    samples = metadata.duration_seconds_from_header * metadata.sample_rate
    if verify and not samples.is_integer():
        raise ValueError(
            f"Number of samples {samples} is not an integer"
            + (f" in {source}" if isinstance(source, (str, Path)) else "")
        )
    return int(samples)


def bytes_from_archive(archive: Path | str, offset: int, file_size: int) -> bytes:
    with Path(archive).open("rb") as path, mmap.mmap(path.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_o:
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
