# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Write manifest files."""

import os
import tarfile
from pathlib import Path

import polars as pl
from tqdm import tqdm

from spidr.data.utils import bytes_from_archive, num_samples


class CompressedArchiveError(Exception):
    """Archive must be uncompressed to be read."""


class NoAudioFileError(Exception):
    """No audio file found in the archive."""

    def __init__(self, archive: Path | str, file_extension: str) -> None:
        super().__init__(f"No file found in {archive} with extension `{file_extension}`.")


def write_manifest(dataset: Path | str, output: Path | str, file_extension: str = ".wav") -> None:
    """Write a manifest file containing the file paths and their number of samples.

    Each line contains the absolute path of the file and its number of samples.
    """
    files_info = [
        {"path": str(name), "num_samples": num_samples(name)}
        for name in tqdm(list(Path(dataset).resolve().rglob(f"*{file_extension}")))
    ]
    (
        pl.from_dicts(files_info)
        .lazy()
        .with_columns(pl.col("path").str.strip_suffix(file_extension).str.split(os.sep).list.get(-1).alias("fileid"))
        .select("fileid", "path", "num_samples")
        .sink_csv(output)
    )


def write_manifest_fairseq(dataset: Path | str, output: Path | str, file_extension: str = ".wav") -> None:
    """Write a manifest file containing the file paths and their number of samples.

    First line is the root directory of the dataset.
    Each line contains the relative path of the file and its number of samples.
    """
    lines = [str(Path(dataset).resolve())]
    paths = list(Path(dataset).rglob(f"*{file_extension}"))
    lines += [f"{name.relative_to(dataset)}\t{num_samples(name)}" for name in tqdm(paths)]
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest_tar(dataset: Path | str, output: Path | str, file_extension: str = ".wav") -> None:
    if not tarfile.is_tarfile(dataset):
        raise ValueError(dataset)
    with tarfile.open(dataset, mode="r") as tar_file:
        if tar_file.mode != "r":
            raise CompressedArchiveError
        infolist = tar_file.getmembers()
    bytes_info = [
        {
            "path": info.name,
            "num_samples": num_samples(bytes_from_archive(dataset, info.offset_data, info.size)),
            "byte_offset": info.offset_data,
            "byte_size": info.size,
        }
        for info in tqdm(infolist)
        if not info.isdir() and info.name.endswith(file_extension)
    ]
    if not bytes_info:
        raise NoAudioFileError(dataset, file_extension)
    (
        pl.from_dicts(bytes_info)
        .lazy()
        .with_columns(
            pl.col("path").str.strip_suffix(file_extension).str.split(os.sep).list.get(-1).alias("fileid"),
            pl.lit(dataset).alias("archive"),
        )
        .select("fileid", "path", "num_samples", "archive", "byte_offset", "byte_size")
        .sink_csv(output)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write manifest files.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset directory or uncompressed tar file.")
    parser.add_argument("output", type=Path, help="Path to the output manifest file.")
    parser.add_argument("--ext", type=str, default=".wav", help="Extension of audio files. (default: %(default)s)")
    parser.add_argument("--fairseq", action="store_true", help="Write a Fairseq-style TSV manifest, instead of CSV.")
    args = parser.parse_args()

    root = args.dataset.resolve()
    if root.is_dir():
        (write_manifest_fairseq if args.fairseq else write_manifest)(root, args.output, args.ext)
    elif root.suffix == ".tar":
        write_manifest_tar(root, args.ext)
    else:
        raise parser.error(f"{root} is not a directory or a tar file.")
