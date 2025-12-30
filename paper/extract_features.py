# Copyright (c) 2025 Meta Platforms, Inc. and affiliates. # noqa: INP001
"""Extract SpidR features from audio files."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from spidr.data import speech_dataset
from spidr.models import spidr_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract SpidR features.")
    parser.add_argument("manifest", type=Path, help="Path to the manifest file.")
    parser.add_argument("output", type=Path, help="Output directory for extracted features.")
    parser.add_argument("--layer", type=int, default=6, help="Layer number to extract features from (default: 6).")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = spidr_base().to(device)
    dataset = speech_dataset(args.manifest, normalize=True)
    paths = dataset.manifest["path"].to_list()
    with torch.inference_mode():
        for i, waveform in enumerate(tqdm(dataset)):
            outputs = model.get_intermediate_outputs(waveform.to(device).unsqueeze(0))
            output_file = args.output / Path(paths[i]).with_suffix(".pt").name
            torch.save(outputs[args.layer - 1].squeeze().cpu(), output_file)
