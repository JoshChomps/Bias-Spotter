"""
Dataset Download Script — Universal Bias Corpus.

Automates the fetching of news, social media (Reddit), and political speech
datasets to create a "Universal" training corpus for the Bias Spotter.

Auto-Downloads:
- GUS-Net (Token-level bias)
- LIAR (Fact-checking)
- BABE (Media bias)
- GoEmotions (Reddit/Social Media) via HF Datasets API

Manual Downloads (Required):
- SemEval-2020 Task 11 (Anchor)
- SemEval-2023 Task 3 (Expansion)

Saves to: training/data/<dataset_name>/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests


# Dataset registry
DATASETS = {
    "semeval_2020": {
        "name": "SemEval-2020 Task 11",
        "url": "https://zenodo.org/records/3952415",
        "description": "14 propaganda techniques in news articles. Anchor dataset.",
        "manual_download": True,
        "instructions": (
            "1. Go to https://zenodo.org/records/3952415\n"
            "2. Download the dataset files\n"
            "3. Extract to training/data/semeval_2020/"
        ),
    },
    "semeval_2023": {
        "name": "SemEval-2023 Task 3",
        "url": "https://propaganda.math.unipd.it/semeval2023task3/",
        "description": "Multilingual persuasion technique detection. English subset.",
        "manual_download": True,
        "instructions": (
            "1. Go to https://propaganda.math.unipd.it/semeval2023task3/\n"
            "2. Register and download the English data\n"
            "3. Extract to training/data/semeval_2023/"
        ),
    },
    "gus_net": {
        "name": "GUS-Net (Bias Annotations)",
        "url": "https://github.com/Ethical-Spectacle/fairly",
        "description": "69k+ token-level bias annotations (religion, race, gender, politics).",
        "manual_download": False,
        "git_clone": "https://github.com/Ethical-Spectacle/fairly.git",
    },
    "liar": {
        "name": "LIAR Dataset",
        "url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        "description": "12,800 labeled political statements from PolitiFact.",
        "manual_download": False,
        "direct_download": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
    },
    "babe": {
        "name": "BABE (Expert Media Bias)",
        "url": "https://github.com/Media-Bias-Group/MBIC",
        "description": "3,700+ sentences annotated for media bias by experts.",
        "manual_download": False,
        "git_clone": "https://github.com/Media-Bias-Group/MBIC.git",
    },
    "goemotions": {
        "name": "GoEmotions (Social Media)",
        "url": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
        "description": "58k Reddit comments with 27 emotion labels (Signal for Loaded Language).",
        "manual_download": False,
        "hf_dataset": "google-research-datasets/go_emotions",
    },
}


def ensure_data_dir() -> Path:
    """Create the data directory if it doesn't exist."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_file(url: str, dest: Path) -> None:
    """Download a file with a progress indicator."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def git_clone(url: str, dest: Path) -> None:
    """Clone a git repository."""
    if dest.exists():
        print(f"Directory {dest} already exists. Skipping clone.")
        return
    print(f"Cloning {url} into {dest}...")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def hf_download(dataset_name: str, dest: Path) -> None:
    """Download a dataset via the HuggingFace Datasets API."""
    from datasets import load_dataset

    print(f"Fetching HF dataset {dataset_name}...")
    ds = load_dataset(dataset_name)
    ds.save_to_disk(str(dest))
    print(f"Saved to {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Bias Corpus Downloader")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all datasets and download instructions",
    )
    args = parser.parse_args()

    data_dir = ensure_data_dir()

    if args.list:
        print("=" * 70)
        for key, info in DATASETS.items():
            print(f"{key:15} | {info['name']}")
        return

    to_download = DATASETS.keys() if args.dataset == "all" else [args.dataset]

    for key in to_download:
        info = DATASETS[key]
        dest = data_dir / key

        print(f"\n📦 Processing: {info['name']}")

        if info.get("manual_download"):
            if not dest.exists():
                print(f"   ⚠️  MANUAL DOWNLOAD REQUIRED:")
                print(f"   {info['instructions']}")
            else:
                print(f"   ✅ Manual directory found at {dest}")
            continue

        try:
            if "git_clone" in info:
                git_clone(info["git_clone"], dest)
            elif "direct_download" in info:
                zip_path = data_dir / f"{key}.zip"
                download_file(info["direct_download"], zip_path)
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(dest)
                zip_path.unlink()
            elif "hf_dataset" in info:
                hf_download(info["hf_dataset"], dest)
        except Exception as e:
            print(f"   ❌ Error processing {key}: {e}")

    print("\n" + "=" * 70)
    print("Download process complete.")
    print(f"Data location: {data_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
