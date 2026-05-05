"""
Dataset Download Script — Automated download of all training datasets.

Downloads:
1. SemEval-2020 Task 11 (anchor dataset)
2. GUS-Net (token-level bias annotations)
3. LIAR (political fact-checking)
4. Hyperpartisan News (political bias)

Saves to: training/data/<dataset_name>/

Usage:
    python download_datasets.py
    python download_datasets.py --dataset semeval  # download only one

STATUS: Stub — download URLs and structure defined. Will need manual
download for some datasets (SemEval requires Zenodo access).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
        "description": "69k+ token-level bias annotations across religion, race, gender, politics.",
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
    "hyperpartisan": {
        "name": "Hyperpartisan News",
        "url": "https://zenodo.org/records/1489920",
        "description": "Labels articles as hyperpartisan or not.",
        "manual_download": True,
        "instructions": (
            "1. Go to https://zenodo.org/records/1489920\n"
            "2. Download the dataset\n"
            "3. Extract to training/data/hyperpartisan/"
        ),
    },
    "babe": {
        "name": "BABE (Expert Media Bias)",
        "url": "https://github.com/Media-Bias-Group/MBIC",
        "description": "3,700+ sentences annotated for media bias by trained experts.",
        "manual_download": False,
        "git_clone": "https://github.com/Media-Bias-Group/MBIC.git",
    },
}


def ensure_data_dir() -> Path:
    """Create the data directory if it doesn't exist."""
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def print_download_instructions() -> None:
    """Print download instructions for all datasets."""
    print("=" * 70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    for key, info in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"📦 {info['name']}")
        print(f"   URL: {info['url']}")
        print(f"   Description: {info['description']}")

        if info.get("manual_download"):
            print(f"   ⚠️  MANUAL DOWNLOAD REQUIRED:")
            print(f"   {info['instructions']}")
        elif info.get("git_clone"):
            print(f"   ✅ Can be auto-downloaded via git clone")
        elif info.get("direct_download"):
            print(f"   ✅ Can be auto-downloaded")

    print(f"\n{'=' * 70}")
    print("Save all datasets to: training/data/<dataset_name>/")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download training datasets")
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

    if args.list:
        print_download_instructions()
        return

    data_dir = ensure_data_dir()
    print(f"Data directory: {data_dir}")

    # TODO: Implement actual downloads for auto-downloadable datasets
    # For now, print instructions
    print_download_instructions()


if __name__ == "__main__":
    main()
