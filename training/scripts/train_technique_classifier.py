"""
Technique Classifier Training Script — Phase 2.

Fine-tunes DeBERTa-v3-base on SemEval-2020 Task 11 Technique Classification.

Goal: Working end-to-end training loop with F1 ~0.55-0.60 baseline.

Usage:
    python train_technique_classifier.py
    python train_technique_classifier.py --model deberta-v3-base --epochs 5
    python train_technique_classifier.py --model deberta-v3-large --focal-loss

STATUS: Stub — will be implemented in Phase 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the universal technique classification model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="answerdotai/ModernBERT-large",
        help="Pretrained model name or path",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--focal-loss", action="store_true", default=True, help="Use focal loss for class imbalance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "semeval_2020"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "models" / "technique_classifier"),
    )
    args = parser.parse_args()

    print("=" * 50)
    print("TECHNIQUE CLASSIFIER TRAINING — Phase 2")
    print("=" * 50)
    print(f"Model:         {args.model}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Focal loss:    {args.focal_loss}")
    print(f"Data dir:      {args.data_dir}")
    print(f"Output dir:    {args.output_dir}")
    print()

    # TODO (Phase 2): Implement the training loop
    # Steps:
    # 1. Load SemEval-2020 TC data into HuggingFace datasets format
    # 2. Tokenize with the model's tokenizer
    # 3. Handle class imbalance (focal loss or class weights)
    # 4. Train with AdamW optimizer
    # 5. Evaluate on test set using macro F1
    # 6. Save checkpoint + model card
    print("⚠️  Training not yet implemented. Complete Phase 1 data setup first.")
    print("   Run `python download_datasets.py` to get started.")


if __name__ == "__main__":
    main()
