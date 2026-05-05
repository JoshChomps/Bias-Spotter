"""
Span Identifier Training Script — Phase 3.

Fine-tunes DeBERTa-v3 with a token classification head for span identification
using BIO tags. Identifies WHERE in the text persuasion techniques are used.

The span identifier works in sequence with the technique classifier:
1. Span model finds WHERE (BIO tagging)
2. Technique model classifies WHAT (14 categories)

Usage:
    python train_span_identifier.py
    python train_span_identifier.py --model deberta-v3-large --context-window 512

STATUS: Stub — will be implemented in Phase 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the span identification model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Pretrained model name or path",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--context-window", type=int, default=512, help="Token context window size")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "semeval_2020"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "models" / "span_identifier"),
    )
    args = parser.parse_args()

    print("=" * 50)
    print("SPAN IDENTIFIER TRAINING — Phase 3")
    print("=" * 50)
    print(f"Model:          {args.model}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Context window: {args.context_window}")
    print(f"Data dir:       {args.data_dir}")
    print(f"Output dir:     {args.output_dir}")
    print()

    # TODO (Phase 3): Implement the training loop
    # Steps:
    # 1. Load SemEval-2020 SI data
    # 2. Convert to BIO tagging format at token level
    # 3. Tokenize with sliding window for long documents
    # 4. Train token classification head
    # 5. Evaluate using span-level F1
    # 6. Save checkpoint + model card
    print("⚠️  Training not yet implemented. Complete Phase 2 first.")


if __name__ == "__main__":
    main()
