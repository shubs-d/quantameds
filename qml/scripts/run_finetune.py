import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import Config
from quantum_kc.training.finetune import finetune_vqc, run_cross_validation

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser(description="Stage 2: Supervised VQC Finetuning")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--fold', type=int, default=1, help='Fold to run (1-5)')
    parser.add_argument('--cross-val', action='store_true', help='Run cross validation')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches per epoch')
    parser.add_argument('--encoder-checkpoint', type=str, default='checkpoints/pretrained_encoder.pt', help='Path to encoder checkpoint')
    args = parser.parse_args()

    config = Config()
    config.FINETUNE_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR_FINETUNE = args.lr
    
    if args.smoke_test:
        config.FINETUNE_EPOCHS = 1
        
    if args.cross_val:
        results = run_cross_validation(config)
        print("Cross-validation completed.")
        print(results)
    else:
        results = finetune_vqc(config, fold=args.fold)
        print(f"Finetuning for fold {args.fold} completed.")
        print(results)

if __name__ == "__main__":
    main()
