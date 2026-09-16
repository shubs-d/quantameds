import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import Config
from quantum_kc.training.pretrain import pretrain_autoencoder

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain Autoencoder")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test')
    parser.add_argument('--max-batches', type=int, default=None, help='Max batches per epoch')
    args = parser.parse_args()

    config = Config()
    config.PRETRAIN_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LR_PRETRAIN = args.lr
    
    if args.smoke_test:
        config.PRETRAIN_EPOCHS = 1
        
    metrics = pretrain_autoencoder(config)
    print("Pretraining completed.")
    print(metrics)

if __name__ == "__main__":
    main()
