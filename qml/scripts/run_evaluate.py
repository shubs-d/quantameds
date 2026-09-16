import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantum_kc.config import Config
from quantum_kc.evaluation.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Model")
    parser.add_argument('--model-path', type=str, default='checkpoints/hybrid_classifier_best.pt', help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results (defaults to config.RESULTS_DIR)')
    args = parser.parse_args()

    config = Config()
    
    results = evaluate_model(config, model_path=args.model_path, output_dir=args.output_dir)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
