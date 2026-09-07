import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        
    def __call__(self, metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            return False
            
        if self.mode == 'min':
            if metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric > self.best_metric + self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                
        return self.counter >= self.patience

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                metrics['auc_roc'] = float(roc_auc_score(y_true, prob))
            else:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        except ValueError:
            metrics['auc_roc'] = None
    return metrics

def log_gradient_variance(model: nn.Module, tag: str = 'quantum') -> float:
    """Log variance of gradients for quantum parameters."""
    grads = []
    for name, param in model.named_parameters():
        if ('vqc' in name or 'quantum' in name) and param.grad is not None:
            grads.append(param.grad.view(-1))
            
    if not grads:
        return 0.0
        
    all_grads = torch.cat(grads)
    variance = float(torch.var(all_grads).item())
    
    if variance < 1e-6:
        logger.warning(f"[{tag}] Low gradient variance ({variance:.2e}). Possible barren plateau.")
        
    return variance

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32)
