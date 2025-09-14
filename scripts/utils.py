import torch
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def init_wandb(config):
    """Initialize Weights & Biases logging."""
    if not config.get('enable_wandb', True):
        print("WandB logging is disabled.")
        return None
    
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        entity=config['wandb_entity'],
        config=config
    )
    return wandb


def calculate_accuracy(output, target, topk=(1, 5)):
    """Calculate top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Create and return a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    plt.tight_layout()

    return fig, cm


def extract_xy_coordinates(data):
    """
    Extract only x,y coordinates from the landmark data.

    Input: (N, C, T, V, M) where:
        N = batch size
        C = 3 (x, y, z coordinates)
        T = number of frames
        V = 21 (number of landmarks)
        M = 2 (number of hands)
    
    Output: (N, 2, T, V, M) where 2 represents only x,y coordinates
    """
    # Input shape: (N, C, T, V, M) where C=3
    # We want to extract only the first 2 channels (x, y), dropping z
    
    if data.shape[1] != 3:
        raise ValueError(f"Expected 3 channels (x,y,z) but got {data.shape[1]}")
    
    # Extract only x,y coordinates (first 2 channels)
    data_xy = data[:, :2, :, :, :]  # (N, 2, T, V, M)
    
    return data_xy


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, filename='checkpoint.pth'):
    """Save model checkpoint."""
    import os
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config
    }

    filepath = os.path.join(config['checkpoint_dir'], filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def log_metrics_to_wandb(metrics_dict, step=None):
    """Log metrics to WandB."""
    if not hasattr(wandb, 'run') or wandb.run is None:
        return  # WandB is not initialized, skip logging
    
    if step is not None:
        wandb.log(metrics_dict, step=step)
    else:
        wandb.log(metrics_dict)