import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Import your modules
from isl_dataloader import create_data_loaders
from st_gcn import HandSTGCN
from utils import (
    init_wandb, calculate_accuracy, plot_confusion_matrix,
    extract_xy_coordinates, AverageMeter, save_checkpoint,
    log_metrics_to_wandb
)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, config):
    """Train for one epoch."""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')

    for batch_idx, (data, labels, lengths, metadata) in enumerate(pbar):
        # Move to device
        data = data.to(device)
        labels = labels.to(device)

        # Extract only x,y coordinates (ignore z)
        data_xy = extract_xy_coordinates(data)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data_xy)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # Update meters
        batch_size = data.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0].item(), batch_size)
        top5.update(acc5[0].item(), batch_size)

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'Acc@5': f'{top5.avg:.2f}%'
        })

        # Log to WandB
        if batch_idx % config['log_interval'] == 0:
            step = epoch * len(dataloader) + batch_idx
            log_metrics_to_wandb({
                'train/loss': losses.val,
                'train/acc1': top1.val,
                'train/acc5': top5.val,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }, step=step)

    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg
    }


def evaluate(model, dataloader, criterion, device, epoch, config, phase='val'):
    """Evaluate the model."""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Evaluating {phase}')

        for data, labels, lengths, metadata in pbar:
            # Move to device
            data = data.to(device)
            labels = labels.to(device)

            # Extract only x,y coordinates (ignore z)
            data_xy = extract_xy_coordinates(data)

            # Forward pass
            outputs = model(data_xy)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

            # Store predictions for confusion matrix
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update meters
            batch_size = data.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)
            top5.update(acc5[0].item(), batch_size)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%',
                'Acc@5': f'{top5.avg:.2f}%'
            })

    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


def main(config_path):
    """Main training function."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize WandB
    wandb_run = init_wandb(config)

    # Create data loaders
    print("Creating data loaders...")
    dataloaders = create_data_loaders(
        root_dir=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_world_landmarks=config['use_world_landmarks'],
        use_hand_landmarks=config['use_hand_landmarks'],
        normalize=config['normalize'],
        max_frames=config['max_frames'],
        shuffle_train=True
    )

    # Get number of classes from dataset
    train_dataset = dataloaders['train'].dataset
    num_classes = train_dataset.get_num_classes()
    class_names = train_dataset.get_class_names()

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")

    # Update config with dynamic values
    config['num_classes'] = num_classes

    # Initialize model
    print("Initializing model...")
    model = HandSTGCN(
        in_channels=2,  # x, y coordinates only
        num_class=num_classes,
        edge_importance_weighting=config['edge_importance_weighting'],
        dropout=config['dropout']
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['cosine_t_max'],
        eta_min=config['cosine_eta_min']
    )

    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 50}")

        # Train
        train_metrics = train_epoch(
            model, dataloaders['train'], criterion, optimizer,
            device, epoch, config
        )

        # Validate
        if epoch % config['val_interval'] == 0:
            val_metrics = evaluate(
                model, dataloaders.get('val', dataloaders.get('test')),
                criterion, device, epoch, config, phase='val'
            )

            # Log epoch metrics to WandB
            log_metrics_to_wandb({
                'epoch': epoch,
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_acc1': train_metrics['acc1'],
                'train/epoch_acc5': train_metrics['acc5'],
                'val/epoch_loss': val_metrics['loss'],
                'val/epoch_acc1': val_metrics['acc1'],
                'val/epoch_acc5': val_metrics['acc5'],
            })

            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc@1: {train_metrics['acc1']:.2f}%, "
                  f"Acc@5: {train_metrics['acc5']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc@1: {val_metrics['acc1']:.2f}%, "
                  f"Acc@5: {val_metrics['acc5']:.2f}%")

            # Save best model
            if val_metrics['acc1'] > best_val_acc:
                best_val_acc = val_metrics['acc1']
                if config['save_model']:
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, config,
                        val_metrics, filename='best_model.pth'
                    )
                print(f"New best validation accuracy: {best_val_acc:.2f}%")

        # Step scheduler
        scheduler.step()

    # Final evaluation on test set
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)

    if 'test' in dataloaders:
        test_metrics = evaluate(
            model, dataloaders['test'], criterion,
            device, config['epochs'], config, phase='test'
        )

        print(f"\nTest - Loss: {test_metrics['loss']:.4f}, "
              f"Acc@1: {test_metrics['acc1']:.2f}%, "
              f"Acc@5: {test_metrics['acc5']:.2f}%")

        # Create and log confusion matrix
        fig, cm = plot_confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions'],
            class_names,
            title='Test Set Confusion Matrix'
        )

        # Log to WandB
        if wandb_run is not None:
            wandb_run.log({
                'test/final_loss': test_metrics['loss'],
                'test/final_acc1': test_metrics['acc1'],
                'test/final_acc5': test_metrics['acc5'],
                'test/confusion_matrix': wandb_run.Image(fig)
            })

        # Save confusion matrix
        os.makedirs('results', exist_ok=True)
        fig.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Confusion matrix saved to results/confusion_matrix.png")

    # Save final model
    if config['save_model']:
        save_checkpoint(
            model, optimizer, scheduler, config['epochs'], config,
            test_metrics if 'test' in dataloaders else val_metrics,
            filename='final_model.pth'
        )

    print("\nTraining completed!")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ISL ST-GCN model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    main(args.config)