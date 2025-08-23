#!/usr/bin/env python3
"""
Example usage of the ISL Video DataLoader

This script demonstrates how to use the ISL video dataloader for training a model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from scripts.isl_dataloader import create_data_loaders
from tqdm import tqdm


class SimpleLSTMClassifier(nn.Module):
    """
    Simple LSTM classifier for ISL video classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(SimpleLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x, lengths):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: (batch_size,) actual sequence lengths
        """
        batch_size = x.size(0)
        
        # Pack padded sequences
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_packed)
        
        # Use the last hidden state from both directions
        # hidden: (num_layers * 2, batch_size, hidden_size)
        forward_hidden = hidden[-2]  # Last layer, forward direction
        backward_hidden = hidden[-1]  # Last layer, backward direction
        
        # Concatenate forward and backward hidden states
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classify
        output = self.classifier(final_hidden)
        return output


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, labels, lengths, metadata) in enumerate(pbar):
        data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (data, labels, lengths, metadata) in enumerate(pbar):
            data, labels, lengths = data.to(device), labels.to(device), lengths.to(device)
            
            outputs = model(data, lengths)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
    
    return running_loss / len(dataloader), correct / total


def main():
    """Main training function."""
    # Configuration
    config = {
        'root_dir': 'h5_output',
        'batch_size': 16,
        'num_workers': 4,
        'use_world_landmarks': True,
        'use_hand_landmarks': True,
        'normalize': True,
        'max_frames': 150,  # Limit sequence length
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataloaders
    print("Creating dataloaders...")
    dataloaders = create_data_loaders(
        root_dir=config['root_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_world_landmarks=config['use_world_landmarks'],
        use_hand_landmarks=config['use_hand_landmarks'],
        normalize=config['normalize'],
        max_frames=config['max_frames']
    )
    
    # Get dataset info
    train_dataset = dataloaders['train'].dataset
    num_classes = train_dataset.get_num_classes()
    
    # Calculate input size
    # Each hand has 21 landmarks with 3 coordinates = 21 * 3 = 63
    # 2 hands = 2 * 63 = 126
    # If using both hand and world landmarks: 126 * 2 = 252
    input_size = 126  # hand landmarks only
    if config['use_world_landmarks'] and config['use_hand_landmarks']:
        input_size = 252
    elif config['use_world_landmarks']:
        input_size = 126
    
    print(f"Input size: {input_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.get_class_names()[:5]}...")  # Show first 5
    print()
    
    # Create model
    model = SimpleLSTMClassifier(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=num_classes,
        dropout=config['dropout']
    )
    
    device = torch.device(config['device'])
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on device: {device}")
    print()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        # Validate
        if 'validation' in dataloaders:
            val_loss, val_acc = validate(
                model, dataloaders['validation'], criterion, device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
        else:
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        print()
    
    # Test on test set if available
    if 'test' in dataloaders:
        print("Testing on test set...")
        model.load_state_dict(torch.load('best_model.pth'))
        test_loss, test_acc = validate(
            model, dataloaders['test'], criterion, device
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
