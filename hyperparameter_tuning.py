"""
Hyperparameter tuning script for CAPTCHA model.
Tests different combinations of learning rates, epochs, and convolutional layers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime

from data_loader import create_dataloaders, CaptchaDataset
from models import BaselineCNN
from evaluate import calculate_accuracy


class FlexibleBaselineCNN(nn.Module):
    """
    Flexible baseline CNN with configurable number of convolutional layers.
    """
    
    def __init__(self, num_chars=36, seq_length=5, num_conv_layers=4):
        """
        Args:
            num_chars: Number of possible characters (36 for alphanumeric)
            seq_length: Length of CAPTCHA sequence (5)
            num_conv_layers: Number of convolutional blocks (2-6)
        """
        super(FlexibleBaselineCNN, self).__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        self.num_conv_layers = num_conv_layers
        
        # Build convolutional layers dynamically
        conv_layers = []
        bn_layers = []
        pool_layers = []
        
        channels = [1, 64, 128, 256, 512, 512, 512]  # Channel progression
        
        for i in range(num_conv_layers):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            bn_layers.append(nn.BatchNorm2d(out_channels))
            pool_layers.append(nn.MaxPool2d(2, 2))
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.pool_layers = nn.ModuleList(pool_layers)
        
        # Calculate flattened size after all pooling
        # (50, 200) -> after num_conv_layers pooling: (50/2^num_conv_layers, 200/2^num_conv_layers)
        height = 50 // (2 ** num_conv_layers)
        width = 200 // (2 ** num_conv_layers)
        final_channels = channels[num_conv_layers]
        self.flatten_size = final_channels * height * width
        
        # Fully connected layers
        if num_conv_layers <= 3:
            self.fc1 = nn.Linear(self.flatten_size, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
        else:
            self.fc1 = nn.Linear(self.flatten_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = None
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.3) if num_conv_layers <= 3 else None
        
        # 5 separate output heads
        fc_out_size = 256 if num_conv_layers <= 3 else 256
        self.heads = nn.ModuleList([
            nn.Linear(fc_out_size, num_chars) for _ in range(seq_length)
        ])
    
    def forward(self, x):
        # Convolutional layers
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = torch.nn.functional.relu(bn(conv(x)))
            x = pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        
        if self.fc3 is not None:
            x = torch.nn.functional.relu(self.fc3(x))
            x = self.dropout3(x)
        
        # 5 separate predictions
        outputs = [head(x) for head in self.heads]
        
        return torch.stack(outputs, dim=1)  # Shape: (batch, 5, num_chars)


def train_and_evaluate(lr, epochs, num_conv_layers, batch_size=32, data_dir='samples'):
    """Train and evaluate a model with given hyperparameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir,
        train_split=0.8,
        batch_size=batch_size,
        augment_train=True
    )
    
    # Create model
    num_chars = CaptchaDataset.NUM_CHARS
    model = FlexibleBaselineCNN(num_chars=num_chars, seq_length=5, num_conv_layers=num_conv_layers)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_seq_acc = 0.0
    best_char_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = 0
            for i in range(5):
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss = loss / 5
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                loss = 0
                for i in range(5):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                loss = loss / 5
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=2)
                for i in range(predictions.size(0)):
                    all_predictions.append(predictions[i].cpu().numpy())
                    all_labels.append(labels[i].cpu().numpy())
        
        # Calculate accuracies
        char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
        
        scheduler.step(val_loss / len(val_loader))
        
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            best_char_acc = char_acc
    
    return best_char_acc, best_seq_acc


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for CAPTCHA model')
    parser.add_argument('--data_dir', type=str, default='samples',
                        help='Directory containing CAPTCHA images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--results_file', type=str, default='hyperparameter_results.json',
                        help='File to save results')
    
    args = parser.parse_args()
    
    # Define hyperparameter combinations to test
    learning_rates = [0.001, 0.0005, 0.0003, 0.0001]
    epochs_list = [50, 75, 100]
    num_conv_layers_list = [3, 4, 5]
    
    results = []
    total_experiments = len(learning_rates) * len(epochs_list) * len(num_conv_layers_list)
    experiment_num = 0
    
    print(f"Starting hyperparameter tuning...")
    print(f"Total experiments: {total_experiments}")
    print("=" * 80)
    
    for lr in learning_rates:
        for epochs in epochs_list:
            for num_conv_layers in num_conv_layers_list:
                experiment_num += 1
                print(f"\nExperiment {experiment_num}/{total_experiments}")
                print(f"LR: {lr}, Epochs: {epochs}, Conv Layers: {num_conv_layers}")
                print("-" * 80)
                
                try:
                    char_acc, seq_acc = train_and_evaluate(
                        lr=lr,
                        epochs=epochs,
                        num_conv_layers=num_conv_layers,
                        batch_size=args.batch_size,
                        data_dir=args.data_dir
                    )
                    
                    result = {
                        'lr': lr,
                        'epochs': epochs,
                        'num_conv_layers': num_conv_layers,
                        'char_accuracy': float(char_acc),
                        'seq_accuracy': float(seq_acc),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    print(f"✓ Completed: Char Acc: {char_acc:.2%}, Seq Acc: {seq_acc:.2%}")
                    
                    # Save results after each experiment
                    with open(args.results_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Sort results by sequence accuracy
    results.sort(key=lambda x: x['seq_accuracy'], reverse=True)
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. LR: {result['lr']}, Epochs: {result['epochs']}, Conv Layers: {result['num_conv_layers']}")
        print(f"   Character Accuracy: {result['char_accuracy']:.2%}")
        print(f"   Sequence Accuracy: {result['seq_accuracy']:.2%}")
    
    # Save final results
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ All results saved to {args.results_file}")


if __name__ == '__main__':
    main()

