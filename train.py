"""
Training script for CAPTCHA recognition models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import time

from data_loader import create_dataloaders, CaptchaDataset
from models import BaselineCNN, CRNN
from evaluate import calculate_accuracy


def train_epoch(model, train_loader, criterion, optimizer, device, model_type='baseline'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, labels, label_strs in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'baseline':
            # Baseline CNN: 5 separate outputs
            outputs = model(images)  # (batch, 5, num_chars)
            
            # Calculate loss for each character position
            loss = 0
            for i in range(5):
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss = loss / 5
            
        else:  # CRNN
            # CRNN: CTC loss
            outputs = model(images)  # (batch, seq_len, num_chars + 1)
            outputs = outputs.log_softmax(2)  # Apply log_softmax for CTC
            
            # Prepare for CTC: (seq_len, batch, num_chars + 1)
            outputs = outputs.permute(1, 0, 2)
            
            # Create input lengths (all same for our case)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            
            # Create target lengths (all 5 for CAPTCHA)
            target_lengths = torch.full((images.size(0),), 5, dtype=torch.long)
            
            # CTC loss
            loss = criterion(outputs, labels, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device, model_type='baseline'):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, label_strs in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            if model_type == 'baseline':
                outputs = model(images)
                
                # Calculate loss
                loss = 0
                for i in range(5):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                loss = loss / 5
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=2)  # (batch, 5)
                
            else:  # CRNN
                outputs = model(images)
                outputs = outputs.log_softmax(2)
                outputs = outputs.permute(1, 0, 2)
                
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
                target_lengths = torch.full((images.size(0),), 5, dtype=torch.long)
                
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                
                # Decode CTC predictions
                predictions = decode_ctc(outputs, input_lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and labels
            for i in range(predictions.size(0)):
                all_predictions.append(predictions[i].cpu().numpy())
                all_labels.append(labels[i].cpu().numpy())
    
    # Calculate accuracies
    char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
    
    return total_loss / num_batches, char_acc, seq_acc


def decode_ctc(ctc_output, input_lengths):
    """
    Decode CTC output using greedy decoding.
    Args:
        ctc_output: (seq_len, batch, num_chars + 1) log probabilities
        input_lengths: Lengths of input sequences
    Returns:
        predictions: (batch, 5) tensor of predicted character indices
    """
    batch_size = ctc_output.size(1)
    predictions = []
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        probs = ctc_output[:seq_len, b, :].argmax(dim=1)  # Greedy decoding
        
        # Remove blank tokens (last index) and duplicates
        decoded = []
        prev = -1
        for idx in probs:
            if idx.item() < ctc_output.size(2) - 1:  # Not blank token
                if idx.item() != prev:  # Remove duplicates
                    decoded.append(idx.item())
                prev = idx.item()
        
        # Pad or truncate to length 5
        if len(decoded) < 5:
            decoded.extend([0] * (5 - len(decoded)))
        else:
            decoded = decoded[:5]
        
        predictions.append(torch.tensor(decoded, dtype=torch.long))
    
    return torch.stack(predictions)


def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA recognition model')
    parser.add_argument('--data_dir', type=str, default='samples',
                        help='Directory containing CAPTCHA images')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'crnn'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        train_split=0.8,
        batch_size=args.batch_size,
        augment_train=args.augment
    )
    
    # Create model
    num_chars = CaptchaDataset.NUM_CHARS
    if args.model_type == 'baseline':
        model = BaselineCNN(num_chars=num_chars, seq_length=5)
        criterion = nn.CrossEntropyLoss()
    else:  # CRNN
        model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
        criterion = nn.CTCLoss(blank=num_chars, reduction='mean', zero_infinity=True)
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_seq_acc = 0.0
    print(f'\nStarting training ({args.model_type} model)...')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model_type)
        
        # Validate
        val_loss, char_acc, seq_acc = validate(model, val_loader, criterion, device, args.model_type)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Character', char_acc, epoch)
        writer.add_scalar('Accuracy/Sequence', seq_acc, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Character Accuracy: {char_acc:.2%}, Sequence Accuracy: {seq_acc:.2%}')
        
        # Save best model (by sequence accuracy, or character accuracy if seq_acc is 0)
        should_save = False
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            should_save = True
        elif seq_acc == 0 and best_seq_acc == 0 and epoch == 0:
            # Save first model if no sequence accuracy yet
            should_save = True
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'char_acc': char_acc,
                'seq_acc': seq_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'best_{args.model_type}.pth'))
            if seq_acc > 0:
                print(f'✓ Saved best model (Sequence Accuracy: {seq_acc:.2%})')
            else:
                print(f'✓ Saved model (Character Accuracy: {char_acc:.2%})')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'char_acc': char_acc,
                'seq_acc': seq_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f'\nTraining completed! Best sequence accuracy: {best_seq_acc:.2%}')


if __name__ == '__main__':
    main()

