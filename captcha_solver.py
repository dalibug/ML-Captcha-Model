"""
CAPTCHA Recognition - All-in-One Script
A complete solution for training and evaluating CAPTCHA recognition models.
"""
import os
import re
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


# ============================================================================
# Dataset Classes
# ============================================================================

class CaptchaDataset(Dataset):
    """Dataset class for CAPTCHA images."""
    
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    NUM_CHARS = len(CHARS)
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
    SEQ_LENGTH = 5
    
    def __init__(self, data_dir, image_files=None, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        
        if image_files is None:
            self.image_files = []
            for file in os.listdir(data_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = os.path.splitext(file)[0]
                    if len(label) == self.SEQ_LENGTH:
                        self.image_files.append(file)
        else:
            self.image_files = image_files
        
        if self.augment and self.transform is None:
            self.transform = self._get_augmentation_transforms()
        elif self.transform is None:
            self.transform = self._get_base_transforms()
    
    def _get_base_transforms(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((50, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _get_augmentation_transforms(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((50, 200)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: self._add_noise(x))
        ])
    
    def _add_noise(self, tensor):
        noise = torch.randn_like(tensor) * 0.05
        return torch.clamp(tensor + noise, -1, 1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        label_str = os.path.splitext(self.image_files[idx])[0]
        label = torch.zeros(self.SEQ_LENGTH, dtype=torch.long)
        for i, char in enumerate(label_str):
            if char in self.CHAR_TO_IDX:
                label[i] = self.CHAR_TO_IDX[char]
            else:
                label[i] = 0
        
        return image, label, label_str
    
    @staticmethod
    def decode_label(label_tensor):
        if isinstance(label_tensor, torch.Tensor):
            label_tensor = label_tensor.cpu().numpy()
        return ''.join([CaptchaDataset.IDX_TO_CHAR[int(idx)] for idx in label_tensor])


def create_dataloaders(data_dir, train_split=0.8, batch_size=32, augment_train=True, num_workers=0):
    """Create train and validation dataloaders."""
    image_files = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(file)[0]
            if len(label) == CaptchaDataset.SEQ_LENGTH:
                image_files.append(file)
    
    random.seed(42)
    random.shuffle(image_files)
    
    train_size = int(train_split * len(image_files))
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    train_dataset = CaptchaDataset(data_dir, train_files, augment=augment_train)
    val_dataset = CaptchaDataset(data_dir, val_files, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# Model Architectures
# ============================================================================

class BaselineCNN(nn.Module):
    """Baseline CNN with 5 separate output heads."""
    
    def __init__(self, num_chars=36, seq_length=5):
        super(BaselineCNN, self).__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.flatten_size = 512 * 3 * 12
        
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        
        self.heads = nn.ModuleList([
            nn.Linear(256, num_chars) for _ in range(seq_length)
        ])
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization for ReLU layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=1)


class CRNN(nn.Module):
    """CRNN: CNN encoder + BiLSTM + CTC loss."""
    
    def __init__(self, num_chars=36, hidden_size=128, num_layers=2):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))
        
        self.conv_to_lstm = nn.Conv2d(256, 256, kernel_size=(3, 1), padding=0)
        
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.conv_to_lstm(x)
        
        batch_size, channels, height, width = x.size()
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# ============================================================================
# Training Functions
# ============================================================================

def decode_ctc(ctc_output, input_lengths):
    """Decode CTC output using greedy decoding."""
    batch_size = ctc_output.size(1)
    predictions = []
    
    for b in range(batch_size):
        seq_len = input_lengths[b].item()
        probs = ctc_output[:seq_len, b, :].argmax(dim=1)
        
        decoded = []
        prev = -1
        for idx in probs:
            if idx.item() < ctc_output.size(2) - 1:
                if idx.item() != prev:
                    decoded.append(idx.item())
                prev = idx.item()
        
        if len(decoded) < 5:
            decoded.extend([0] * (5 - len(decoded)))
        else:
            decoded = decoded[:5]
        
        predictions.append(torch.tensor(decoded, dtype=torch.long))
    
    return torch.stack(predictions)


def train_epoch(model, train_loader, criterion, optimizer, device, model_type='baseline', max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, labels, label_strs in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        if model_type == 'baseline':
            outputs = model(images)
            loss = 0
            for i in range(5):
                loss += criterion(outputs[:, i, :], labels[:, i])
            loss = loss / 5
        else:  # CRNN
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
            target_lengths = torch.full((images.size(0),), 5, dtype=torch.long)
            loss = criterion(outputs, labels, input_lengths, target_lengths)
        
        loss.backward()
        
        # Gradient clipping for training stability
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
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
                loss = 0
                for i in range(5):
                    loss += criterion(outputs[:, i, :], labels[:, i])
                loss = loss / 5
                predictions = torch.argmax(outputs, dim=2)
            else:  # CRNN
                outputs = model(images)
                outputs = outputs.log_softmax(2)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
                target_lengths = torch.full((images.size(0),), 5, dtype=torch.long)
                loss = criterion(outputs, labels, input_lengths, target_lengths)
                predictions = decode_ctc(outputs, input_lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            for i in range(predictions.size(0)):
                all_predictions.append(predictions[i].cpu().numpy())
                all_labels.append(labels[i].cpu().numpy())
    
    char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
    return total_loss / num_batches, char_acc, seq_acc


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_accuracy(predictions, labels):
    """Calculate per-character and full-sequence accuracy."""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    char_correct = (predictions == labels).sum()
    char_total = predictions.size
    char_acc = char_correct / char_total
    
    seq_correct = (predictions == labels).all(axis=1).sum()
    seq_total = len(predictions)
    seq_acc = seq_correct / seq_total
    
    return char_acc, seq_acc


def evaluate_model(model, dataloader, device, model_type='baseline'):
    """Evaluate a model on a dataset."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_label_strs = []
    
    with torch.no_grad():
        for images, labels, label_strs in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            if model_type == 'baseline':
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=2)
            else:  # CRNN
                outputs = model(images)
                outputs = outputs.log_softmax(2)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
                predictions = decode_ctc(outputs, input_lengths)
            
            for i in range(predictions.size(0)):
                all_predictions.append(predictions[i].cpu().numpy())
                all_labels.append(labels[i].cpu().numpy())
                all_label_strs.append(label_strs[i])
    
    char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
    pred_strings = [CaptchaDataset.decode_label(pred) for pred in all_predictions]
    
    return char_acc, seq_acc, pred_strings, all_label_strs


def predict_image(model, image_path, device, model_type='baseline'):
    """Predict CAPTCHA from a single image file."""
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if model_type == 'baseline':
            outputs = model(image)
            predictions = torch.argmax(outputs, dim=2)
        else:  # CRNN
            outputs = model(image)
            outputs = outputs.log_softmax(2)
            outputs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((1,), outputs.size(0), dtype=torch.long)
            predictions = decode_ctc(outputs, input_lengths)
    
    prediction = CaptchaDataset.decode_label(predictions[0])
    return prediction


# ============================================================================
# Main Functions
# ============================================================================

def train(args):
    """Main training function."""
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        print(f'Created directories: {args.save_dir}, {args.log_dir}')
    except Exception as e:
        print(f'Error creating directories: {e}')
        raise
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_loader, val_loader = create_dataloaders(
        args.data_dir, train_split=0.8,
        batch_size=args.batch_size, augment_train=args.augment
    )
    
    num_chars = CaptchaDataset.NUM_CHARS
    if args.model_type == 'baseline':
        model = BaselineCNN(num_chars=num_chars, seq_length=5)
        # Use label smoothing for better generalization (optional, can be disabled)
        label_smoothing = getattr(args, 'label_smoothing', 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
        criterion = nn.CTCLoss(blank=num_chars, reduction='mean', zero_infinity=True)
    
    model = model.to(device)
    
    # Use slightly lower initial learning rate for better stability
    initial_lr = args.lr if args.lr > 0 else 0.0005
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    
    # Use cosine annealing with warm restarts for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Alternative: ReduceLROnPlateau (uncomment to use instead)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    # )
    
    writer = SummaryWriter(args.log_dir)
    best_seq_acc = 0.0
    
    print(f'\nStarting training ({args.model_type} model)...')
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model_type, max_grad_norm=1.0)
        val_loss, char_acc, seq_acc = validate(model, val_loader, criterion, device, args.model_type)
        
        # Step scheduler (adjust based on scheduler type)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Character', char_acc, epoch)
        writer.add_scalar('Accuracy/Sequence', seq_acc, epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Character Accuracy: {char_acc:.2%}, Sequence Accuracy: {seq_acc:.2%}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        should_save = False
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            should_save = True
        elif seq_acc == 0 and best_seq_acc == 0 and epoch == 0:
            should_save = True
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'char_acc': char_acc,
                'seq_acc': seq_acc,
                'model_type': args.model_type
            }
            best_path = os.path.join(args.save_dir, f'best_{args.model_type}.pth')
            try:
                torch.save(checkpoint, best_path)
                if seq_acc > 0:
                    print(f'✓ Saved best model (Sequence Accuracy: {seq_acc:.2%})')
                else:
                    print(f'✓ Saved model (Character Accuracy: {char_acc:.2%})')
            except Exception as e:
                print(f'⚠ Warning: Failed to save best model to {best_path}: {e}')
        
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'char_acc': char_acc,
                'seq_acc': seq_acc,
                'model_type': args.model_type
            }
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f'✓ Saved checkpoint: {checkpoint_path}')
            except Exception as e:
                print(f'⚠ Warning: Failed to save checkpoint to {checkpoint_path}: {e}')
                # Try alternative location in current directory
                try:
                    alt_path = f'./checkpoint_epoch_{epoch+1}.pth'
                    torch.save(checkpoint, alt_path)
                    print(f'✓ Saved checkpoint to alternative location: {alt_path}')
                except Exception as e2:
                    print(f'✗ Failed to save checkpoint to alternative location: {e2}')
    
    writer.close()
    print(f'\nTraining completed! Best sequence accuracy: {best_seq_acc:.2%}')


def evaluate(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    _, val_loader = create_dataloaders(
        args.data_dir, train_split=0.8,
        batch_size=args.batch_size, augment_train=False
    )
    
    num_chars = CaptchaDataset.NUM_CHARS
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get model type from checkpoint or args
    model_type = checkpoint.get('model_type', args.model_type)
    
    if model_type == 'baseline':
        model = BaselineCNN(num_chars=num_chars, seq_length=5)
    else:
        model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    char_acc, seq_acc, pred_strings, label_strings = evaluate_model(
        model, val_loader, device, model_type
    )
    
    print(f'\nEvaluation Results:')
    print(f'Character Accuracy: {char_acc:.2%}')
    print(f'Sequence Accuracy: {seq_acc:.2%}')
    
    print("\nSample Predictions:")
    print("-" * 50)
    for i in range(min(20, len(pred_strings))):
        status = "✓" if pred_strings[i] == label_strings[i] else "✗"
        print(f"{status} Predicted: {pred_strings[i]:<10} Actual: {label_strings[i]}")


def predict(args):
    """Predict CAPTCHA from a single image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_type = checkpoint.get('model_type', args.model_type)
    
    num_chars = CaptchaDataset.NUM_CHARS
    if model_type == 'baseline':
        model = BaselineCNN(num_chars=num_chars, seq_length=5)
    else:
        model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    prediction = predict_image(model, args.image_path, device, model_type)
    print(f'\nPrediction: {prediction}')


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CAPTCHA Recognition - All-in-One')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data_dir', type=str, default='samples', help='Directory with CAPTCHA images')
    train_parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'crnn'])
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate (default: 0.0005)')
    train_parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (default: 0.1, set to 0 to disable)')
    train_parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    train_parser.add_argument('--save_dir', type=str, default='checkpoints')
    train_parser.add_argument('--log_dir', type=str, default='logs')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--data_dir', type=str, default='samples')
    eval_parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'crnn'])
    eval_parser.add_argument('--batch_size', type=int, default=32)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict CAPTCHA from an image')
    predict_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    predict_parser.add_argument('--image_path', type=str, required=True, help='Path to CAPTCHA image')
    predict_parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'crnn'])
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


