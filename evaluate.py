"""
Evaluation utilities for CAPTCHA recognition models.
"""
import torch
import numpy as np
from data_loader import CaptchaDataset


def calculate_accuracy(predictions, labels):
    """
    Calculate per-character and full-sequence accuracy.
    
    Args:
        predictions: List of prediction arrays (each of length 5)
        labels: List of label arrays (each of length 5)
    
    Returns:
        char_acc: Per-character accuracy
        seq_acc: Full-sequence accuracy
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Per-character accuracy
    char_correct = (predictions == labels).sum()
    char_total = predictions.size
    char_acc = char_correct / char_total
    
    # Full-sequence accuracy
    seq_correct = (predictions == labels).all(axis=1).sum()
    seq_total = len(predictions)
    seq_acc = seq_correct / seq_total
    
    return char_acc, seq_acc


def evaluate_model(model, dataloader, device, model_type='baseline'):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run on
        model_type: 'baseline' or 'crnn'
    
    Returns:
        char_acc, seq_acc, predictions, labels
    """
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
                predictions = torch.argmax(outputs, dim=2)  # (batch, 5)
            else:  # CRNN
                outputs = model(images)
                outputs = outputs.log_softmax(2)
                outputs = outputs.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
                predictions = decode_ctc(outputs, input_lengths)
            
            # Store predictions and labels
            for i in range(predictions.size(0)):
                all_predictions.append(predictions[i].cpu().numpy())
                all_labels.append(labels[i].cpu().numpy())
                all_label_strs.append(label_strs[i])
    
    # Calculate accuracies
    char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
    
    # Decode predictions to strings
    pred_strings = [CaptchaDataset.decode_label(pred) for pred in all_predictions]
    
    return char_acc, seq_acc, pred_strings, all_label_strs


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


def print_sample_predictions(pred_strings, label_strings, num_samples=10):
    """Print sample predictions for inspection."""
    print("\nSample Predictions:")
    print("-" * 50)
    correct = 0
    for i in range(min(num_samples, len(pred_strings))):
        is_correct = pred_strings[i] == label_strings[i]
        if is_correct:
            correct += 1
        status = "✓" if is_correct else "✗"
        print(f"{status} Predicted: {pred_strings[i]:<10} Actual: {label_strings[i]}")
    print("-" * 50)
    if num_samples < len(pred_strings):
        print(f"Showing {num_samples} of {len(pred_strings)} samples")


if __name__ == '__main__':
    import argparse
    from data_loader import create_dataloaders
    from models import BaselineCNN, CRNN
    
    parser = argparse.ArgumentParser(description='Evaluate CAPTCHA recognition model')
    parser.add_argument('--data_dir', type=str, default='samples',
                        help='Directory containing CAPTCHA images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='baseline',
                        choices=['baseline', 'crnn'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloader
    _, val_loader = create_dataloaders(
        args.data_dir,
        train_split=0.8,
        batch_size=args.batch_size,
        augment_train=False
    )
    
    # Load model
    num_chars = CaptchaDataset.NUM_CHARS
    if args.model_type == 'baseline':
        model = BaselineCNN(num_chars=num_chars, seq_length=5)
    else:
        model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    char_acc, seq_acc, pred_strings, label_strings = evaluate_model(
        model, val_loader, device, args.model_type
    )
    
    print(f'\nEvaluation Results:')
    print(f'Character Accuracy: {char_acc:.2%}')
    print(f'Sequence Accuracy: {seq_acc:.2%}')
    
    # Print samples
    print_sample_predictions(pred_strings, label_strings, num_samples=20)

