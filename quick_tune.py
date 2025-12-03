"""
Quick hyperparameter tuning - tests a few key combinations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime

from data_loader import create_dataloaders, CaptchaDataset
from hyperparameter_tuning import FlexibleBaselineCNN
from evaluate import calculate_accuracy


def quick_train(lr, epochs, num_conv_layers, batch_size=32, data_dir='samples'):
    """Quick training and evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader = create_dataloaders(
        data_dir, train_split=0.8, batch_size=batch_size, augment_train=True
    )
    
    num_chars = CaptchaDataset.NUM_CHARS
    model = FlexibleBaselineCNN(num_chars=num_chars, seq_length=5, num_conv_layers=num_conv_layers)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_seq_acc = 0.0
    best_char_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = sum(criterion(outputs[:, i, :], labels[:, i]) for i in range(5)) / 5
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=2)
                for i in range(predictions.size(0)):
                    all_predictions.append(predictions[i].cpu().numpy())
                    all_labels.append(labels[i].cpu().numpy())
        
        char_acc, seq_acc = calculate_accuracy(all_predictions, all_labels)
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            best_char_acc = char_acc
    
    return best_char_acc, best_seq_acc


def main():
    # Test combinations: (lr, epochs, conv_layers)
    combinations = [
        (0.001, 50, 4),   # Baseline
        (0.0005, 75, 4),  # Lower LR, more epochs
        (0.0003, 100, 4), # Even lower LR
        (0.001, 50, 3),   # Fewer conv layers
        (0.001, 50, 5),   # More conv layers
        (0.0005, 100, 3), # Lower LR, fewer layers
        (0.0005, 100, 5), # Lower LR, more layers
    ]
    
    results = []
    
    print("=" * 80)
    print("QUICK HYPERPARAMETER TUNING")
    print("=" * 80)
    
    for i, (lr, epochs, num_conv) in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Testing: LR={lr}, Epochs={epochs}, Conv Layers={num_conv}")
        print("-" * 80)
        
        try:
            char_acc, seq_acc = quick_train(lr, epochs, num_conv)
            results.append({
                'lr': lr,
                'epochs': epochs,
                'num_conv_layers': num_conv,
                'char_accuracy': float(char_acc),
                'seq_accuracy': float(seq_acc)
            })
            print(f"✓ Char Acc: {char_acc:.2%}, Seq Acc: {seq_acc:.2%}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Sort by sequence accuracy
    results.sort(key=lambda x: x['seq_accuracy'], reverse=True)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i}. LR={r['lr']}, Epochs={r['epochs']}, Conv={r['num_conv_layers']}")
        print(f"   Char: {r['char_accuracy']:.2%}, Seq: {r['seq_accuracy']:.2%}")
    
    # Save results
    with open('quick_tune_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to quick_tune_results.json")


if __name__ == '__main__':
    main()

