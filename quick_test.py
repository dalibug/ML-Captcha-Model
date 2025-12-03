"""
Quick test script to verify the setup and data loading.
"""
import torch
from data_loader import CaptchaDataset, create_dataloaders
from models import BaselineCNN, CRNN

def test_data_loading():
    """Test if data can be loaded correctly."""
    print("Testing data loading...")
    try:
        dataset = CaptchaDataset('samples', augment=False)
        print(f"✓ Found {len(dataset)} images")
        
        # Test loading one sample
        image, label, label_str = dataset[0]
        decoded = CaptchaDataset.decode_label(label)
        print(f"✓ Sample loaded: {label_str} -> {decoded}")
        print(f"  Image shape: {image.shape}")
        print(f"  Label shape: {label.shape}")
        
        # Test dataloaders
        train_loader, val_loader = create_dataloaders('samples', batch_size=4, augment_train=False)
        print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Test one batch
        images, labels, label_strs = next(iter(train_loader))
        print(f"✓ Batch shape: images {images.shape}, labels {labels.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """Test if models can be created and run forward pass."""
    print("\nTesting models...")
    try:
        num_chars = CaptchaDataset.NUM_CHARS
        
        # Test Baseline CNN
        baseline_model = BaselineCNN(num_chars=num_chars, seq_length=5)
        print("✓ BaselineCNN created")
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 50, 200)  # (batch, channels, height, width)
        output = baseline_model(dummy_input)
        print(f"✓ BaselineCNN forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        # Test CRNN
        crnn_model = CRNN(num_chars=num_chars, hidden_size=128, num_layers=2)
        print("✓ CRNN created")
        
        output = crnn_model(dummy_input)
        print(f"✓ CRNN forward pass: input {dummy_input.shape} -> output {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("CAPTCHA Model Quick Test")
    print("=" * 50)
    
    data_ok = test_data_loading()
    models_ok = test_models()
    
    print("\n" + "=" * 50)
    if data_ok and models_ok:
        print("✓ All tests passed! Ready to train.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 50)

