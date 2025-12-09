# CAPTCHA Recognition with Deep Learning

A deep learning project to automatically solve 5-character CAPTCHA images using CNN and CRNN architectures.

**Team Members:**
- Dalia Cabrera Hurtado
- Ahmed Torki
- Justin Ho
- Erica Lopez-Hernandez

## Project Overview

This project implements two model architectures for CAPTCHA recognition:

1. **Baseline Model**: CNN with 5 separate output heads (one per character position)
   - Target: ≥70% sequence accuracy
   
2. **Stretch Goal**: CRNN (CNN encoder + BiLSTM + CTC loss)
   - Target: ≥90% sequence accuracy

## Dataset

- **Source**: [CAPTCHA Version 2 Images (Kaggle)](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
- **Size**: 1,070 grayscale 200×50 CAPTCHA images
- **Format**: 5 alphanumeric characters per image
- **Characters**: 0-9, a-z (36 possible characters)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML-Captcha-Model
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
ML-Captcha-Model/
├── captcha_solver.py   # PyTorch implementation (all-in-one)
├── captcha_keras.py    # TensorFlow/Keras implementation (lecture-style)
├── requirements.txt    # Python dependencies
├── samples/            # CAPTCHA images (place dataset here)
├── checkpoints/        # Saved model checkpoints (created during training)
└── logs/               # TensorBoard logs (created during training)
```

## Usage

### 🔥 PyTorch Version (captcha_solver.py)

Everything in one file with PyTorch/torchvision.

#### Training the Baseline Model

```bash
python captcha_solver.py train --model_type baseline --epochs 50 --batch_size 32 --augment
```

#### Training the CRNN Model

```bash
python captcha_solver.py train --model_type crnn --epochs 50 --batch_size 32 --augment
```

#### Evaluation

```bash
python captcha_solver.py evaluate --checkpoint checkpoints/best_baseline.pth
```

#### Predict a Single Image

```bash
python captcha_solver.py predict --checkpoint checkpoints/best_baseline.pth --image_path samples/abc12.png
```

#### Training Options

- `--data_dir`: Directory containing CAPTCHA images (default: `samples`)
- `--model_type`: Model architecture - `baseline` or `crnn` (default: `baseline`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--augment`: Enable data augmentation (rotation, distortion, noise)
- `--save_dir`: Directory to save checkpoints (default: `checkpoints`)
- `--log_dir`: Directory for TensorBoard logs (default: `logs`)

### 🎓 TensorFlow/Keras Version (captcha_keras.py)

Follows CST463 lecture patterns: batch processing, data augmentation, transfer learning.

#### Training Baseline Model

```bash
python captcha_keras.py --model baseline
```

#### Training Improved Model (Data Augmentation + VGG16 Transfer Learning)

```bash
python captcha_keras.py --model improved
```

This version includes:
- **Batch Processing**: Uses `tf.data.Dataset` with prefetching
- **Data Augmentation**: RandomFlip, RandomRotation, RandomZoom
- **Transfer Learning**: VGG16 pretrained on ImageNet
- **Fine-tuning**: Unfreezes top conv blocks for better performance

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

## Model Architectures

### Baseline CNN

- **Architecture**: 4 convolutional blocks + 2 fully connected layers
- **Output**: 5 separate softmax heads (one per character position)
- **Loss**: Cross-entropy loss averaged across 5 positions
- **Advantages**: Simple, interpretable, fast training

### CRNN (Stretch Goal)

- **Architecture**: CNN encoder + Bidirectional LSTM + CTC loss
- **Output**: Sequence prediction with CTC decoding
- **Loss**: CTC loss (handles variable-length sequences)
- **Advantages**: Better sequence modeling, handles alignment automatically

## Data Preprocessing

- **Grayscale conversion**: All images converted to single channel
- **Resize**: Images resized to 50×200 pixels
- **Normalization**: Pixel values normalized to [-1, 1]

## Data Augmentation (Optional)

When `--augment` flag is used:
- Random rotation (±5 degrees)
- Random translation (±5%)
- Brightness/contrast jitter
- Random noise injection

## Metrics

The model reports two accuracy metrics:

1. **Per-Character Accuracy**: Percentage of correctly predicted individual characters
2. **Sequence Accuracy**: Percentage of CAPTCHAs where all 5 characters are correct

## Expected Performance

- **Baseline Goal**: ≥70% sequence accuracy
- **Stretch Goal**: ≥90% sequence accuracy

## Computational Requirements

- **Training Time**: ~5-10 minutes per epoch on Google Colab free GPU
- **Dataset Size**: <10MB
- **Memory**: ~2-4GB GPU memory recommended
- **No special hardware required** - works on CPU (slower) or GPU (faster)

## Example Output

```
Epoch 1/50
Training: 100%|████████| 27/27 [00:45<00:00]
Validating: 100%|██████| 7/7 [00:08<00:00]
Train Loss: 1.2345, Val Loss: 1.1234
Character Accuracy: 85.23%, Sequence Accuracy: 72.45%
✓ Saved best model (Sequence Accuracy: 72.45%)
```

## Troubleshooting

### Out of Memory Error
- Reduce `--batch_size` (e.g., `--batch_size 16`)

### Low Accuracy
- Enable data augmentation: `--augment`
- Train for more epochs: `--epochs 100`
- Try the CRNN model for better sequence modeling

### Slow Training
- Use GPU if available (automatically detected)
- Reduce batch size if memory constrained
- Disable augmentation for faster training

## License

This project is for educational purposes (CST463 course project).

## References

- Dataset: [CAPTCHA Version 2 Images](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images)
- PyTorch Documentation: https://pytorch.org/docs/
