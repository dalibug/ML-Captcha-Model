"""
CAPTCHA Recognition - TensorFlow/Keras (Lecture-Style)
Follows CST463 lecture patterns: batch processing, data augmentation, transfer learning.

Dataset structure:
samples/
  abc12.png
  xyz34.png
  ... (filenames are the labels)
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image
import random

# ============================================================================
# Configuration
# ============================================================================

img_size = (50, 200)  # CAPTCHA images are 50x200
batch_size = 32
data_dir = "samples"

# Character mapping
CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
NUM_CHARS = len(CHARS)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
SEQ_LENGTH = 5  # 5-character CAPTCHA

# ============================================================================
# Data Loading (Lecture: image_dataset_from_directory style)
# ============================================================================

def create_tf_dataset_from_directory(data_dir, subset='train', train_split=0.8):
    """
    Create tf.data.Dataset from CAPTCHA images.
    Follows lecture pattern: image_dataset_from_directory -> tf.data.Dataset
    """
    # Get all image files
    image_files = []
    labels = []
    
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            label_str = os.path.splitext(file)[0]
            if len(label_str) == SEQ_LENGTH:
                image_files.append(os.path.join(data_dir, file))
                # Convert label string to indices
                label_indices = [CHAR_TO_IDX.get(c, 0) for c in label_str]
                labels.append(label_indices)
    
    # Shuffle and split
    random.seed(42)
    combined = list(zip(image_files, labels))
    random.shuffle(combined)
    image_files, labels = zip(*combined)
    
    split_idx = int(train_split * len(image_files))
    
    if subset == 'train':
        image_files = list(image_files[:split_idx])
        labels = list(labels[:split_idx])
    else:  # validation
        image_files = list(image_files[split_idx:])
        labels = list(labels[split_idx:])
    
    # Create dataset
    def load_and_preprocess_image(image_path):
        """Load and preprocess single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        return image
    
    def load_image_and_label(image_path, label):
        """Load image and return with label as list for multi-output model."""
        image = load_and_preprocess_image(image_path)
        # For multi-output model, return labels as list (one per output head)
        label_list = [label[i] for i in range(SEQ_LENGTH)]
        return image, label_list
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_files, labels))
    dataset = dataset.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Lecture: prefetch for performance
    
    return dataset

# ============================================================================
# Baseline Model (Lecture: Simple CNN - Conv2D + MaxPooling2D stack)
# ============================================================================

def create_baseline_model():
    """
    Baseline CNN following lecture pattern:
    - Rescaling layer
    - Conv2D + MaxPooling2D blocks
    - Flatten + Dense layers
    - 5 separate output heads (one per character position)
    """
    inputs = keras.Input(shape=img_size + (1,))
    
    # Rescaling (lecture pattern)
    x = layers.Rescaling(1./255)(inputs)
    x = (x - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Conv blocks (lecture: Conv2D + MaxPooling2D)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Flatten + Dense (lecture pattern)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    # 5 separate output heads (one per character position)
    outputs = []
    for i in range(SEQ_LENGTH):
        head = layers.Dense(NUM_CHARS, activation="softmax", name=f"char_{i}")(x)
        outputs.append(head)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# ============================================================================
# Improved Model (Lecture: Data Augmentation + Transfer Learning)
# ============================================================================

def create_improved_model():
    """
    Improved model following lecture patterns:
    - Data augmentation: RandomFlip, RandomRotation, RandomZoom
    - Transfer learning: VGG16 conv base (frozen)
    - Fine-tuning: unfreeze top conv blocks
    """
    inputs = keras.Input(shape=img_size + (1,))
    
    # Data augmentation block (lecture: RandomFlip/Rotation/Zoom)
    # Note: Converting grayscale to RGB for pretrained models
    x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inputs)
    x = layers.Rescaling(1./255)(x)
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),  # ±5 degrees
        layers.RandomZoom(0.1),        # ±10%
        layers.RandomTranslation(0.05, 0.05),  # ±5% translation
    ])
    x = data_augmentation(x)
    
    # Pretrained conv base (lecture: VGG16, no top classifier)
    # Resize to VGG16 input size (224, 224)
    x = layers.Resizing(224, 224)(x)
    
    conv_base = keras.applications.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze conv base (lecture: freeze borrowed layers first)
    conv_base.trainable = False
    
    x = conv_base(x)
    
    # Global average pooling (better than flatten for transfer learning)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    # 5 separate output heads
    outputs = []
    for i in range(SEQ_LENGTH):
        head = layers.Dense(NUM_CHARS, activation="softmax", name=f"char_{i}")(x)
        outputs.append(head)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model, conv_base

# ============================================================================
# Training Functions
# ============================================================================

def train_baseline():
    """Train baseline model (lecture: simple CNN training)."""
    print("=" * 60)
    print("Training Baseline CNN Model")
    print("=" * 60)
    
    # Load data (lecture: batch processing with tf.data)
    print("\n[Lecture: Batch Processing] Loading data with tf.data.Dataset...")
    train_ds = create_tf_dataset_from_directory(data_dir, subset='train')
    val_ds = create_tf_dataset_from_directory(data_dir, subset='validation')
    
    print(f"Training samples: {len(list(train_ds.unbatch()))}")
    print(f"Validation samples: {len(list(val_ds.unbatch()))}")
    
    # Create model (lecture: simple CNN architecture)
    print("\n[Lecture: Basic CNN] Creating baseline model...")
    model = create_baseline_model()
    
    # Compile (lecture: standard compilation)
    # For multi-output model, use list of losses (one per output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=["sparse_categorical_crossentropy"] * SEQ_LENGTH,
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Callbacks (lecture: ModelCheckpoint pattern)
    print("\n[Lecture: Checkpointing] Setting up ModelCheckpoint callback...")
    checkpoint_cb = ModelCheckpoint(
        filepath="baseline_captcha_cnn.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
    
    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train (lecture: fit on tf.data.Dataset)
    print("\n[Training] Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    # Evaluate best model
    print("\n[Evaluation] Loading best model...")
    best_model = keras.models.load_model("baseline_captcha_cnn.keras")
    
    # Calculate accuracies
    val_predictions = best_model.predict(val_ds)
    
    # Extract true labels (val_ds returns labels as list of 5 tensors)
    val_labels = []
    for _, label_list in val_ds:
        batch_size_actual = label_list[0].shape[0]
        for i in range(batch_size_actual):
            sample_labels = [label_list[j][i].numpy() for j in range(SEQ_LENGTH)]
            val_labels.append(sample_labels)
    val_labels = np.array(val_labels)
    
    # Per-character accuracy
    char_correct = 0
    char_total = 0
    seq_correct = 0
    
    for i in range(len(val_labels)):
        pred_chars = [np.argmax(val_predictions[j][i]) for j in range(SEQ_LENGTH)]
        true_chars = val_labels[i]
        
        for j in range(SEQ_LENGTH):
            if pred_chars[j] == true_chars[j]:
                char_correct += 1
            char_total += 1
        
        if all(pred_chars[j] == true_chars[j] for j in range(SEQ_LENGTH)):
            seq_correct += 1
    
    char_acc = char_correct / char_total
    seq_acc = seq_correct / len(val_labels)
    
    print(f"\nBaseline Results:")
    print(f"Character Accuracy: {char_acc:.2%}")
    print(f"Sequence Accuracy: {seq_acc:.2%}")
    
    return best_model, history

def train_improved():
    """Train improved model (lecture: data augmentation + transfer learning)."""
    print("=" * 60)
    print("Training Improved Model (Data Augmentation + Transfer Learning)")
    print("=" * 60)
    
    # Load data
    print("\n[Lecture: Batch Processing] Loading data...")
    train_ds = create_tf_dataset_from_directory(data_dir, subset='train')
    val_ds = create_tf_dataset_from_directory(data_dir, subset='validation')
    
    # Create model (lecture: data augmentation + transfer learning)
    print("\n[Lecture: Data Augmentation + Transfer Learning] Creating improved model...")
    model, conv_base = create_improved_model()
    
    # Compile with lower learning rate for transfer learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=["sparse_categorical_crossentropy"] * SEQ_LENGTH,
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        filepath="improved_captcha_vgg16.keras",
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
    
    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train head-only (lecture: freeze conv base first)
    print("\n[Lecture: Transfer Learning] Training head with frozen conv base...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    # Fine-tuning (lecture: unfreeze top conv blocks)
    print("\n[Lecture: Fine-tuning] Unfreezing top conv blocks...")
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == "block5_conv1":  # Unfreeze from block5 upward
            set_trainable = True
        layer.trainable = set_trainable
    
    # Recompile with smaller learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=["sparse_categorical_crossentropy"] * SEQ_LENGTH,
        metrics=["accuracy"]
    )
    
    print("\n[Lecture: Fine-tuning] Training with unfrozen top layers...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    # Evaluate
    print("\n[Evaluation] Loading best model...")
    best_model = keras.models.load_model("improved_captcha_vgg16.keras")
    
    val_predictions = best_model.predict(val_ds)
    
    # Extract true labels (val_ds returns labels as list of 5 tensors)
    val_labels = []
    for _, label_list in val_ds:
        batch_size_actual = label_list[0].shape[0]
        for i in range(batch_size_actual):
            sample_labels = [label_list[j][i].numpy() for j in range(SEQ_LENGTH)]
            val_labels.append(sample_labels)
    val_labels = np.array(val_labels)
    
    char_correct = 0
    char_total = 0
    seq_correct = 0
    
    for i in range(len(val_labels)):
        pred_chars = [np.argmax(val_predictions[j][i]) for j in range(SEQ_LENGTH)]
        true_chars = val_labels[i]
        
        for j in range(SEQ_LENGTH):
            if pred_chars[j] == true_chars[j]:
                char_correct += 1
            char_total += 1
        
        if all(pred_chars[j] == true_chars[j] for j in range(SEQ_LENGTH)):
            seq_correct += 1
    
    char_acc = char_correct / char_total
    seq_acc = seq_correct / len(val_labels)
    
    print(f"\nImproved Model Results:")
    print(f"Character Accuracy: {char_acc:.2%}")
    print(f"Sequence Accuracy: {seq_acc:.2%}")
    
    return best_model, history1, history2

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Recognition - Keras (Lecture Style)')
    parser.add_argument('--model', type=str, default='baseline', 
                       choices=['baseline', 'improved'],
                       help='Model to train: baseline or improved')
    parser.add_argument('--data_dir', type=str, default='samples',
                       help='Directory containing CAPTCHA images')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    if args.model == 'baseline':
        train_baseline()
    else:
        train_improved()

