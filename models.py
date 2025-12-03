"""
Model architectures for CAPTCHA recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Baseline CNN model with 5 separate output heads (one per character).
    Each head predicts a single character using softmax.
    """
    
    def __init__(self, num_chars=36, seq_length=5):
        """
        Args:
            num_chars: Number of possible characters (36 for alphanumeric)
            seq_length: Length of CAPTCHA sequence (5)
        """
        super(BaselineCNN, self).__init__()
        self.num_chars = num_chars
        self.seq_length = seq_length
        
        # Feature extraction layers - improved architecture
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
        
        # Calculate flattened size: (50, 200) -> (3, 12) after 4 pooling layers
        # 50/16 = 3.125 -> 3, 200/16 = 12.5 -> 12
        self.flatten_size = 512 * 3 * 12
        
        # Shared fully connected layers - increased capacity
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.3)
        
        # 5 separate output heads (one per character position)
        self.heads = nn.ModuleList([
            nn.Linear(256, num_chars) for _ in range(seq_length)
        ])
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # 5 separate predictions
        outputs = [head(x) for head in self.heads]
        
        return torch.stack(outputs, dim=1)  # Shape: (batch, 5, num_chars)


class CRNN(nn.Module):
    """
    CRNN model: CNN encoder + BiLSTM + CTC loss for sequence prediction.
    This is the stretch goal architecture.
    """
    
    def __init__(self, num_chars=36, hidden_size=128, num_layers=2):
        """
        Args:
            num_chars: Number of possible characters (36 for alphanumeric)
            hidden_size: Hidden size for LSTM
            num_layers: Number of LSTM layers
        """
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.hidden_size = hidden_size
        
        # CNN Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 1))  # Keep width
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d((2, 1))  # Keep width
        
        # After pooling: (50, 200) -> (3, 50) -> channels*height = 256*3 = 768
        # Add a linear layer to reduce to 256 for LSTM
        self.conv_to_lstm = nn.Conv2d(256, 256, kernel_size=(3, 1), padding=0)  # Reduce height to 1
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_chars + 1)  # +1 for CTC blank token
    
    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Reduce height dimension to 1: (batch, 256, 3, width) -> (batch, 256, 1, width)
        x = self.conv_to_lstm(x)
        
        # Reshape for LSTM: (batch, channels, height, width) -> (batch, width, channels)
        batch_size, channels, height, width = x.size()
        x = x.squeeze(2)  # Remove height dimension: (batch, channels, width)
        x = x.permute(0, 2, 1)  # (batch, width, channels)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Output layer
        x = self.fc(x)  # (batch, width, num_chars + 1)
        
        return x  # Shape: (batch, sequence_length, num_chars + 1)

