#!/bin/bash

# Script to fix temporary directory issues

echo "Checking temporary directories..."

# Check disk space
echo -e "\n=== Disk Space ==="
df -h /

# Check /tmp directory
echo -e "\n=== /tmp Directory ==="
if [ -d "/tmp" ]; then
    echo "✓ /tmp exists"
    ls -ld /tmp
    if [ -w "/tmp" ]; then
        echo "✓ /tmp is writable"
    else
        echo "✗ /tmp is not writable"
        echo "Attempting to fix permissions..."
        sudo chmod 1777 /tmp
    fi
else
    echo "✗ /tmp does not exist"
    echo "Creating /tmp directory..."
    sudo mkdir -p /tmp
    sudo chmod 1777 /tmp
fi

# Check user-specific temp directory
USER_TEMP="/var/folders/7s/y2ptxpt11h7fpjjqb4cphgcc0000gn/T/"
echo -e "\n=== User Temp Directory ==="
if [ -d "$USER_TEMP" ]; then
    echo "✓ $USER_TEMP exists"
    ls -ld "$USER_TEMP"
    if [ -w "$USER_TEMP" ]; then
        echo "✓ User temp directory is writable"
    else
        echo "✗ User temp directory is not writable"
    fi
else
    echo "✗ User temp directory does not exist"
fi

# Create a test file
echo -e "\n=== Testing temp file creation ==="
TEST_FILE=$(mktemp 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ Successfully created test temp file: $TEST_FILE"
    rm -f "$TEST_FILE"
else
    echo "✗ Failed to create temp file: $TEST_FILE"
    echo -e "\nSuggested fixes:"
    echo "1. Clear disk space if disk is full"
    echo "2. Restart your computer to clean up temp files"
    echo "3. Manually set TMPDIR: export TMPDIR=~/tmp && mkdir -p ~/tmp"
fi

echo -e "\n=== Python temp dir test ==="
python3 << 'EOF'
import tempfile
try:
    temp_dir = tempfile.gettempdir()
    print(f"✓ Python temp directory: {temp_dir}")
    
    # Try to create a temp file
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        print(f"✓ Successfully created temp file: {tmp.name}")
except Exception as e:
    print(f"✗ Python temp directory error: {e}")
    print("\nTo fix:")
    print("1. Run: export TMPDIR=~/tmp && mkdir -p ~/tmp")
    print("2. Add to ~/.zshrc: export TMPDIR=~/tmp")
EOF

echo -e "\n=== Done ==="



