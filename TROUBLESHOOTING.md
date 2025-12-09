# Troubleshooting Guide

## Issues Encountered and Fixed

### 1. Checkpoint Save Failure

**Problem:** Training failed at epoch 10 with:
```
RuntimeError: File checkpoints/checkpoint_epoch_10.pth cannot be opened.
```

**Root Cause:** The checkpoint directory didn't exist or had permission issues.

**Solutions Applied:**
- ✅ Added explicit error handling with try-catch blocks for checkpoint saving
- ✅ Added verbose logging when directories are created
- ✅ Added fallback to save checkpoints in current directory if primary location fails
- ✅ The code now creates the `checkpoints` directory automatically with `exist_ok=True`

### 2. Temporary Directory Error

**Problem:** Training failed with:
```
FileNotFoundError: [Errno 2] No usable temporary directory found in ['/var/folders/...', '/tmp', ...]
```

**Root Cause:** System temporary directories were inaccessible due to:
- Disk space full
- Permission issues
- System temp directory corruption

**Solutions:**

#### Quick Fix (Recommended)
Set a custom temp directory in your home folder:
```bash
export TMPDIR=~/tmp
mkdir -p ~/tmp
```

Add this to your `~/.zshrc` to make it permanent:
```bash
echo 'export TMPDIR=~/tmp' >> ~/.zshrc
source ~/.zshrc
```

#### Check and Fix Script
Run the diagnostic script:
```bash
chmod +x fix_temp_dir.sh
./fix_temp_dir.sh
```

This will:
- Check disk space
- Verify `/tmp` directory exists and is writable
- Test temp file creation
- Provide specific fix recommendations

#### Manual Fixes

1. **Check disk space:**
   ```bash
   df -h /
   ```
   If disk is full, free up space.

2. **Fix /tmp permissions:**
   ```bash
   sudo chmod 1777 /tmp
   ```

3. **Restart your computer** to clean up system temp files

4. **Clear temporary files manually:**
   ```bash
   # Clear user temp (be careful!)
   rm -rf ~/Library/Caches/*
   
   # Clear system temp (requires admin)
   sudo rm -rf /private/var/folders/*/*/T/*
   ```

### 3. Training Successfully

After applying fixes, training should work normally:
```bash
python captcha_solver.py train --model_type baseline --epochs 100 --batch_size 32 --augment
```

The improved error handling will now:
- ✅ Show clear messages when directories are created
- ✅ Warn if checkpoint saving fails (but continue training)
- ✅ Attempt to save to alternative location if primary fails
- ✅ Not crash the entire training if one checkpoint save fails

## Current Status

✅ Checkpoint directory exists with saved models:
- `checkpoints/best_baseline.pth`
- `checkpoints/checkpoint_epoch_*.pth` (every 10 epochs)

✅ Logs directory exists with TensorBoard logs

✅ Code improvements applied with better error handling

## Best Practices

1. **Before training:**
   ```bash
   # Check disk space
   df -h
   
   # Set temp directory
   export TMPDIR=~/tmp
   mkdir -p ~/tmp
   
   # Verify directories exist
   ls -la checkpoints/ logs/
   ```

2. **During training:**
   - Monitor disk space: `watch -n 60 df -h`
   - Watch for checkpoint save messages
   - If training fails, check the last error message

3. **After training:**
   - Verify checkpoints were saved: `ls -lh checkpoints/`
   - Check TensorBoard logs: `tensorboard --logdir=logs`

## Additional Help

If you still encounter issues:

1. Check available disk space: `df -h`
2. Check checkpoint directory permissions: `ls -la checkpoints/`
3. Try running with a different save directory: `--save_dir ~/captcha_checkpoints`
4. Check Python temp directory: `python -c "import tempfile; print(tempfile.gettempdir())"`

## Performance Notes

Based on your training run:
- Training samples: 856
- Validation samples: 214
- After 10 epochs: ~10% character accuracy, 0% sequence accuracy

This is expected for early training. The model typically needs:
- 50-100 epochs to reach good character accuracy (>70%)
- 100-200 epochs for high sequence accuracy (>50%)

Consider:
- Continuing training for full 100 epochs
- Using data augmentation (already enabled with `--augment`)
- Trying different model types: `--model_type crnn`



