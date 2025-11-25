# Checkpoints

This folder contains saved model checkpoints and training histories.

## Contents

- **`*.pth`** - PyTorch Model Checkpoints
  - Saved model weights after training
  - Format: `{model_name}.pth`
  - Contains: `model_state_dict`, `epoch`, `val_metric`

## Usage

### Load Checkpoint

```python
from utils import load_checkpoint

epoch, val_metric = load_checkpoint(model, 'checkpoints/best_model_v1.pth', device=device)
print(f"Loaded model from epoch {epoch} with validation metric: {val_metric:.4f}")
```

### Save Checkpoint

Checkpoints are automatically saved during training when `save_best_model=True` is set in `train_loop()`.

## Notes

- Checkpoints should be backed up regularly
- Large checkpoint files (.pth) should not be committed to the Git repository
- Best Practice: Use `.gitignore` for `*.pth` files
