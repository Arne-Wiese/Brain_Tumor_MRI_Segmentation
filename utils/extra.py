"""
Extra QOL functions for BraTS 2020 brain tumor segmentation.
"""

import torch.backends

def get_device():
    # Set device - prioritize MPS for Mac, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")

    return device