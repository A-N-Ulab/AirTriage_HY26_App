"""Accelerator selection shared by the tracker and the notebook helpers.

Kept in its own module so that `tracker` does not have to import
`video_utils` (which pulls in IPython) just to pick a device.
"""


def get_device():
    """Best available accelerator: CUDA, Apple Silicon, or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"