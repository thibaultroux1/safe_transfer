import numpy as np


def clip_norm(x, window_size):
    """Normalize coordinate to [-1, 1] based on window size."""
    return np.clip(x / (window_size / 2) - 1, -1, 1)


def unclip_norm(norm_x, window_size):
    """Convert normalized coordinate back to window coordinates."""
    return (norm_x + 1) * (window_size / 2)


def distance_normalized(x1, y1, x2, y2):
    """Calculate distance between two normalized points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
