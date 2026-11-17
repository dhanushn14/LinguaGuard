"""
Adaptive Differential Privacy (DP) Module
Implements client-side gradient/parameter clipping and calibrated Gaussian noise injection.
Supports both .privatize() and .apply_differential_privacy() interfaces.
"""

import numpy as np
from configs.config import DP_CLIP_VALUE, DP_NOISE_MULTIPLIER


class AdaptiveDifferentialPrivacy:
    """
    Lightweight Adaptive Differential Privacy:
    - In-place parameter clipping
    - Calibrated Gaussian noise addition
    - Avoids excessive memory allocations
    """
    def __init__(self, clip_value: float = DP_CLIP_VALUE, noise_multiplier: float = DP_NOISE_MULTIPLIER):
        self.clip_value = float(clip_value)
        self.clip_norm = self.clip_value
        self.noise_multiplier = float(noise_multiplier)

    def privatize(self, parameters):
        """
        Applies clipping and Gaussian noise to a list of NumPy parameter arrays in-place.
        """
        for i in range(len(parameters)):
            param = parameters[i].astype(np.float32, copy=False)

            # In-place clipping
            np.clip(param, -self.clip_value, self.clip_value, out=param)

            # Calibrated Gaussian noise
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_multiplier,
                size=param.shape
            ).astype(np.float32)

            # In-place addition (memory efficient)
            param += noise
            parameters[i] = param

        return parameters

    def apply_differential_privacy(self, parameters):
        """
        Alias for privatize() to maintain backward compatibility.
        """
        return self.privatize(parameters)
