r"""
Secure Aggregation Simulation Module
Simulates zero-sum masking for federated model updates across clients.
Each client adds a calibrated noise mask that cancels out upon summation at the server (\sum_i mask_i = 0).
"""

import numpy as np
from configs.config import SECURE_AGG_SCALE


class SecureAggregation:
    """
    Lightweight Secure Aggregation Simulation:
    Masks model parameters before transmitting them to the central server.
    """
    def __init__(self, mask_scale: float = SECURE_AGG_SCALE):
        self.mask_scale = float(mask_scale)

    def generate_mask(self, parameters):
        masks = []
        for param in parameters:
            mask = np.random.normal(
                loc=0.0,
                scale=self.mask_scale,
                size=param.shape
            ).astype(np.float32)
            masks.append(mask)
        return masks

    def mask_parameters(self, parameters):
        masks = self.generate_mask(parameters)
        masked_parameters = []
        for param, mask in zip(parameters, masks):
            masked_parameters.append(param + mask)
        return masked_parameters
