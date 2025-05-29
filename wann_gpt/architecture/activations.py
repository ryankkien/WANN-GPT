"""
diverse activation functions for wann architectures
supports evolution of activation types per layer/neuron
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable
from enum import Enum

class ActivationType(Enum):
    """enumeration of supported activation functions"""
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    SINE = "sine"
    COSINE = "cosine"
    GAUSSIAN = "gaussian"
    STEP = "step"
    LINEAR = "linear"
    ABS = "abs"
    SQUARE = "square"
    SOFTPLUS = "softplus"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"

class ActivationRegistry:
    """registry for activation functions used in wann evolution"""
    
    def __init__(self):
        self._activations: Dict[ActivationType, Callable] = {}
        self._register_default_activations()
    
    def _register_default_activations(self):
        """register all available activation functions"""
        
        # standard activations
        self._activations[ActivationType.RELU] = torch.relu
        self._activations[ActivationType.GELU] = torch.nn.functional.gelu
        self._activations[ActivationType.TANH] = torch.tanh
        self._activations[ActivationType.SIGMOID] = torch.sigmoid
        self._activations[ActivationType.SWISH] = lambda x: x * torch.sigmoid(x)
        self._activations[ActivationType.SOFTPLUS] = torch.nn.functional.softplus
        self._activations[ActivationType.ELU] = torch.nn.functional.elu
        self._activations[ActivationType.LEAKY_RELU] = lambda x: torch.nn.functional.leaky_relu(x, 0.1)
        
        # trigonometric activations
        self._activations[ActivationType.SINE] = torch.sin
        self._activations[ActivationType.COSINE] = torch.cos
        
        # other non-standard activations
        self._activations[ActivationType.GAUSSIAN] = lambda x: torch.exp(-x**2)
        self._activations[ActivationType.STEP] = lambda x: (x > 0).float()
        self._activations[ActivationType.LINEAR] = lambda x: x
        self._activations[ActivationType.ABS] = torch.abs
        self._activations[ActivationType.SQUARE] = lambda x: x**2
    
    def get_activation(self, activation_type: ActivationType) -> Callable:
        """get activation function by type"""
        if activation_type not in self._activations:
            raise ValueError(f"unsupported activation type: {activation_type}")
        return self._activations[activation_type]
    
    def get_random_activation(self) -> ActivationType:
        """get random activation type for mutations"""
        return np.random.choice(list(ActivationType))
    
    def get_all_types(self) -> list[ActivationType]:
        """get all available activation types"""
        return list(ActivationType)

class EvolvableActivation(nn.Module):
    """activation layer that can evolve during architecture search"""
    
    def __init__(self, initial_type: ActivationType = ActivationType.RELU):
        super().__init__()
        self.registry = ActivationRegistry()
        self.activation_type = initial_type
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply current activation function"""
        activation_fn = self.registry.get_activation(self.activation_type)
        return activation_fn(x)
    
    def mutate_activation(self):
        """mutate to random activation type"""
        self.activation_type = self.registry.get_random_activation()
    
    def set_activation(self, activation_type: ActivationType):
        """set specific activation type"""
        self.activation_type = activation_type
    
    def clone(self) -> 'EvolvableActivation':
        """create copy with same activation"""
        return EvolvableActivation(self.activation_type)

def create_cuda_activation_kernel():
    """cuda kernel for efficient activation computation"""
    # this would be implemented with cupy or custom cuda
    # for now we'll use pytorch's built-in cuda support
    pass 