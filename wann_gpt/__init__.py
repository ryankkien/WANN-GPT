"""
Weight-Agnostic Neural Networks for GPT-2 Transformers

This package implements the WANN approach for transformer architectures,
allowing for architecture evolution without weight training.
"""

__version__ = "0.2.0"
__author__ = "Research Team"

# core architecture
from .architecture import WannGPT, WannTransformerBlock

# evolution components
from .evolution import EvolutionEngine, ArchitectureGenome

# evaluation system
from .evaluation import SharedWeightEvaluator

# cuda optimization
from .cuda_kernels import CudaAttentionKernel, CudaFeedForwardKernel

# new comprehensive modules
from .benchmarking import WannBenchmarkSuite, BenchmarkResult
from .datasets import (
    load_classification_data, load_generation_data, DatasetRegistry,
    create_custom_classification_dataset, create_custom_generation_dataset
)
from .config import (
    WannGPTConfig, ConfigPresets, load_config,
    ModelConfig, EvolutionConfig, DataConfig, TrainingConfig, 
    BenchmarkConfig, LoggingConfig
)

__all__ = [
    # core architecture
    "WannGPT",
    "WannTransformerBlock", 
    
    # evolution system
    "EvolutionEngine",
    "ArchitectureGenome",
    
    # evaluation
    "SharedWeightEvaluator",
    
    # cuda kernels
    "CudaAttentionKernel",
    "CudaFeedForwardKernel",
    
    # benchmarking and analysis
    "WannBenchmarkSuite",
    "BenchmarkResult",
    
    # dataset integration
    "load_classification_data",
    "load_generation_data", 
    "DatasetRegistry",
    "create_custom_classification_dataset",
    "create_custom_generation_dataset",
    
    # configuration system
    "WannGPTConfig",
    "ConfigPresets", 
    "load_config",
    "ModelConfig",
    "EvolutionConfig", 
    "DataConfig",
    "TrainingConfig",
    "BenchmarkConfig",
    "LoggingConfig",
] 