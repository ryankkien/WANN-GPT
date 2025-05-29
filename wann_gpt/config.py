"""
configuration management for wann-gpt
centralized configuration with validation and presets
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    """configuration for wann transformer model"""
    vocab_size: int = 1000
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_length: int = 512
    dropout: float = 0.1
    num_classes: Optional[int] = None
    embedding_type: str = "random"  # "random", "onehot"
    causal: bool = True

@dataclass 
class EvolutionConfig:
    """configuration for evolutionary algorithm"""
    population_size: int = 20
    num_generations: int = 50
    mutation_rate: float = 0.8
    crossover_rate: float = 0.3
    elitism_rate: float = 0.1
    selection_strategy: str = "adaptive"  # "tournament", "nsga2", "adaptive"
    complexity_weight: float = 0.1
    max_layers: int = 8
    max_heads: int = 8
    embed_dim: int = 512
    vocab_size: int = 1000
    max_length: int = 512  # max sequence length for genomes
    num_classes: Optional[int] = None  # for classification tasks
    weight_samples: List[float] = field(default_factory=lambda: [-2.5, -1.0, -0.5, 0.5, 1.0, 2.5])
    parallel_evaluation: bool = False
    adaptive_mutation: bool = True
    fitness_stagnation_threshold: int = 10
    tournament_size: int = 3
    pareto_selection_pressure: float = 1.0

@dataclass
class DataConfig:
    """configuration for datasets"""
    dataset_name: str = "imdb"
    task_type: str = "classification"  # "classification", "generation"
    batch_size: int = 16
    max_length: int = 256
    vocab_size: int = 1000
    subset_size: Optional[int] = None
    train_split: float = 0.8
    data_dir: str = "./data"

@dataclass
class TrainingConfig:
    """configuration for evaluation and training"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    parallel_evaluation: bool = False
    weight_samples: List[float] = field(default_factory=lambda: [-2.5, -1.0, -0.5, 0.5, 1.0, 2.5])
    num_weight_samples: int = 20
    weight_sensitivity_range: tuple = (-3.0, 3.0)
    ensemble_weights: List[float] = field(default_factory=lambda: [-1.0, -0.5, 0.5, 1.0])
    save_checkpoints: bool = True
    checkpoint_interval: int = 10

@dataclass
class BenchmarkConfig:
    """configuration for benchmarking and analysis"""
    save_plots: bool = True
    output_dir: str = "./benchmark_results"
    plot_format: str = "png"  # "png", "pdf", "both"
    generate_interactive_plots: bool = True
    detailed_analysis: bool = True
    ablation_study: bool = False
    comparison_baselines: List[str] = field(default_factory=list)

@dataclass
class LoggingConfig:
    """configuration for logging and monitoring"""
    use_wandb: bool = False
    wandb_project: str = "wann-gpt"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"
    save_logs: bool = True
    log_dir: str = "./logs"
    log_interval: int = 1

@dataclass
class WannGPTConfig:
    """main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # experiment metadata
    experiment_name: str = "wann_gpt_experiment"
    description: str = "weight-agnostic gpt evolution experiment"
    seed: int = 42
    
    def __post_init__(self):
        """validate and adjust configuration after initialization"""
        self._validate()
        self._adjust_dependent_params()
    
    def _validate(self):
        """validate configuration parameters"""
        
        # model validation
        assert self.model.vocab_size > 0, "vocab_size must be positive"
        assert self.model.embed_dim > 0, "embed_dim must be positive"
        assert self.model.num_layers > 0, "num_layers must be positive"
        assert self.model.num_heads > 0, "num_heads must be positive"
        assert self.model.embed_dim % self.model.num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # evolution validation
        assert 0 < self.evolution.population_size, "population_size must be positive"
        assert 0 < self.evolution.num_generations, "num_generations must be positive"
        assert 0 <= self.evolution.mutation_rate <= 1, "mutation_rate must be in [0,1]"
        assert 0 <= self.evolution.crossover_rate <= 1, "crossover_rate must be in [0,1]"
        assert 0 <= self.evolution.elitism_rate <= 1, "elitism_rate must be in [0,1]"
        assert self.evolution.selection_strategy in ["tournament", "nsga2", "adaptive"], "invalid selection strategy"
        
        # data validation
        assert self.data.task_type in ["classification", "generation"], "invalid task type"
        assert self.data.batch_size > 0, "batch_size must be positive"
        assert 0 < self.data.train_split <= 1, "train_split must be in (0,1]"
        
        # ensure consistency between model and data configs
        if self.data.task_type == "classification" and self.model.num_classes is None:
            print("warning: classification task but num_classes not set")
    
    def _adjust_dependent_params(self):
        """adjust dependent parameters"""
        
        # sync vocab sizes
        if self.model.vocab_size != self.data.vocab_size:
            print(f"syncing vocab_size: model={self.model.vocab_size}, data={self.data.vocab_size}")
            self.model.vocab_size = self.data.vocab_size
        
        # sync evolution config with model config
        if self.evolution.embed_dim != self.model.embed_dim:
            self.evolution.embed_dim = self.model.embed_dim
        
        if self.evolution.vocab_size != self.model.vocab_size:
            self.evolution.vocab_size = self.model.vocab_size
        
        # sync max_length between model, data, and evolution configs
        if self.model.max_length != self.data.max_length:
            self.model.max_length = self.data.max_length
        
        if self.evolution.max_length != self.model.max_length:
            self.evolution.max_length = self.model.max_length
        
        # set device for parallel evaluation
        if self.training.parallel_evaluation and not torch.cuda.is_available():
            print("warning: parallel evaluation requested but cuda not available")
            self.training.parallel_evaluation = False
        
        # sync num_classes for classification tasks
        if self.data.task_type == "classification" and self.model.num_classes is not None:
            if self.evolution.num_classes != self.model.num_classes:
                self.evolution.num_classes = self.model.num_classes
    
    def to_dict(self) -> Dict[str, Any]:
        """convert config to dictionary"""
        return asdict(self)
    
    def save(self, path: Union[str, Path], format: str = "yaml"):
        """save configuration to file"""
        path = Path(path)
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"unsupported format: {format}")
        
        print(f"configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'WannGPTConfig':
        """load configuration from file"""
        path = Path(path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WannGPTConfig':
        """create config from dictionary"""
        
        # extract nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        evolution_config = EvolutionConfig(**config_dict.get('evolution', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        benchmark_config = BenchmarkConfig(**config_dict.get('benchmark', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'evolution', 'data', 'training', 'benchmark', 'logging']}
        
        return cls(
            model=model_config,
            evolution=evolution_config,
            data=data_config,
            training=training_config,
            benchmark=benchmark_config,
            logging=logging_config,
            **main_config
        )

class ConfigPresets:
    """predefined configuration presets"""
    
    @staticmethod
    def classification_small() -> WannGPTConfig:
        """small configuration for classification tasks"""
        return WannGPTConfig(
            experiment_name="classification_small",
            model=ModelConfig(
                vocab_size=500,
                embed_dim=128,
                num_layers=3,
                num_heads=4,
                max_length=128,
                num_classes=2
            ),
            evolution=EvolutionConfig(
                population_size=15,
                num_generations=30,
                max_layers=6,
                max_heads=6
            ),
            data=DataConfig(
                dataset_name="imdb",
                task_type="classification",
                batch_size=8,
                max_length=128,
                vocab_size=500,
                subset_size=1000
            )
        )
    
    @staticmethod
    def classification_large() -> WannGPTConfig:
        """large configuration for classification tasks"""
        return WannGPTConfig(
            experiment_name="classification_large",
            model=ModelConfig(
                vocab_size=2000,
                embed_dim=512,
                num_layers=8,
                num_heads=8,
                max_length=512,
                num_classes=4
            ),
            evolution=EvolutionConfig(
                population_size=50,
                num_generations=100,
                max_layers=12,
                max_heads=16
            ),
            data=DataConfig(
                dataset_name="ag_news",
                task_type="classification",
                batch_size=16,
                max_length=512,
                vocab_size=2000
            ),
            training=TrainingConfig(
                parallel_evaluation=True
            )
        )
    
    @staticmethod
    def generation_small() -> WannGPTConfig:
        """small configuration for generation tasks"""
        return WannGPTConfig(
            experiment_name="generation_small",
            model=ModelConfig(
                vocab_size=300,
                embed_dim=128,
                num_layers=3,
                num_heads=4,
                max_length=64,
                causal=True
            ),
            evolution=EvolutionConfig(
                population_size=12,
                num_generations=40,
                max_layers=6,
                max_heads=6,
                weight_samples=[-1.5, -0.8, -0.3, 0.3, 0.8, 1.5]
            ),
            data=DataConfig(
                dataset_name="tiny_stories",
                task_type="generation",
                batch_size=4,
                max_length=64,
                vocab_size=300,
                subset_size=800
            )
        )
    
    @staticmethod
    def generation_large() -> WannGPTConfig:
        """large configuration for generation tasks"""
        return WannGPTConfig(
            experiment_name="generation_large",
            model=ModelConfig(
                vocab_size=1000,
                embed_dim=256,
                num_layers=6,
                num_heads=8,
                max_length=256,
                causal=True
            ),
            evolution=EvolutionConfig(
                population_size=30,
                num_generations=80,
                max_layers=10,
                max_heads=12
            ),
            data=DataConfig(
                dataset_name="wikitext",
                task_type="generation",
                batch_size=8,
                max_length=256,
                vocab_size=1000
            ),
            training=TrainingConfig(
                parallel_evaluation=True
            )
        )
    
    @staticmethod
    def debug() -> WannGPTConfig:
        """minimal configuration for debugging"""
        return WannGPTConfig(
            experiment_name="debug",
            model=ModelConfig(
                vocab_size=100,
                embed_dim=64,
                num_layers=2,
                num_heads=2,
                max_length=32,
                num_classes=2
            ),
            evolution=EvolutionConfig(
                population_size=5,
                num_generations=3,
                max_layers=3,
                max_heads=3
            ),
            data=DataConfig(
                dataset_name="imdb",
                task_type="classification",
                batch_size=2,
                max_length=32,
                vocab_size=100,
                subset_size=50
            ),
            training=TrainingConfig(
                weight_samples=[-1.0, 1.0],
                save_checkpoints=False
            ),
            benchmark=BenchmarkConfig(
                save_plots=False,
                detailed_analysis=False
            )
        )

def load_config(config_path: Optional[Union[str, Path]] = None,
               preset: Optional[str] = None,
               overrides: Optional[Dict[str, Any]] = None) -> WannGPTConfig:
    """load configuration with optional overrides"""
    
    if preset:
        # load preset configuration
        preset_map = {
            "classification_small": ConfigPresets.classification_small,
            "classification_large": ConfigPresets.classification_large,
            "generation_small": ConfigPresets.generation_small,
            "generation_large": ConfigPresets.generation_large,
            "debug": ConfigPresets.debug
        }
        
        if preset not in preset_map:
            raise ValueError(f"unknown preset: {preset}. available: {list(preset_map.keys())}")
        
        config = preset_map[preset]()
        
    elif config_path:
        # load from file
        config = WannGPTConfig.load(config_path)
        
    else:
        # use default
        config = WannGPTConfig()
    
    # apply overrides
    if overrides:
        config_dict = config.to_dict()
        _deep_update(config_dict, overrides)
        config = WannGPTConfig.from_dict(config_dict)
    
    return config

def _deep_update(base_dict: Dict, update_dict: Dict):
    """recursively update nested dictionary"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value 