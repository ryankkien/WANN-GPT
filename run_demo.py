#!/usr/bin/env python3
"""
Simple demonstration script for WANN-GPT
Shows basic usage and capabilities
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from wann_gpt import *

def demo_config_system():
    """demonstrate configuration system"""
    print("=" * 50)
    print("Configuration System Demo")
    print("=" * 50)
    
    # load debug preset
    config = load_config(preset="debug")
    print(f"loaded preset: {config.experiment_name}")
    print(f"population size: {config.evolution.population_size}")
    print(f"vocab size: {config.data.vocab_size}")
    
    # demonstrate overrides
    overrides = {
        "evolution": {"population_size": 3},
        "data": {"subset_size": 20}
    }
    
    config = load_config(preset="debug", overrides=overrides)
    print(f"\nwith overrides:")
    print(f"population size: {config.evolution.population_size}")
    print(f"subset size: {config.data.subset_size}")

def demo_dataset_system():
    """demonstrate dataset integration"""
    print("\n" + "=" * 50)
    print("Dataset System Demo")
    print("=" * 50)
    
    # show available datasets
    datasets = DatasetRegistry.list_datasets()
    print(f"available datasets: {datasets}")
    
    # load debug config to get consistent parameters
    config = load_config(preset="debug")
    
    # load small classification dataset
    print("\nloading classification dataset...")
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=config.data.vocab_size,
        max_length=config.data.max_length,
        subset_size=20,
        batch_size=config.data.batch_size
    )
    
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"test samples: {len(test_loader.dataset)}")
    print(f"num classes: {num_classes}")
    
    # show sample batch
    batch = next(iter(train_loader))
    print(f"\nsample batch shape: {batch['input_ids'].shape}")
    print(f"sample labels: {batch['labels'][:5].tolist()}")

def demo_evolution():
    """demonstrate evolution system"""
    print("\n" + "=" * 50)
    print("Evolution System Demo")
    print("=" * 50)
    
    # use minimal config for demo
    config = load_config(preset="debug")
    
    # force CPU for debugging
    config.training.device = "cpu"
    
    # load tiny dataset
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=config.data.vocab_size,
        max_length=config.data.max_length,
        subset_size=config.data.subset_size,
        batch_size=config.data.batch_size
    )
    
    # set num_classes in both model and evolution configs
    config.model.num_classes = num_classes
    config.evolution.num_classes = num_classes
    
    # create evolution components
    evaluator = SharedWeightEvaluator(device=config.training.device)
    engine = EvolutionEngine(config.evolution, evaluator, save_dir="./demo_results")
    
    print(f"running evolution with {config.evolution.population_size} individuals for {config.evolution.num_generations} generations...")
    
    # run evolution
    best_genome = engine.evolve(
        dataloader=train_loader,
        task_type="classification",
        initialization_strategy="mixed",
        log_wandb=True
    )
    
    print(f"evolution completed!")
    print(f"best fitness: {best_genome.get_fitness('classification'):.4f}")
    print(f"complexity: {best_genome.calculate_complexity()}")

def demo_benchmarking():
    """demonstrate benchmarking system"""
    print("\n" + "=" * 50)
    print("Benchmarking System Demo")
    print("=" * 50)
    
    # create simple model for testing
    config = load_config(preset="debug")
    
    # force CPU for debugging
    config.training.device = "cpu"
    
    # create test dataset
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=config.data.vocab_size,
        max_length=config.data.max_length,
        subset_size=20,
        batch_size=config.data.batch_size
    )
    
    # set num_classes in both model and evolution configs
    config.model.num_classes = num_classes
    config.evolution.num_classes = num_classes
    
    # create simple model
    genome = ArchitectureGenome.create_simple(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_layers=2,
        num_classes=num_classes
    )
    
    # properly set max_length from config
    genome.max_length = config.model.max_length
    
    evaluator = SharedWeightEvaluator(device=config.training.device)
    model = evaluator.instantiate_from_genome(genome)
    
    print(f"created model with {model.get_total_complexity()} connections")
    
    # run basic evaluation
    result = evaluator.evaluate_classification(model, test_loader)
    print(f"accuracy: {result.mean_performance:.4f} ± {result.std_performance:.4f}")
    
    # run benchmark suite
    benchmark_suite = WannBenchmarkSuite(device=config.training.device)
    
    print("\nrunning comprehensive benchmark...")
    benchmark_result = benchmark_suite.comprehensive_benchmark(
        model=model,
        test_loader=test_loader,
        task_type="classification",
        save_plots=False,  # disable plots for demo
        output_dir="./demo_benchmark"
    )
    
    print(f"benchmark completed!")
    print(f"  accuracy: {benchmark_result.accuracy:.4f}")
    print(f"  robustness: {benchmark_result.robustness_score:.4f}")
    print(f"  weight sensitivity: {benchmark_result.weight_sensitivity:.4f}")

def main():
    """run all demonstrations"""
    print("WANN-GPT System Demonstration")
    print("=" * 60)
    
    try:
        # set device to CPU for debugging
        device = "cpu"  # Force CPU for debugging
        print(f"using device: {device}")
        
        # run demonstrations
        demo_config_system()
        demo_dataset_system()
        demo_evolution()
        demo_benchmarking()
        
        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")
        print("Check ./demo_results/ and ./demo_benchmark/ for outputs")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This might be due to missing dependencies or system limitations")
        print("Please check requirements.txt and ensure all dependencies are installed")

if __name__ == "__main__":
    main() 