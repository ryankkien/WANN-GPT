"""
comprehensive benchmark example for wann-gpt
demonstrates full system capabilities including:
- real dataset integration
- comprehensive benchmarking
- ablation studies
- architecture comparison
- configuration management
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from wann_gpt.config import load_config, ConfigPresets
from wann_gpt.datasets import load_classification_data, load_generation_data, DatasetRegistry
from wann_gpt.evolution import EvolutionEngine
from wann_gpt.evaluation import SharedWeightEvaluator
from wann_gpt.benchmarking import WannBenchmarkSuite
from wann_gpt.architecture import WannGPT
from wann_gpt.evolution.genome import ArchitectureGenome

def run_classification_benchmark(config, output_dir: str = "./classification_results"):
    """run comprehensive classification benchmark"""
    
    print("=" * 60)
    print("CLASSIFICATION BENCHMARK")
    print("=" * 60)
    
    # load dataset
    print(f"loading {config.data.dataset_name} dataset...")
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name=config.data.dataset_name,
        vocab_size=config.data.vocab_size,
        max_length=config.data.max_length,
        subset_size=config.data.subset_size
    )
    
    # update num_classes in config
    config.model.num_classes = num_classes
    
    # create evolution engine
    print("initializing evolution engine...")
    evaluator = SharedWeightEvaluator(device=config.training.device)
    engine = EvolutionEngine(config.evolution, evaluator, save_dir=output_dir)
    
    # run evolution
    print("starting evolution...")
    best_genome = engine.evolve(
        dataloader=train_loader,
        task_type="classification",
        initialization_strategy="mixed",
        log_wandb=config.logging.use_wandb
    )
    
    print(f"evolution completed! best fitness: {best_genome.get_fitness('classification'):.4f}")
    
    # comprehensive benchmarking
    print("\nstarting comprehensive benchmarking...")
    benchmark_suite = WannBenchmarkSuite(device=config.training.device)
    
    # instantiate best model
    best_model = evaluator.instantiate_from_genome(best_genome)
    
    # run full benchmark suite
    benchmark_result = benchmark_suite.comprehensive_benchmark(
        model=best_model,
        test_loader=test_loader,
        task_type="classification",
        save_plots=config.benchmark.save_plots,
        output_dir=f"{output_dir}/benchmarks"
    )
    
    print(f"\nbenchmark results:")
    print(f"  accuracy: {benchmark_result.accuracy:.4f}")
    print(f"  complexity: {benchmark_result.complexity}")
    print(f"  robustness score: {benchmark_result.robustness_score:.4f}")
    print(f"  weight sensitivity: {benchmark_result.weight_sensitivity:.4f}")
    print(f"  ensemble improvement: {benchmark_result.ensemble_improvement:.4f}")
    print(f"  inference time: {benchmark_result.inference_time:.6f}s per sample")
    
    # ablation study
    if config.benchmark.ablation_study:
        print("\nperforming ablation study...")
        ablation_results = benchmark_suite.ablation_study(
            model=best_model,
            genome=best_genome,
            test_loader=test_loader,
            task_type="classification",
            output_dir=f"{output_dir}/ablation"
        )
        
        print("ablation study completed!")
        print(f"  baseline performance: {ablation_results['baseline']:.4f}")
        print(f"  zero weight performance: {ablation_results['zero_weight']:.4f}")
    
    # create baseline comparisons
    print("\ncreating baseline comparisons...")
    baselines = create_baseline_models(config, num_classes)
    
    model_list = [(best_model, "Evolved WANN")]
    model_list.extend(baselines)
    
    comparison_results = benchmark_suite.compare_architectures(
        models_and_names=model_list,
        test_loader=test_loader,
        task_type="classification",
        output_dir=f"{output_dir}/comparison"
    )
    
    print("architecture comparison completed!")
    for name, result in comparison_results:
        print(f"  {name}: accuracy={result.accuracy:.4f}, complexity={result.complexity}")
    
    return best_genome, benchmark_result

def run_generation_benchmark(config, output_dir: str = "./generation_results"):
    """run comprehensive generation benchmark"""
    
    print("=" * 60)
    print("GENERATION BENCHMARK")
    print("=" * 60)
    
    # load dataset
    print(f"loading {config.data.dataset_name} dataset...")
    train_loader, test_loader = load_generation_data(
        dataset_name=config.data.dataset_name,
        vocab_size=config.data.vocab_size,
        max_length=config.data.max_length,
        subset_size=config.data.subset_size
    )
    
    # create evolution engine
    print("initializing evolution engine...")
    evaluator = SharedWeightEvaluator(device=config.training.device)
    engine = EvolutionEngine(config.evolution, evaluator, save_dir=output_dir)
    
    # run evolution
    print("starting evolution...")
    best_genome = engine.evolve(
        dataloader=train_loader,
        task_type="generation",
        initialization_strategy="mixed",
        log_wandb=config.logging.use_wandb
    )
    
    print(f"evolution completed! best fitness: {best_genome.get_fitness('generation'):.4f}")
    
    # comprehensive benchmarking
    print("\nstarting comprehensive benchmarking...")
    benchmark_suite = WannBenchmarkSuite(device=config.training.device)
    
    # instantiate best model
    best_model = evaluator.instantiate_from_genome(best_genome)
    
    # run full benchmark suite
    benchmark_result = benchmark_suite.comprehensive_benchmark(
        model=best_model,
        test_loader=test_loader,
        task_type="generation",
        save_plots=config.benchmark.save_plots,
        output_dir=f"{output_dir}/benchmarks"
    )
    
    print(f"\nbenchmark results:")
    print(f"  perplexity: {benchmark_result.perplexity:.4f}")
    print(f"  complexity: {benchmark_result.complexity}")
    print(f"  robustness score: {benchmark_result.robustness_score:.4f}")
    print(f"  weight sensitivity: {benchmark_result.weight_sensitivity:.4f}")
    print(f"  inference time: {benchmark_result.inference_time:.6f}s per sample")
    
    # text generation samples
    print("\ngenerating sample text...")
    generate_text_samples(best_model, config.data.vocab_size, output_dir)
    
    return best_genome, benchmark_result

def create_baseline_models(config, num_classes=None):
    """create baseline models for comparison"""
    
    baselines = []
    
    # simple 1-layer model
    simple_genome = ArchitectureGenome.create_simple(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_layers=1,
        num_classes=num_classes
    )
    evaluator = SharedWeightEvaluator(device=config.training.device)
    simple_model = evaluator.instantiate_from_genome(simple_genome)
    baselines.append((simple_model, "Simple 1-Layer"))
    
    # standard transformer (random weights)
    standard_genome = ArchitectureGenome.create_standard(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_classes=num_classes
    )
    standard_model = evaluator.instantiate_from_genome(standard_genome)
    baselines.append((standard_model, "Standard Transformer"))
    
    return baselines

def generate_text_samples(model, vocab_size, output_dir):
    """generate and save text samples"""
    
    model.eval()
    samples_dir = Path(output_dir) / "text_samples"
    samples_dir.mkdir(exist_ok=True)
    
    weight_values = [-1.0, 0.0, 1.0]
    
    for weight in weight_values:
        model.set_shared_weight(weight)
        
        # create random prompt
        prompt = torch.randint(0, vocab_size, (1, 5)).to(model.device)
        
        # generate text
        generated = model.generate(
            prompt,
            max_new_tokens=20,
            temperature=1.0,
            top_k=50
        )
        
        # save sample
        with open(samples_dir / f"sample_weight_{weight}.txt", 'w') as f:
            f.write(f"prompt tokens: {prompt[0].tolist()}\n")
            f.write(f"generated tokens: {generated[0, 5:].tolist()}\n")
        
        print(f"  weight {weight:+.1f}: generated {len(generated[0]) - 5} tokens")

def demonstrate_config_system():
    """demonstrate configuration system features"""
    
    print("=" * 60)
    print("CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    
    # show available presets
    presets = ["classification_small", "classification_large", "generation_small", "generation_large", "debug"]
    print(f"available presets: {presets}")
    
    # load and save configs
    for preset in presets[:2]:  # demo first 2 presets
        print(f"\nloading preset: {preset}")
        config = load_config(preset=preset)
        
        # save configuration
        config_path = f"./configs/{preset}.yaml"
        Path("./configs").mkdir(exist_ok=True)
        config.save(config_path)
        print(f"saved to: {config_path}")
    
    # demonstrate overrides
    print("\ndemonstrating config overrides...")
    overrides = {
        "evolution": {"population_size": 10, "num_generations": 5},
        "data": {"subset_size": 100}
    }
    
    config = load_config(preset="debug", overrides=overrides)
    print(f"overridden population size: {config.evolution.population_size}")
    print(f"overridden subset size: {config.data.subset_size}")

def demonstrate_dataset_system():
    """demonstrate dataset integration features"""
    
    print("=" * 60)
    print("DATASET SYSTEM DEMO")  
    print("=" * 60)
    
    # show available datasets
    datasets = DatasetRegistry.list_datasets()
    print(f"available datasets: {datasets}")
    
    # load sample datasets
    print("\nloading sample classification dataset...")
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=100,
        max_length=32,
        subset_size=50
    )
    print(f"loaded imdb: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test, {num_classes} classes")
    
    print("\nloading sample generation dataset...")
    train_loader, test_loader = load_generation_data(
        dataset_name="tiny_stories",
        vocab_size=100,
        max_length=32,
        subset_size=50
    )
    print(f"loaded tiny_stories: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

def main():
    """main function"""
    
    parser = argparse.ArgumentParser(description="comprehensive wann-gpt benchmark")
    parser.add_argument("--task", choices=["classification", "generation", "both", "demo"], 
                       default="both", help="task to run")
    parser.add_argument("--preset", type=str, help="configuration preset to use")
    parser.add_argument("--config", type=str, help="configuration file path")
    parser.add_argument("--output", type=str, default="./results", help="output directory")
    parser.add_argument("--demo-config", action="store_true", help="demonstrate config system")
    parser.add_argument("--demo-datasets", action="store_true", help="demonstrate dataset system")
    
    args = parser.parse_args()
    
    # create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # demonstrations
    if args.demo_config:
        demonstrate_config_system()
        return
    
    if args.demo_datasets:
        demonstrate_dataset_system()
        return
    
    if args.task == "demo":
        demonstrate_config_system()
        demonstrate_dataset_system()
        return
    
    # set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # load configuration
    if args.preset:
        config = load_config(preset=args.preset)
    elif args.config:
        config = load_config(config_path=args.config)
    else:
        # use appropriate default based on task
        if args.task == "classification":
            config = load_config(preset="classification_small")
        elif args.task == "generation":
            config = load_config(preset="generation_small")
        else:
            config = load_config(preset="debug")
    
    print(f"using configuration: {config.experiment_name}")
    print(f"task: {args.task}")
    print(f"device: {config.training.device}")
    
    results = {}
    
    # run classification benchmark
    if args.task in ["classification", "both"]:
        classification_config = config
        classification_config.data.task_type = "classification"
        
        best_genome, benchmark_result = run_classification_benchmark(
            classification_config, 
            output_dir=str(output_dir / "classification")
        )
        
        results["classification"] = {
            "best_genome": best_genome,
            "benchmark_result": benchmark_result
        }
    
    # run generation benchmark
    if args.task in ["generation", "both"]:
        generation_config = config
        generation_config.data.task_type = "generation"
        generation_config.model.num_classes = None
        
        best_genome, benchmark_result = run_generation_benchmark(
            generation_config,
            output_dir=str(output_dir / "generation")
        )
        
        results["generation"] = {
            "best_genome": best_genome,
            "benchmark_result": benchmark_result
        }
    
    # final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    if "classification" in results:
        cls_result = results["classification"]["benchmark_result"]
        print(f"classification:")
        print(f"  accuracy: {cls_result.accuracy:.4f}")
        print(f"  complexity: {cls_result.complexity}")
        print(f"  robustness: {cls_result.robustness_score:.4f}")
    
    if "generation" in results:
        gen_result = results["generation"]["benchmark_result"]
        print(f"generation:")
        print(f"  perplexity: {gen_result.perplexity:.4f}")
        print(f"  complexity: {gen_result.complexity}")
        print(f"  robustness: {gen_result.robustness_score:.4f}")
    
    print(f"\nall results saved to: {output_dir}")
    print("benchmark complete!")

if __name__ == "__main__":
    main() 