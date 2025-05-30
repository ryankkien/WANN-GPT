#!/usr/bin/env python3
"""
script to examine and test evolved model architecture from example_hybrid_evolution.py
provides comprehensive analysis of the evolved hybrid model
"""

import torch
import json
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from wann_gpt import (
    HybridWannGPT, HeadOnlyGenome, SharedWeightEvaluator, 
    load_classification_data, WannBenchmarkSuite
)

def load_evolved_model(evolution_dir="./head_evolution_complex"):
    """load the best evolved model from evolution results"""
    
    evolution_path = Path(evolution_dir)
    
    # load best genome json
    genome_file = evolution_path / "best_head_genome.json"
    if not genome_file.exists():
        raise FileNotFoundError(f"no genome file found at {genome_file}")
    
    with open(genome_file, 'r') as f:
        genome_data = json.load(f)
    
    print("loaded genome data:")
    for key, value in genome_data.items():
        print(f"  {key}: {value}")
    
    # create genome object
    genome = HeadOnlyGenome(
        embed_dim=genome_data["embed_dim"],
        vocab_size=genome_data["vocab_size"],
        num_classes=genome_data["num_classes"]
    )
    
    # set genome properties
    genome.lm_head_sparsity = genome_data["lm_head_sparsity"]
    genome.classifier_sparsity = genome_data["classifier_sparsity"] 
    genome.fitness_scores = genome_data["fitness_scores"]
    genome.generation = genome_data["generation"]
    
    # instantiate model
    evaluator = SharedWeightEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    model = evaluator.instantiate_hybrid_from_genome(genome, model_name="gpt2")
    
    return model, genome, evaluator

def analyze_architecture(model, genome):
    """analyze the evolved architecture in detail"""
    
    print("\n" + "=" * 60)
    print("ARCHITECTURE ANALYSIS")
    print("=" * 60)
    
    # get architecture info
    arch_info = model.get_architecture_info()
    
    print("\nbasic architecture:")
    print(f"  embedding dimension: {arch_info['embed_dim']}")
    print(f"  vocabulary size: {arch_info['vocab_size']}")
    print(f"  number of layers: {arch_info['num_layers']}")
    print(f"  number of heads: {arch_info['num_heads']}")
    print(f"  max sequence length: {arch_info['max_length']}")
    print(f"  number of classes: {arch_info['num_classes']}")
    print(f"  shared weight: {arch_info['shared_weight']}")
    print(f"  total complexity: {arch_info['total_complexity']:,}")
    
    # analyze sparsity
    print("\nsparsity analysis:")
    print(f"  lm head sparsity: {genome.lm_head_sparsity:.3f}")
    print(f"  classifier sparsity: {genome.classifier_sparsity:.3f}")
    
    # calculate connection counts
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'connection_mask'):
        lm_connections = model.lm_head.connection_mask.sum().item()
        max_lm_connections = model.lm_head.connection_mask.numel()
        lm_sparsity = 1.0 - (lm_connections / max_lm_connections)
        
        print(f"  lm head connections: {lm_connections:,} / {max_lm_connections:,} ({(1-lm_sparsity)*100:.1f}% dense)")
    
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'connection_mask'):
        clf_connections = model.classifier.connection_mask.sum().item()
        max_clf_connections = model.classifier.connection_mask.numel()
        clf_sparsity = 1.0 - (clf_connections / max_clf_connections)
        
        print(f"  classifier connections: {clf_connections:,} / {max_clf_connections:,} ({(1-clf_sparsity)*100:.1f}% dense)")
    
    # fitness information
    print("\nfitness information:")
    for task, fitness in genome.fitness_scores.items():
        print(f"  {task} fitness: {fitness:.4f}")
    print(f"  generation: {genome.generation}")

def test_weight_sensitivity(model, evaluator, test_loader):
    """test model sensitivity to different shared weights"""
    
    print("\n" + "=" * 60)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # test different weights
    test_weights = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    results = {}
    
    model.eval()
    with torch.no_grad():
        for weight in test_weights:
            model.set_shared_weight(weight)
            
            correct = 0
            total = 0
            
            for batch in test_loader:
                input_ids = batch['input_ids'].to(evaluator.device)
                labels = batch['labels'].to(evaluator.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(evaluator.device)
                
                logits = model(input_ids, attention_mask, task="classification")
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            results[weight] = accuracy
            
            print(f"weight {weight:5.1f}: accuracy = {accuracy:.4f}")
    
    # find best and worst weights
    best_weight = max(results.keys(), key=results.get)
    worst_weight = min(results.keys(), key=results.get)
    
    print(f"\nbest weight: {best_weight:.1f} (accuracy: {results[best_weight]:.4f})")
    print(f"worst weight: {worst_weight:.1f} (accuracy: {results[worst_weight]:.4f})")
    print(f"mean accuracy: {sum(results.values()) / len(results):.4f}")
    print(f"accuracy range: {results[best_weight] - results[worst_weight]:.4f}")
    
    # plot weight sensitivity
    plt.figure(figsize=(10, 6))
    weights = list(results.keys())
    accuracies = list(results.values())
    
    plt.plot(weights, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('shared weight value')
    plt.ylabel('classification accuracy')
    plt.title('weight sensitivity analysis')
    plt.grid(True, alpha=0.3)
    
    # highlight best weight
    plt.axvline(x=best_weight, color='red', linestyle='--', alpha=0.7, 
                label=f'best weight: {best_weight:.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weight_sensitivity.png', dpi=300)
    plt.show()
    
    return results

def run_comprehensive_benchmark(model, test_loader, evaluator):
    """run comprehensive benchmarking suite"""
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARKING")
    print("=" * 60)
    
    # create benchmark suite
    benchmark_suite = WannBenchmarkSuite(device=evaluator.device)
    
    # run comprehensive benchmark
    benchmark_result = benchmark_suite.comprehensive_benchmark(
        model=model,
        test_loader=test_loader,
        task_type="classification",
        save_plots=True,
        output_dir="./model_analysis_results"
    )
    
    print("\nbenchmark results:")
    print(f"  accuracy: {benchmark_result.accuracy:.4f}")
    print(f"  complexity: {benchmark_result.complexity:,}")
    print(f"  robustness score: {benchmark_result.robustness_score:.4f}")
    print(f"  weight sensitivity: {benchmark_result.weight_sensitivity:.4f}")
    print(f"  ensemble improvement: {benchmark_result.ensemble_improvement:.4f}")
    print(f"  inference time: {benchmark_result.inference_time:.6f}s per sample")
    print(f"  memory usage: {benchmark_result.memory_usage:.2f}mb")
    
    return benchmark_result

def test_ensemble_performance(model, evaluator, test_loader):
    """test ensemble performance across multiple weights"""
    
    print("\n" + "=" * 60)  
    print("ENSEMBLE PERFORMANCE TESTING")
    print("=" * 60)
    
    # use evaluator's default weight samples
    weight_samples = evaluator.default_weight_samples
    print(f"testing ensemble with weights: {weight_samples}")
    
    # evaluate ensemble
    ensemble_result = evaluator.ensemble_evaluate(
        model, test_loader, weight_samples=weight_samples
    )
    
    print(f"\nensemble results:")
    print(f"  ensemble accuracy: {ensemble_result['ensemble_accuracy']:.4f}")
    print(f"  individual accuracies: {[f'{acc:.4f}' for acc in ensemble_result['individual_accuracies']]}")
    print(f"  agreement rate: {ensemble_result['agreement_rate']:.4f}")
    print(f"  prediction diversity: {ensemble_result['prediction_diversity']:.4f}")
    
    return ensemble_result

def inspect_connections(model):
    """inspect the connection patterns in detail"""
    
    print("\n" + "=" * 60)
    print("CONNECTION PATTERN INSPECTION")
    print("=" * 60)
    
    # analyze lm head connections
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'connection_mask'):
        lm_mask = model.lm_head.connection_mask.cpu().numpy()
        
        print(f"\nlm head connection pattern:")
        print(f"  shape: {lm_mask.shape}")
        print(f"  total connections: {lm_mask.sum():.0f}")
        print(f"  density: {lm_mask.mean():.3f}")
        
        # visualize connection pattern (sample if too large)
        if lm_mask.shape[0] <= 100 and lm_mask.shape[1] <= 100:
            plt.figure(figsize=(8, 6))
            plt.imshow(lm_mask, cmap='Blues', aspect='auto')
            plt.title('lm head connection pattern')
            plt.xlabel('input features')
            plt.ylabel('output features')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('lm_head_connections.png', dpi=300)
            plt.show()
        else:
            # sample for visualization
            sample_size = 100
            sample_mask = lm_mask[:sample_size, :sample_size]
            
            plt.figure(figsize=(8, 6))
            plt.imshow(sample_mask, cmap='Blues', aspect='auto')
            plt.title(f'lm head connection pattern (sample {sample_size}x{sample_size})')
            plt.xlabel('input features (sample)')
            plt.ylabel('output features (sample)')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('lm_head_connections_sample.png', dpi=300)
            plt.show()
    
    # analyze classifier connections
    if hasattr(model, 'classifier') and hasattr(model.classifier, 'connection_mask'):
        clf_mask = model.classifier.connection_mask.cpu().numpy()
        
        print(f"\nclassifier connection pattern:")
        print(f"  shape: {clf_mask.shape}")
        print(f"  total connections: {clf_mask.sum():.0f}")
        print(f"  density: {clf_mask.mean():.3f}")
        
        # visualize classifier connections
        plt.figure(figsize=(8, 6))
        plt.imshow(clf_mask, cmap='Reds', aspect='auto')
        plt.title('classifier connection pattern')
        plt.xlabel('input features')
        plt.ylabel('output classes')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('classifier_connections.png', dpi=300)
        plt.show()

def main():
    """main function to examine evolved model"""
    
    print("examining evolved hybrid model from example_hybrid_evolution.py")
    print("=" * 80)
    
    # load evolved model
    try:
        model, genome, evaluator = load_evolved_model()
        print("✓ successfully loaded evolved model")
    except FileNotFoundError as e:
        print(f"✗ error loading model: {e}")
        print("make sure you've run example_hybrid_evolution.py first")
        return
    
    # load test data
    print("\nloading test data...")
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=50257,
        max_length=1024,
        batch_size=8,
        subset_size=200  # smaller subset for testing
    )
    print(f"✓ loaded test data with {num_classes} classes")
    
    # run analysis
    analyze_architecture(model, genome)
    
    weight_results = test_weight_sensitivity(model, evaluator, test_loader)
    
    ensemble_results = test_ensemble_performance(model, evaluator, test_loader)
    
    inspect_connections(model)
    
    benchmark_results = run_comprehensive_benchmark(model, test_loader, evaluator)
    
    # summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"evolved model performance:")
    print(f"  best single weight accuracy: {max(weight_results.values()):.4f}")
    print(f"  ensemble accuracy: {ensemble_results['ensemble_accuracy']:.4f}")
    print(f"  benchmark accuracy: {benchmark_results.accuracy:.4f}")
    print(f"  model complexity: {benchmark_results.complexity:,} connections")
    print(f"  robustness score: {benchmark_results.robustness_score:.4f}")
    
    print(f"\nfiles generated:")
    print(f"  weight_sensitivity.png - weight sensitivity curve")
    print(f"  lm_head_connections.png - language model head connections")
    print(f"  classifier_connections.png - classifier head connections") 
    print(f"  ./model_analysis_results/ - comprehensive benchmark results")
    
    print(f"\nnext steps:")
    print(f"  1. examine generated plots to understand architecture")
    print(f"  2. check ./model_analysis_results/ for detailed benchmarks")
    print(f"  3. try different shared weights for your specific use case")
    print(f"  4. compare with baseline models using the benchmarking suite")

if __name__ == "__main__":
    main() 