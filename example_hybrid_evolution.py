#!/usr/bin/env python3
"""
full-scale hybrid gpt-2 evolution with comprehensive settings
production-ready script for evolving sparse output heads on frozen gpt-2
"""

import torch
import argparse
import json
import time
from pathlib import Path
from wann_gpt import (
    HybridWannGPT, HeadOnlyGenome, HeadOnlyEvolutionEngine, 
    SharedWeightEvaluator, load_classification_data, 
    WannGPTConfig, ConfigPresets, WannBenchmarkSuite
)

def create_full_config():
    """create comprehensive configuration for full evolution"""
    
    config = ConfigPresets.classification_small()
    
    # scale up for full evolution
    config.model.embed_dim = 768  # gpt-2 embedding dimension
    config.model.vocab_size = 50257  # gpt-2 vocabulary size
    config.model.num_layers = 12  # gpt-2 layers  
    config.model.num_heads = 12  # gpt-2 heads
    config.model.max_length = 512  # reasonable sequence length for efficiency
    
    # comprehensive evolution settings
    config.evolution.embed_dim = 768
    config.evolution.vocab_size = 50257
    config.evolution.population_size = 50  # large population for diversity
    config.evolution.num_generations = 50  # substantial evolution time
    config.evolution.complexity_weight = 0.0005  # balance performance vs efficiency
    config.evolution.mutation_rate = 0.15  # moderate mutation
    config.evolution.crossover_rate = 0.7  # high crossover for mixing
    config.evolution.elite_size = 5  # preserve best individuals
    config.evolution.tournament_size = 3  # selection pressure
    config.evolution.fitness_stagnation_threshold = 15  # early stopping
    
    # data settings for robust evaluation
    config.data.batch_size = 16  # efficient batch size
    config.data.max_length = 512
    
    return config

def setup_comprehensive_evaluation(config, dataset_name="imdb", subset_size=5000):
    """setup comprehensive training and evaluation data"""
    
    print(f"setting up comprehensive evaluation...")
    print(f"dataset: {dataset_name}")
    print(f"subset size: {subset_size:,} samples")
    
    # load larger dataset for robust evaluation
    train_loader, test_loader, num_classes = load_classification_data(
        dataset_name=dataset_name,
        vocab_size=config.model.vocab_size,
        max_length=config.model.max_length,
        batch_size=config.data.batch_size,
        subset_size=subset_size
    )
    
    config.model.num_classes = num_classes
    print(f"loaded dataset: {num_classes} classes, {subset_size:,} samples")
    
    return train_loader, test_loader, num_classes

def run_initial_baseline_tests(config, train_loader, test_loader):
    """run baseline tests to understand task difficulty"""
    
    print(f"\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    
    evaluator = SharedWeightEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # test random baseline
    print(f"\ntesting random baseline...")
    random_genome = HeadOnlyGenome(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_classes=config.model.num_classes
    )
    # create random sparsity and connection patterns
    random_genome.lm_head_sparsity = 0.5
    random_genome.classifier_sparsity = 0.5
    random_genome.randomize_connections()
    
    random_model = evaluator.instantiate_hybrid_from_genome(random_genome, model_name="gpt2")
    random_result = evaluator.evaluate_classification(random_model, test_loader)
    
    print(f"random baseline accuracy: {random_result.mean_performance:.4f} ± {random_result.std_performance:.4f}")
    
    # test dense baseline
    print(f"\ntesting dense baseline...")
    dense_genome = HeadOnlyGenome.create_dense(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_classes=config.model.num_classes
    )
    
    dense_model = evaluator.instantiate_hybrid_from_genome(dense_genome, model_name="gpt2")
    dense_result = evaluator.evaluate_classification(dense_model, test_loader)
    
    print(f"dense baseline accuracy: {dense_result.mean_performance:.4f} ± {dense_result.std_performance:.4f}")
    
    # test sparse baseline
    print(f"\ntesting sparse baseline...")
    sparse_genome = HeadOnlyGenome.create_sparse(
        embed_dim=config.model.embed_dim,
        vocab_size=config.model.vocab_size,
        num_classes=config.model.num_classes,
        sparsity=0.8
    )
    
    sparse_model = evaluator.instantiate_hybrid_from_genome(sparse_genome, model_name="gpt2")
    sparse_result = evaluator.evaluate_classification(sparse_model, test_loader)
    
    print(f"sparse baseline accuracy: {sparse_result.mean_performance:.4f} ± {sparse_result.std_performance:.4f}")
    
    print(f"\nbaseline summary:")
    print(f"  task appears {'easy' if random_result.mean_performance > 0.7 else 'moderate' if random_result.mean_performance > 0.5 else 'challenging'}")
    print(f"  evolution target: >{max(random_result.mean_performance, dense_result.mean_performance, sparse_result.mean_performance):.3f}")
    
    return {
        'random_accuracy': random_result.mean_performance,
        'dense_accuracy': dense_result.mean_performance,
        'sparse_accuracy': sparse_result.mean_performance,
        'target_accuracy': max(random_result.mean_performance, dense_result.mean_performance, sparse_result.mean_performance)
    }

def run_full_scale_evolution(config, train_loader, test_loader, save_dir="./full_hybrid_evolution"):
    """run comprehensive evolution with advanced features"""
    
    print(f"\n" + "=" * 60)
    print("FULL-SCALE EVOLUTION")
    print("=" * 60)
    
    print(f"evolution configuration:")
    print(f"  population size: {config.evolution.population_size}")
    print(f"  generations: {config.evolution.num_generations}")
    print(f"  mutation rate: {config.evolution.mutation_rate}")
    print(f"  crossover rate: {config.evolution.crossover_rate}")
    print(f"  elite size: {config.evolution.elite_size}")
    print(f"  tournament size: {config.evolution.tournament_size}")
    print(f"  complexity weight: {config.evolution.complexity_weight}")
    print(f"  early stopping: {config.evolution.fitness_stagnation_threshold} generations")
    
    # create evaluator with comprehensive weight sampling
    evaluator = SharedWeightEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    evaluator.default_weight_samples = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    
    print(f"  weight samples: {evaluator.default_weight_samples}")
    print(f"  device: {evaluator.device}")
    
    # create evolution engine
    evolution_engine = HeadOnlyEvolutionEngine(
        config=config.evolution,
        evaluator=evaluator,
        save_dir=save_dir,
        model_name="gpt2"
    )
    
    # run evolution with timing
    start_time = time.time()
    print(f"\nstarting full-scale evolution...")
    
    best_genome = evolution_engine.evolve(
        train_loader,
        task_type="classification",
        initialization_strategy="mixed"  # diverse initialization
    )
    
    evolution_time = time.time() - start_time
    
    print(f"\nevolution completed!")
    print(f"  time taken: {evolution_time/60:.1f} minutes")
    print(f"  final generation: {best_genome.generation if hasattr(best_genome, 'generation') else 'unknown'}")
    print(f"  best fitness: {best_genome.get_fitness('classification'):.4f}")
    print(f"  best complexity: {best_genome.calculate_complexity():,}")
    
    return best_genome, evolution_engine

def comprehensive_model_analysis(best_genome, evaluator, test_loader, config):
    """perform comprehensive analysis of the best evolved model"""
    
    print(f"\n" + "=" * 60)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 60)
    
    # instantiate best model
    best_model = evaluator.instantiate_hybrid_from_genome(best_genome, model_name="gpt2")
    
    # detailed weight sensitivity analysis
    print(f"\ndetailed weight sensitivity analysis...")
    weight_range = (-4.0, 4.0)
    num_weights = 17
    test_weights = torch.linspace(weight_range[0], weight_range[1], num_weights).tolist()
    
    weight_results = {}
    best_model.eval()
    
    with torch.no_grad():
        for weight in test_weights:
            best_model.set_shared_weight(weight)
            
            correct = 0
            total = 0
            
            for batch in test_loader:
                input_ids = batch['input_ids'].to(evaluator.device)
                labels = batch['labels'].to(evaluator.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(evaluator.device)
                
                logits = best_model(input_ids, attention_mask, task="classification")
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            accuracy = correct / total if total > 0 else 0.0
            weight_results[weight] = accuracy
            
            if weight % 1.0 == 0:  # print every integer weight
                print(f"  weight {weight:5.1f}: accuracy = {accuracy:.4f}")
    
    # find optimal performance
    best_weight = max(weight_results.keys(), key=weight_results.get)
    best_accuracy = weight_results[best_weight]
    mean_accuracy = sum(weight_results.values()) / len(weight_results)
    accuracy_std = torch.tensor(list(weight_results.values())).std().item()
    
    print(f"\nweight sensitivity summary:")
    print(f"  best weight: {best_weight:.1f} (accuracy: {best_accuracy:.4f})")
    print(f"  mean accuracy: {mean_accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"  robustness range: {max(weight_results.values()) - min(weight_results.values()):.4f}")
    
    # ensemble evaluation
    print(f"\nensemble evaluation...")
    ensemble_weights = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ensemble_result = evaluator.ensemble_evaluate(best_model, test_loader, weight_samples=ensemble_weights)
    
    print(f"  ensemble accuracy: {ensemble_result['ensemble_accuracy']:.4f}")
    print(f"  individual accuracies: {[f'{acc:.4f}' for acc in ensemble_result['individual_accuracies']]}")
    print(f"  ensemble improvement: +{ensemble_result['ensemble_accuracy'] - best_accuracy:.4f}")
    
    # complexity analysis
    total_connections = best_genome.calculate_complexity()
    max_lm_connections = config.model.embed_dim * config.model.vocab_size
    max_clf_connections = config.model.embed_dim * config.model.num_classes
    max_total = max_lm_connections + max_clf_connections
    sparsity_ratio = 1.0 - (total_connections / max_total)
    
    print(f"\ncomplexity analysis:")
    print(f"  active connections: {total_connections:,}")
    print(f"  maximum possible: {max_total:,}")
    print(f"  sparsity ratio: {sparsity_ratio:.1%}")
    print(f"  parameter efficiency: {total_connections/1000000:.1f}M evolved parameters")
    
    # architecture details
    arch_info = best_model.get_architecture_info()
    print(f"\nfinal architecture:")
    print(f"  model type: {arch_info.get('architecture_type', 'hybrid_wann_gpt2')}")
    print(f"  embedding dim: {arch_info['embed_dim']}")
    print(f"  vocabulary: {arch_info['vocab_size']:,}")
    print(f"  sequence length: {arch_info['max_length']}")
    print(f"  transformer layers: {arch_info['num_layers']}")
    print(f"  attention heads: {arch_info['num_heads']}")
    print(f"  output classes: {arch_info['num_classes']}")
    print(f"  lm head sparsity: {best_genome.lm_head_sparsity:.3f}")
    print(f"  classifier sparsity: {best_genome.classifier_sparsity:.3f}")
    
    return {
        'best_model': best_model,
        'weight_results': weight_results,
        'ensemble_result': ensemble_result,
        'best_weight': best_weight,
        'best_accuracy': best_accuracy,
        'sparsity_ratio': sparsity_ratio,
        'total_connections': total_connections
    }

def run_comprehensive_benchmarking(best_model, test_loader, analysis_results):
    """run comprehensive benchmarking suite"""
    
    print(f"\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARKING")
    print("=" * 60)
    
    # set optimal weight
    best_model.set_shared_weight(analysis_results['best_weight'])
    
    # create benchmark suite
    benchmark_suite = WannBenchmarkSuite(device=best_model.device if hasattr(best_model, 'device') else 'cpu')
    
    # run comprehensive benchmark
    benchmark_result = benchmark_suite.comprehensive_benchmark(
        model=best_model,
        test_loader=test_loader,
        task_type="classification",
        save_plots=True,
        output_dir="./full_evolution_benchmarks"
    )
    
    print(f"\nbenchmark results:")
    print(f"  accuracy: {benchmark_result.accuracy:.4f}")
    print(f"  complexity: {benchmark_result.complexity:,}")
    print(f"  robustness score: {benchmark_result.robustness_score:.4f}")
    print(f"  weight sensitivity: {benchmark_result.weight_sensitivity:.4f}")
    print(f"  ensemble improvement: {benchmark_result.ensemble_improvement:.4f}")
    print(f"  inference time: {benchmark_result.inference_time:.6f}s per sample")
    print(f"  memory usage: {benchmark_result.memory_usage:.2f}mb")
    
    return benchmark_result

def save_complete_production_model(best_model, best_genome, analysis_results, benchmark_result, config, save_dir):
    """save complete production-ready model with all metadata"""
    
    print(f"\n" + "=" * 60)
    print("SAVING PRODUCTION MODEL")
    print("=" * 60)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # set optimal weight
    best_model.set_shared_weight(analysis_results['best_weight'])
    
    # comprehensive model save
    model_path = save_path / "production_hybrid_model.pt"
    
    save_data = {
        # model components
        'model_state_dict': best_model.state_dict(),
        'model_name': "gpt2",
        
        # genome information
        'genome_data': {
            'embed_dim': best_genome.embed_dim,
            'vocab_size': best_genome.vocab_size,
            'num_classes': best_genome.num_classes,
            'lm_head_sparsity': best_genome.lm_head_sparsity,
            'classifier_sparsity': best_genome.classifier_sparsity,
            'fitness_scores': best_genome.fitness_scores,
            'generation': getattr(best_genome, 'generation', 0),
            'complexity': best_genome.calculate_complexity()
        },
        
        # architecture details
        'architecture_info': best_model.get_architecture_info(),
        
        # performance results
        'performance': {
            'optimal_weight': analysis_results['best_weight'],
            'best_accuracy': analysis_results['best_accuracy'],
            'ensemble_accuracy': analysis_results['ensemble_result']['ensemble_accuracy'],
            'weight_results': analysis_results['weight_results'],
            'benchmark_results': {
                'accuracy': benchmark_result.accuracy,
                'robustness_score': benchmark_result.robustness_score,
                'weight_sensitivity': benchmark_result.weight_sensitivity,
                'inference_time': benchmark_result.inference_time,
                'memory_usage': benchmark_result.memory_usage
            }
        },
        
        # efficiency metrics
        'efficiency': {
            'total_connections': analysis_results['total_connections'],
            'sparsity_ratio': analysis_results['sparsity_ratio'],
            'gpt2_parameters': 124000000,  # ~124M
            'evolved_parameters': analysis_results['total_connections'],
            'parameter_efficiency': analysis_results['total_connections'] / (124000000 + analysis_results['total_connections'])
        },
        
        # configuration used
        'training_config': {
            'population_size': config.evolution.population_size,
            'num_generations': config.evolution.num_generations,
            'mutation_rate': config.evolution.mutation_rate,
            'crossover_rate': config.evolution.crossover_rate,
            'complexity_weight': config.evolution.complexity_weight
        },
        
        # metadata
        'metadata': {
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_version': torch.__version__,
            'device_used': str(best_model.device if hasattr(best_model, 'device') else 'cpu'),
            'model_type': 'hybrid_wann_gpt2_production'
        }
    }
    
    torch.save(save_data, model_path)
    
    # save readable summary
    summary_path = save_path / "model_summary.json"
    summary = {
        'model_overview': {
            'type': 'Hybrid WANN-GPT2 with Evolved Sparse Heads',
            'backbone': 'GPT-2 (124M parameters, frozen)',
            'evolved_components': 'Sparse output heads for classification',
            'total_size_mb': (124000000 + analysis_results['total_connections']) * 4 / 1024 / 1024  # approximate
        },
        'performance_summary': {
            'best_accuracy': f"{analysis_results['best_accuracy']:.4f}",
            'ensemble_accuracy': f"{analysis_results['ensemble_result']['ensemble_accuracy']:.4f}",
            'optimal_weight': f"{analysis_results['best_weight']:.2f}",
            'robustness_score': f"{benchmark_result.robustness_score:.4f}"
        },
        'efficiency_summary': {
            'sparsity_ratio': f"{analysis_results['sparsity_ratio']:.1%}",
            'evolved_parameters': f"{analysis_results['total_connections']/1000000:.1f}M",
            'inference_time_per_sample': f"{benchmark_result.inference_time:.6f}s",
            'memory_usage': f"{benchmark_result.memory_usage:.1f}MB"
        },
        'deployment_info': {
            'load_command': 'CompleteModelLoader("production_hybrid_model.pt")',
            'optimal_weight': analysis_results['best_weight'],
            'recommended_batch_size': config.data.batch_size,
            'max_sequence_length': config.model.max_length
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ production model saved:")
    print(f"  model file: {model_path}")
    print(f"  summary: {summary_path}")
    print(f"  benchmarks: ./full_evolution_benchmarks/")
    print(f"  model size: ~{(124000000 + analysis_results['total_connections']) * 4 / 1024 / 1024:.0f}MB")
    print(f"  ready for deployment!")

def main():
    """main function for full-scale hybrid evolution"""
    
    parser = argparse.ArgumentParser(description="Full-scale hybrid WANN-GPT evolution")
    parser.add_argument("--dataset", default="imdb", help="Dataset to use (imdb, ag_news, etc.)")
    parser.add_argument("--subset-size", type=int, default=5000, help="Dataset subset size")
    parser.add_argument("--population", type=int, default=50, help="Population size")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--save-dir", default="./full_hybrid_evolution", help="Save directory")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline tests")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip comprehensive benchmarks")
    
    args = parser.parse_args()
    
    print("full-scale hybrid wann-gpt evolution")
    print("=" * 80)
    
    # create configuration
    config = create_full_config()
    config.evolution.population_size = args.population
    config.evolution.num_generations = args.generations
    
    print(f"configuration:")
    print(f"  dataset: {args.dataset}")
    print(f"  subset size: {args.subset_size:,}")
    print(f"  population: {args.population}")
    print(f"  generations: {args.generations}")
    print(f"  save directory: {args.save_dir}")
    
    # setup data
    train_loader, test_loader, num_classes = setup_comprehensive_evaluation(
        config, args.dataset, args.subset_size
    )
    
    # baseline tests
    if not args.skip_baselines:
        baseline_results = run_initial_baseline_tests(config, train_loader, test_loader)
    
    # run evolution
    best_genome, evolution_engine = run_full_scale_evolution(
        config, train_loader, test_loader, args.save_dir
    )
    
    # comprehensive analysis
    evaluator = SharedWeightEvaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    analysis_results = comprehensive_model_analysis(best_genome, evaluator, test_loader, config)
    
    # benchmarking
    if not args.skip_benchmarks:
        benchmark_result = run_comprehensive_benchmarking(
            analysis_results['best_model'], test_loader, analysis_results
        )
    else:
        # create minimal benchmark result
        benchmark_result = type('obj', (object,), {
            'accuracy': analysis_results['best_accuracy'],
            'robustness_score': 0.0,
            'weight_sensitivity': 0.0,
            'inference_time': 0.0,
            'memory_usage': 0.0
        })()
    
    # save production model
    save_complete_production_model(
        analysis_results['best_model'], best_genome, analysis_results, 
        benchmark_result, config, args.save_dir
    )
    
    # final summary
    print(f"\n" + "=" * 80)
    print("EVOLUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\nfinal results:")
    print(f"  best accuracy: {analysis_results['best_accuracy']:.4f}")
    print(f"  ensemble accuracy: {analysis_results['ensemble_result']['ensemble_accuracy']:.4f}")
    print(f"  optimal weight: {analysis_results['best_weight']:.2f}")
    print(f"  sparsity achieved: {analysis_results['sparsity_ratio']:.1%}")
    print(f"  evolved parameters: {analysis_results['total_connections']/1000000:.1f}M")
    
    print(f"\nfiles created:")
    print(f"  production model: {args.save_dir}/production_hybrid_model.pt")
    print(f"  model summary: {args.save_dir}/model_summary.json")
    print(f"  evolution data: {args.save_dir}/")
    if not args.skip_benchmarks:
        print(f"  benchmarks: ./full_evolution_benchmarks/")
    
    print(f"\nto use your model:")
    print(f"  from load_complete_model import CompleteModelLoader")
    print(f"  loader = CompleteModelLoader('{args.save_dir}/production_hybrid_model.pt')")
    print(f"  results = loader.predict(['your text here'])")

if __name__ == "__main__":
    main() 