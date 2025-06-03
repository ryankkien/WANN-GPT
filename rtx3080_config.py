#!/usr/bin/env python3
"""
rtx 3080 optimized configuration for wann-gpt
optimized for 10gb rtx 3080 gpu memory and good performance
"""

import torch
import argparse
import time
import psutil
import GPUtil
import threading
from pathlib import Path
import wandb
from wann_gpt import (
    HybridWannGPT, HeadOnlyGenome, HeadOnlyEvolutionEngine,
    SharedWeightEvaluator, load_generation_data,
    WannGPTConfig, ConfigPresets, WannBenchmarkSuite
)
import os

class SystemMonitor:
    """monitor system resources in real-time for wandb logging"""
    
    def __init__(self, log_interval=30, use_wandb=True):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.use_wandb = use_wandb
        
    def start_monitoring(self):
        """start background monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """background monitoring loop"""
        while self.monitoring:
            try:
                #cpu and ram metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                ram_used_gb = memory.used / (1024**3)
                ram_percent = memory.percent
                
                metrics = {
                    "system/cpu_percent": cpu_percent,
                    "system/ram_used_gb": ram_used_gb,
                    "system/ram_percent": ram_percent
                }
                
                #gpu metrics
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    gpu_metrics = {
                        "system/gpu_memory_used_gb": gpu_memory_used,
                        "system/gpu_memory_cached_gb": gpu_memory_cached,
                        "system/gpu_memory_total_gb": gpu_memory_total,
                        "system/gpu_memory_usage_percent": (gpu_memory_used / gpu_memory_total) * 100
                    }
                    
                    #add gpu utilization if gputil is available
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu_metrics.update({
                                "system/gpu_utilization_percent": gpus[0].load * 100,
                                "system/gpu_temperature_c": gpus[0].temperature
                            })
                    except Exception:
                        pass  #gputil might not work on all systems
                    
                    metrics.update(gpu_metrics)
                
                #log to wandb if enabled
                if self.use_wandb:
                    wandb.log(metrics)
                    
                #print metrics occasionally for non-wandb runs
                elif not hasattr(self, '_last_print') or time.time() - self._last_print > 300:  #every 5 min
                    print(f"system: cpu {cpu_percent:.1f}%, ram {ram_used_gb:.1f}gb ({ram_percent:.1f}%)")
                    if torch.cuda.is_available():
                        print(f"gpu: memory {gpu_memory_used:.1f}gb ({(gpu_memory_used/gpu_memory_total)*100:.1f}%)")
                    self._last_print = time.time()
                    
            except Exception as e:
                print(f"monitoring error: {e}")
                
            time.sleep(self.log_interval)

def create_rtx3080_config():
    """create rtx 3080 optimized configuration for 10gb memory limit"""
    
    config = ConfigPresets.generation_large()
    
    #model architecture - scaled for 10gb memory
    config.model.embed_dim = 768  #reduced from 1536 to save significant memory
    config.model.vocab_size = 50257  #full gpt-2 vocabulary
    config.model.num_layers = 12  #reduced from 20 for memory efficiency
    config.model.num_heads = 12  #768/12 = 64 per head (standard gpt-2 ratio)
    config.model.max_length = 512  #reduced from 1024 for memory
    config.model.causal = True  #ensure causal masking for generation
    
    #evolution settings - optimized for 10gb memory limit
    config.evolution.embed_dim = 768
    config.evolution.vocab_size = 50257
    config.evolution.population_size = 8  #small population to fit in memory
    config.evolution.num_generations = 50  #moderate evolution time
    config.evolution.complexity_weight = 0.0002  #allow some complexity
    config.evolution.mutation_rate = 0.25  #higher mutation for exploration with small population
    config.evolution.crossover_rate = 0.7  #moderate crossover
    config.evolution.elitism_rate = 0.25  #preserve 25% best (2 out of 8)
    config.evolution.tournament_size = 3  #smaller tournament for small population
    config.evolution.fitness_stagnation_threshold = 15  #moderate patience
    config.evolution.max_layers = 16  #reduced from 24
    config.evolution.max_heads = 24  #reduced from 48
    config.evolution.parallel_evaluation = False  #disable parallel to save memory
    config.evolution.selection_strategy = "tournament"
    
    #memory management flags - critical for rtx 3080
    config.evolution.clear_cache_between_evals = True  #clear cuda cache between evaluations
    config.evolution.sequential_evaluation = True  #evaluate one genome at a time
    
    #fix num_heads for headonlygenome compatibility
    config.evolution.num_heads = 12  #768/12 = 64 per head
    config.evolution.num_layers = 12  #explicitly set for headonlygenome
    
    #conservative batch size for 10gb memory
    config.data.batch_size = 8  #small batch size to fit in memory
    config.data.max_length = 512  #shorter sequences
    config.data.subset_size = 25000  #smaller dataset for faster training
    config.data.task_type = "generation"
    config.data.dataset_name = "wikitext"
    
    #moderate weight sampling
    config.evolution.weight_samples = [
        -2, -1, -0.5, 0, 0.5, 1, 2
    ]
    
    #training optimizations for rtx 3080
    config.training.parallel_evaluation = False  #disable to save memory
    config.training.device = "cuda"
    config.training.mixed_precision = True  #enable mixed precision for memory savings
    config.training.use_fp16 = True  #use fp16 throughout
    config.training.gradient_checkpointing = True  #save memory
    
    return config

def create_rtx3080_conservative_config():
    """create ultra-conservative rtx 3080 configuration for guaranteed fit"""
    
    config = ConfigPresets.generation_large()
    
    #ultra-conservative scaling for rtx 3080
    config.model.embed_dim = 512  #very small for guaranteed fit
    config.model.vocab_size = 50257
    config.model.num_layers = 8  #shallow network
    config.model.num_heads = 8  #512/8 = 64 per head
    config.model.max_length = 256  #short sequences
    config.model.causal = True
    
    #minimal evolution settings
    config.evolution.embed_dim = 512
    config.evolution.vocab_size = 50257
    config.evolution.population_size = 4  #tiny population
    config.evolution.num_generations = 25  #short evolution
    config.evolution.complexity_weight = 0.0005
    config.evolution.mutation_rate = 0.3
    config.evolution.crossover_rate = 0.6
    config.evolution.elitism_rate = 0.5  #preserve 50% (2 out of 4)
    config.evolution.tournament_size = 2
    config.evolution.fitness_stagnation_threshold = 10
    config.evolution.max_layers = 12
    config.evolution.max_heads = 16
    config.evolution.parallel_evaluation = False
    config.evolution.selection_strategy = "tournament"
    
    #memory management
    config.evolution.clear_cache_between_evals = True
    config.evolution.sequential_evaluation = True
    
    #fix num_heads for headonlygenome compatibility
    config.evolution.num_heads = 8  #512/8 = 64 per head
    config.evolution.num_layers = 8
    
    #minimal batch size
    config.data.batch_size = 4  #very small batch
    config.data.max_length = 256
    config.data.subset_size = 10000  #small dataset
    config.data.task_type = "generation"
    config.data.dataset_name = "wikitext"
    
    #simple weight sampling
    config.evolution.weight_samples = [
        -1, 0, 1
    ]
    
    #training optimizations
    config.training.parallel_evaluation = False
    config.training.device = "cuda"
    config.training.mixed_precision = True
    config.training.use_fp16 = True
    config.training.gradient_checkpointing = True
    
    return config

def run_rtx3080_evolution(dataset_name="wikitext", mode="standard", wandb_project="wann-gpt-rtx3080-evolution"):
    """run rtx 3080 optimized evolution"""
    
    print("=" * 80)
    print("RTX 3080 OPTIMIZED EVOLUTION")
    print("=" * 80)
    
    #check if wandb is disabled
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    
    #initialize wandb
    if use_wandb:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        wandb.init(
            project=wandb_project,
            name=f"rtx3080-{mode}-{dataset_name}-{int(time.time())}",
            tags=["rtx3080", mode, "evolution", dataset_name],
            config={
                "dataset": dataset_name,
                "mode": mode,
                "gpu_type": "rtx3080" if torch.cuda.is_available() else "cpu"
            }
        )
        print(f"✓ wandb initialized: {wandb.run.name}")
        print(f"  project: {wandb.run.project}")
        print(f"  url: {wandb.run.url}")
    else:
        print("wandb disabled - logging to console only")
    
    #start system monitoring
    monitor = SystemMonitor(log_interval=30, use_wandb=use_wandb)
    monitor.start_monitoring()
    
    #check gpu specs
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_cores = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"gpu: {gpu_name}")
        print(f"memory: {gpu_memory:.1f} gb")
        print(f"cuda cores: {cuda_cores}")
        
        #log gpu specs to wandb
        if use_wandb:
            wandb.config.update({
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory,
                "cuda_cores": cuda_cores
            })
    else:
        print("warning: cuda not available!")
        if use_wandb:
            wandb.config.update({"gpu_available": False})
        return
    
    #create rtx 3080 optimized config
    config = create_rtx3080_config() if mode == "standard" else create_rtx3080_conservative_config()
    
    #log configuration to wandb
    config_dict = {
        "model/embed_dim": config.model.embed_dim,
        "model/vocab_size": config.model.vocab_size,
        "model/num_layers": config.model.num_layers,
        "model/num_heads": config.model.num_heads,
        "model/max_length": config.model.max_length,
        "evolution/population_size": config.evolution.population_size,
        "evolution/num_generations": config.evolution.num_generations,
        "evolution/mutation_rate": config.evolution.mutation_rate,
        "evolution/crossover_rate": config.evolution.crossover_rate,
        "evolution/elitism_rate": config.evolution.elitism_rate,
        "evolution/tournament_size": config.evolution.tournament_size,
        "evolution/max_layers": config.evolution.max_layers,
        "evolution/max_heads": config.evolution.max_heads,
        "data/batch_size": config.data.batch_size,
        "data/max_length": config.data.max_length,
        "data/subset_size": config.data.subset_size,
        "evolution/weight_samples_count": len(config.evolution.weight_samples),
        "evolution/complexity_weight": config.evolution.complexity_weight
    }
    if use_wandb:
        wandb.config.update(config_dict, allow_val_change=True)
    
    print(f"\nrtx 3080 configuration:")
    print(f"  embed dimension: {config.model.embed_dim}")
    print(f"  vocabulary size: {config.model.vocab_size:,}")
    print(f"  max layers: {config.model.num_layers}")
    print(f"  attention heads: {config.model.num_heads}")
    print(f"  sequence length: {config.model.max_length}")
    print(f"  population size: {config.evolution.population_size}")
    print(f"  generations: {config.evolution.num_generations}")
    print(f"  batch size: {config.data.batch_size}")
    print(f"  dataset size: {config.data.subset_size:,} samples")
    print(f"  weight samples: {len(config.evolution.weight_samples)} points")
    
    #estimate memory usage
    estimated_memory = estimate_rtx3080_memory_usage(config)
    print(f"  estimated memory: {estimated_memory:.1f} gb")
    if use_wandb:
        wandb.log({"config/estimated_memory_gb": estimated_memory})
    
    #load dataset
    print(f"\nloading {dataset_name} dataset...")
    train_loader, test_loader = load_generation_data(
        dataset_name=dataset_name,
        vocab_size=config.model.vocab_size,
        max_length=config.model.max_length,
        batch_size=config.data.batch_size,
        subset_size=config.data.subset_size
    )
    
    print(f"dataset loaded:")
    print(f"  train batches: {len(train_loader):,}")
    print(f"  test batches: {len(test_loader):,}")
    print(f"  task: text generation")
    
    #log dataset info
    if use_wandb:
        wandb.log({
            "dataset/train_batches": len(train_loader),
            "dataset/test_batches": len(test_loader),
            "dataset/task_type": "text_generation"
        })
    
    #create rtx 3080 optimized evaluator
    evaluator = SharedWeightEvaluator(device="cuda")
    evaluator.default_weight_samples = config.evolution.weight_samples
    evaluator.use_mixed_precision = True  #enable mixed precision for memory savings
    evaluator.parallel_evaluation = False  #disable parallel to save memory
    evaluator.clear_cache_after_eval = True  #clear cuda cache after each evaluation
    
    #create evolution engine with rtx 3080 optimizations
    evolution_engine = HeadOnlyEvolutionEngine(
        config=config.evolution,
        evaluator=evaluator,
        save_dir="./rtx3080_results",
        model_name="gpt2"
    )
    
    #add callback for evolution tracking
    def evolution_callback(generation, population, best_fitness, avg_fitness, best_genome):
        """callback to log evolution metrics to wandb"""
        complexity_stats = [genome.calculate_complexity() for genome in population]
        
        metrics = {
            "evolution/generation": generation,
            "evolution/best_fitness": best_fitness,
            "evolution/avg_fitness": avg_fitness,
            "evolution/fitness_std": torch.std(torch.tensor([g.get_fitness('generation') for g in population if g.get_fitness('generation') is not None])).item(),
            "evolution/population_size": len(population),
            "evolution/avg_complexity": sum(complexity_stats) / len(complexity_stats),
            "evolution/max_complexity": max(complexity_stats),
            "evolution/min_complexity": min(complexity_stats),
            "evolution/best_genome_layers": best_genome.num_layers if best_genome else 0,
            "evolution/best_genome_heads": best_genome.num_heads if best_genome else 0,
            "evolution/lm_head_sparsity": best_genome.lm_head_sparsity if best_genome else 0,
            "evolution/classifier_sparsity": best_genome.classifier_sparsity if best_genome else 0
        }
        
        if use_wandb:
            wandb.log(metrics)
        else:
            print(f"generation {generation}: best_fitness={best_fitness:.6f}, avg_fitness={avg_fitness:.6f}, "
                  f"avg_complexity={metrics['evolution/avg_complexity']:.0f}")
    
    #monkey patch the evolution engine to add our callback
    original_evaluate_population = evolution_engine.evaluate_population
    
    def tracked_evaluate_population(population, train_loader, task_type):
        results = original_evaluate_population(population, train_loader, task_type)
        
        #calculate metrics for logging
        fitnesses = [genome.get_fitness(task_type) for genome in population if genome.get_fitness(task_type) is not None]
        if fitnesses:
            best_fitness = max(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_genome = max(population, key=lambda g: g.get_fitness(task_type) if g.get_fitness(task_type) is not None else float('-inf'))
            
            #call our tracking callback
            evolution_callback(
                generation=evolution_engine.generation,
                population=population,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                best_genome=best_genome
            )
        
        return results
    
    evolution_engine.evaluate_population = tracked_evaluate_population
    
    #run rtx 3080 optimized evolution
    print(f"\nstarting rtx 3080 optimized evolution...")
    estimated_runtime_hours = estimate_runtime(config)
    print(f"estimated runtime: {estimated_runtime_hours:.1f} hours")
    if use_wandb:
        wandb.log({"config/estimated_runtime_hours": estimated_runtime_hours})
    
    start_time = time.time()
    
    try:
        best_genome = evolution_engine.evolve(
            train_loader,
            task_type="generation",
            initialization_strategy="mixed"
        )
        
        evolution_time = time.time() - start_time
        
        print(f"\nevolution completed!")
        print(f"  time taken: {evolution_time/3600:.2f} hours")
        print(f"  best fitness: {best_genome.get_fitness('generation'):.6f}")
        print(f"  best complexity: {best_genome.calculate_complexity():,}")
        
        #log final evolution results
        if use_wandb:
            wandb.log({
                "results/evolution_time_hours": evolution_time / 3600,
                "results/best_fitness": best_genome.get_fitness('generation'),
                "results/best_complexity": best_genome.calculate_complexity(),
                "results/final_layers": best_genome.num_layers,
                "results/final_heads": best_genome.num_heads,
                "results/final_lm_head_sparsity": best_genome.lm_head_sparsity,
                "results/final_classifier_sparsity": best_genome.classifier_sparsity
            })
        
        #comprehensive benchmarking
        print(f"\nrunning rtx 3080 comprehensive benchmark...")
        
        model = evaluator.instantiate_hybrid_from_genome(best_genome)
        benchmark_suite = WannBenchmarkSuite(device="cuda")
        
        benchmark_result = benchmark_suite.comprehensive_benchmark(
            model=model,
            test_loader=test_loader,
            task_type="generation",
            save_plots=True,
            output_dir="./rtx3080_benchmark_results"
        )
        
        print(f"benchmark completed!")
        print(f"  final perplexity: {benchmark_result.perplexity:.6f}")
        print(f"  generation quality: {benchmark_result.generation_quality:.6f}")
        print(f"  robustness score: {benchmark_result.robustness_score:.6f}")
        
        #log benchmark results
        if use_wandb:
            wandb.log({
                "benchmark/perplexity": benchmark_result.perplexity,
                "benchmark/generation_quality": benchmark_result.generation_quality,
                "benchmark/robustness_score": benchmark_result.robustness_score,
                "benchmark/total_time_hours": (time.time() - start_time) / 3600
            })
        
        #save genome and model files
        torch.save(best_genome, "./rtx3080_results/best_genome.pt")
        torch.save(model.state_dict(), "./rtx3080_results/best_model.pt")
        
        #save model artifact to wandb if enabled
        if use_wandb:
            model_artifact = wandb.Artifact(
                name=f"rtx3080-evolved-model-{dataset_name}",
                type="model",
                description=f"best evolved model from rtx 3080 run on {dataset_name}"
            )
            
            model_artifact.add_file("./rtx3080_results/best_genome.pt")
            model_artifact.add_file("./rtx3080_results/best_model.pt")
            wandb.log_artifact(model_artifact)
        
    except Exception as e:
        print(f"evolution failed: {e}")
        if use_wandb:
            wandb.log({"error": str(e)})
        raise
    finally:
        #stop monitoring and finish wandb
        monitor.stop_monitoring()
        if use_wandb:
            wandb.finish()
    
    return best_genome, benchmark_result

def estimate_rtx3080_memory_usage(config):
    """estimate memory usage for rtx 3080 configuration"""
    
    #model parameters
    embed_params = config.model.vocab_size * config.model.embed_dim
    attention_params = config.model.num_layers * 4 * config.model.embed_dim * config.model.embed_dim
    ffn_params = config.model.num_layers * 2 * config.model.embed_dim * (4 * config.model.embed_dim)
    total_model_params = embed_params + attention_params + ffn_params
    
    #memory per model (fp16 + gradients + optimizer states)
    memory_per_model_gb = total_model_params * 2 * 3 / 1e9
    
    #batch memory
    batch_memory_gb = (config.data.batch_size * config.data.max_length * config.model.embed_dim * 2) / 1e9  #fp16
    
    #total memory estimate with overhead
    active_models = 1  #sequential evaluation
    memory_gb = (memory_per_model_gb * active_models + batch_memory_gb) * 1.3  #1.3x for overhead
    
    return memory_gb

def estimate_runtime(config):
    """estimate runtime in hours for rtx 3080"""
    
    complexity_factor = (
        config.evolution.population_size * 
        config.evolution.num_generations * 
        len(config.evolution.weight_samples) *
        config.model.num_layers
    ) / 100000  #adjusted for rtx 3080 performance
    
    return complexity_factor * 0.2  #rough hours estimate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run rtx 3080 optimized wann-gpt evolution")
    parser.add_argument("--dataset", default="wikitext", help="dataset to use (wikitext, tiny_stories, etc.)")
    parser.add_argument("--save-config", action="store_true", help="save rtx 3080 config to file")
    parser.add_argument("--mode", choices=["standard", "conservative"], default="standard", 
                       help="configuration mode: standard (balanced) or conservative (ultra-safe)")
    parser.add_argument("--wandb-project", default="wann-gpt-rtx3080-evolution", help="wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    
    args = parser.parse_args()
    
    #set wandb mode
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"
    
    if args.save_config:
        if args.mode == "standard":
            config = create_rtx3080_config()
            config.save("rtx3080_config.yaml")
            print("rtx 3080 standard configuration saved to rtx3080_config.yaml")
        else:
            config = create_rtx3080_conservative_config()
            config.save("rtx3080_conservative_config.yaml")
            print("rtx 3080 conservative configuration saved to rtx3080_conservative_config.yaml")
    else:
        if args.mode == "conservative":
            print("running rtx 3080 conservative evolution...")
            config = create_rtx3080_conservative_config()
            print(f"conservative config - batch size: {config.data.batch_size}, population: {config.evolution.population_size}")
            
        best_genome, benchmark = run_rtx3080_evolution(args.dataset, mode=args.mode, wandb_project=args.wandb_project)
        print(f"\nrtx 3080 evolution completed successfully!")
        print(f"results saved to ./rtx3080_results/")
        print(f"benchmarks saved to ./rtx3080_benchmark_results/") 