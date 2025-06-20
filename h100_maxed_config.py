#!/usr/bin/env python3
"""
h100 maximum performance configuration for wann-gpt
optimized for 80gb h100 gpu memory and high performance
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

def create_h100_maxed_config():
    """create h100 optimized configuration for maximum performance within 80gb memory limit"""
    
    config = ConfigPresets.generation_large()  # switch to generation preset
    
    #model architecture - balanced for memory efficiency
    config.model.embed_dim = 1536  # reduced from 2048 to save memory
    config.model.vocab_size = 50257  # full gpt-2 vocabulary
    config.model.num_layers = 20  # reduced from 24
    config.model.num_heads = 24  # reduced from 32 (1536/24 = 64 per head)
    config.model.max_length = 1024  # reduced from 2048 for memory
    config.model.causal = True  # ensure causal masking for generation
    
    #evolution settings - optimized for 80gb memory limit
    config.evolution.embed_dim = 1536
    config.evolution.vocab_size = 50257
    config.evolution.population_size = 25  # drastically reduced from 400 to fit in memory
    config.evolution.num_generations = 100  # keep extensive evolution time
    config.evolution.complexity_weight = 0.0001  # allow complex architectures
    config.evolution.mutation_rate = 0.2  # higher mutation for exploration
    config.evolution.crossover_rate = 0.8  # high crossover for mixing
    config.evolution.elitism_rate = 0.1  # preserve 10% best individuals (10 out of 100)
    config.evolution.tournament_size = 5  # higher selection pressure
    config.evolution.fitness_stagnation_threshold = 25  # longer patience
    config.evolution.max_layers = 24  # reduced from 32
    config.evolution.max_heads = 48  # reduced from 64
    config.evolution.parallel_evaluation = False  # disable parallel to save memory
    config.evolution.selection_strategy = "tournament"  # use tournament selection for head evolution
    
    #memory management flags
    config.evolution.clear_cache_between_evals = True  # clear cuda cache between evaluations
    config.evolution.sequential_evaluation = True  # evaluate one genome at a time
    
    # fix num_heads for headonlygenome compatibility
    # gpt-2 uses 768 embed_dim with 12 heads (768/12=64 per head)
    # for 2048 embed_dim, valid num_heads could be: 16 (128 per head), 32 (64 per head), or 64 (32 per head)
    config.evolution.num_heads = 32  # set to 32 so 2048/32 = 64 per head
    config.evolution.num_layers = 24  # explicitly set for headonlygenome
    
    # maximize batch size for h100 memory - aggressive scaling since only using 4.6gb/81gb
    config.data.batch_size = 32  # massive increase from 64 (8x larger)
    config.data.max_length = 2048  # long sequences
    config.data.subset_size = 100000  # double dataset size for robust training
    config.data.task_type = "generation"  # set to generation task
    config.data.dataset_name = "wikitext"  # better dataset for generation
    
    # thorough weight sampling
    config.evolution.weight_samples = [
        -3, -2, -1, 0, 1, 2, 3
    ]
    
    # training optimizations for h100
    config.training.parallel_evaluation = True
    config.training.device = "cuda"
    config.training.mixed_precision = False  # disable mixed precision`n    config.training.use_fp16 = True  # use pure fp16 throughout
    config.training.gradient_checkpointing = True  # save memory
    
    return config

def create_h100_conservative_config():
    """create conservative h100 configuration for gradual scaling"""
    
    config = ConfigPresets.generation_large()  # switch to generation
    
    # moderate scaling for h100
    config.model.embed_dim = 1536  # larger but not maxed
    config.model.vocab_size = 50257  # full gpt-2 vocabulary
    config.model.num_layers = 18  # deeper than default
    config.model.num_heads = 24  # more heads
    config.model.max_length = 1536  # longer sequences
    config.model.causal = True  # ensure causal masking for generation
    
    # moderate evolution settings
    config.evolution.embed_dim = 1536
    config.evolution.vocab_size = 50257
    config.evolution.population_size = 20  # 3x default
    config.evolution.num_generations = 75
    config.evolution.complexity_weight = 0.0002
    config.evolution.mutation_rate = 0.18
    config.evolution.crossover_rate = 0.75
    config.evolution.elitism_rate = 0.1  # preserve 10% best individuals (15 out of 150)
    config.evolution.tournament_size = 4
    config.evolution.fitness_stagnation_threshold = 20
    config.evolution.max_layers = 24
    config.evolution.max_heads = 48
    config.evolution.parallel_evaluation = True
    config.evolution.selection_strategy = "tournament"  # use tournament selection for head evolution
    
    # fix num_heads for headonlygenome compatibility
    # for 1536 embed_dim: valid num_heads are 12 (128 per head), 16 (96), 24 (64), 32 (48), 48 (32)
    config.evolution.num_heads = 24  # set to 24 so 1536/24 = 64 per head
    config.evolution.num_layers = 18  # explicitly set for headonlygenome
    
    # conservative but substantial batch size increase
    config.data.batch_size = 256  # 4x increase from current
    config.data.max_length = 1536
    config.data.subset_size = 75000
    config.data.task_type = "generation"  # set to generation task
    config.data.dataset_name = "wikitext"  # better dataset for generation
    
    # thorough weight sampling
    config.evolution.weight_samples = [
        -3.5, -2.5, -1.5, -1.0, -0.5, -0.25, 
        0.0, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5
    ]
    
    # training optimizations
    config.training.parallel_evaluation = True
    config.training.device = "cuda"
    config.training.mixed_precision = False  # disable mixed precision`n    config.training.use_fp16 = True  # use pure fp16 throughout
    config.training.gradient_checkpointing = True
    
    return config

def run_h100_maxed_evolution(dataset_name="imdb", mode="maxed", wandb_project="wann-gpt-h100-evolution"):
    """run maximum performance evolution on h100"""
    
    print("=" * 80)
    print("H100 MAXIMUM PERFORMANCE EVOLUTION")
    print("=" * 80)
    
    #check if wandb is disabled - fix the logic
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    
    #initialize wandb
    if use_wandb:
        #optionally set api key if provided via environment variable
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        wandb.init(
            project=wandb_project,
            name=f"h100-{mode}-{dataset_name}-{int(time.time())}",
            tags=["h100", mode, "evolution", dataset_name],
            config={
                "dataset": dataset_name,
                "mode": mode,
                "gpu_type": "h100" if torch.cuda.is_available() else "cpu"
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
    
    # check gpu specs
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
    
    # create h100 optimized config
    config = create_h100_maxed_config() if mode == "maxed" else create_h100_conservative_config()
    
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
    
    print(f"\nh100 configuration:")
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
    
    # estimate memory usage
    estimated_memory = estimate_h100_memory_usage(config)
    print(f"  estimated memory: {estimated_memory:.1f} gb")
    if use_wandb:
        wandb.log({"config/estimated_memory_gb": estimated_memory})
    
    # load large dataset
    print(f"\nloading {dataset_name} dataset...")
    train_loader, test_loader = load_generation_data(  # generation doesn't return num_classes
        dataset_name=dataset_name,
        vocab_size=config.model.vocab_size,
        max_length=config.model.max_length,
        batch_size=config.data.batch_size,
        subset_size=config.data.subset_size
    )
    
    # no need to set num_classes for generation tasks
    
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
    
    # create h100 optimized evaluator
    evaluator = SharedWeightEvaluator(device="cuda")
    evaluator.default_weight_samples = config.evolution.weight_samples
    evaluator.use_mixed_precision = False  # fp16 for speed
    evaluator.parallel_evaluation = False  # disable parallel to save memory
    evaluator.clear_cache_after_eval = True  # clear cuda cache after each evaluation
    
    # create evolution engine with h100 optimizations
    evolution_engine = HeadOnlyEvolutionEngine(
        config=config.evolution,
        evaluator=evaluator,
        save_dir="./h100_maxed_results",
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
    
    # run maximum performance evolution
    print(f"\nstarting h100 maximum performance evolution...")
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
        
        # comprehensive benchmarking
        print(f"\nrunning h100 comprehensive benchmark...")
        
        model = evaluator.instantiate_hybrid_from_genome(best_genome)  # use the correct method for hybrid models
        benchmark_suite = WannBenchmarkSuite(device="cuda")
        
        benchmark_result = benchmark_suite.comprehensive_benchmark(
            model=model,
            test_loader=test_loader,
            task_type="generation",
            save_plots=True,
            output_dir="./h100_benchmark_results"
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
        torch.save(best_genome, "./h100_maxed_results/best_genome.pt")
        torch.save(model.state_dict(), "./h100_maxed_results/best_model.pt")
        
        #save model artifact to wandb if enabled
        if use_wandb:
            model_artifact = wandb.Artifact(
                name=f"h100-evolved-model-{dataset_name}",
                type="model",
                description=f"best evolved model from h100 run on {dataset_name}"
            )
            
            model_artifact.add_file("./h100_maxed_results/best_genome.pt")
            model_artifact.add_file("./h100_maxed_results/best_model.pt")
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

def estimate_h100_memory_usage(config):
    """estimate memory usage for h100 configuration with more accurate calculation"""
    
    #model parameters
    embed_params = config.model.vocab_size * config.model.embed_dim
    attention_params = config.model.num_layers * 4 * config.model.embed_dim * config.model.embed_dim  # q,k,v,o projections
    ffn_params = config.model.num_layers * 2 * config.model.embed_dim * (4 * config.model.embed_dim)  # mlp layers
    total_model_params = embed_params + attention_params + ffn_params
    
    #memory per model (including gradients and optimizer states)
    memory_per_model_gb = total_model_params * 2 * 3 / 1e9  # fp16: 2 bytes per param + gradients + optimizer states
    
    #batch memory
    batch_memory_gb = (config.data.batch_size * config.data.max_length * config.model.embed_dim * 4) / 1e9
    
    #total memory estimate
    #note: with sequential evaluation, we only keep one model active at a time
    active_models = 1  # sequential evaluation
    memory_gb = (memory_per_model_gb * active_models + batch_memory_gb) * 1.5  # 1.5x for overhead
    
    return memory_gb

def estimate_runtime(config):
    """estimate runtime in hours"""
    
    # rough estimate based on config complexity
    complexity_factor = (
        config.evolution.population_size * 
        config.evolution.num_generations * 
        len(config.evolution.weight_samples) *
        config.model.num_layers
    ) / 1000000
    
    return complexity_factor * 0.1  # rough hours estimate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run h100 maximum performance wann-gpt evolution")
    parser.add_argument("--dataset", default="wikitext", help="dataset to use (wikitext, tiny_stories, etc.)")
    parser.add_argument("--save-config", action="store_true", help="save h100 config to file")
    parser.add_argument("--mode", choices=["maxed", "conservative"], default="maxed", 
                       help="configuration mode: maxed (aggressive) or conservative (gradual scaling)")
    parser.add_argument("--wandb-project", default="wann-gpt-h100-evolution", help="wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    
    args = parser.parse_args()
    
    #set wandb mode
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"
    
    if args.save_config:
        if args.mode == "maxed":
            config = create_h100_maxed_config()
            config.save("h100_maxed_config.yaml")
            print("h100 maxed configuration saved to h100_maxed_config.yaml")
        else:
            config = create_h100_conservative_config()
            config.save("h100_conservative_config.yaml")
            print("h100 conservative configuration saved to h100_conservative_config.yaml")
    else:
        if args.mode == "conservative":
            print("running h100 conservative evolution...")
            #for conservative mode, we pass the mode to the function instead of double-initializing wandb
            config = create_h100_conservative_config()
            print(f"conservative config - batch size: {config.data.batch_size}, population: {config.evolution.population_size}")
            
        best_genome, benchmark = run_h100_maxed_evolution(args.dataset, mode=args.mode, wandb_project=args.wandb_project)
        print(f"\nh100 evolution completed successfully!")
        print(f"results saved to ./h100_maxed_results/")
        print(f"benchmarks saved to ./h100_benchmark_results/") 
