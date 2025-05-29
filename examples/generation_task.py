"""
example demonstrating wann-gpt on text generation tasks
shows evolution for causal language modeling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer

from wann_gpt.architecture import WannGPT
from wann_gpt.evolution import EvolutionEngine, ArchitectureGenome
from wann_gpt.config import EvolutionConfig
from wann_gpt.evaluation import SharedWeightEvaluator

class SimpleLanguageDataset(Dataset):
    """simple language modeling dataset for demonstration"""
    
    def __init__(self, size: int = 1000, vocab_size: int = 1000, seq_len: int = 64):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # generate random sequences with some structure
        # pattern: sequences that follow a simple grammar
        self.sequences = []
        
        for _ in range(size):
            # create sequence with simple patterns
            seq = []
            
            # start token
            seq.append(0)
            
            # add pattern: repeat some tokens
            for _ in range(seq_len - 1):
                if len(seq) > 5 and np.random.random() < 0.3:
                    # repeat a previous token
                    repeat_idx = np.random.randint(0, min(5, len(seq)))
                    seq.append(seq[-(repeat_idx+1)])
                else:
                    # random token
                    seq.append(np.random.randint(1, vocab_size))
            
            self.sequences.append(torch.tensor(seq[:seq_len]))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            'input_ids': sequence,
            'attention_mask': torch.ones(len(sequence))
        }

def create_language_dataloaders(vocab_size: int = 500, seq_len: int = 64, batch_size: int = 16):
    """create language modeling dataloaders"""
    
    train_dataset = SimpleLanguageDataset(size=1200, vocab_size=vocab_size, seq_len=seq_len)
    test_dataset = SimpleLanguageDataset(size=300, vocab_size=vocab_size, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def generate_sample_text(model: WannGPT, evaluator: SharedWeightEvaluator, 
                        weight_samples: list, vocab_size: int, 
                        prompt_length: int = 10, max_new_tokens: int = 20):
    """generate sample text with different weight values"""
    
    # create random prompt
    prompt = torch.randint(0, vocab_size, (1, prompt_length)).to(model.device)
    
    print(f"prompt tokens: {prompt[0].tolist()}")
    
    for i, weight in enumerate(weight_samples[:3]):  # test first 3 weights
        model.set_shared_weight(weight)
        
        generated = model.generate(
            prompt.clone(), 
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=50
        )
        
        generated_tokens = generated[0, prompt_length:].tolist()
        print(f"weight {weight:+.1f}: {generated_tokens}")

def main():
    """main function for language modeling evolution"""
    
    # configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    # dataset parameters
    vocab_size = 500  # smaller vocab for demo
    seq_len = 48      # shorter sequences
    embed_dim = 256   # reasonable embedding size
    batch_size = 8    # smaller batch size
    
    # create datasets
    print("creating language modeling datasets...")
    train_loader, test_loader = create_language_dataloaders(vocab_size, seq_len, batch_size)
    
    # evolution configuration for generation
    config = EvolutionConfig(
        population_size=15,          # smaller population for demo
        num_generations=30,          # more generations for language modeling
        mutation_rate=0.7,
        crossover_rate=0.2,
        elitism_rate=0.15,           # slightly higher elitism
        selection_strategy="nsga2",  # use nsga-ii for multi-objective
        complexity_weight=0.05,      # lower complexity weight for generation
        max_layers=8,
        max_heads=6,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        max_length=seq_len,         # set max_length to sequence length
        weight_samples=[-1.5, -0.8, -0.3, 0.3, 0.8, 1.5],  # different range
        adaptive_mutation=True,
        fitness_stagnation_threshold=12
    )
    
    # create evaluator and evolution engine
    print("initializing evolution engine...")
    evaluator = SharedWeightEvaluator(device=device)
    engine = EvolutionEngine(config, evaluator, save_dir="./generation_results")
    
    # run evolution for language modeling
    print("starting evolution for text generation...")
    best_genome = engine.evolve(
        dataloader=train_loader,
        task_type="generation",
        initialization_strategy="mixed",
        log_wandb=False
    )
    
    print("evolution completed!")
    print(f"best fitness (neg perplexity): {best_genome.get_fitness('generation'):.4f}")
    print(f"best complexity: {best_genome.calculate_complexity()}")
    
    # test the best evolved architecture
    print("\ntesting best architecture...")
    best_model = evaluator.instantiate_from_genome(best_genome)
    
    # evaluate on test set
    test_result = evaluator.evaluate_generation(best_model, test_loader)
    print(f"test performance (neg perplexity): {test_result.mean_performance:.4f} ± {test_result.std_performance:.4f}")
    print(f"best weight performance: {test_result.best_performance:.4f}")
    
    # analyze weight sensitivity for generation
    print("\nanalyzing weight sensitivity...")
    sensitivity_analysis = evaluator.analyze_weight_sensitivity(
        best_model, test_loader, 
        weight_range=(-2.0, 2.0), 
        num_samples=12,
        task_type="generation"
    )
    
    print(f"weight sensitivity: {sensitivity_analysis['weight_sensitivity']:.4f}")
    print(f"optimal weight: {sensitivity_analysis['best_weight']:.2f}")
    
    # generate sample text
    print("\ngenerating sample text...")
    generate_sample_text(
        best_model, evaluator, config.weight_samples, 
        vocab_size, prompt_length=8, max_new_tokens=15
    )
    
    # display architecture info
    print("\nbest architecture details:")
    arch_info = best_genome.get_architecture_info()
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # compare with simple baseline
    print("\ncomparing with baseline architectures...")
    
    # create simple baseline
    baseline_genome = ArchitectureGenome.create_simple(embed_dim, vocab_size, num_layers=1)
    baseline_genome.max_length = seq_len  # set max_length to match sequence length
    baseline_result = evaluator.evaluate_genome(baseline_genome, test_loader, "generation")
    
    print(f"baseline (1 layer): {baseline_result.mean_performance:.4f}")
    print(f"evolved model improvement: {best_genome.get_fitness('generation') - baseline_result.mean_performance:.4f}")
    
    # show pareto front
    pareto_front = engine.get_pareto_front("generation")
    print(f"\npareto front contains {len(pareto_front)} solutions")
    
    for i, genome in enumerate(pareto_front[:3]):
        print(f"  solution {i+1}: fitness={genome.get_fitness('generation'):.4f}, "
              f"complexity={genome.calculate_complexity()}, "
              f"layers={len(genome.get_active_layers())}")

if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(123)
    np.random.seed(123)
    
    main() 