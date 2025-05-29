"""
basic example of evolving weight-agnostic transformer architectures
demonstrates classification task on a simple dataset
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

class SimpleTextDataset(Dataset):
    """simple text classification dataset for demonstration"""
    
    def __init__(self, size: int = 1000, vocab_size: int = 1000, seq_len: int = 32):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # generate random text sequences
        self.texts = torch.randint(0, vocab_size, (size, seq_len))
        
        # simple classification: positive if sum of tokens is even
        token_sums = self.texts.sum(dim=1)
        self.labels = (token_sums % 2).long()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.texts[idx],
            'labels': self.labels[idx],
            'attention_mask': torch.ones(self.seq_len)
        }

def create_simple_dataloaders(vocab_size: int = 1000, seq_len: int = 32, batch_size: int = 32):
    """create simple train/test dataloaders"""
    
    train_dataset = SimpleTextDataset(size=800, vocab_size=vocab_size, seq_len=seq_len)
    test_dataset = SimpleTextDataset(size=200, vocab_size=vocab_size, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    """main function demonstrating wann-gpt evolution"""
    
    # configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    # dataset parameters
    vocab_size = 1000
    seq_len = 32
    num_classes = 2
    embed_dim = 128
    batch_size = 16
    
    # create datasets
    print("creating datasets...")
    train_loader, test_loader = create_simple_dataloaders(vocab_size, seq_len, batch_size)
    
    # evolution configuration
    config = EvolutionConfig(
        population_size=20,          # small population for demo
        num_generations=25,          # few generations for demo
        mutation_rate=0.8,
        crossover_rate=0.3,
        elitism_rate=0.1,
        selection_strategy="adaptive",
        complexity_weight=0.1,
        max_layers=6,
        max_heads=8,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        max_length=seq_len,         # set max_length to sequence length
        weight_samples=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
        adaptive_mutation=True,
        fitness_stagnation_threshold=8
    )
    
    # create evaluator and evolution engine
    print("initializing evolution engine...")
    evaluator = SharedWeightEvaluator(device=device)
    engine = EvolutionEngine(config, evaluator, save_dir="./evolution_results")
    
    # set classification task parameters in genomes
    for i in range(config.population_size):
        genome = ArchitectureGenome.create_simple(embed_dim, vocab_size, num_layers=2)
        genome.num_classes = num_classes
    
    # run evolution
    print("starting evolution...")
    best_genome = engine.evolve(
        dataloader=train_loader,
        task_type="classification", 
        initialization_strategy="mixed",
        log_wandb=False  # set to true if you have wandb configured
    )
    
    print("evolution completed!")
    print(f"best fitness: {best_genome.get_fitness('classification'):.4f}")
    print(f"best complexity: {best_genome.calculate_complexity()}")
    
    # test the best evolved architecture
    print("\ntesting best architecture...")
    best_model = evaluator.instantiate_from_genome(best_genome)
    
    # evaluate on test set
    test_result = evaluator.evaluate_classification(best_model, test_loader)
    print(f"test accuracy: {test_result.mean_performance:.4f} ± {test_result.std_performance:.4f}")
    print(f"best weight performance: {test_result.best_performance:.4f}")
    
    # analyze weight sensitivity
    print("\nanalyzing weight sensitivity...")
    sensitivity_analysis = evaluator.analyze_weight_sensitivity(
        best_model, test_loader, weight_range=(-3.0, 3.0), num_samples=15
    )
    
    print(f"weight sensitivity: {sensitivity_analysis['weight_sensitivity']:.4f}")
    print(f"optimal weight: {sensitivity_analysis['best_weight']:.2f}")
    
    # test ensemble performance
    print("\ntesting ensemble performance...")
    ensemble_result = evaluator.ensemble_evaluate(
        best_model, test_loader, weight_samples=config.weight_samples
    )
    print(f"ensemble accuracy: {ensemble_result['ensemble_accuracy']:.4f}")
    
    # display architecture info
    print("\nbest architecture details:")
    arch_info = best_genome.get_architecture_info()
    for key, value in arch_info.items():
        print(f"  {key}: {value}")
    
    # get pareto front
    pareto_front = engine.get_pareto_front("classification")
    print(f"\npareto front contains {len(pareto_front)} solutions")
    
    for i, genome in enumerate(pareto_front[:3]):  # show top 3
        print(f"  solution {i+1}: fitness={genome.get_fitness('classification'):.4f}, "
              f"complexity={genome.calculate_complexity()}")

if __name__ == "__main__":
    # set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 