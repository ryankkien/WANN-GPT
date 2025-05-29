"""
main evolution engine for wann architecture search
orchestrates the evolutionary algorithm with evaluation, selection, and mutation
"""

import os
import json
import pickle
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import wandb

from .genome import ArchitectureGenome
from .mutations import MutationOperators, SpecializedMutations
from .selection import (
    SelectionStrategies, MultiObjectiveSelection, 
    AdaptiveSelection, select_parents_for_reproduction
)
from ..evaluation import SharedWeightEvaluator
from ..config import EvolutionConfig

@dataclass
class GenerationStats:
    """statistics for a single generation"""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_complexity: int
    mean_complexity: float
    diversity_score: float
    mutation_stats: Dict

class EvolutionEngine:
    """main engine for evolving weight-agnostic transformer architectures"""
    
    def __init__(self, config: EvolutionConfig, evaluator: SharedWeightEvaluator,
                 save_dir: str = "./evolution_results"):
        self.config = config
        self.evaluator = evaluator
        self.save_dir = save_dir
        
        # create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # evolution components
        self.mutation_operators = MutationOperators()
        self.multi_obj_selector = MultiObjectiveSelection(config.complexity_weight)
        self.adaptive_selector = AdaptiveSelection()
        
        # evolution state
        self.population: List[ArchitectureGenome] = []
        self.generation = 0
        self.evolution_history: List[GenerationStats] = []
        self.best_individual: Optional[ArchitectureGenome] = None
        
        # tracking
        self.fitness_stagnation_count = 0
        self.last_best_fitness = float('-inf')
        
        # logging
        self.use_wandb = False
    
    def initialize_population(self, initialization_strategy: str = "mixed") -> List[ArchitectureGenome]:
        """initialize population with diverse architectures"""
        
        population = []
        
        if initialization_strategy == "mixed":
            # mix of different initialization strategies
            
            # 20% minimal architectures
            minimal_count = max(1, self.config.population_size // 5)
            for _ in range(minimal_count):
                genome = ArchitectureGenome.create_minimal(
                    self.config.embed_dim, self.config.vocab_size)
                genome.num_classes = self.config.num_classes  # set num_classes for classification
                genome.max_length = self.config.max_length  # set max_length
                population.append(genome)
            
            # 30% simple architectures (1-3 layers)
            simple_count = max(1, (self.config.population_size * 3) // 10)
            for _ in range(simple_count):
                num_layers = random.randint(1, 3)
                genome = ArchitectureGenome.create_simple(
                    self.config.embed_dim, self.config.vocab_size, num_layers, self.config.num_classes)
                genome.max_length = self.config.max_length  # set max_length
                population.append(genome)
            
            # 50% random architectures
            remaining = self.config.population_size - len(population)
            for _ in range(remaining):
                genome = self._create_random_genome()
                population.append(genome)
        
        elif initialization_strategy == "minimal":
            # start with only minimal architectures
            for _ in range(self.config.population_size):
                genome = ArchitectureGenome.create_minimal(
                    self.config.embed_dim, self.config.vocab_size)
                genome.num_classes = self.config.num_classes  # set num_classes for classification
                genome.max_length = self.config.max_length  # set max_length
                population.append(genome)
        
        elif initialization_strategy == "random":
            # completely random initialization
            for _ in range(self.config.population_size):
                genome = self._create_random_genome()
                population.append(genome)
        
        else:
            raise ValueError(f"unknown initialization strategy: {initialization_strategy}")
        
        return population
    
    def _create_random_genome(self) -> ArchitectureGenome:
        """create a random genome"""
        genome = ArchitectureGenome(
            embed_dim=self.config.embed_dim,
            vocab_size=self.config.vocab_size,
            max_layers=self.config.max_layers,
            max_heads=self.config.max_heads
        )
        genome.num_classes = self.config.num_classes  # set num_classes for classification
        genome.max_length = self.config.max_length  # set max_length from config
        
        # add random number of layers
        num_layers = random.randint(1, min(6, self.config.max_layers))
        for _ in range(num_layers):
            genome.add_layer()
        
        # add some random connections
        for _ in range(random.randint(0, num_layers)):
            if len(genome.layers) >= 2:
                from_layer = random.randint(0, len(genome.layers) - 2)
                to_layer = random.randint(from_layer + 1, len(genome.layers) - 1)
                genome.add_connection(from_layer, to_layer, 'skip')
        
        return genome
    
    def evaluate_population(self, population: List[ArchitectureGenome],
                           dataloader, task_type: str = "classification") -> List[ArchitectureGenome]:
        """evaluate entire population"""
        
        results = self.evaluator.batch_evaluate_genomes(
            population, dataloader, task_type, 
            self.config.weight_samples, self.config.parallel_evaluation
        )
        
        # update genome fitness scores
        for genome, result in zip(population, results):
            genome.set_fitness(task_type, result.mean_performance)
            genome.generation = self.generation
        
        return population
    
    def selection(self, population: List[ArchitectureGenome],
                 task_type: str = "classification") -> List[ArchitectureGenome]:
        """select individuals for next generation"""
        
        num_select = self.config.population_size
        
        if self.config.selection_strategy == "tournament":
            selected = []
            for _ in range(num_select):
                selected.append(SelectionStrategies.tournament_selection(
                    population, self.config.tournament_size, task_type))
        
        elif self.config.selection_strategy == "nsga2":
            selected = self.multi_obj_selector.nsga2_selection(
                population, num_select, task_type)
        
        elif self.config.selection_strategy == "adaptive":
            selected = self.adaptive_selector.select(
                population, num_select, task_type)
        
        elif self.config.selection_strategy == "weighted_sum":
            selected = self.multi_obj_selector.weighted_sum_selection(
                population, num_select, task_type)
        
        else:
            raise ValueError(f"unknown selection strategy: {self.config.selection_strategy}")
        
        return selected
    
    def reproduction(self, population: List[ArchitectureGenome],
                    task_type: str = "classification") -> List[ArchitectureGenome]:
        """create offspring through mutation and crossover"""
        
        # calculate how many offspring to create
        num_elite = max(1, int(self.config.population_size * self.config.elitism_rate))
        num_offspring = self.config.population_size - num_elite
        
        # elite individuals (best performers)
        elite = sorted(population, key=lambda x: x.get_fitness(task_type), reverse=True)[:num_elite]
        
        # create offspring
        offspring = []
        
        # crossover + mutation
        if self.config.crossover_rate > 0:
            num_crossover = int(num_offspring * self.config.crossover_rate)
            parent_pairs = select_parents_for_reproduction(
                population, num_crossover, "tournament", task_type)
            
            for parent1, parent2 in parent_pairs:
                if random.random() < self.config.crossover_rate:
                    child = parent1.crossover(parent2)
                else:
                    child = parent1.clone()
                
                # apply mutation
                if random.random() < self.config.mutation_rate:
                    if self.config.adaptive_mutation:
                        child = self.mutation_operators.adaptive_mutation(
                            child, self.generation, self.fitness_stagnation_count)
                    else:
                        child = self.mutation_operators.mutate(child)
                
                offspring.append(child)
        
        # mutation only for remaining offspring
        remaining_offspring = num_offspring - len(offspring)
        for _ in range(remaining_offspring):
            parent = SelectionStrategies.tournament_selection(population, task=task_type)
            
            if self.config.adaptive_mutation:
                child = self.mutation_operators.adaptive_mutation(
                    parent, self.generation, self.fitness_stagnation_count)
            else:
                child = self.mutation_operators.mutate(parent)
            
            offspring.append(child)
        
        # combine elite and offspring
        new_population = elite + offspring
        
        return new_population
    
    def calculate_diversity(self, population: List[ArchitectureGenome]) -> float:
        """calculate population diversity score"""
        if len(population) < 2:
            return 0.0
        
        diversity_metrics = []
        
        for i, genome1 in enumerate(population):
            for j, genome2 in enumerate(population[i+1:], i+1):
                # compare number of layers
                layer_diff = abs(len(genome1.get_active_layers()) - len(genome2.get_active_layers()))
                
                # compare complexity
                complexity_diff = abs(genome1.calculate_complexity() - genome2.calculate_complexity())
                
                # compare activation types
                activations1 = [layer.activation_type for layer in genome1.get_active_layers()]
                activations2 = [layer.activation_type for layer in genome2.get_active_layers()]
                activation_diff = len(set(activations1).symmetric_difference(set(activations2)))
                
                # combined diversity metric
                diversity = layer_diff + complexity_diff / 1000.0 + activation_diff
                diversity_metrics.append(diversity)
        
        return np.mean(diversity_metrics) if diversity_metrics else 0.0
    
    def track_generation_stats(self, population: List[ArchitectureGenome],
                              task_type: str = "classification") -> GenerationStats:
        """calculate and track statistics for current generation"""
        
        fitnesses = [genome.get_fitness(task_type) for genome in population]
        complexities = [genome.calculate_complexity() for genome in population]
        
        stats = GenerationStats(
            generation=self.generation,
            best_fitness=max(fitnesses),
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            best_complexity=min(complexities),
            mean_complexity=np.mean(complexities),
            diversity_score=self.calculate_diversity(population),
            mutation_stats=self.mutation_operators.get_mutation_stats()
        )
        
        # update best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.last_best_fitness:
            self.best_individual = population[best_idx].clone()
            self.last_best_fitness = fitnesses[best_idx]
            self.fitness_stagnation_count = 0
        else:
            self.fitness_stagnation_count += 1
        
        self.evolution_history.append(stats)
        
        return stats
    
    def evolve(self, dataloader, task_type: str = "classification",
              initialization_strategy: str = "mixed",
              log_wandb: bool = False) -> ArchitectureGenome:
        """run the complete evolutionary search"""
        
        self.use_wandb = log_wandb
        
        if log_wandb:
            wandb.init(project="wann-gpt-evolution", config=self.config.__dict__)
        
        print("initializing population...")
        self.population = self.initialize_population(initialization_strategy)
        
        print("starting evolution...")
        for generation in tqdm(range(self.config.num_generations), desc="evolution"):
            self.generation = generation
            
            # evaluate population
            self.population = self.evaluate_population(
                self.population, dataloader, task_type)
            
            # track statistics
            stats = self.track_generation_stats(self.population, task_type)
            
            # log progress
            print(f"generation {generation}: best_fitness={stats.best_fitness:.4f}, "
                  f"mean_fitness={stats.mean_fitness:.4f}, "
                  f"diversity={stats.diversity_score:.4f}")
            
            if log_wandb:
                wandb.log({
                    "generation": generation,
                    "best_fitness": stats.best_fitness,
                    "mean_fitness": stats.mean_fitness,
                    "std_fitness": stats.std_fitness,
                    "best_complexity": stats.best_complexity,
                    "mean_complexity": stats.mean_complexity,
                    "diversity_score": stats.diversity_score,
                    "fitness_stagnation": self.fitness_stagnation_count
                })
            
            # save checkpoint
            if generation % 10 == 0:
                self.save_checkpoint(generation)
            
            # early stopping if converged
            if self.fitness_stagnation_count > self.config.fitness_stagnation_threshold:
                print(f"early stopping due to fitness stagnation at generation {generation}")
                break
            
            # create next generation
            if generation < self.config.num_generations - 1:
                self.population = self.selection(self.population, task_type)
                self.population = self.reproduction(self.population, task_type)
        
        # final evaluation and save
        self.population = self.evaluate_population(self.population, dataloader, task_type)
        self.save_results()
        
        if log_wandb:
            wandb.finish()
        
        return self.best_individual
    
    def save_checkpoint(self, generation: int):
        """save evolution checkpoint"""
        checkpoint = {
            "generation": generation,
            "population": [genome.to_dict() for genome in self.population],
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "evolution_history": self.evolution_history,
            "config": self.config.__dict__,
            "fitness_stagnation_count": self.fitness_stagnation_count,
            "last_best_fitness": self.last_best_fitness
        }
        
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_gen_{generation}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, checkpoint_path: str):
        """load evolution checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.generation = checkpoint["generation"]
        self.population = [ArchitectureGenome.from_dict(genome_dict) 
                          for genome_dict in checkpoint["population"]]
        
        if checkpoint["best_individual"]:
            self.best_individual = ArchitectureGenome.from_dict(checkpoint["best_individual"])
        
        self.evolution_history = checkpoint["evolution_history"]
        self.fitness_stagnation_count = checkpoint["fitness_stagnation_count"]
        self.last_best_fitness = checkpoint["last_best_fitness"]
    
    def save_results(self):
        """save final evolution results"""
        results = {
            "config": self.config.__dict__,
            "evolution_history": [
                {
                    "generation": stats.generation,
                    "best_fitness": stats.best_fitness,
                    "mean_fitness": stats.mean_fitness,
                    "std_fitness": stats.std_fitness,
                    "best_complexity": stats.best_complexity,
                    "mean_complexity": stats.mean_complexity,
                    "diversity_score": stats.diversity_score
                }
                for stats in self.evolution_history
            ],
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "final_population": [genome.to_dict() for genome in self.population]
        }
        
        # save as json
        results_path = os.path.join(self.save_dir, "evolution_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # save best individual separately
        if self.best_individual:
            best_path = os.path.join(self.save_dir, "best_individual.json")
            with open(best_path, 'w') as f:
                json.dump(self.best_individual.to_dict(), f, indent=2)
        
        print(f"results saved to {self.save_dir}")
    
    def get_pareto_front(self, task_type: str = "classification") -> List[ArchitectureGenome]:
        """get pareto-optimal solutions from final population"""
        if not self.population:
            return []
        
        fronts = self.multi_obj_selector.fast_non_dominated_sort(self.population, task_type)
        
        if fronts and fronts[0]:
            return [self.population[i] for i in fronts[0]]
        else:
            return [] 