"""
main evolution engine for wann architecture search
orchestrates the evolutionary algorithm with evaluation, selection, and mutation
"""

import os
import json
import pickle
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import wandb

from .genome import ArchitectureGenome, HeadOnlyGenome
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
            if self.evaluator:  # Ensure evaluator exists
                self.evaluator.wandb_run = wandb.run
        
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
                log_data = {
                    "generation": generation,
                    "best_fitness": stats.best_fitness,
                    "mean_fitness": stats.mean_fitness,
                    "std_fitness": stats.std_fitness,
                    "best_complexity": stats.best_complexity,
                    "mean_complexity": stats.mean_complexity,
                    "diversity_score": stats.diversity_score,
                    "fitness_stagnation": self.fitness_stagnation_count,
                    "mutation_stats": stats.mutation_stats  # Log mutation statistics
                }
                
                if self.best_individual:
                    log_data["best_individual_details"] = {
                        "num_layers": len(self.best_individual.get_active_layers()),
                        "complexity": self.best_individual.calculate_complexity(),
                        "fitness": self.best_individual.get_fitness(task_type),
                        "activation_functions": [layer.activation_type for layer in self.best_individual.get_active_layers()]
                    }
                    
                    # Log best genome structure if a new best is found
                    # A new best is found if fitness_stagnation_count is 0
                    if self.fitness_stagnation_count == 0:
                        log_data["best_genome_structure"] = self.best_individual.to_dict()
                
                wandb.log(log_data)
            
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

class HeadOnlyEvolutionEngine:
    """specialized engine for evolving only output heads with frozen gpt-2 backbone"""
    
    def __init__(self, config: EvolutionConfig, evaluator: SharedWeightEvaluator,
                 save_dir: str = "./head_evolution_results", model_name: str = "gpt2"):
        self.config = config
        self.evaluator = evaluator
        self.save_dir = save_dir
        self.model_name = model_name
        
        # create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # evolution components adapted for head-only evolution
        self.multi_obj_selector = MultiObjectiveSelection(config.complexity_weight)
        
        # evolution state
        self.population: List[HeadOnlyGenome] = []
        self.generation = 0
        self.evolution_history: List[GenerationStats] = []
        self.best_individual: Optional[HeadOnlyGenome] = None
        
        # tracking
        self.fitness_stagnation_count = 0
        self.last_best_fitness = float('-inf')
        
        # logging
        self.use_wandb = False
    
    def initialize_population(self, initialization_strategy: str = "mixed") -> List[HeadOnlyGenome]:
        """initialize population with diverse head configurations"""
        
        population = []
        
        if initialization_strategy == "mixed":
            # mix of dense and sparse heads
            
            # 30% fully dense heads
            dense_count = max(1, (self.config.population_size * 3) // 10)
            for _ in range(dense_count):
                genome = HeadOnlyGenome.create_dense(
                    self.config.embed_dim, self.config.vocab_size, self.config.num_classes)
                population.append(genome)
            
            # 40% sparse heads with varying sparsity
            sparse_count = max(1, (self.config.population_size * 4) // 10)
            for _ in range(sparse_count):
                sparsity = random.uniform(0.1, 0.7)
                genome = HeadOnlyGenome.create_sparse(
                    self.config.embed_dim, self.config.vocab_size, self.config.num_classes, sparsity)
                population.append(genome)
            
            # 30% random connection patterns
            remaining = self.config.population_size - len(population)
            for _ in range(remaining):
                genome = self._create_random_head_genome()
                population.append(genome)
        
        elif initialization_strategy == "dense":
            # start with dense heads
            for _ in range(self.config.population_size):
                genome = HeadOnlyGenome.create_dense(
                    self.config.embed_dim, self.config.vocab_size, self.config.num_classes)
                population.append(genome)
        
        elif initialization_strategy == "sparse":
            # start with sparse heads
            for _ in range(self.config.population_size):
                sparsity = random.uniform(0.3, 0.8)
                genome = HeadOnlyGenome.create_sparse(
                    self.config.embed_dim, self.config.vocab_size, self.config.num_classes, sparsity)
                population.append(genome)
        
        return population
    
    def _create_random_head_genome(self) -> HeadOnlyGenome:
        """create genome with random head configuration"""
        genome = HeadOnlyGenome(
            embed_dim=self.config.embed_dim,
            vocab_size=self.config.vocab_size,
            num_classes=self.config.num_classes
        )
        
        # random sparsity levels
        genome.lm_head_sparsity = random.uniform(0.0, 0.8)
        if genome.num_classes is not None:
            genome.classifier_sparsity = random.uniform(0.0, 0.8)
        
        # generate random connection patterns
        genome.randomize_connections()
        
        return genome
    
    def evaluate_population(self, population: List[HeadOnlyGenome],
                           dataloader, task_type: str = "classification") -> List[HeadOnlyGenome]:
        """evaluate population using hybrid models"""
        
        # convert head genomes to hybrid models and evaluate
        results = []
        for genome in tqdm(population, desc=f"evaluating generation {self.generation}"):
            try:
                # create hybrid model from head genome
                model = self.evaluator.instantiate_hybrid_from_genome(
                    genome, self.model_name, freeze_backbone=True)
                
                # evaluate model
                if task_type == "classification":
                    result = self.evaluator.evaluate_classification(model, dataloader)
                elif task_type == "generation":
                    result = self.evaluator.evaluate_generation(model, dataloader)
                else:
                    raise ValueError(f"unsupported task type: {task_type}")
                
                # update genome fitness
                genome.set_fitness(task_type, result.mean_performance)
                genome.generation = self.generation
                results.append(result)
                
            except Exception as e:
                print(f"evaluation failed for genome: {e}")
                # assign low fitness for failed evaluations
                genome.set_fitness(task_type, -1000.0)
                results.append(None)
        
        return population
    
    def selection(self, population: List[HeadOnlyGenome],
                 task_type: str = "classification") -> List[HeadOnlyGenome]:
        """select head genomes for next generation"""
        
        num_select = self.config.population_size
        
        if self.config.selection_strategy == "tournament":
            selected = []
            for _ in range(num_select):
                # tournament selection adapted for head genomes
                tournament = random.sample(population, min(self.config.tournament_size, len(population)))
                winner = max(tournament, key=lambda x: x.get_fitness(task_type))
                selected.append(winner)
        
        elif self.config.selection_strategy == "weighted_sum":
            # sort by combined score (fitness - complexity penalty)
            scored_pop = []
            for genome in population:
                raw_fitness = genome.get_fitness(task_type)
                complexity = genome.calculate_complexity()
                
                # normalize complexity penalty
                max_complexity = self.config.embed_dim * (self.config.vocab_size + (genome.num_classes or 0))
                normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0.0
                
                # combined score with complexity penalty
                combined_score = raw_fitness - self.config.complexity_weight * normalized_complexity
                
                scored_pop.append((combined_score, raw_fitness, complexity, genome))
            
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            selected = [genome for _, _, _, genome in scored_pop[:num_select]]
            
            # print top performers for debugging
            if self.generation % 5 == 0:  # every 5 generations
                print(f"\ntop 3 performers (gen {self.generation}):")
                for i, (score, fitness, complexity, genome) in enumerate(scored_pop[:3]):
                    sparsity = f"lm:{genome.lm_head_sparsity:.2f}, cls:{genome.classifier_sparsity:.2f}"
                    print(f"  #{i+1}: score={score:.4f}, fitness={fitness:.4f}, complexity={complexity:,}, sparsity=({sparsity})")
        
        else:
            # default to fitness-based selection with complexity consideration
            def fitness_with_penalty(genome):
                raw_fitness = genome.get_fitness(task_type)
                complexity = genome.calculate_complexity()
                max_complexity = self.config.embed_dim * (self.config.vocab_size + (genome.num_classes or 0))
                normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0.0
                return raw_fitness - self.config.complexity_weight * normalized_complexity
            
            selected = sorted(population, key=fitness_with_penalty, reverse=True)[:num_select]
        
        return selected
    
    def reproduction(self, population: List[HeadOnlyGenome],
                    task_type: str = "classification") -> List[HeadOnlyGenome]:
        """create offspring through mutation and crossover"""
        
        # calculate offspring counts
        num_elite = max(1, int(self.config.population_size * self.config.elitism_rate))
        num_offspring = self.config.population_size - num_elite
        
        # elite individuals
        elite = sorted(population, key=lambda x: x.get_fitness(task_type), reverse=True)[:num_elite]
        
        # create offspring
        offspring = []
        
        for _ in range(num_offspring):
            if random.random() < self.config.crossover_rate and len(population) > 1:
                # crossover
                parent1, parent2 = random.sample(population, 2)
                child = parent1.crossover(parent2)
            else:
                # mutation only
                parent = random.choice(population)
                child = parent.clone()
            
            # apply diverse mutations with different probabilities
            mutation_applied = False
            
            # sparsity mutations (most common)
            if random.random() < self.config.mutation_rate:
                child.mutate_sparsity(mutation_rate=0.4)
                mutation_applied = True
            
            # aggressive sparsity changes (less common)
            if random.random() < 0.15:  # 15% chance
                # randomly increase or decrease sparsity significantly
                if random.random() < 0.5:
                    child.lm_head_sparsity = min(0.95, child.lm_head_sparsity + random.uniform(0.1, 0.3))
                else:
                    child.lm_head_sparsity = max(0.0, child.lm_head_sparsity - random.uniform(0.1, 0.3))
                
                if child.num_classes is not None:
                    if random.random() < 0.5:
                        child.classifier_sparsity = min(0.95, child.classifier_sparsity + random.uniform(0.1, 0.3))
                    else:
                        child.classifier_sparsity = max(0.0, child.classifier_sparsity - random.uniform(0.1, 0.3))
                mutation_applied = True
            
            # connection pattern regeneration (least common but most disruptive)
            if random.random() < 0.1:  # 10% chance to completely randomize connections
                child.randomize_connections()
                mutation_applied = True
            
            # selective connection mutations (medium frequency)
            if random.random() < 0.25 and child.lm_head_connections:  # 25% chance
                # flip some random connections in lm head
                for i in range(len(child.lm_head_connections)):
                    for j in range(len(child.lm_head_connections[i])):
                        if random.random() < 0.05:  # 5% of connections get flipped
                            child.lm_head_connections[i][j] = 1 - child.lm_head_connections[i][j]
                mutation_applied = True
            
            if random.random() < 0.25 and child.classifier_connections:  # 25% chance
                # flip some random connections in classifier
                for i in range(len(child.classifier_connections)):
                    for j in range(len(child.classifier_connections[i])):
                        if random.random() < 0.05:  # 5% of connections get flipped
                            child.classifier_connections[i][j] = 1 - child.classifier_connections[i][j]
                mutation_applied = True
            
            # ensure at least one mutation is applied
            if not mutation_applied:
                child.mutate_sparsity(mutation_rate=0.2)
            
            offspring.append(child)
        
        return elite + offspring
    
    def evolve(self, dataloader, task_type: str = "classification",
              initialization_strategy: str = "mixed",
              log_wandb: bool = False) -> HeadOnlyGenome:
        """run head-only evolution"""
        
        self.use_wandb = log_wandb
        
        # initialize population
        print("initializing head population...")
        self.population = self.initialize_population(initialization_strategy)
        
        for generation in range(self.config.num_generations):
            self.generation = generation
            print(f"\ngeneration {generation + 1}/{self.config.num_generations}")
            
            # evaluate population
            self.population = self.evaluate_population(self.population, dataloader, task_type)
            
            # track best individual
            current_best = max(self.population, key=lambda x: x.get_fitness(task_type))
            if self.best_individual is None or current_best.get_fitness(task_type) > self.best_individual.get_fitness(task_type):
                self.best_individual = current_best.clone()
                self.fitness_stagnation_count = 0
            else:
                self.fitness_stagnation_count += 1
            
            # log progress
            best_fitness = current_best.get_fitness(task_type)
            mean_fitness = np.mean([g.get_fitness(task_type) for g in self.population])
            best_complexity = current_best.calculate_complexity()
            
            print(f"best fitness: {best_fitness:.4f}, mean: {mean_fitness:.4f}, complexity: {best_complexity}")
            
            # early stopping
            if self.fitness_stagnation_count >= self.config.fitness_stagnation_threshold:
                print(f"early stopping: fitness stagnated for {self.fitness_stagnation_count} generations")
                break
            
            # selection and reproduction
            if generation < self.config.num_generations - 1:
                selected = self.selection(self.population, task_type)
                self.population = self.reproduction(selected, task_type)
            
            # save checkpoint
            if generation % 10 == 0:
                self.save_checkpoint(generation)
        
        print(f"\nevolution completed. best fitness: {self.best_individual.get_fitness(task_type):.4f}")
        self.save_results()
        
        return self.best_individual
    
    def save_checkpoint(self, generation: int):
        """save evolution checkpoint"""
        checkpoint = {
            'generation': generation,
            'population': [genome.to_dict() for genome in self.population],
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'fitness_stagnation_count': self.fitness_stagnation_count
        }
        
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_gen_{generation}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def save_results(self):
        """save final evolution results"""
        if self.best_individual:
            # save best individual
            best_path = os.path.join(self.save_dir, "best_head_genome.json")
            with open(best_path, 'w') as f:
                json.dump(self.best_individual.to_dict(), f, indent=2)
            
            print(f"saved best head genome to {best_path}") 