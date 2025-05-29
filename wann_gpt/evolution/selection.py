"""
selection strategies for evolutionary architecture search
implements multi-objective selection balancing performance and complexity
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Callable
from .genome import ArchitectureGenome

class SelectionStrategies:
    """collection of selection strategies for evolution"""
    
    @staticmethod
    def tournament_selection(population: List[ArchitectureGenome], 
                           tournament_size: int = 3,
                           task: str = "classification") -> ArchitectureGenome:
        """tournament selection based on fitness"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # select best individual from tournament
        best = tournament[0]
        for individual in tournament[1:]:
            if individual.get_fitness(task) > best.get_fitness(task):
                best = individual
        
        return best
    
    @staticmethod
    def roulette_selection(population: List[ArchitectureGenome],
                          task: str = "classification") -> ArchitectureGenome:
        """roulette wheel selection based on fitness"""
        fitnesses = [individual.get_fitness(task) for individual in population]
        
        # handle negative fitness by shifting
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness + 1e-6 for f in fitnesses]
        
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)
        
        # select based on probability proportional to fitness
        selection_probs = [f / total_fitness for f in fitnesses]
        selected_idx = np.random.choice(len(population), p=selection_probs)
        
        return population[selected_idx]
    
    @staticmethod
    def rank_selection(population: List[ArchitectureGenome],
                      task: str = "classification",
                      pressure: float = 2.0) -> ArchitectureGenome:
        """rank-based selection"""
        # sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.get_fitness(task))
        
        # assign ranks (1 = worst, n = best)
        ranks = list(range(1, len(population) + 1))
        
        # linear ranking probability
        n = len(population)
        probs = [(pressure - 2) * (rank - 1) / (n - 1) + 2 - pressure 
                for rank in ranks]
        probs = [max(0, p) for p in probs]  # ensure non-negative
        
        # normalize probabilities
        total_prob = sum(probs)
        if total_prob > 0:
            probs = [p / total_prob for p in probs]
            selected_idx = np.random.choice(len(population), p=probs)
            return sorted_pop[selected_idx]
        else:
            return random.choice(population)

class MultiObjectiveSelection:
    """multi-objective selection for balancing performance and complexity"""
    
    def __init__(self, complexity_weight: float = 0.1):
        self.complexity_weight = complexity_weight
    
    def pareto_dominance(self, individual1: ArchitectureGenome, 
                        individual2: ArchitectureGenome,
                        task: str = "classification") -> int:
        """check pareto dominance between two individuals
        returns: 1 if individual1 dominates, -1 if individual2 dominates, 0 if non-dominated
        """
        
        # get objectives: performance (maximize), complexity (minimize)
        perf1 = individual1.get_fitness(task)
        perf2 = individual2.get_fitness(task)
        complexity1 = individual1.calculate_complexity()
        complexity2 = individual2.calculate_complexity()
        
        # individual1 dominates if it's better or equal in all objectives
        # and strictly better in at least one
        better_perf = perf1 >= perf2
        better_complexity = complexity1 <= complexity2
        
        strictly_better_perf = perf1 > perf2
        strictly_better_complexity = complexity1 < complexity2
        
        if better_perf and better_complexity and (strictly_better_perf or strictly_better_complexity):
            return 1  # individual1 dominates
        elif perf2 >= perf1 and complexity2 <= complexity1 and (perf2 > perf1 or complexity2 < complexity1):
            return -1  # individual2 dominates
        else:
            return 0  # non-dominated
    
    def fast_non_dominated_sort(self, population: List[ArchitectureGenome],
                               task: str = "classification") -> List[List[int]]:
        """fast non-dominated sorting (nsga-ii algorithm)"""
        n = len(population)
        
        # for each individual, track domination count and dominated solutions
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        # compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dominance = self.pareto_dominance(population[i], population[j], task)
                
                if dominance == 1:  # i dominates j
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif dominance == -1:  # j dominates i
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        
        # first front: individuals with domination count 0
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
                current_front += 1
            else:
                break
        
        return fronts
    
    def crowding_distance(self, front: List[int], population: List[ArchitectureGenome],
                         task: str = "classification") -> List[float]:
        """calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        
        # get objectives for this front
        performances = [population[i].get_fitness(task) for i in front]
        complexities = [population[i].calculate_complexity() for i in front]
        
        # normalize objectives
        perf_range = max(performances) - min(performances)
        comp_range = max(complexities) - min(complexities)
        
        if perf_range == 0:
            perf_range = 1.0
        if comp_range == 0:
            comp_range = 1.0
        
        # sort by each objective and assign crowding distance
        for obj_idx, values in enumerate([performances, complexities]):
            # sort individuals by this objective
            sorted_indices = sorted(range(len(front)), key=lambda i: values[i])
            
            # boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # calculate distance for middle points
            for i in range(1, len(sorted_indices) - 1):
                if distances[sorted_indices[i]] != float('inf'):
                    distance_contribution = (values[sorted_indices[i + 1]] - 
                                           values[sorted_indices[i - 1]])
                    if obj_idx == 0:  # performance
                        distance_contribution /= perf_range
                    else:  # complexity
                        distance_contribution /= comp_range
                    
                    distances[sorted_indices[i]] += distance_contribution
        
        return distances
    
    def nsga2_selection(self, population: List[ArchitectureGenome],
                       num_select: int, task: str = "classification") -> List[ArchitectureGenome]:
        """nsga-ii selection"""
        if num_select >= len(population):
            return population.copy()
        
        # non-dominated sorting
        fronts = self.fast_non_dominated_sort(population, task)
        
        selected = []
        front_idx = 0
        
        # select entire fronts until we can't fit another complete front
        while front_idx < len(fronts) and len(selected) + len(fronts[front_idx]) <= num_select:
            selected.extend([population[i] for i in fronts[front_idx]])
            front_idx += 1
        
        # if we need to select partial front, use crowding distance
        if len(selected) < num_select and front_idx < len(fronts):
            remaining = num_select - len(selected)
            current_front = fronts[front_idx]
            
            # calculate crowding distances
            distances = self.crowding_distance(current_front, population, task)
            
            # sort by crowding distance (descending)
            sorted_front = sorted(zip(current_front, distances), 
                                key=lambda x: x[1], reverse=True)
            
            # select top individuals by crowding distance
            for i in range(remaining):
                selected.append(population[sorted_front[i][0]])
        
        return selected
    
    def weighted_sum_selection(self, population: List[ArchitectureGenome],
                              num_select: int, task: str = "classification") -> List[ArchitectureGenome]:
        """selection using weighted sum of objectives"""
        
        # calculate weighted scores
        scores = []
        for individual in population:
            performance = individual.get_fitness(task)
            complexity = individual.calculate_complexity()
            
            # normalize complexity (lower is better, so we subtract from max)
            max_complexity = max(ind.calculate_complexity() for ind in population)
            normalized_complexity = (max_complexity - complexity) / max_complexity if max_complexity > 0 else 0
            
            # weighted sum
            score = performance + self.complexity_weight * normalized_complexity
            scores.append(score)
        
        # select top individuals
        selected_indices = sorted(range(len(population)), 
                                key=lambda i: scores[i], reverse=True)[:num_select]
        
        return [population[i] for i in selected_indices]
    
    def epsilon_constraint_selection(self, population: List[ArchitectureGenome],
                                   num_select: int, 
                                   complexity_threshold: int,
                                   task: str = "classification") -> List[ArchitectureGenome]:
        """epsilon constraint method: maximize performance subject to complexity constraint"""
        
        # filter individuals that meet complexity constraint
        feasible = [ind for ind in population 
                   if ind.calculate_complexity() <= complexity_threshold]
        
        if not feasible:
            # if no feasible solutions, relax constraint and select least complex
            feasible = sorted(population, key=lambda x: x.calculate_complexity())[:num_select]
        
        # select best performing feasible solutions
        feasible.sort(key=lambda x: x.get_fitness(task), reverse=True)
        
        return feasible[:num_select]

class AdaptiveSelection:
    """adaptive selection that changes strategy based on evolution progress"""
    
    def __init__(self):
        self.multi_obj_selector = MultiObjectiveSelection()
        self.generation_count = 0
        self.fitness_history = []
    
    def select(self, population: List[ArchitectureGenome], num_select: int,
              task: str = "classification") -> List[ArchitectureGenome]:
        """adaptive selection strategy"""
        
        # track fitness progress
        current_best_fitness = max(ind.get_fitness(task) for ind in population)
        self.fitness_history.append(current_best_fitness)
        
        # detect stagnation
        stagnation_threshold = 5
        is_stagnating = (len(self.fitness_history) >= stagnation_threshold and
                        all(abs(f - current_best_fitness) < 1e-6 
                            for f in self.fitness_history[-stagnation_threshold:]))
        
        # adapt selection strategy
        if self.generation_count < 10:
            # early generations: focus on diversity (nsga-ii)
            selected = self.multi_obj_selector.nsga2_selection(population, num_select, task)
        elif is_stagnating:
            # stagnation: increase selection pressure (tournament)
            selected = []
            for _ in range(num_select):
                selected.append(SelectionStrategies.tournament_selection(
                    population, tournament_size=5, task=task))
        else:
            # normal evolution: balanced approach (weighted sum)
            selected = self.multi_obj_selector.weighted_sum_selection(
                population, num_select, task)
        
        self.generation_count += 1
        return selected
    
    def reset(self):
        """reset adaptive selection state"""
        self.generation_count = 0
        self.fitness_history = []

def select_parents_for_reproduction(population: List[ArchitectureGenome],
                                  num_offspring: int,
                                  selection_strategy: str = "tournament",
                                  task: str = "classification") -> List[Tuple[ArchitectureGenome, ArchitectureGenome]]:
    """select parent pairs for reproduction"""
    
    parent_pairs = []
    
    for _ in range(num_offspring):
        if selection_strategy == "tournament":
            parent1 = SelectionStrategies.tournament_selection(population, task=task)
            parent2 = SelectionStrategies.tournament_selection(population, task=task)
        elif selection_strategy == "roulette":
            parent1 = SelectionStrategies.roulette_selection(population, task=task)
            parent2 = SelectionStrategies.roulette_selection(population, task=task)
        elif selection_strategy == "rank":
            parent1 = SelectionStrategies.rank_selection(population, task=task)
            parent2 = SelectionStrategies.rank_selection(population, task=task)
        else:
            # default to random selection
            parent1 = random.choice(population)
            parent2 = random.choice(population)
        
        parent_pairs.append((parent1, parent2))
    
    return parent_pairs