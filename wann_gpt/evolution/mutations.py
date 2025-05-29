"""
mutation operators for evolving transformer architectures
implements the core wann mutation strategies adapted for transformers
"""

import numpy as np
import random
from typing import List, Callable
from .genome import ArchitectureGenome, LayerGene, ConnectionGene
from ..architecture.activations import ActivationType

class MutationOperators:
    """collection of mutation operators for architecture evolution"""
    
    def __init__(self, mutation_rates: dict = None):
        # default mutation rates
        default_rates = {
            'add_layer': 0.1,
            'remove_layer': 0.05,
            'add_connection': 0.15,
            'remove_connection': 0.1,
            'change_activation': 0.2,
            'add_head': 0.1,
            'remove_head': 0.05,
            'prune_connections': 0.1,
            'toggle_skip': 0.1,
            'mutate_head_count': 0.1
        }
        
        self.mutation_rates = mutation_rates or default_rates
        
        # list of available mutations
        self.mutations: List[Callable] = [
            self.add_layer_mutation,
            self.remove_layer_mutation,
            self.add_connection_mutation,
            self.remove_connection_mutation,
            self.change_activation_mutation,
            self.add_head_mutation,
            self.remove_head_mutation,
            self.prune_connections_mutation,
            self.toggle_skip_mutation,
            self.mutate_head_count_mutation
        ]
    
    def mutate(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """apply random mutations to genome"""
        mutated = genome.clone()
        
        # apply each type of mutation with its probability
        for mutation_fn in self.mutations:
            mutation_name = mutation_fn.__name__.replace('_mutation', '')
            if random.random() < self.mutation_rates.get(mutation_name, 0.1):
                try:
                    mutation_fn(mutated)
                except Exception as e:
                    # log mutation failure but continue
                    print(f"mutation {mutation_name} failed: {e}")
        
        return mutated
    
    def add_layer_mutation(self, genome: ArchitectureGenome):
        """add a new transformer layer (insert layer/node mutation)"""
        if len(genome.layers) >= genome.max_layers:
            return  # cannot add more layers
        
        # create new layer with random properties
        layer_id = len(genome.layers)
        new_layer = LayerGene(
            layer_id=layer_id,
            is_active=True,
            num_heads=random.randint(1, min(8, genome.embed_dim // 64)),
            hidden_dim=genome.embed_dim * random.choice([1, 2, 4]),
            activation_type=random.choice(list(ActivationType)),
            skip_attention=random.random() > 0.2,  # 80% chance to keep attention
            skip_feedforward=random.random() > 0.2  # 80% chance to keep feedforward
        )
        
        genome.layers.append(new_layer)
        
        # add connections to/from existing layers
        if len(genome.layers) > 1:
            # connect to previous layer
            prev_layer = random.randint(0, len(genome.layers) - 2)
            genome.add_connection(prev_layer, layer_id, 'sequential')
            
            # maybe add skip connection to earlier layer
            if len(genome.layers) > 2 and random.random() < 0.3:
                skip_layer = random.randint(0, len(genome.layers) - 3)
                genome.add_connection(skip_layer, layer_id, 'skip')
    
    def remove_layer_mutation(self, genome: ArchitectureGenome):
        """remove a layer by deactivating it"""
        active_layers = genome.get_active_layers()
        if len(active_layers) <= 1:
            return  # keep at least one layer
        
        # select random active layer to remove
        layer_to_remove = random.choice(active_layers)
        genome.remove_layer(layer_to_remove.layer_id)
    
    def add_connection_mutation(self, genome: ArchitectureGenome):
        """add a new connection between layers"""
        active_layers = genome.get_active_layers()
        if len(active_layers) < 2:
            return
        
        # select two different layers
        from_layer = random.choice(active_layers)
        to_layer = random.choice(active_layers)
        
        # ensure from_layer comes before to_layer
        if from_layer.layer_id >= to_layer.layer_id:
            from_layer, to_layer = to_layer, from_layer
        
        if from_layer.layer_id == to_layer.layer_id:
            return  # cannot connect layer to itself
        
        # add connection
        connection_type = random.choice(['skip', 'attention', 'feedforward'])
        genome.add_connection(from_layer.layer_id, to_layer.layer_id, connection_type)
    
    def remove_connection_mutation(self, genome: ArchitectureGenome):
        """remove an existing connection"""
        active_connections = genome.get_active_connections()
        if not active_connections:
            return
        
        # select random connection to remove
        conn_to_remove = random.choice(active_connections)
        genome.remove_connection(conn_to_remove.from_layer, 
                               conn_to_remove.to_layer,
                               conn_to_remove.connection_type)
    
    def change_activation_mutation(self, genome: ArchitectureGenome):
        """change activation function of a random layer"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        # select random layer and new activation
        layer = random.choice(active_layers)
        new_activation = random.choice(list(ActivationType))
        
        # ensure it's different from current activation
        if layer.activation_type != new_activation:
            layer.activation_type = new_activation
    
    def add_head_mutation(self, genome: ArchitectureGenome):
        """add attention head to a random layer"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        layer = random.choice(active_layers)
        if layer.num_heads < genome.max_heads:
            layer.num_heads += 1
    
    def remove_head_mutation(self, genome: ArchitectureGenome):
        """remove attention head from a random layer"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        layer = random.choice(active_layers)
        if layer.num_heads > 1:
            layer.num_heads -= 1
    
    def prune_connections_mutation(self, genome: ArchitectureGenome):
        """randomly prune connections in a layer"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        layer = random.choice(active_layers)
        connection_type = random.choice(['attention', 'feedforward'])
        keep_prob = random.uniform(0.5, 0.9)  # keep 50-90% of connections
        
        genome.prune_connections(layer.layer_id, connection_type, keep_prob)
    
    def toggle_skip_mutation(self, genome: ArchitectureGenome):
        """toggle skip connections in a layer"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        layer = random.choice(active_layers)
        connection_type = random.choice(['attention', 'feedforward'])
        
        genome.toggle_skip_connection(layer.layer_id, connection_type)
    
    def mutate_head_count_mutation(self, genome: ArchitectureGenome):
        """randomly adjust number of attention heads"""
        active_layers = genome.get_active_layers()
        if not active_layers:
            return
        
        layer = random.choice(active_layers)
        
        # increase or decrease heads
        if random.random() < 0.5 and layer.num_heads > 1:
            layer.num_heads -= 1
        elif layer.num_heads < genome.max_heads:
            layer.num_heads += 1
    
    def adaptive_mutation(self, genome: ArchitectureGenome, 
                         generation: int, fitness_stagnation: int) -> ArchitectureGenome:
        """apply adaptive mutation based on evolution progress"""
        mutated = genome.clone()
        
        # increase mutation rate if fitness is stagnating
        mutation_multiplier = 1.0
        if fitness_stagnation > 10:
            mutation_multiplier = 2.0
        elif fitness_stagnation > 5:
            mutation_multiplier = 1.5
        
        # adjust mutation rates based on generation
        adapted_rates = {}
        for name, rate in self.mutation_rates.items():
            adapted_rates[name] = min(rate * mutation_multiplier, 0.8)
        
        # apply mutations with adapted rates
        for mutation_fn in self.mutations:
            mutation_name = mutation_fn.__name__.replace('_mutation', '')
            if random.random() < adapted_rates.get(mutation_name, 0.1):
                try:
                    mutation_fn(mutated)
                except Exception as e:
                    print(f"adaptive mutation {mutation_name} failed: {e}")
        
        return mutated
    
    def large_mutation(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """apply large-scale structural mutations"""
        mutated = genome.clone()
        
        # apply multiple mutations
        num_mutations = random.randint(2, 5)
        for _ in range(num_mutations):
            mutation_fn = random.choice(self.mutations)
            try:
                mutation_fn(mutated)
            except Exception as e:
                print(f"large mutation failed: {e}")
        
        return mutated
    
    def complexity_aware_mutation(self, genome: ArchitectureGenome, 
                                target_complexity: int) -> ArchitectureGenome:
        """apply mutations guided by complexity target"""
        mutated = genome.clone()
        current_complexity = mutated.calculate_complexity()
        
        # if too complex, prefer structure-reducing mutations
        if current_complexity > target_complexity:
            reducing_mutations = [
                self.remove_layer_mutation,
                self.remove_connection_mutation,
                self.remove_head_mutation,
                self.prune_connections_mutation
            ]
            mutation_fn = random.choice(reducing_mutations)
            mutation_fn(mutated)
        
        # if too simple, prefer structure-adding mutations
        elif current_complexity < target_complexity * 0.8:
            adding_mutations = [
                self.add_layer_mutation,
                self.add_connection_mutation,
                self.add_head_mutation
            ]
            mutation_fn = random.choice(adding_mutations)
            mutation_fn(mutated)
        
        # otherwise, apply regular mutations
        else:
            regular_mutations = [
                self.change_activation_mutation,
                self.toggle_skip_mutation,
                self.mutate_head_count_mutation
            ]
            mutation_fn = random.choice(regular_mutations)
            mutation_fn(mutated)
        
        return mutated
    
    def set_mutation_rate(self, mutation_type: str, rate: float):
        """set mutation rate for specific type"""
        if mutation_type in self.mutation_rates:
            self.mutation_rates[mutation_type] = max(0.0, min(1.0, rate))
    
    def get_mutation_stats(self) -> dict:
        """get current mutation rates"""
        return self.mutation_rates.copy()

class SpecializedMutations:
    """specialized mutations for specific scenarios"""
    
    @staticmethod
    def classification_focused_mutation(genome: ArchitectureGenome) -> ArchitectureGenome:
        """mutations focused on improving classification performance"""
        mutated = genome.clone()
        
        # favor attention mechanisms for classification
        if random.random() < 0.3:
            active_layers = mutated.get_active_layers()
            if active_layers:
                layer = random.choice(active_layers)
                if layer.num_heads < 8:
                    layer.num_heads += 1
        
        # ensure skip connections are active (good for classification)
        for layer in mutated.get_active_layers():
            if random.random() < 0.8:
                layer.skip_attention = True
                layer.skip_feedforward = True
        
        return mutated
    
    @staticmethod
    def generation_focused_mutation(genome: ArchitectureGenome) -> ArchitectureGenome:
        """mutations focused on improving generation performance"""
        mutated = genome.clone()
        
        # favor deeper networks for generation
        if len(mutated.layers) < 6 and random.random() < 0.4:
            MutationOperators().add_layer_mutation(mutated)
        
        # ensure causal structure is maintained
        mutated.causal = True
        
        return mutated
    
    @staticmethod
    def minimal_mutation(genome: ArchitectureGenome) -> ArchitectureGenome:
        """very conservative mutations for fine-tuning"""
        mutated = genome.clone()
        
        # only change activations or prune slightly
        if random.random() < 0.5:
            MutationOperators().change_activation_mutation(mutated)
        else:
            active_layers = mutated.get_active_layers()
            if active_layers:
                layer = random.choice(active_layers)
                # light pruning
                mutated.prune_connections(layer.layer_id, 'feedforward', 0.95)
        
        return mutated 