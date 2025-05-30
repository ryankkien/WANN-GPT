"""
genome representation for evolvable transformer architectures
encodes layers, connections, heads, and activation functions
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import copy

from ..architecture.activations import ActivationType

@dataclass
class LayerGene:
    """represents a single transformer layer in the genome"""
    layer_id: int
    is_active: bool = True
    num_heads: int = 8
    hidden_dim: int = 2048
    activation_type: ActivationType = ActivationType.RELU
    skip_attention: bool = True
    skip_feedforward: bool = True
    
    # connection masks (will be populated during instantiation)
    attention_connections: Optional[Dict] = None
    feedforward_connections: Optional[Dict] = None
    
    def __post_init__(self):
        if self.attention_connections is None:
            self.attention_connections = {}
        if self.feedforward_connections is None:
            self.feedforward_connections = {}

@dataclass
class ConnectionGene:
    """represents a connection between layers"""
    from_layer: int
    to_layer: int
    connection_type: str  # 'attention', 'feedforward', 'skip'
    weight: float = 1.0
    is_active: bool = True

class ArchitectureGenome:
    """genome encoding for transformer architecture evolution"""
    
    def __init__(self, embed_dim: int = 512, vocab_size: int = 1000,
                 max_layers: int = 12, max_heads: int = 16):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.max_heads = max_heads
        
        # core architecture genes
        self.layers: List[LayerGene] = []
        self.connections: List[ConnectionGene] = []
        
        # global parameters
        self.num_classes: Optional[int] = None
        self.max_length: int = 1024
        self.dropout: float = 0.1
        self.embedding_type: str = "random"
        self.causal: bool = True
        
        # fitness tracking
        self.fitness_scores: Dict[str, float] = {}
        self.complexity_score: int = 0
        self.generation: int = 0
    
    @classmethod
    def create_minimal(cls, embed_dim: int = 512, vocab_size: int = 1000) -> 'ArchitectureGenome':
        """create minimal genome with no hidden layers"""
        genome = cls(embed_dim, vocab_size)
        
        # add direct input-output connection (no hidden layers)
        genome.connections.append(
            ConnectionGene(from_layer=-1, to_layer=-2, 
                         connection_type='direct', is_active=True)
        )
        
        return genome
    
    @classmethod 
    def create_simple(cls, embed_dim: int = 512, vocab_size: int = 1000,
                     num_layers: int = 1, num_classes: Optional[int] = None) -> 'ArchitectureGenome':
        """create simple genome with minimal architecture"""
        
        genome = cls(embed_dim, vocab_size)
        genome.num_classes = num_classes
        
        # add minimal layers
        for i in range(num_layers):
            layer = LayerGene(
                layer_id=i,
                is_active=True,
                num_heads=4,
                hidden_dim=embed_dim * 2,
                activation_type=ActivationType.RELU
            )
            genome.layers.append(layer)
        
        # add basic connections
        for i in range(num_layers - 1):
            genome.add_connection(i, i + 1, 'feedforward')
        
        return genome
    
    @classmethod
    def create_standard(cls, embed_dim: int = 512, vocab_size: int = 1000,
                       num_layers: int = 6, num_heads: int = 8,
                       num_classes: Optional[int] = None) -> 'ArchitectureGenome':
        """create standard transformer architecture genome"""
        
        genome = cls(embed_dim, vocab_size, max_layers=num_layers, max_heads=num_heads)
        genome.num_classes = num_classes
        
        # add standard transformer layers
        for i in range(num_layers):
            layer = LayerGene(
                layer_id=i,
                is_active=True,
                num_heads=num_heads,
                hidden_dim=embed_dim * 4,  # standard 4x expansion
                activation_type=ActivationType.GELU,  # gpt-2 uses gelu
                skip_attention=True,
                skip_feedforward=True
            )
            genome.layers.append(layer)
        
        # add sequential connections (standard transformer stack)
        for i in range(num_layers - 1):
            genome.add_connection(i, i + 1, 'attention')
            genome.add_connection(i, i + 1, 'feedforward')
        
        return genome
    
    def add_layer(self, layer_gene: LayerGene = None) -> int:
        """add a new layer to the genome"""
        if len(self.layers) >= self.max_layers:
            return -1  # cannot add more layers
        
        if layer_gene is None:
            # create random layer
            layer_id = len(self.layers)
            layer_gene = LayerGene(
                layer_id=layer_id,
                is_active=True,
                num_heads=np.random.randint(1, min(self.max_heads, self.embed_dim // 32) + 1),
                hidden_dim=self.embed_dim * np.random.choice([1, 2, 4]),
                activation_type=np.random.choice(list(ActivationType))
            )
        
        self.layers.append(layer_gene)
        return layer_gene.layer_id
    
    def remove_layer(self, layer_id: int):
        """remove layer by deactivating it"""
        for layer in self.layers:
            if layer.layer_id == layer_id:
                layer.is_active = False
                break
        
        # deactivate connections involving this layer
        for conn in self.connections:
            if conn.from_layer == layer_id or conn.to_layer == layer_id:
                conn.is_active = False
    
    def add_connection(self, from_layer: int, to_layer: int, 
                      connection_type: str = 'skip'):
        """add connection between layers"""
        # check if connection already exists
        for conn in self.connections:
            if (conn.from_layer == from_layer and 
                conn.to_layer == to_layer and
                conn.connection_type == connection_type):
                conn.is_active = True  # reactivate if exists
                return
        
        # add new connection
        self.connections.append(
            ConnectionGene(from_layer, to_layer, connection_type)
        )
    
    def remove_connection(self, from_layer: int, to_layer: int, 
                         connection_type: str = None):
        """remove connection by deactivating it"""
        for conn in self.connections:
            if (conn.from_layer == from_layer and 
                conn.to_layer == to_layer and 
                (connection_type is None or conn.connection_type == connection_type)):
                conn.is_active = False
    
    def mutate_activation(self, layer_id: int):
        """mutate activation function of specific layer"""
        for layer in self.layers:
            if layer.layer_id == layer_id and layer.is_active:
                layer.activation_type = np.random.choice(list(ActivationType))
                break
    
    def add_attention_head(self, layer_id: int):
        """add attention head to specific layer"""
        for layer in self.layers:
            if layer.layer_id == layer_id and layer.is_active:
                if layer.num_heads < self.max_heads:
                    layer.num_heads += 1
                break
    
    def remove_attention_head(self, layer_id: int):
        """remove attention head from specific layer"""
        for layer in self.layers:
            if layer.layer_id == layer_id and layer.is_active:
                if layer.num_heads > 1:
                    layer.num_heads -= 1
                break
    
    def toggle_skip_connection(self, layer_id: int, connection_type: str):
        """toggle skip connection for layer"""
        for layer in self.layers:
            if layer.layer_id == layer_id and layer.is_active:
                if connection_type == 'attention':
                    layer.skip_attention = not layer.skip_attention
                elif connection_type == 'feedforward':
                    layer.skip_feedforward = not layer.skip_feedforward
                break
    
    def prune_connections(self, layer_id: int, connection_type: str, 
                         keep_prob: float = 0.7):
        """randomly prune connections in a layer"""
        for layer in self.layers:
            if layer.layer_id == layer_id and layer.is_active:
                if connection_type == 'attention':
                    # prune attention connections
                    if 'q_proj' not in layer.attention_connections:
                        layer.attention_connections['q_proj'] = np.random.random((self.embed_dim, self.embed_dim)) < keep_prob
                    else:
                        layer.attention_connections['q_proj'] = (
                            layer.attention_connections['q_proj'] & 
                            (np.random.random((self.embed_dim, self.embed_dim)) < keep_prob)
                        )
                elif connection_type == 'feedforward':
                    # prune feedforward connections
                    if 'linear1' not in layer.feedforward_connections:
                        layer.feedforward_connections['linear1'] = np.random.random((layer.hidden_dim, self.embed_dim)) < keep_prob
                    else:
                        layer.feedforward_connections['linear1'] = (
                            layer.feedforward_connections['linear1'] &
                            (np.random.random((layer.hidden_dim, self.embed_dim)) < keep_prob)
                        )
                break
    
    def get_active_layers(self) -> List[LayerGene]:
        """get list of active layers"""
        return [layer for layer in self.layers if layer.is_active]
    
    def get_active_connections(self) -> List[ConnectionGene]:
        """get list of active connections"""
        return [conn for conn in self.connections if conn.is_active]
    
    def calculate_complexity(self) -> int:
        """calculate total complexity (number of connections)"""
        complexity = 0
        
        for layer in self.get_active_layers():
            # count attention connections
            if layer.skip_attention:
                complexity += self.embed_dim * self.embed_dim * 4  # q, k, v, out projections
                
            # count feedforward connections  
            if layer.skip_feedforward:
                complexity += self.embed_dim * layer.hidden_dim * 2  # two linear layers
        
        # count output connections
        complexity += self.embed_dim * self.vocab_size  # lm head
        if self.num_classes:
            complexity += self.embed_dim * self.num_classes  # classifier
        
        self.complexity_score = complexity
        return complexity
    
    def set_fitness(self, task: str, score: float):
        """set fitness score for specific task"""
        self.fitness_scores[task] = score
    
    def get_fitness(self, task: str) -> float:
        """get fitness score for specific task"""
        return self.fitness_scores.get(task, 0.0)
    
    def get_multi_objective_score(self, complexity_weight: float = 0.1) -> Tuple[float, int]:
        """get multi-objective score (performance, complexity)"""
        # average performance across tasks
        if self.fitness_scores:
            avg_performance = np.mean(list(self.fitness_scores.values()))
        else:
            avg_performance = 0.0
        
        complexity = self.calculate_complexity()
        
        # weighted score
        score = avg_performance - complexity_weight * complexity
        
        return score, complexity
    
    def clone(self) -> 'ArchitectureGenome':
        """create deep copy of genome"""
        return copy.deepcopy(self)
    
    def crossover(self, other: 'ArchitectureGenome') -> 'ArchitectureGenome':
        """create offspring through crossover with another genome"""
        offspring = self.clone()
        
        # randomly inherit layers from either parent
        for i, layer in enumerate(offspring.layers):
            if i < len(other.layers) and np.random.random() < 0.5:
                offspring.layers[i] = copy.deepcopy(other.layers[i])
        
        # inherit some connections from other parent
        for conn in other.get_active_connections():
            if np.random.random() < 0.3:  # 30% chance to inherit connection
                offspring.add_connection(conn.from_layer, conn.to_layer, 
                                       conn.connection_type)
        
        return offspring
    
    def to_dict(self) -> Dict:
        """serialize genome to dictionary"""
        return {
            'embed_dim': int(self.embed_dim),
            'vocab_size': int(self.vocab_size),
            'max_layers': int(self.max_layers),
            'max_heads': int(self.max_heads),
            'layers': [
                {
                    'layer_id': int(layer.layer_id),
                    'is_active': bool(layer.is_active),
                    'num_heads': int(layer.num_heads),
                    'hidden_dim': int(layer.hidden_dim),
                    'activation_type': layer.activation_type.value,
                    'skip_attention': bool(layer.skip_attention),
                    'skip_feedforward': bool(layer.skip_feedforward)
                }
                for layer in self.layers
            ],
            'connections': [
                {
                    'from_layer': int(conn.from_layer),
                    'to_layer': int(conn.to_layer),
                    'connection_type': str(conn.connection_type),
                    'weight': float(conn.weight),
                    'is_active': bool(conn.is_active)
                }
                for conn in self.connections
            ],
            'num_classes': int(self.num_classes) if self.num_classes is not None else None,
            'max_length': int(self.max_length),
            'dropout': float(self.dropout),
            'embedding_type': str(self.embedding_type),
            'causal': bool(self.causal),
            'fitness_scores': {str(k): float(v) for k, v in self.fitness_scores.items()},
            'complexity_score': int(self.complexity_score),
            'generation': int(self.generation)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchitectureGenome':
        """deserialize genome from dictionary"""
        genome = cls(data['embed_dim'], data['vocab_size'], 
                    data['max_layers'], data['max_heads'])
        
        # restore layers
        for layer_data in data['layers']:
            layer = LayerGene(
                layer_id=layer_data['layer_id'],
                is_active=layer_data['is_active'],
                num_heads=layer_data['num_heads'],
                hidden_dim=layer_data['hidden_dim'],
                activation_type=ActivationType(layer_data['activation_type']),
                skip_attention=layer_data['skip_attention'],
                skip_feedforward=layer_data['skip_feedforward']
            )
            genome.layers.append(layer)
        
        # restore connections
        for conn_data in data['connections']:
            conn = ConnectionGene(
                from_layer=conn_data['from_layer'],
                to_layer=conn_data['to_layer'],
                connection_type=conn_data['connection_type'],
                weight=conn_data['weight'],
                is_active=conn_data['is_active']
            )
            genome.connections.append(conn)
        
        # restore other attributes
        genome.num_classes = data.get('num_classes')
        genome.max_length = data.get('max_length', 1024)
        genome.dropout = data.get('dropout', 0.1)
        genome.embedding_type = data.get('embedding_type', 'random')
        genome.causal = data.get('causal', True)
        genome.fitness_scores = data.get('fitness_scores', {})
        genome.complexity_score = data.get('complexity_score', 0)
        genome.generation = data.get('generation', 0)
        
        return genome

class HeadOnlyGenome:
    """specialized genome for evolving only output heads with frozen gpt-2 backbone"""
    
    def __init__(self, embed_dim: int = 768, vocab_size: int = 50257,
                 num_classes: Optional[int] = None):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # gpt-2 backbone parameters (frozen)
        self.num_layers = 12  # standard gpt-2
        self.num_heads = 12   # standard gpt-2
        self.max_length = 1024
        self.dropout = 0.1
        self.causal = True
        
        # evolvable head parameters
        self.lm_head_sparsity = 0.0  # fraction of connections to prune
        self.classifier_sparsity = 0.0  # for classification head
        
        # connection masks for heads (if specific patterns needed)
        self.lm_head_connections: Optional[List[List[int]]] = None
        self.classifier_connections: Optional[List[List[int]]] = None
        
        # fitness tracking
        self.fitness_scores: Dict[str, float] = {}
        self.complexity_score: int = 0
        self.generation: int = 0
    
    @classmethod
    def create_dense(cls, embed_dim: int = 768, vocab_size: int = 50257,
                    num_classes: Optional[int] = None) -> 'HeadOnlyGenome':
        """create genome with fully connected heads"""
        genome = cls(embed_dim, vocab_size, num_classes)
        genome.lm_head_sparsity = 0.0
        genome.classifier_sparsity = 0.0
        return genome
    
    @classmethod 
    def create_sparse(cls, embed_dim: int = 768, vocab_size: int = 50257,
                     num_classes: Optional[int] = None,
                     sparsity: float = 0.5) -> 'HeadOnlyGenome':
        """create genome with sparse heads"""
        genome = cls(embed_dim, vocab_size, num_classes)
        genome.lm_head_sparsity = sparsity
        genome.classifier_sparsity = sparsity
        return genome
    
    def mutate_sparsity(self, mutation_rate: float = 0.1):
        """mutate sparsity levels of heads"""
        if np.random.random() < mutation_rate:
            # mutate lm head sparsity
            delta = np.random.normal(0, 0.1)
            self.lm_head_sparsity = np.clip(self.lm_head_sparsity + delta, 0.0, 0.9)
        
        if np.random.random() < mutation_rate and self.num_classes is not None:
            # mutate classifier sparsity
            delta = np.random.normal(0, 0.1)
            self.classifier_sparsity = np.clip(self.classifier_sparsity + delta, 0.0, 0.9)
    
    def randomize_connections(self):
        """randomly generate connection masks for heads"""
        # language modeling head connections
        lm_keep_prob = 1.0 - self.lm_head_sparsity
        lm_mask = (np.random.random((self.vocab_size, self.embed_dim)) < lm_keep_prob).astype(int)
        self.lm_head_connections = lm_mask.tolist()
        
        # classification head connections (if applicable)
        if self.num_classes is not None:
            cls_keep_prob = 1.0 - self.classifier_sparsity
            cls_mask = (np.random.random((self.num_classes, self.embed_dim)) < cls_keep_prob).astype(int)
            self.classifier_connections = cls_mask.tolist()
    
    def calculate_complexity(self) -> int:
        """calculate complexity based on active connections in heads"""
        complexity = 0
        
        # language modeling head complexity
        if self.lm_head_connections:
            complexity += sum(sum(row) for row in self.lm_head_connections)
        else:
            # estimate based on sparsity
            total_lm_connections = self.vocab_size * self.embed_dim
            complexity += int(total_lm_connections * (1.0 - self.lm_head_sparsity))
        
        # classification head complexity
        if self.num_classes is not None:
            if self.classifier_connections:
                complexity += sum(sum(row) for row in self.classifier_connections)
            else:
                # estimate based on sparsity
                total_cls_connections = self.num_classes * self.embed_dim
                complexity += int(total_cls_connections * (1.0 - self.classifier_sparsity))
        
        self.complexity_score = complexity
        return complexity
    
    def set_fitness(self, task: str, score: float):
        """set fitness score for specific task"""
        self.fitness_scores[task] = score
    
    def get_fitness(self, task: str) -> float:
        """get fitness score for specific task"""
        return self.fitness_scores.get(task, 0.0)
    
    def get_multi_objective_score(self, complexity_weight: float = 0.01) -> Tuple[float, int]:
        """get multi-objective score combining performance and complexity"""
        # use average fitness across all tasks
        avg_fitness = np.mean(list(self.fitness_scores.values())) if self.fitness_scores else 0.0
        complexity = self.calculate_complexity()
        
        # penalize high complexity
        combined_score = avg_fitness - complexity_weight * complexity
        
        return combined_score, complexity
    
    def clone(self) -> 'HeadOnlyGenome':
        """create a copy of this genome"""
        clone = HeadOnlyGenome(self.embed_dim, self.vocab_size, self.num_classes)
        clone.lm_head_sparsity = self.lm_head_sparsity
        clone.classifier_sparsity = self.classifier_sparsity
        
        # deep copy connection masks
        if self.lm_head_connections:
            clone.lm_head_connections = [row.copy() for row in self.lm_head_connections]
        if self.classifier_connections:
            clone.classifier_connections = [row.copy() for row in self.classifier_connections]
        
        clone.fitness_scores = self.fitness_scores.copy()
        clone.generation = self.generation
        
        return clone
    
    def crossover(self, other: 'HeadOnlyGenome') -> 'HeadOnlyGenome':
        """create offspring through crossover"""
        child = self.clone()
        
        # crossover sparsity values
        if np.random.random() < 0.5:
            child.lm_head_sparsity = other.lm_head_sparsity
        if np.random.random() < 0.5:
            child.classifier_sparsity = other.classifier_sparsity
        
        # crossover connection masks (if they exist)
        if self.lm_head_connections and other.lm_head_connections:
            # randomly mix connections
            for i in range(len(child.lm_head_connections)):
                for j in range(len(child.lm_head_connections[i])):
                    if np.random.random() < 0.5:
                        child.lm_head_connections[i][j] = other.lm_head_connections[i][j]
        
        if (self.classifier_connections and other.classifier_connections and 
            child.classifier_connections):
            for i in range(len(child.classifier_connections)):
                for j in range(len(child.classifier_connections[i])):
                    if np.random.random() < 0.5:
                        child.classifier_connections[i][j] = other.classifier_connections[i][j]
        
        child.generation = max(self.generation, other.generation) + 1
        return child
    
    def to_dict(self) -> Dict:
        """convert genome to dictionary for serialization"""
        return {
            'embed_dim': self.embed_dim,
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_length': self.max_length,
            'dropout': self.dropout,
            'lm_head_sparsity': self.lm_head_sparsity,
            'classifier_sparsity': self.classifier_sparsity,
            'lm_head_connections': self.lm_head_connections,
            'classifier_connections': self.classifier_connections,
            'fitness_scores': self.fitness_scores,
            'complexity_score': self.complexity_score,
            'generation': self.generation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HeadOnlyGenome':
        """create genome from dictionary"""
        genome = cls(
            embed_dim=data['embed_dim'],
            vocab_size=data['vocab_size'],
            num_classes=data.get('num_classes')
        )
        
        genome.num_layers = data.get('num_layers', 12)
        genome.num_heads = data.get('num_heads', 12)
        genome.max_length = data.get('max_length', 1024)
        genome.dropout = data.get('dropout', 0.1)
        genome.lm_head_sparsity = data.get('lm_head_sparsity', 0.0)
        genome.classifier_sparsity = data.get('classifier_sparsity', 0.0)
        genome.lm_head_connections = data.get('lm_head_connections')
        genome.classifier_connections = data.get('classifier_connections')
        genome.fitness_scores = data.get('fitness_scores', {})
        genome.complexity_score = data.get('complexity_score', 0)
        genome.generation = data.get('generation', 0)
        
        return genome 