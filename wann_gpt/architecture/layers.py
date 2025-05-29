"""
core layers for weight-agnostic transformer architecture
all layers support shared weight parameter and connection masks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .activations import EvolvableActivation, ActivationType

class FixedEmbedding(nn.Module):
    """fixed embedding layer for weight-agnostic networks
    uses either one-hot or fixed random embeddings
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, embedding_type: str = "random"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        
        if embedding_type == "onehot":
            # one-hot encoding (vocab_size must equal embed_dim)
            assert vocab_size == embed_dim, "for onehot, vocab_size must equal embed_dim"
            self.register_buffer("embedding_matrix", torch.eye(vocab_size))
        elif embedding_type == "random":
            # fixed random embedding matrix
            embedding_matrix = torch.randn(vocab_size, embed_dim)
            embedding_matrix = F.normalize(embedding_matrix, dim=1)  # normalize for stability
            self.register_buffer("embedding_matrix", embedding_matrix)
        else:
            raise ValueError(f"unsupported embedding type: {embedding_type}")
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """lookup embeddings for input tokens"""
        return F.embedding(input_ids, self.embedding_matrix)

class PositionalEncoding(nn.Module):
    """fixed sinusoidal positional encoding"""
    
    def __init__(self, embed_dim: int, max_length: int = 1024):
        super().__init__()
        
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """add positional encoding to input"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class SharedWeightLinear(nn.Module):
    """linear layer using shared weight parameter and connection mask"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # connection mask (1 = connection exists, 0 = no connection)
        # initialized as fully connected, evolution will sparsify
        mask = torch.ones(out_features, in_features)
        self.register_buffer("connection_mask", mask)
        
        if bias:
            # bias uses shared weight too
            bias_mask = torch.ones(out_features)
            self.register_buffer("bias_mask", bias_mask)
    
    def forward(self, x: torch.Tensor, shared_weight: float = 1.0) -> torch.Tensor:
        """forward pass with shared weight parameter"""
        # apply connection mask and shared weight
        weight = self.connection_mask * shared_weight
        output = F.linear(x, weight, bias=None)
        
        if self.use_bias:
            bias = self.bias_mask * shared_weight
            output = output + bias
        
        return output
    
    def prune_connections(self, keep_prob: float = 0.5):
        """randomly prune connections for evolution"""
        self.connection_mask = (torch.rand_like(self.connection_mask) < keep_prob).float()
    
    def add_connection(self, in_idx: int, out_idx: int):
        """add a specific connection"""
        self.connection_mask[out_idx, in_idx] = 1.0
    
    def remove_connection(self, in_idx: int, out_idx: int):
        """remove a specific connection"""
        self.connection_mask[out_idx, in_idx] = 0.0
    
    def get_complexity(self) -> int:
        """get number of active connections"""
        return int(self.connection_mask.sum().item())

class WannAttention(nn.Module):
    """multi-head self-attention with shared weights and evolvable structure"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # shared weight linear projections
        self.q_proj = SharedWeightLinear(embed_dim, embed_dim)
        self.k_proj = SharedWeightLinear(embed_dim, embed_dim)
        self.v_proj = SharedWeightLinear(embed_dim, embed_dim)
        self.out_proj = SharedWeightLinear(embed_dim, embed_dim)
        
        # head masks for evolving number of heads
        self.register_buffer("head_mask", torch.ones(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, shared_weight: float = 1.0, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """multi-head attention forward pass"""
        batch_size, seq_len, embed_dim = x.shape
        
        # compute q, k, v projections
        q = self.q_proj(x, shared_weight)
        k = self.k_proj(x, shared_weight) 
        v = self.v_proj(x, shared_weight)
        
        # reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # apply head mask (for evolving number of heads)
        q = q * self.head_mask.view(1, -1, 1, 1)
        k = k * self.head_mask.view(1, -1, 1, 1)
        v = v * self.head_mask.view(1, -1, 1, 1)
        
        # compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # apply causal mask
        if self.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(scores.device)
            scores.masked_fill_(causal_mask, float('-inf'))
        
        # apply additional attention mask if provided
        if attention_mask is not None:
            # attention_mask is [batch_size, seq_len]
            # need to expand to [batch_size, num_heads, seq_len, seq_len]
            # create mask where 0 -> -inf and 1 -> 0
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attention_mask = (1.0 - attention_mask) * -10000.0  # convert to additive mask
            scores = scores + attention_mask
        
        # softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output, shared_weight)
        
        return output
    
    def add_head(self):
        """add a new attention head (for evolution)"""
        if self.num_heads < self.embed_dim:  # reasonable limit
            # expand head mask
            new_mask = torch.cat([self.head_mask, torch.ones(1)], dim=0)
            self.register_buffer("head_mask", new_mask)
            self.num_heads += 1
            self.head_dim = self.embed_dim // self.num_heads
    
    def remove_head(self, head_idx: int):
        """remove a specific attention head"""
        if self.num_heads > 1 and head_idx < self.num_heads:
            self.head_mask[head_idx] = 0.0
    
    def get_active_heads(self) -> int:
        """get number of active heads"""
        return int(self.head_mask.sum().item())

class WannFeedForward(nn.Module):
    """feed-forward network with shared weights and evolvable activations"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, 
                 activation_type: ActivationType = ActivationType.RELU,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # two linear layers with shared weights
        self.linear1 = SharedWeightLinear(embed_dim, hidden_dim)
        self.linear2 = SharedWeightLinear(hidden_dim, embed_dim)
        
        # evolvable activation function
        self.activation = EvolvableActivation(activation_type)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, shared_weight: float = 1.0) -> torch.Tensor:
        """feed-forward pass with shared weights"""
        x = self.linear1(x, shared_weight)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x, shared_weight)
        return x
    
    def mutate_activation(self):
        """mutate activation function for evolution"""
        self.activation.mutate_activation()
    
    def get_complexity(self) -> int:
        """get total number of connections"""
        return self.linear1.get_complexity() + self.linear2.get_complexity()

class LayerNormFixed(nn.Module):
    """layer normalization with fixed parameters (no learned scale/bias)"""
    
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # fixed scale and bias
        self.register_buffer("weight", torch.ones(embed_dim))
        self.register_buffer("bias", torch.zeros(embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """apply layer normalization with fixed parameters"""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias 