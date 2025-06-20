"""
main transformer architecture for hybrid gpt-2 with weight-agnostic heads
keeps original gpt-2 weights for transformer layers, only final heads are evolutionary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union
from transformers import GPT2Model, GPT2Config
from .layers import SharedWeightLinear
from .activations import ActivationType

class HybridWannGPT(nn.Module):
    """hybrid architecture: pretrained gpt-2 backbone + weight-agnostic heads"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 768, 
                 num_layers: int = 12, num_heads: int = 12,
                 max_length: int = 1024, dropout: float = 0.1,
                 num_classes: int = None, model_name: str = "gpt2",
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # load pretrained gpt-2 backbone
        self.gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=embed_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            embd_pdrop=dropout
        )
        
        # try to load pretrained model, fallback to random initialization
        try:
            import warnings
            import sys
            from io import StringIO
            
            # temporarily suppress warnings and stdout to hide size mismatch messages
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gpt2_backbone = GPT2Model.from_pretrained(
                    model_name, 
                    config=self.gpt2_config,
                    ignore_mismatched_sizes=True
                )
            
            # restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            print(f"loaded pretrained {model_name} weights")
        except:
            # restore stdout/stderr in case of exception
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            print(f"failed to load {model_name}, using random initialization")
            self.gpt2_backbone = GPT2Model(self.gpt2_config)
        
        # freeze backbone if specified
        if freeze_backbone:
            for param in self.gpt2_backbone.parameters():
                param.requires_grad = False
            print("froze gpt-2 backbone parameters")
        
        # weight-agnostic output heads (these are evolutionary)
        self._setup_output_heads()
        
        # shared weight parameter (only affects heads)
        self.shared_weight = 1.0
    
    def _setup_output_heads(self):
        """setup weight-agnostic output heads for different tasks"""
        
        # language modeling head (weight-agnostic)
        self.lm_head = SharedWeightLinear(self.embed_dim, self.vocab_size, bias=False)
        
        # classification head (weight-agnostic, if num_classes specified)
        if self.num_classes is not None:
            self.classifier = SharedWeightLinear(self.embed_dim, self.num_classes, bias=False)
        
        # special tokens for classification
        if self.num_classes is not None:
            self.register_buffer("cls_token_id", torch.tensor(self.vocab_size - 1))
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                task: str = "generation") -> torch.Tensor:
        """forward pass: gpt-2 backbone + weight-agnostic heads"""
        
        batch_size, seq_len = input_ids.shape
        
        # add [cls] token for classification
        if task == "classification" and self.num_classes is not None:
            # truncate input if needed to leave room for [cls] token
            if seq_len >= self.max_length:
                input_ids = input_ids[:, :(self.max_length - 1)]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :(self.max_length - 1)]
                seq_len = input_ids.shape[1]
            
            cls_tokens = self.cls_token_id.expand(batch_size, 1).to(input_ids.device)
            input_ids = torch.cat([input_ids, cls_tokens], dim=1)
            seq_len += 1
            
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, cls_mask], dim=1)
        
        # forward through gpt-2 backbone (keeps original weights)
        outputs = self.gpt2_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, embed_dim]
        
        # task-specific weight-agnostic heads
        if task == "generation":
            return self.lm_head(hidden_states, self.shared_weight)
        elif task == "classification" and self.num_classes is not None:
            # use [cls] token representation
            cls_representation = hidden_states[:, -1, :]  # last token ([cls])
            return self.classifier(cls_representation, self.shared_weight)
        else:
            raise ValueError(f"unsupported task: {task}")
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """generate text continuation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # forward pass
                logits = self.forward(input_ids, task="generation")
                
                # get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # stop if max length reached
                if input_ids.size(1) >= self.max_length:
                    break
        
        return input_ids
    
    def set_shared_weight(self, weight: float):
        """set shared weight parameter (only affects heads)"""
        self.shared_weight = weight
    
    def get_total_complexity(self) -> int:
        """get total complexity of weight-agnostic components (heads only)"""
        complexity = 0
        
        # language modeling head complexity
        complexity += self.lm_head.get_complexity()
        
        # classification head complexity (if exists)
        if hasattr(self, 'classifier'):
            complexity += self.classifier.get_complexity()
        
        return complexity
    
    def get_architecture_info(self) -> Dict[str, Union[int, float, List]]:
        """get architecture information"""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_length": self.max_length,
            "num_classes": self.num_classes,
            "shared_weight": self.shared_weight,
            "total_complexity": self.get_total_complexity(),
            "freeze_backbone": self.freeze_backbone,
            "architecture_type": "hybrid_wann_gpt2"
        }
    
    def clone(self) -> 'HybridWannGPT':
        """create a copy of this model with same architecture"""
        clone = HybridWannGPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_length=self.max_length,
            dropout=0.1,  # default
            num_classes=self.num_classes,
            freeze_backbone=self.freeze_backbone
        )
        
        # copy head connection masks
        clone.lm_head.connection_mask = self.lm_head.connection_mask.clone()
        if hasattr(self, 'classifier') and hasattr(clone, 'classifier'):
            clone.classifier.connection_mask = self.classifier.connection_mask.clone()
        
        # copy shared weight
        clone.shared_weight = self.shared_weight
        
        return clone 

class WannTransformerBlock(nn.Module):
    """single transformer block with shared weights and evolvable structure"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, hidden_dim: int = None,
                 activation_type: ActivationType = ActivationType.RELU,
                 dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or 4 * embed_dim
        
        #load the other layers for compatibility
        from .layers import WannAttention, WannFeedForward, LayerNormFixed, FixedEmbedding, PositionalEncoding
        
        # self-attention layer
        self.attention = WannAttention(embed_dim, num_heads, dropout, causal)
        self.norm1 = LayerNormFixed(embed_dim)
        
        # feed-forward layer
        self.feedforward = WannFeedForward(embed_dim, self.hidden_dim, 
                                          activation_type, dropout)
        self.norm2 = LayerNormFixed(embed_dim)
        
        # evolvable skip connections
        self.register_buffer("skip_attention", torch.tensor(1.0))
        self.register_buffer("skip_feedforward", torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor, shared_weight: float = 1.0,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """transformer block forward pass"""
        
        # self-attention with residual connection
        if self.skip_attention > 0.5:  # skip connection active
            attn_out = self.attention(x, shared_weight, attention_mask)
            x = x + attn_out
            x = self.norm1(x)
        
        # feed-forward with residual connection  
        if self.skip_feedforward > 0.5:  # skip connection active
            ff_out = self.feedforward(x, shared_weight)
            x = x + ff_out
            x = self.norm2(x)
        
        return x
    
    def mutate_activation(self):
        """mutate feed-forward activation for evolution"""
        self.feedforward.mutate_activation()
    
    def toggle_skip_attention(self):
        """toggle attention skip connection"""
        self.skip_attention = 1.0 - self.skip_attention
    
    def toggle_skip_feedforward(self):
        """toggle feedforward skip connection"""
        self.skip_feedforward = 1.0 - self.skip_feedforward
    
    def get_complexity(self) -> int:
        """get total number of connections in this block"""
        complexity = 0
        if self.skip_attention > 0.5:
            #count attention connections
            complexity += (self.attention.q_proj.get_complexity() + 
                         self.attention.k_proj.get_complexity() +
                         self.attention.v_proj.get_complexity() +
                         self.attention.out_proj.get_complexity())
        if self.skip_feedforward > 0.5:
            complexity += self.feedforward.get_complexity()
        return complexity

class WannGPT(nn.Module):
    """original weight-agnostic gpt-2 transformer for classification and generation"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 512, 
                 num_layers: int = 6, num_heads: int = 8,
                 max_length: int = 1024, dropout: float = 0.1,
                 num_classes: int = None, embedding_type: str = "random",
                 causal: bool = True):
        super().__init__()
        
        #load the other layers for compatibility
        from .layers import FixedEmbedding, PositionalEncoding
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.num_classes = num_classes
        self.causal = causal
        
        # embedding layers
        self.token_embedding = FixedEmbedding(vocab_size, embed_dim, embedding_type)
        self.position_embedding = PositionalEncoding(embed_dim, max_length)
        self.dropout = nn.Dropout(dropout)
        
        # transformer blocks
        self.layers = nn.ModuleList([
            WannTransformerBlock(embed_dim, num_heads, 
                               hidden_dim=4*embed_dim, 
                               dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])
        
        # layer masks for evolving number of layers
        self.register_buffer("layer_mask", torch.ones(num_layers))
        
        # output heads
        self._setup_output_heads()
        
        # shared weight parameter
        self.shared_weight = 1.0
    
    def _setup_output_heads(self):
        """setup output heads for different tasks"""
        
        # language modeling head (for generation)
        self.lm_head = SharedWeightLinear(self.embed_dim, self.vocab_size, bias=False)
        
        # classification head (if num_classes specified)
        if self.num_classes is not None:
            self.classifier = SharedWeightLinear(self.embed_dim, self.num_classes, bias=False)
        
        # special token embeddings - use a valid token ID within vocab_size
        # reserve the last token for [cls]
        self.register_buffer("cls_token_id", torch.tensor(self.vocab_size - 1))  # [cls] token
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                task: str = "generation") -> torch.Tensor:
        """forward pass for specified task"""
        
        batch_size, seq_len = input_ids.shape
        
        # add [cls] token for classification
        if task == "classification" and self.num_classes is not None:
            # truncate input if needed to leave room for [cls] token
            if seq_len >= self.max_length:
                input_ids = input_ids[:, :(self.max_length - 1)]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :(self.max_length - 1)]
                seq_len = input_ids.shape[1]
            
            cls_tokens = self.cls_token_id.expand(batch_size, 1).to(input_ids.device)
            input_ids = torch.cat([input_ids, cls_tokens], dim=1)
            seq_len += 1
            
            if attention_mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, cls_mask], dim=1)
        
        # embedding
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        # transformer layers
        for i, layer in enumerate(self.layers):
            if self.layer_mask[i] > 0.5:  # layer is active
                x = layer(x, self.shared_weight, attention_mask)
        
        # task-specific output
        if task == "generation":
            return self.lm_head(x, self.shared_weight)
        elif task == "classification" and self.num_classes is not None:
            # use [cls] token representation
            cls_representation = x[:, -1, :]  # last token ([cls])
            return self.classifier(cls_representation, self.shared_weight)
        else:
            raise ValueError(f"unsupported task: {task}")
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """generate text continuation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # forward pass
                logits = self.forward(input_ids, task="generation")
                
                # get logits for last token
                logits = logits[:, -1, :] / temperature
                
                # apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # stop if max length reached
                if input_ids.size(1) >= self.max_length:
                    break
        
        return input_ids
    
    def set_shared_weight(self, weight: float):
        """set the shared weight parameter"""
        self.shared_weight = weight
        
        # update all layers with new weight
        for layer in self.layers:
            layer.shared_weight = weight
    
    def add_layer(self):
        """add a new transformer layer (for evolution)"""
        new_layer = WannTransformerBlock(self.embed_dim, self.num_heads,
                                       hidden_dim=4*self.embed_dim,
                                       causal=self.causal)
        self.layers.append(new_layer)
        
        # extend layer mask
        new_mask = torch.cat([self.layer_mask, torch.ones(1)], dim=0)
        self.register_buffer("layer_mask", new_mask)
        self.num_layers += 1
    
    def remove_layer(self, layer_idx: int):
        """remove a layer by masking it out"""
        if 0 <= layer_idx < self.num_layers:
            self.layer_mask[layer_idx] = 0.0
    
    def get_active_layers(self) -> int:
        """get number of active layers"""
        return int(self.layer_mask.sum().item())
    
    def get_total_complexity(self) -> int:
        """calculate total model complexity (number of active connections)"""
        
        complexity = 0
        
        # embedding complexity
        complexity += self.vocab_size * self.embed_dim
        
        # layer complexities
        for i, layer in enumerate(self.layers):
            if self.layer_mask[i] > 0.5:  # layer is active
                # attention complexity
                if hasattr(layer.attention, 'head_mask'):
                    active_heads = layer.attention.head_mask.sum().item()
                    head_dim = self.embed_dim // layer.attention.num_heads
                    
                    # q, k, v projections + output projection
                    complexity += int(active_heads * head_dim * self.embed_dim * 4)
                else:
                    # fallback: assume all heads active
                    complexity += self.embed_dim * self.embed_dim * 4
                
                # feedforward complexity
                complexity += self.embed_dim * layer.feedforward.hidden_dim  # first linear
                complexity += layer.feedforward.hidden_dim * self.embed_dim  # second linear
        
        # output head complexity
        if hasattr(self, 'lm_head'):
            complexity += self.embed_dim * self.vocab_size
        
        if hasattr(self, 'classifier') and self.classifier is not None:
            complexity += self.embed_dim * self.num_classes
        
        return complexity
    
    def mutate_layer_activation(self, layer_idx: int):
        """mutate activation function of specific layer"""
        if 0 <= layer_idx < self.num_layers:
            self.layers[layer_idx].mutate_activation()
    
    def clone(self) -> 'WannGPT':
        """create a copy of this model for evolution"""
        # create new model with same architecture
        clone = WannGPT(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_length=self.max_length,
            num_classes=self.num_classes,
            causal=self.causal
        )
        
        # copy state dict (including masks and buffers)
        clone.load_state_dict(self.state_dict())
        clone.shared_weight = self.shared_weight
        
        return clone 
    
    def get_architecture_info(self) -> Dict[str, Union[int, float, List]]:
        """get detailed architecture information"""
        
        active_layers = (self.layer_mask > 0.5).sum().item()
        
        layer_info = []
        for i, layer in enumerate(self.layers):
            if self.layer_mask[i] > 0.5:
                layer_data = {
                    'layer_id': i,
                    'num_heads': layer.attention.num_heads,
                    'hidden_dim': layer.feedforward.hidden_dim if hasattr(layer.feedforward, 'hidden_dim') else 'unknown'
                }
                
                if hasattr(layer.attention, 'head_mask'):
                    layer_data['active_heads'] = layer.attention.head_mask.sum().item()
                
                layer_info.append(layer_data)
        
        return {
            'total_layers': self.num_layers,
            'active_layers': active_layers,
            'embed_dim': self.embed_dim,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'num_classes': self.num_classes,
            'total_complexity': self.get_total_complexity(),
            'shared_weight': self.shared_weight,
            'layer_details': layer_info
        } 