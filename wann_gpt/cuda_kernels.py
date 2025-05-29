"""
cuda kernels for optimized wann transformer operations
implements fast gpu evaluation for shared weight networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# try to import cupy for custom cuda kernels
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class CudaAttentionKernel:
    """optimized cuda kernels for attention computation with shared weights"""
    
    def __init__(self, embed_dim: int, num_heads: int, device: str = "cuda"):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device
        self.scale = self.head_dim ** -0.5
        
        # compile custom kernels if cupy is available
        if CUPY_AVAILABLE:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """compile custom cuda kernels for attention"""
        
        # fused qkv projection kernel
        self.qkv_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void fused_qkv_projection(
            const float* input,           // [batch, seq_len, embed_dim]
            const float* q_mask,          // [embed_dim, embed_dim]
            const float* k_mask,          // [embed_dim, embed_dim]
            const float* v_mask,          // [embed_dim, embed_dim]
            float* q_output,              // [batch, seq_len, embed_dim]
            float* k_output,              // [batch, seq_len, embed_dim]
            float* v_output,              // [batch, seq_len, embed_dim]
            float shared_weight,
            int batch_size,
            int seq_len,
            int embed_dim
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = batch_size * seq_len * embed_dim;
            
            if (idx < total_elements) {
                int batch_idx = idx / (seq_len * embed_dim);
                int remaining = idx % (seq_len * embed_dim);
                int seq_idx = remaining / embed_dim;
                int out_dim = remaining % embed_dim;
                
                float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
                
                // compute q, k, v projections simultaneously
                for (int in_dim = 0; in_dim < embed_dim; in_dim++) {
                    int input_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + in_dim;
                    float input_val = input[input_idx];
                    
                    int weight_idx = out_dim * embed_dim + in_dim;
                    
                    q_sum += input_val * q_mask[weight_idx] * shared_weight;
                    k_sum += input_val * k_mask[weight_idx] * shared_weight;
                    v_sum += input_val * v_mask[weight_idx] * shared_weight;
                }
                
                q_output[idx] = q_sum;
                k_output[idx] = k_sum;
                v_output[idx] = v_sum;
            }
        }
        ''', 'fused_qkv_projection')
        
        # fused attention kernel (flashattention-inspired)
        self.attention_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void fused_attention(
            const float* q,               // [batch, num_heads, seq_len, head_dim]
            const float* k,               // [batch, num_heads, seq_len, head_dim]
            const float* v,               // [batch, num_heads, seq_len, head_dim]
            float* output,                // [batch, num_heads, seq_len, head_dim]
            const bool* causal_mask,      // [seq_len, seq_len]
            float scale,
            int batch_size,
            int num_heads,
            int seq_len,
            int head_dim
        ) {
            // simplified flash attention implementation
            // each block handles one attention head for one sequence
            
            int head_idx = blockIdx.x;
            int batch_idx = blockIdx.y;
            int query_idx = threadIdx.x;
            
            if (head_idx < num_heads && batch_idx < batch_size && query_idx < seq_len) {
                
                // compute attention scores for this query
                float max_score = -INFINITY;
                
                // find max score for numerical stability
                for (int key_idx = 0; key_idx <= query_idx; key_idx++) {
                    if (!causal_mask[query_idx * seq_len + key_idx]) continue;
                    
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + d;
                        int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim + d;
                        score += q[q_offset] * k[k_offset];
                    }
                    score *= scale;
                    max_score = fmaxf(max_score, score);
                }
                
                // compute softmax and weighted sum
                float sum_exp = 0.0f;
                for (int key_idx = 0; key_idx <= query_idx; key_idx++) {
                    if (!causal_mask[query_idx * seq_len + key_idx]) continue;
                    
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + d;
                        int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim + d;
                        score += q[q_offset] * k[k_offset];
                    }
                    score = (score * scale) - max_score;
                    sum_exp += expf(score);
                }
                
                // final attention computation
                for (int d = 0; d < head_dim; d++) {
                    float weighted_sum = 0.0f;
                    
                    for (int key_idx = 0; key_idx <= query_idx; key_idx++) {
                        if (!causal_mask[query_idx * seq_len + key_idx]) continue;
                        
                        float score = 0.0f;
                        for (int d2 = 0; d2 < head_dim; d2++) {
                            int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + d2;
                            int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim + d2;
                            score += q[q_offset] * k[k_offset];
                        }
                        score = expf((score * scale) - max_score) / sum_exp;
                        
                        int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim + d;
                        weighted_sum += score * v[v_offset];
                    }
                    
                    int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + d;
                    output[out_offset] = weighted_sum;
                }
            }
        }
        ''', 'fused_attention')
    
    def fused_qkv_projection(self, x: torch.Tensor, q_mask: torch.Tensor,
                            k_mask: torch.Tensor, v_mask: torch.Tensor,
                            shared_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fused q, k, v projection using custom cuda kernel"""
        
        if not CUPY_AVAILABLE:
            # fallback to pytorch implementation
            return self._pytorch_qkv_projection(x, q_mask, k_mask, v_mask, shared_weight)
        
        batch_size, seq_len, embed_dim = x.shape
        
        # allocate outputs
        q_output = torch.zeros_like(x)
        k_output = torch.zeros_like(x)
        v_output = torch.zeros_like(x)
        
        # convert to cupy arrays
        x_cp = cp.asarray(x.detach())
        q_mask_cp = cp.asarray(q_mask.detach())
        k_mask_cp = cp.asarray(k_mask.detach())
        v_mask_cp = cp.asarray(v_mask.detach())
        q_out_cp = cp.asarray(q_output.detach())
        k_out_cp = cp.asarray(k_output.detach())
        v_out_cp = cp.asarray(v_output.detach())
        
        # launch kernel
        total_elements = batch_size * seq_len * embed_dim
        block_size = 256
        grid_size = (total_elements + block_size - 1) // block_size
        
        self.qkv_kernel(
            (grid_size,), (block_size,),
            (x_cp, q_mask_cp, k_mask_cp, v_mask_cp,
             q_out_cp, k_out_cp, v_out_cp,
             shared_weight, batch_size, seq_len, embed_dim)
        )
        
        # convert back to pytorch
        q_output = torch.as_tensor(q_out_cp, device=x.device)
        k_output = torch.as_tensor(k_out_cp, device=x.device)
        v_output = torch.as_tensor(v_out_cp, device=x.device)
        
        return q_output, k_output, v_output
    
    def _pytorch_qkv_projection(self, x: torch.Tensor, q_mask: torch.Tensor,
                               k_mask: torch.Tensor, v_mask: torch.Tensor,
                               shared_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pytorch fallback for qkv projection"""
        
        # apply masks and shared weight
        q_weight = q_mask * shared_weight
        k_weight = k_mask * shared_weight
        v_weight = v_mask * shared_weight
        
        # compute projections
        q = F.linear(x, q_weight)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)
        
        return q, k, v
    
    def fused_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       causal_mask: torch.Tensor) -> torch.Tensor:
        """fused attention computation using custom cuda kernel"""
        
        if not CUPY_AVAILABLE:
            # fallback to pytorch implementation
            return self._pytorch_attention(q, k, v, causal_mask)
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        output = torch.zeros_like(q)
        
        # convert to cupy arrays
        q_cp = cp.asarray(q.detach())
        k_cp = cp.asarray(k.detach())
        v_cp = cp.asarray(v.detach())
        mask_cp = cp.asarray(causal_mask.detach())
        out_cp = cp.asarray(output.detach())
        
        # launch kernel
        grid_size = (num_heads, batch_size)
        block_size = min(seq_len, 1024)
        
        self.attention_kernel(
            grid_size, (block_size,),
            (q_cp, k_cp, v_cp, out_cp, mask_cp,
             self.scale, batch_size, num_heads, seq_len, head_dim)
        )
        
        # convert back to pytorch
        output = torch.as_tensor(out_cp, device=q.device)
        
        return output
    
    def _pytorch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          causal_mask: torch.Tensor) -> torch.Tensor:
        """pytorch fallback for attention computation"""
        
        # compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # apply causal mask
        scores.masked_fill_(causal_mask, float('-inf'))
        
        # softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output

class CudaFeedForwardKernel:
    """optimized cuda kernels for feedforward computation with shared weights"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, device: str = "cuda"):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        if CUPY_AVAILABLE:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """compile custom cuda kernels for feedforward"""
        
        # fused feedforward kernel
        self.feedforward_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void fused_feedforward(
            const float* input,           // [batch, seq_len, embed_dim]
            const float* linear1_mask,    // [hidden_dim, embed_dim]
            const float* linear2_mask,    // [embed_dim, hidden_dim]
            float* output,                // [batch, seq_len, embed_dim]
            float shared_weight,
            int batch_size,
            int seq_len,
            int embed_dim,
            int hidden_dim,
            int activation_type            // 0=relu, 1=gelu, 2=tanh, etc.
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_elements = batch_size * seq_len * embed_dim;
            
            if (idx < total_elements) {
                int batch_idx = idx / (seq_len * embed_dim);
                int remaining = idx % (seq_len * embed_dim);
                int seq_idx = remaining / embed_dim;
                int out_dim = remaining % embed_dim;
                
                // first linear layer
                float hidden_sum = 0.0f;
                for (int in_dim = 0; in_dim < embed_dim; in_dim++) {
                    int input_idx = batch_idx * seq_len * embed_dim + seq_idx * embed_dim + in_dim;
                    float input_val = input[input_idx];
                    
                    for (int h = 0; h < hidden_dim; h++) {
                        int weight_idx = h * embed_dim + in_dim;
                        float hidden_val = input_val * linear1_mask[weight_idx] * shared_weight;
                        
                        // apply activation function
                        switch(activation_type) {
                            case 0: // relu
                                hidden_val = fmaxf(0.0f, hidden_val);
                                break;
                            case 1: // gelu (approximation)
                                hidden_val = hidden_val * 0.5f * (1.0f + tanhf(0.7978845608f * (hidden_val + 0.044715f * hidden_val * hidden_val * hidden_val)));
                                break;
                            case 2: // tanh
                                hidden_val = tanhf(hidden_val);
                                break;
                            default: // linear
                                break;
                        }
                        
                        // second linear layer
                        int weight2_idx = out_dim * hidden_dim + h;
                        hidden_sum += hidden_val * linear2_mask[weight2_idx] * shared_weight;
                    }
                }
                
                output[idx] = hidden_sum;
            }
        }
        ''', 'fused_feedforward')
    
    def fused_feedforward(self, x: torch.Tensor, linear1_mask: torch.Tensor,
                         linear2_mask: torch.Tensor, shared_weight: float,
                         activation_type: int = 0) -> torch.Tensor:
        """fused feedforward computation using custom cuda kernel"""
        
        if not CUPY_AVAILABLE:
            # fallback to pytorch implementation
            return self._pytorch_feedforward(x, linear1_mask, linear2_mask, 
                                           shared_weight, activation_type)
        
        batch_size, seq_len, embed_dim = x.shape
        output = torch.zeros_like(x)
        
        # convert to cupy arrays
        x_cp = cp.asarray(x.detach())
        linear1_cp = cp.asarray(linear1_mask.detach())
        linear2_cp = cp.asarray(linear2_mask.detach())
        out_cp = cp.asarray(output.detach())
        
        # launch kernel
        total_elements = batch_size * seq_len * embed_dim
        block_size = 256
        grid_size = (total_elements + block_size - 1) // block_size
        
        self.feedforward_kernel(
            (grid_size,), (block_size,),
            (x_cp, linear1_cp, linear2_cp, out_cp,
             shared_weight, batch_size, seq_len, embed_dim, self.hidden_dim, activation_type)
        )
        
        # convert back to pytorch
        output = torch.as_tensor(out_cp, device=x.device)
        
        return output
    
    def _pytorch_feedforward(self, x: torch.Tensor, linear1_mask: torch.Tensor,
                           linear2_mask: torch.Tensor, shared_weight: float,
                           activation_type: int = 0) -> torch.Tensor:
        """pytorch fallback for feedforward computation"""
        
        # first linear layer
        weight1 = linear1_mask * shared_weight
        hidden = F.linear(x, weight1)
        
        # activation function
        if activation_type == 0:  # relu
            hidden = F.relu(hidden)
        elif activation_type == 1:  # gelu
            hidden = F.gelu(hidden)
        elif activation_type == 2:  # tanh
            hidden = torch.tanh(hidden)
        # else: linear (no activation)
        
        # second linear layer
        weight2 = linear2_mask * shared_weight
        output = F.linear(hidden, weight2)
        
        return output

class CudaSharedWeightEvaluator:
    """cuda-optimized evaluator for shared weight networks"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.kernels = {}
    
    def get_attention_kernel(self, embed_dim: int, num_heads: int) -> CudaAttentionKernel:
        """get or create attention kernel for given dimensions"""
        key = (embed_dim, num_heads)
        if key not in self.kernels:
            self.kernels[key] = CudaAttentionKernel(embed_dim, num_heads, self.device)
        return self.kernels[key]
    
    def get_feedforward_kernel(self, embed_dim: int, hidden_dim: int) -> CudaFeedForwardKernel:
        """get or create feedforward kernel for given dimensions"""
        key = (embed_dim, hidden_dim, 'ff')
        if key not in self.kernels:
            self.kernels[key] = CudaFeedForwardKernel(embed_dim, hidden_dim, self.device)
        return self.kernels[key]
    
    def parallel_weight_evaluation(self, model, input_batch: torch.Tensor,
                                 weight_samples: list, task_type: str = "classification") -> torch.Tensor:
        """evaluate model with multiple weight values in parallel"""
        
        batch_size, seq_len = input_batch.shape[:2]
        num_weights = len(weight_samples)
        
        # expand batch to include weight dimension
        expanded_input = input_batch.unsqueeze(0).repeat(num_weights, 1, 1)
        expanded_input = expanded_input.view(num_weights * batch_size, seq_len)
        
        # create weight tensor
        weight_tensor = torch.tensor(weight_samples, device=self.device)
        weight_tensor = weight_tensor.view(num_weights, 1, 1).expand(num_weights, batch_size, 1)
        weight_tensor = weight_tensor.contiguous().view(num_weights * batch_size, 1)
        
        # forward pass with expanded batch
        with torch.no_grad():
            # this would require modifying the model to accept weight tensor
            # for now, we'll use sequential evaluation
            outputs = []
            for weight in weight_samples:
                model.set_shared_weight(weight)
                output = model(input_batch, task=task_type)
                outputs.append(output)
            
            # stack outputs
            parallel_output = torch.stack(outputs, dim=0)
        
        return parallel_output

# utility functions for cuda optimization
def optimize_memory_layout(tensor: torch.Tensor) -> torch.Tensor:
    """optimize tensor memory layout for cuda access"""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

def get_optimal_block_size(problem_size: int, max_threads: int = 1024) -> Tuple[int, int]:
    """get optimal cuda block configuration"""
    if problem_size <= max_threads:
        return problem_size, 1
    else:
        block_size = min(max_threads, 256)  # reasonable default
        grid_size = (problem_size + block_size - 1) // block_size
        return block_size, grid_size

def warmup_cuda_kernels(device: str = "cuda"):
    """warmup cuda kernels for better performance"""
    if not torch.cuda.is_available():
        return
    
    # create dummy tensors and run operations to warm up kernels
    dummy = torch.randn(32, 64, 512, device=device)
    _ = F.linear(dummy, torch.randn(512, 512, device=device))
    _ = F.relu(dummy)
    _ = F.softmax(dummy, dim=-1)
    
    # synchronize to ensure kernels are loaded
    torch.cuda.synchronize() 