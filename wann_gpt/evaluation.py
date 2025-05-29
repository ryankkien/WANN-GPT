"""
shared weight evaluation system for wann architectures
tests networks with different weight values without training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

from .architecture.transformer import WannGPT
from .evolution.genome import ArchitectureGenome

@dataclass
class EvaluationResult:
    """results from shared weight evaluation"""
    mean_performance: float
    std_performance: float
    best_performance: float
    worst_performance: float
    weight_samples: List[float]
    performance_per_weight: List[float]
    complexity: int
    task_type: str

class SharedWeightEvaluator:
    """evaluator for weight-agnostic architectures"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # default weight sample range as in original wann paper
        self.default_weight_samples = [-2.5, -1.0, -0.5, 0.5, 1.0, 2.5]
        
        # performance metrics storage
        self.evaluation_history = []
    
    def instantiate_from_genome(self, genome: ArchitectureGenome) -> WannGPT:
        """create wanngpt model from genome specification"""
        
        # create base model
        model = WannGPT(
            vocab_size=genome.vocab_size,
            embed_dim=genome.embed_dim,
            num_layers=len(genome.get_active_layers()),
            num_heads=8,  # will be adjusted per layer
            max_length=genome.max_length,
            dropout=genome.dropout,
            num_classes=genome.num_classes,
            embedding_type=genome.embedding_type,
            causal=genome.causal
        )
        
        # apply genome configuration to model
        active_layers = genome.get_active_layers()
        
        for i, layer_gene in enumerate(active_layers):
            if i < len(model.layers):
                layer = model.layers[i]
                
                # set number of heads
                layer.attention.num_heads = layer_gene.num_heads
                layer.attention.head_dim = genome.embed_dim // layer_gene.num_heads
                
                # update head mask
                head_mask = torch.zeros(layer_gene.num_heads)
                head_mask[:layer_gene.num_heads] = 1.0
                layer.attention.register_buffer("head_mask", head_mask)
                
                # set activation function
                layer.feedforward.activation.set_activation(layer_gene.activation_type)
                
                # set skip connections
                layer.skip_attention = torch.tensor(1.0 if layer_gene.skip_attention else 0.0)
                layer.skip_feedforward = torch.tensor(1.0 if layer_gene.skip_feedforward else 0.0)
                
                # apply connection masks if specified
                if layer_gene.attention_connections:
                    for proj_name, mask in layer_gene.attention_connections.items():
                        if hasattr(layer.attention, proj_name):
                            proj_layer = getattr(layer.attention, proj_name)
                            if hasattr(proj_layer, 'connection_mask'):
                                proj_layer.connection_mask = torch.tensor(mask, dtype=torch.float32)
                
                if layer_gene.feedforward_connections:
                    for ff_name, mask in layer_gene.feedforward_connections.items():
                        if hasattr(layer.feedforward, ff_name):
                            ff_layer = getattr(layer.feedforward, ff_name)
                            if hasattr(ff_layer, 'connection_mask'):
                                ff_layer.connection_mask = torch.tensor(mask, dtype=torch.float32)
        
        return model.to(self.device)
    
    def evaluate_classification(self, model: WannGPT, dataloader, 
                              weight_samples: List[float] = None) -> EvaluationResult:
        """evaluate model on classification task with multiple weight values"""
        
        if weight_samples is None:
            weight_samples = self.default_weight_samples
        
        model.eval()
        performances = []
        
        with torch.no_grad():
            for weight in weight_samples:
                model.set_shared_weight(weight)
                
                correct = 0
                total = 0
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    # forward pass
                    logits = model(input_ids, attention_mask, task="classification")
                    predictions = torch.argmax(logits, dim=-1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                
                accuracy = correct / total if total > 0 else 0.0
                performances.append(accuracy)
        
        # calculate statistics
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        best_perf = np.max(performances)
        worst_perf = np.min(performances)
        complexity = model.get_total_complexity()
        
        result = EvaluationResult(
            mean_performance=mean_perf,
            std_performance=std_perf,
            best_performance=best_perf,
            worst_performance=worst_perf,
            weight_samples=weight_samples,
            performance_per_weight=performances,
            complexity=complexity,
            task_type="classification"
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_generation(self, model: WannGPT, dataloader,
                           weight_samples: List[float] = None,
                           max_length: int = 512) -> EvaluationResult:
        """evaluate model on generation task (perplexity)"""
        
        if weight_samples is None:
            weight_samples = self.default_weight_samples
        
        model.eval()
        performances = []  # lower perplexity is better
        
        with torch.no_grad():
            for weight in weight_samples:
                model.set_shared_weight(weight)
                
                total_loss = 0.0
                total_tokens = 0
                
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    # truncate to max length
                    seq_len = min(input_ids.size(1), max_length)
                    input_ids = input_ids[:, :seq_len]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, :seq_len]
                    
                    # forward pass
                    logits = model(input_ids, attention_mask, task="generation")
                    
                    # calculate loss for next token prediction
                    if seq_len > 1:
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()
                        
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                            reduction='sum'
                        )
                        
                        total_loss += loss.item()
                        total_tokens += shift_labels.numel()
                
                # calculate perplexity
                if total_tokens > 0:
                    avg_loss = total_loss / total_tokens
                    perplexity = torch.exp(torch.tensor(avg_loss)).item()
                else:
                    perplexity = float('inf')
                
                performances.append(perplexity)
        
        # for generation, we want lower perplexity, so we use negative for fitness
        # to make higher values better for consistent optimization
        negative_perplexities = [-p if p != float('inf') else -1000 for p in performances]
        
        mean_perf = np.mean(negative_perplexities)
        std_perf = np.std(negative_perplexities)
        best_perf = np.max(negative_perplexities)
        worst_perf = np.min(negative_perplexities)
        complexity = model.get_total_complexity()
        
        result = EvaluationResult(
            mean_performance=mean_perf,
            std_performance=std_perf,
            best_performance=best_perf,
            worst_performance=worst_perf,
            weight_samples=weight_samples,
            performance_per_weight=negative_perplexities,
            complexity=complexity,
            task_type="generation"
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_genome(self, genome: ArchitectureGenome, dataloader,
                       task_type: str = "classification",
                       weight_samples: List[float] = None) -> EvaluationResult:
        """evaluate genome by instantiating and testing model"""
        
        try:
            # instantiate model from genome
            model = self.instantiate_from_genome(genome)
            
            # evaluate based on task type
            if task_type == "classification":
                result = self.evaluate_classification(model, dataloader, weight_samples)
            elif task_type == "generation":
                result = self.evaluate_generation(model, dataloader, weight_samples)
            else:
                raise ValueError(f"unsupported task type: {task_type}")
            
            # update genome fitness
            genome.set_fitness(task_type, result.mean_performance)
            
            return result
            
        except Exception as e:
            print(f"evaluation failed for genome: {e}")
            # return poor performance for failed evaluation
            return EvaluationResult(
                mean_performance=0.0,
                std_performance=0.0,
                best_performance=0.0,
                worst_performance=0.0,
                weight_samples=weight_samples or self.default_weight_samples,
                performance_per_weight=[0.0] * len(weight_samples or self.default_weight_samples),
                complexity=float('inf'),
                task_type=task_type
            )
    
    def batch_evaluate_genomes(self, genomes: List[ArchitectureGenome],
                              dataloader, task_type: str = "classification",
                              weight_samples: List[float] = None,
                              parallel: bool = False) -> List[EvaluationResult]:
        """evaluate multiple genomes"""
        
        results = []
        
        if parallel and torch.cuda.device_count() > 1:
            # implement parallel evaluation across multiple gpus
            results = self._parallel_evaluate_genomes(genomes, dataloader, task_type, weight_samples)
        else:
            # sequential evaluation
            for i, genome in enumerate(tqdm(genomes, desc=f"evaluating {task_type}")):
                result = self.evaluate_genome(genome, dataloader, task_type, weight_samples)
                results.append(result)
        
        return results
    
    def _parallel_evaluate_genomes(self, genomes: List[ArchitectureGenome],
                                 dataloader, task_type: str,
                                 weight_samples: List[float] = None) -> List[EvaluationResult]:
        """parallel evaluation across multiple gpus"""
        
        import torch.multiprocessing as mp
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.distributed as dist
        
        num_gpus = torch.cuda.device_count()
        print(f"using {num_gpus} gpus for parallel evaluation")
        
        # split genomes across gpus
        genome_chunks = [genomes[i::num_gpus] for i in range(num_gpus)]
        
        # use multiprocessing to evaluate chunks in parallel
        with mp.Pool(processes=num_gpus) as pool:
            args = [(chunk, dataloader, task_type, weight_samples, i) 
                   for i, chunk in enumerate(genome_chunks)]
            
            chunk_results = pool.starmap(self._evaluate_genome_chunk, args)
        
        # flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def _evaluate_genome_chunk(self, genome_chunk: List[ArchitectureGenome],
                             dataloader, task_type: str,
                             weight_samples: List[float], gpu_id: int) -> List[EvaluationResult]:
        """evaluate chunk of genomes on specific gpu"""
        
        # set device for this process
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        
        # create evaluator for this gpu
        evaluator = SharedWeightEvaluator(device=device)
        
        results = []
        for genome in genome_chunk:
            try:
                result = evaluator.evaluate_genome(genome, dataloader, task_type, weight_samples)
                results.append(result)
            except Exception as e:
                print(f"failed to evaluate genome on gpu {gpu_id}: {e}")
                # return poor performance for failed evaluation
                dummy_result = EvaluationResult(
                    mean_performance=0.0,
                    std_performance=0.0,
                    best_performance=0.0,
                    worst_performance=0.0,
                    weight_samples=weight_samples or self.default_weight_samples,
                    performance_per_weight=[0.0] * len(weight_samples or self.default_weight_samples),
                    complexity=float('inf'),
                    task_type=task_type
                )
                results.append(dummy_result)
        
        return results
    
    def ensemble_evaluate(self, model: WannGPT, dataloader,
                         weight_samples: List[float] = None,
                         task_type: str = "classification") -> Dict:
        """evaluate using ensemble of multiple weight instances"""
        
        if weight_samples is None:
            weight_samples = self.default_weight_samples
        
        model.eval()
        ensemble_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # collect predictions from all weight instances
                all_logits = []
                for weight in weight_samples:
                    model.set_shared_weight(weight)
                    logits = model(input_ids, attention_mask, task=task_type)
                    all_logits.append(logits)
                
                # ensemble by averaging logits
                ensemble_logits = torch.stack(all_logits).mean(dim=0)
                predictions = torch.argmax(ensemble_logits, dim=-1)
                
                if task_type == "classification":
                    ensemble_correct += (predictions == labels).sum().item()
                
                total_samples += labels.size(0)
        
        ensemble_accuracy = ensemble_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "ensemble_accuracy": ensemble_accuracy,
            "num_weight_instances": len(weight_samples),
            "total_samples": total_samples
        }
    
    def analyze_weight_sensitivity(self, model: WannGPT, dataloader,
                                  weight_range: Tuple[float, float] = (-3.0, 3.0),
                                  num_samples: int = 20,
                                  task_type: str = "classification") -> Dict:
        """analyze how performance varies with weight value"""
        
        weight_values = np.linspace(weight_range[0], weight_range[1], num_samples)
        performances = []
        
        model.eval()
        
        with torch.no_grad():
            for weight in weight_values:
                model.set_shared_weight(weight)
                
                if task_type == "classification":
                    correct = 0
                    total = 0
                    
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        logits = model(input_ids, attention_mask, task="classification")
                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                    
                    performance = correct / total if total > 0 else 0.0
                
                elif task_type == "generation":
                    total_loss = 0.0
                    total_tokens = 0
                    
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        logits = model(input_ids, attention_mask, task="generation")
                        
                        if input_ids.size(1) > 1:
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = input_ids[:, 1:].contiguous()
                            
                            loss = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction='sum'
                            )
                            
                            total_loss += loss.item()
                            total_tokens += shift_labels.numel()
                    
                    if total_tokens > 0:
                        avg_loss = total_loss / total_tokens
                        perplexity = torch.exp(torch.tensor(avg_loss)).item()
                        performance = -perplexity  # negative for consistency
                    else:
                        performance = -1000
                
                performances.append(performance)
        
        return {
            "weight_values": weight_values.tolist(),
            "performances": performances,
            "mean_performance": np.mean(performances),
            "std_performance": np.std(performances),
            "best_weight": weight_values[np.argmax(performances)],
            "best_performance": np.max(performances),
            "weight_sensitivity": np.std(performances) / np.mean(np.abs(performances)) if np.mean(np.abs(performances)) > 0 else float('inf')
        }
    
    def get_evaluation_summary(self) -> Dict:
        """get summary of all evaluations performed"""
        if not self.evaluation_history:
            return {"message": "no evaluations performed yet"}
        
        classification_results = [r for r in self.evaluation_history if r.task_type == "classification"]
        generation_results = [r for r in self.evaluation_history if r.task_type == "generation"]
        
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "classification_evaluations": len(classification_results),
            "generation_evaluations": len(generation_results)
        }
        
        if classification_results:
            summary["classification_stats"] = {
                "mean_accuracy": np.mean([r.mean_performance for r in classification_results]),
                "best_accuracy": np.max([r.best_performance for r in classification_results]),
                "mean_complexity": np.mean([r.complexity for r in classification_results])
            }
        
        if generation_results:
            summary["generation_stats"] = {
                "mean_neg_perplexity": np.mean([r.mean_performance for r in generation_results]),
                "best_neg_perplexity": np.max([r.best_performance for r in generation_results]),
                "mean_complexity": np.mean([r.complexity for r in generation_results])
            }
        
        return summary 