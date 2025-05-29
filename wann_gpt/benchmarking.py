"""
comprehensive benchmarking and analysis tools for wann-gpt
implements all performance metrics and visualization from plan.txt
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time

from .architecture.transformer import WannGPT
from .evolution.genome import ArchitectureGenome
from .evaluation import SharedWeightEvaluator, EvaluationResult

@dataclass
class BenchmarkResult:
    """comprehensive benchmark results"""
    accuracy: float
    perplexity: float
    complexity: int
    robustness_score: float
    weight_sensitivity: float
    ensemble_improvement: float
    inference_time: float
    memory_usage: float
    
class WannBenchmarkSuite:
    """comprehensive benchmarking suite for wann-gpt models"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.evaluator = SharedWeightEvaluator(device)
        self.results_history = []
        
    def comprehensive_benchmark(self, model: WannGPT, test_loader,
                              task_type: str = "classification",
                              save_plots: bool = True,
                              output_dir: str = "./benchmark_results") -> BenchmarkResult:
        """run complete benchmark suite"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("running comprehensive benchmark...")
        
        # 1. basic performance metrics
        basic_result = self.evaluator.evaluate_classification(model, test_loader) \
                      if task_type == "classification" else \
                      self.evaluator.evaluate_generation(model, test_loader)
        
        # 2. weight sensitivity analysis
        print("analyzing weight sensitivity...")
        sensitivity_analysis = self.evaluator.analyze_weight_sensitivity(
            model, test_loader, weight_range=(-3.0, 3.0), 
            num_samples=30, task_type=task_type
        )
        
        # 3. ensemble performance
        print("testing ensemble performance...")
        ensemble_result = self.evaluator.ensemble_evaluate(
            model, test_loader, task_type=task_type
        )
        
        # 4. inference speed and memory analysis
        print("measuring inference speed...")
        inference_time, memory_usage = self.measure_inference_performance(
            model, test_loader, task_type
        )
        
        # 5. robustness analysis
        print("analyzing robustness...")
        robustness_score = self.analyze_robustness(model, test_loader, task_type)
        
        # compile results
        if task_type == "classification":
            accuracy = basic_result.mean_performance
            perplexity = 0.0
            ensemble_improvement = ensemble_result["ensemble_accuracy"] - accuracy
        else:
            accuracy = 0.0
            perplexity = -basic_result.mean_performance  # convert back to positive perplexity
            ensemble_improvement = 0.0  # todo: implement for generation
        
        result = BenchmarkResult(
            accuracy=accuracy,
            perplexity=perplexity,
            complexity=basic_result.complexity,
            robustness_score=robustness_score,
            weight_sensitivity=sensitivity_analysis["weight_sensitivity"],
            ensemble_improvement=ensemble_improvement,
            inference_time=inference_time,
            memory_usage=memory_usage
        )
        
        # generate visualizations
        if save_plots:
            self.generate_visualizations(
                basic_result, sensitivity_analysis, ensemble_result,
                model, output_path, task_type
            )
        
        # save detailed results
        self.save_benchmark_results(result, basic_result, sensitivity_analysis,
                                  ensemble_result, output_path)
        
        self.results_history.append(result)
        return result
    
    def measure_inference_performance(self, model: WannGPT, dataloader,
                                    task_type: str) -> Tuple[float, float]:
        """measure inference speed and memory usage"""
        
        model.eval()
        model.set_shared_weight(1.0)
        
        # warm up
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 3:  # only warm up on a few batches
                    break
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                _ = model(input_ids, attention_mask, task=task_type)
        
        # measure inference time
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        total_samples = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                _ = model(input_ids, attention_mask, task=task_type)
                total_samples += input_ids.size(0)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        inference_time = (end_time - start_time) / total_samples  # per sample
        
        # measure memory usage
        if self.device == "cuda":
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # mb
        else:
            memory_usage = 0.0
        
        return inference_time, memory_usage
    
    def analyze_robustness(self, model: WannGPT, dataloader, task_type: str) -> float:
        """analyze model robustness across different conditions"""
        
        weight_samples = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        performances = []
        
        model.eval()
        with torch.no_grad():
            for weight in weight_samples:
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
                else:
                    # for generation, use simplified perplexity calculation
                    total_loss = 0.0
                    total_tokens = 0
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(self.device)
                        
                        if input_ids.size(1) > 1:
                            logits = model(input_ids, attention_mask, task="generation")
                            shift_logits = logits[:, :-1, :].contiguous()
                            shift_labels = input_ids[:, 1:].contiguous()
                            
                            loss = torch.nn.functional.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                reduction='sum'
                            )
                            total_loss += loss.item()
                            total_tokens += shift_labels.numel()
                    
                    if total_tokens > 0:
                        avg_loss = total_loss / total_tokens
                        perplexity = torch.exp(torch.tensor(avg_loss)).item()
                        performance = 1.0 / perplexity  # inverse for robustness score
                    else:
                        performance = 0.0
                
                performances.append(performance)
        
        # robustness score: 1 - coefficient of variation
        if np.mean(performances) > 0:
            cv = np.std(performances) / np.mean(performances)
            robustness_score = max(0.0, 1.0 - cv)
        else:
            robustness_score = 0.0
        
        return robustness_score
    
    def generate_visualizations(self, basic_result: EvaluationResult,
                              sensitivity_analysis: Dict,
                              ensemble_result: Dict,
                              model: WannGPT,
                              output_path: Path,
                              task_type: str):
        """generate comprehensive visualizations"""
        
        # 1. weight sensitivity curve
        self.plot_weight_sensitivity(sensitivity_analysis, output_path, task_type)
        
        # 2. performance across weights
        self.plot_performance_distribution(basic_result, output_path, task_type)
        
        # 3. architecture visualization
        self.plot_architecture_summary(model, output_path)
        
        # 4. ensemble comparison
        self.plot_ensemble_comparison(basic_result, ensemble_result, output_path, task_type)
        
        print(f"visualizations saved to {output_path}")
    
    def plot_weight_sensitivity(self, sensitivity_analysis: Dict, 
                              output_path: Path, task_type: str):
        """plot weight sensitivity curve"""
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sensitivity_analysis["weight_values"],
            y=sensitivity_analysis["performances"],
            mode='lines+markers',
            name='Performance',
            line=dict(color='blue', width=2)
        ))
        
        # mark optimal weight
        best_idx = np.argmax(sensitivity_analysis["performances"])
        fig.add_trace(go.Scatter(
            x=[sensitivity_analysis["weight_values"][best_idx]],
            y=[sensitivity_analysis["performances"][best_idx]],
            mode='markers',
            name='Optimal Weight',
            marker=dict(color='red', size=10, symbol='star')
        ))
        
        metric_name = "Accuracy" if task_type == "classification" else "Negative Perplexity"
        fig.update_layout(
            title=f"Weight Sensitivity Analysis ({task_type})",
            xaxis_title="Shared Weight Value",
            yaxis_title=metric_name,
            template="plotly_white"
        )
        
        fig.write_html(output_path / f"weight_sensitivity_{task_type}.html")
        
        # also save as png using matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(sensitivity_analysis["weight_values"], 
                sensitivity_analysis["performances"], 'b-o', linewidth=2)
        plt.axvline(sensitivity_analysis["best_weight"], color='red', 
                   linestyle='--', label=f'Optimal: {sensitivity_analysis["best_weight"]:.2f}')
        plt.xlabel("Shared Weight Value")
        plt.ylabel(metric_name)
        plt.title(f"Weight Sensitivity Analysis ({task_type})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"weight_sensitivity_{task_type}.png", dpi=300)
        plt.close()
    
    def plot_performance_distribution(self, result: EvaluationResult,
                                    output_path: Path, task_type: str):
        """plot performance distribution across weight samples"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # histogram of performances
        ax1.hist(result.performance_per_weight, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(result.mean_performance, color='red', linestyle='--', 
                   label=f'Mean: {result.mean_performance:.4f}')
        ax1.axvline(result.best_performance, color='green', linestyle='--',
                   label=f'Best: {result.best_performance:.4f}')
        ax1.set_xlabel('Performance')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Performance Distribution ({task_type})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # performance vs weight values
        ax2.scatter(result.weight_samples, result.performance_per_weight, 
                   alpha=0.7, s=60, color='orange')
        ax2.plot(result.weight_samples, result.performance_per_weight, 
                'o-', alpha=0.5, color='gray')
        ax2.axhline(result.mean_performance, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Weight Value')
        ax2.set_ylabel('Performance')
        ax2.set_title(f'Performance vs Weight Value ({task_type})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"performance_distribution_{task_type}.png", dpi=300)
        plt.close()
    
    def plot_architecture_summary(self, model: WannGPT, output_path: Path):
        """visualize architecture summary"""
        
        # gather architecture info
        arch_info = {
            'layers': model.num_layers,
            'embedding_dim': model.embed_dim,
            'vocab_size': model.vocab_size,
            'max_length': model.max_length,
            'total_complexity': model.get_total_complexity() if hasattr(model, 'get_total_complexity') else 0
        }
        
        # create architecture summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # layer activity visualization
        if hasattr(model, 'layer_mask'):
            layer_activity = model.layer_mask.cpu().numpy()
            ax1.bar(range(len(layer_activity)), layer_activity, alpha=0.7, color='lightblue')
            ax1.set_xlabel('Layer Index')
            ax1.set_ylabel('Activity (0=Disabled, 1=Active)')
            ax1.set_title('Layer Activity')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Layer mask not available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Layer Activity')
        
        # architecture parameters
        labels = list(arch_info.keys())
        values = list(arch_info.values())
        ax2.barh(labels, values, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Value')
        ax2.set_title('Architecture Parameters')
        ax2.grid(True, alpha=0.3)
        
        # attention head analysis (if available)
        if hasattr(model, 'layers') and len(model.layers) > 0:
            layer = model.layers[0]
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'head_mask'):
                head_activity = layer.attention.head_mask.cpu().numpy()
                ax3.bar(range(len(head_activity)), head_activity, alpha=0.7, color='orange')
                ax3.set_xlabel('Head Index')
                ax3.set_ylabel('Activity')
                ax3.set_title('Attention Head Activity (Layer 0)')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Head mask not available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Attention Head Activity')
        else:
            ax3.text(0.5, 0.5, 'No layers available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Attention Head Activity')
        
        # complexity comparison (placeholder)
        baseline_complexity = model.embed_dim * model.vocab_size  # rough estimate
        current_complexity = arch_info['total_complexity']
        complexities = ['Baseline', 'Current']
        values = [baseline_complexity, current_complexity]
        colors = ['red', 'blue']
        ax4.bar(complexities, values, alpha=0.7, color=colors)
        ax4.set_ylabel('Complexity')
        ax4.set_title('Complexity Comparison')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "architecture_summary.png", dpi=300)
        plt.close()
    
    def plot_ensemble_comparison(self, basic_result: EvaluationResult,
                               ensemble_result: Dict,
                               output_path: Path, task_type: str):
        """compare single vs ensemble performance"""
        
        if task_type == "classification":
            single_perf = basic_result.mean_performance
            ensemble_perf = ensemble_result["ensemble_accuracy"]
            
            methods = ['Single Weight\n(Mean)', 'Ensemble']
            performances = [single_perf, ensemble_perf]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(methods, performances, alpha=0.7, 
                          color=['lightcoral', 'lightblue'])
            
            # add value labels on bars
            for bar, perf in zip(bars, performances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{perf:.4f}', ha='center', va='bottom')
            
            plt.ylabel('Accuracy')
            plt.title('Single vs Ensemble Performance')
            plt.ylim(0, max(performances) * 1.1)
            plt.grid(True, alpha=0.3)
            
            # add improvement annotation
            improvement = ensemble_perf - single_perf
            plt.text(0.5, max(performances) * 0.9, 
                    f'Improvement: +{improvement:.4f}',
                    ha='center', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(output_path / f"ensemble_comparison_{task_type}.png", dpi=300)
            plt.close()
    
    def save_benchmark_results(self, result: BenchmarkResult,
                             basic_result: EvaluationResult,
                             sensitivity_analysis: Dict,
                             ensemble_result: Dict,
                             output_path: Path):
        """save detailed benchmark results"""
        
        detailed_results = {
            "summary": {
                "accuracy": result.accuracy,
                "perplexity": result.perplexity,
                "complexity": result.complexity,
                "robustness_score": result.robustness_score,
                "weight_sensitivity": result.weight_sensitivity,
                "ensemble_improvement": result.ensemble_improvement,
                "inference_time_per_sample": result.inference_time,
                "memory_usage_mb": result.memory_usage
            },
            "basic_evaluation": {
                "mean_performance": basic_result.mean_performance,
                "std_performance": basic_result.std_performance,
                "best_performance": basic_result.best_performance,
                "worst_performance": basic_result.worst_performance,
                "performance_per_weight": basic_result.performance_per_weight,
                "weight_samples": basic_result.weight_samples
            },
            "weight_sensitivity": sensitivity_analysis,
            "ensemble_analysis": ensemble_result
        }
        
        with open(output_path / "benchmark_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"detailed results saved to {output_path / 'benchmark_results.json'}")
    
    def compare_architectures(self, models_and_names: List[Tuple[WannGPT, str]],
                            test_loader, task_type: str = "classification",
                            output_dir: str = "./comparison_results"):
        """compare multiple architectures"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        comparison_results = []
        
        for model, name in models_and_names:
            print(f"benchmarking {name}...")
            result = self.comprehensive_benchmark(
                model, test_loader, task_type, save_plots=False
            )
            comparison_results.append((name, result))
        
        # create comparison plots
        self.plot_architecture_comparison(comparison_results, output_path, task_type)
        
        # save comparison data
        comparison_data = {
            name: {
                "accuracy": result.accuracy,
                "perplexity": result.perplexity,
                "complexity": result.complexity,
                "robustness_score": result.robustness_score,
                "weight_sensitivity": result.weight_sensitivity,
                "ensemble_improvement": result.ensemble_improvement,
                "inference_time": result.inference_time,
                "memory_usage": result.memory_usage
            }
            for name, result in comparison_results
        }
        
        with open(output_path / "architecture_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        return comparison_results
    
    def plot_architecture_comparison(self, comparison_results: List[Tuple[str, BenchmarkResult]],
                                   output_path: Path, task_type: str):
        """plot comparison between architectures"""
        
        names = [name for name, _ in comparison_results]
        
        if task_type == "classification":
            accuracies = [result.accuracy for _, result in comparison_results]
            complexities = [result.complexity for _, result in comparison_results]
            robustness = [result.robustness_score for _, result in comparison_results]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # accuracy comparison
            ax1.bar(names, accuracies, alpha=0.7, color='lightblue')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Classification Accuracy Comparison')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # complexity comparison
            ax2.bar(names, complexities, alpha=0.7, color='lightgreen')
            ax2.set_ylabel('Complexity')
            ax2.set_title('Model Complexity Comparison')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # robustness comparison
            ax3.bar(names, robustness, alpha=0.7, color='orange')
            ax3.set_ylabel('Robustness Score')
            ax3.set_title('Robustness Comparison')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # pareto plot (accuracy vs complexity)
            ax4.scatter(complexities, accuracies, alpha=0.7, s=100)
            for i, name in enumerate(names):
                ax4.annotate(name, (complexities[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
            ax4.set_xlabel('Complexity')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy vs Complexity (Pareto View)')
            ax4.grid(True, alpha=0.3)
            
        else:  # generation
            perplexities = [result.perplexity for _, result in comparison_results]
            complexities = [result.complexity for _, result in comparison_results]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # perplexity comparison
            ax1.bar(names, perplexities, alpha=0.7, color='lightcoral')
            ax1.set_ylabel('Perplexity')
            ax1.set_title('Perplexity Comparison (Lower is Better)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # pareto plot (perplexity vs complexity)
            ax2.scatter(complexities, perplexities, alpha=0.7, s=100)
            for i, name in enumerate(names):
                ax2.annotate(name, (complexities[i], perplexities[i]),
                           xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('Complexity')
            ax2.set_ylabel('Perplexity')
            ax2.set_title('Perplexity vs Complexity (Pareto View)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / f"architecture_comparison_{task_type}.png", dpi=300)
        plt.close()
    
    def ablation_study(self, model: WannGPT, genome: ArchitectureGenome,
                      test_loader, task_type: str = "classification",
                      output_dir: str = "./ablation_results"):
        """perform ablation study on architecture components"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("performing ablation study...")
        
        # baseline performance
        baseline_result = self.evaluator.evaluate_classification(model, test_loader) \
                         if task_type == "classification" else \
                         self.evaluator.evaluate_generation(model, test_loader)
        
        ablation_results = {"baseline": baseline_result.mean_performance}
        
        # test with zero weight (should fail)
        print("testing with zero weight...")
        model.set_shared_weight(0.0)
        zero_weight_result = self.evaluator.evaluate_classification(model, test_loader) \
                           if task_type == "classification" else \
                           self.evaluator.evaluate_generation(model, test_loader)
        ablation_results["zero_weight"] = zero_weight_result.mean_performance
        
        # restore original weight
        model.set_shared_weight(1.0)
        
        # test disabling layers one by one
        if hasattr(model, 'layer_mask'):
            original_mask = model.layer_mask.clone()
            
            for i in range(len(original_mask)):
                if original_mask[i] > 0.5:  # only test active layers
                    print(f"testing without layer {i}...")
                    
                    # disable layer i
                    model.layer_mask[i] = 0.0
                    
                    # evaluate
                    result = self.evaluator.evaluate_classification(model, test_loader) \
                           if task_type == "classification" else \
                           self.evaluator.evaluate_generation(model, test_loader)
                    
                    ablation_results[f"without_layer_{i}"] = result.mean_performance
                    
                    # restore layer
                    model.layer_mask[i] = original_mask[i]
        
        # test disabling attention heads
        if hasattr(model, 'layers') and len(model.layers) > 0:
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'head_mask'):
                    original_head_mask = layer.attention.head_mask.clone()
                    
                    for head_idx in range(len(original_head_mask)):
                        if original_head_mask[head_idx] > 0.5:
                            print(f"testing without head {head_idx} in layer {layer_idx}...")
                            
                            # disable head
                            layer.attention.head_mask[head_idx] = 0.0
                            
                            # evaluate
                            result = self.evaluator.evaluate_classification(model, test_loader) \
                                   if task_type == "classification" else \
                                   self.evaluator.evaluate_generation(model, test_loader)
                            
                            ablation_results[f"without_layer_{layer_idx}_head_{head_idx}"] = result.mean_performance
                            
                            # restore head
                            layer.attention.head_mask[head_idx] = original_head_mask[head_idx]
        
        # plot ablation results
        self.plot_ablation_results(ablation_results, output_path, task_type)
        
        # save ablation data
        with open(output_path / "ablation_results.json", 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        return ablation_results
    
    def plot_ablation_results(self, ablation_results: Dict, output_path: Path, task_type: str):
        """plot ablation study results"""
        
        components = list(ablation_results.keys())
        performances = list(ablation_results.values())
        
        plt.figure(figsize=(12, 8))
        
        # color bars based on performance relative to baseline
        baseline_perf = ablation_results["baseline"]
        colors = ['red' if p < baseline_perf * 0.9 else 
                 'yellow' if p < baseline_perf * 0.95 else 
                 'green' for p in performances]
        
        bars = plt.barh(components, performances, alpha=0.7, color=colors)
        
        # add baseline line
        plt.axvline(baseline_perf, color='blue', linestyle='--', linewidth=2, label='Baseline')
        
        # add value labels
        for bar, perf in zip(bars, performances):
            plt.text(bar.get_width() + max(performances)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{perf:.4f}', va='center')
        
        metric_name = "Accuracy" if task_type == "classification" else "Negative Perplexity"
        plt.xlabel(metric_name)
        plt.ylabel('Component')
        plt.title(f'Ablation Study Results ({task_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path / f"ablation_study_{task_type}.png", dpi=300)
        plt.close()
    
    def generate_final_report(self, output_dir: str = "./final_report"):
        """generate comprehensive final report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.results_history:
            print("no benchmark results available for report")
            return
        
        # aggregate results
        summary_stats = {
            "total_benchmarks": len(self.results_history),
            "average_accuracy": np.mean([r.accuracy for r in self.results_history if r.accuracy > 0]),
            "average_perplexity": np.mean([r.perplexity for r in self.results_history if r.perplexity > 0]),
            "average_complexity": np.mean([r.complexity for r in self.results_history]),
            "average_robustness": np.mean([r.robustness_score for r in self.results_history]),
            "average_inference_time": np.mean([r.inference_time for r in self.results_history])
        }
        
        # create comprehensive plots
        self.plot_benchmark_history(output_path)
        
        # save summary
        with open(output_path / "benchmark_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"final report generated in {output_path}")
        return summary_stats
    
    def plot_benchmark_history(self, output_path: Path):
        """plot history of all benchmarks performed"""
        
        if len(self.results_history) < 2:
            return
        
        accuracies = [r.accuracy for r in self.results_history if r.accuracy > 0]
        complexities = [r.complexity for r in self.results_history]
        robustness_scores = [r.robustness_score for r in self.results_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # accuracy over time
        if accuracies:
            ax1.plot(range(len(accuracies)), accuracies, 'o-', alpha=0.7)
            ax1.set_xlabel('Benchmark Index')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Accuracy Over Benchmarks')
            ax1.grid(True, alpha=0.3)
        
        # complexity over time
        ax2.plot(range(len(complexities)), complexities, 'o-', alpha=0.7, color='orange')
        ax2.set_xlabel('Benchmark Index')
        ax2.set_ylabel('Complexity')
        ax2.set_title('Complexity Over Benchmarks')
        ax2.grid(True, alpha=0.3)
        
        # robustness over time
        ax3.plot(range(len(robustness_scores)), robustness_scores, 'o-', alpha=0.7, color='green')
        ax3.set_xlabel('Benchmark Index')
        ax3.set_ylabel('Robustness Score')
        ax3.set_title('Robustness Over Benchmarks')
        ax3.grid(True, alpha=0.3)
        
        # correlation plot
        if accuracies and len(accuracies) == len(complexities):
            ax4.scatter(complexities[:len(accuracies)], accuracies, alpha=0.7)
            ax4.set_xlabel('Complexity')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Accuracy vs Complexity')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "benchmark_history.png", dpi=300)
        plt.close() 