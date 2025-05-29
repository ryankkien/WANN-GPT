# WANN-GPT: Weight-Agnostic Neural Networks for GPT-2 Transformers

This repository implements Weight-Agnostic Neural Networks (WANNs) applied to GPT-2 transformer architectures. The system evolves transformer architectures that can perform text classification and generation tasks with random or shared weights, without traditional weight training.

## Overview

Weight-Agnostic Neural Networks demonstrate that network architecture alone can encode solutions to tasks, without requiring precise weight optimization. This project adapts the WANN approach to transformer architectures, allowing evolution of GPT-2-like models where:

- All weights in the network share a single scalar value
- Architecture topology (layers, connections, attention heads) encodes the solution
- Networks perform tasks with random weights drawn from a distribution
- Evolution optimizes architecture rather than weights

## Key Features

### 🏗️ **Weight-Agnostic Transformers**
- GPT-2 architectures that work with shared weights
- Support for both text classification and generation
- Evolvable attention heads, layer connections, and activation functions

### 🧬 **Advanced Evolutionary Search**
- Multi-objective evolution balancing performance and complexity
- NSGA-II and adaptive selection strategies
- Comprehensive mutation operators (add/remove layers, heads, connections)
- Parallel evaluation across multiple GPUs

### 📊 **Comprehensive Benchmarking Suite**
- Weight sensitivity analysis and robustness testing
- Ensemble evaluation methods
- Ablation studies to understand component importance
- Architecture comparison and Pareto analysis
- Interactive visualizations and detailed reports

### 🗂️ **Real Dataset Integration**
- Built-in support for IMDb, AG News, WikiText, TinyStories
- Automatic fallback to synthetic data when datasets unavailable
- Custom dataset creation utilities
- Smart tokenization and preprocessing

### ⚙️ **Configuration Management**
- Preset configurations for different scenarios
- YAML/JSON configuration file support
- Runtime configuration overrides
- Validation and dependency resolution

### 🚀 **CUDA Optimization**
- Custom kernels for fast GPU evaluation
- Parallel weight evaluation
- Memory-efficient attention computation
- Multi-GPU support for large populations

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/wann-gpt.git
cd wann-gpt

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- See `requirements.txt` for complete list

## Quick Start

### 🎯 **Simple Demo**

```bash
# Run complete system demonstration
python run_demo.py

# Or run specific demos
python examples/comprehensive_benchmark.py --task demo
```

### 📈 **Classification Task**

```python
from wann_gpt import load_config, load_classification_data, EvolutionEngine, SharedWeightEvaluator

# Load preset configuration
config = load_config(preset="classification_small")

# Load dataset
train_loader, test_loader, num_classes = load_classification_data(
    dataset_name="imdb",
    vocab_size=500,
    max_length=128,
    subset_size=1000
)

# Run evolution
evaluator = SharedWeightEvaluator()
engine = EvolutionEngine(config.evolution, evaluator)

best_genome = engine.evolve(
    dataloader=train_loader,
    task_type="classification"
)

print(f"Best accuracy: {best_genome.get_fitness('classification'):.4f}")
```

### 📝 **Generation Task**

```python
from wann_gpt import load_config, load_generation_data

# Load generation config
config = load_config(preset="generation_small")

# Load text dataset
train_loader, test_loader = load_generation_data(
    dataset_name="tiny_stories",
    vocab_size=300,
    max_length=64
)

# Run evolution for generation
best_genome = engine.evolve(
    dataloader=train_loader,
    task_type="generation"
)

# Generate text samples
model = evaluator.instantiate_from_genome(best_genome)
generated = model.generate(prompt_tokens, max_new_tokens=20)
```

### 🔬 **Comprehensive Benchmarking**

```python
from wann_gpt import WannBenchmarkSuite

# Create benchmark suite
benchmark_suite = WannBenchmarkSuite()

# Run full analysis
benchmark_result = benchmark_suite.comprehensive_benchmark(
    model=evolved_model,
    test_loader=test_loader,
    task_type="classification",
    save_plots=True,
    output_dir="./results"
)

# Perform ablation study
ablation_results = benchmark_suite.ablation_study(
    model=evolved_model,
    genome=best_genome,
    test_loader=test_loader
)

# Compare multiple architectures
comparison = benchmark_suite.compare_architectures([
    (evolved_model, "Evolved WANN"),
    (baseline_model, "Standard Transformer")
], test_loader)
```

## Configuration Presets

The system includes several built-in presets for different scenarios:

| Preset | Description | Best For |
|--------|-------------|----------|
| `classification_small` | Lightweight classification setup | Quick experiments, debugging |
| `classification_large` | Full-scale classification | Production classification tasks |
| `generation_small` | Compact generation model | Simple text generation |
| `generation_large` | Large-scale generation | Complex language modeling |
| `debug` | Minimal configuration | Testing and development |

```python
# Load and customize presets
config = load_config(preset="classification_small")

# Override specific parameters
config = load_config(
    preset="debug",
    overrides={
        "evolution": {"population_size": 10},
        "data": {"subset_size": 100}
    }
)

# Save custom configurations
config.save("my_config.yaml")
```

## Available Datasets

### Classification Datasets
- **IMDb**: Movie review sentiment classification (2 classes)
- **AG News**: News topic classification (4 classes)

### Generation Datasets  
- **WikiText**: Wikipedia articles for language modeling
- **TinyStories**: Simple stories for basic generation

```python
from wann_gpt import DatasetRegistry

# List all available datasets
datasets = DatasetRegistry.list_datasets()
print(datasets)
# {'classification': ['imdb', 'ag_news'], 'generation': ['wikitext', 'tiny_stories']}

# Create custom datasets
from wann_gpt import create_custom_classification_dataset

train_loader, test_loader, num_classes = create_custom_classification_dataset(
    texts=["positive text", "negative text"],
    labels=[1, 0],
    vocab_size=1000
)
```

## Advanced Usage

### 🎛️ **Custom Evolution Parameters**

```python
from wann_gpt import EvolutionConfig

evolution_config = EvolutionConfig(
    population_size=50,
    num_generations=100,
    mutation_rate=0.8,
    selection_strategy="nsga2",  # Multi-objective optimization
    adaptive_mutation=True,
    weight_samples=[-2.0, -1.0, 0.5, 1.0, 2.0]
)
```

### 🔧 **Custom Architecture Components**

```python
from wann_gpt.architecture.activations import ActivationType, ActivationRegistry

# Add custom activation function
registry = ActivationRegistry()
registry.register_activation("swish", lambda x: x * torch.sigmoid(x))
```

### 📈 **Weight Sensitivity Analysis**

```python
# Analyze how performance varies with weight values
sensitivity = evaluator.analyze_weight_sensitivity(
    model=best_model,
    dataloader=test_loader,
    weight_range=(-3.0, 3.0),
    num_samples=20
)

print(f"Weight sensitivity: {sensitivity['weight_sensitivity']:.4f}")
print(f"Optimal weight: {sensitivity['best_weight']:.2f}")
```

### 🎯 **Ensemble Evaluation**

```python
# Test ensemble of multiple weight instances
ensemble_result = evaluator.ensemble_evaluate(
    model=best_model,
    dataloader=test_loader,
    weight_samples=[-1.0, -0.5, 0.5, 1.0]
)

print(f"Ensemble accuracy: {ensemble_result['ensemble_accuracy']:.4f}")
```

## Command Line Interface

```bash
# Run comprehensive benchmarks
python examples/comprehensive_benchmark.py --task classification --preset classification_small

# Run both classification and generation
python examples/comprehensive_benchmark.py --task both --output ./my_results

# Use custom configuration
python examples/comprehensive_benchmark.py --config my_config.yaml

# Enable ablation studies
python examples/comprehensive_benchmark.py --preset debug --task classification
```

## Results and Analysis

### 📊 **Performance Metrics**

The system tracks comprehensive metrics:

- **Classification**: Accuracy across weight samples, ensemble improvement
- **Generation**: Perplexity, token prediction accuracy  
- **Architecture**: Connection count, layer utilization, head efficiency
- **Robustness**: Performance variance across weights, weight sensitivity
- **Efficiency**: Inference time, memory usage, CUDA utilization

### 🎨 **Visualizations**

The benchmarking suite generates rich visualizations:

- Weight sensitivity curves showing performance vs. shared weight value
- Architecture diagrams with layer activity and connection patterns
- Pareto frontiers balancing accuracy vs. complexity
- Evolution progress tracking fitness and diversity over generations
- Ablation study results highlighting critical components

### 📈 **Expected Outcomes**

Based on WANN research, evolved architectures should:

- ✅ Achieve above-chance performance without weight training
- ✅ Show robustness across different weight values  
- ✅ Discover minimal architectures that encode task solutions
- ✅ Demonstrate that topology can substitute for precise weights

## Research Applications

This implementation enables research into:

- **🔍 Architecture Search**: Finding optimal transformer topologies
- **🧠 Inductive Biases**: Understanding what architectures encode vs. what weights learn
- **🔄 Transfer Learning**: Whether evolved architectures generalize across tasks  
- **⚡ Efficiency**: Minimal architectures for specific capabilities
- **🔬 Interpretability**: Understanding how topology encodes solutions

## Examples and Tutorials

The `examples/` directory contains comprehensive demonstrations:

- `basic_evolution.py`: Simple classification evolution
- `generation_task.py`: Text generation with WANN
- `comprehensive_benchmark.py`: Full system capabilities
- `run_demo.py`: Quick system demonstration

## Contributing

Contributions are welcome! Priority areas:

- 🔧 Additional activation functions and mutation operators
- 📊 New evaluation metrics and visualization tools
- 🗂️ More dataset integrations
- ⚡ Performance optimizations and CUDA kernels
- 📚 Documentation and tutorials

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{wann-gpt-2024,
  title={WANN-GPT: Weight-Agnostic Neural Networks for Transformer Architectures},
  author={Research Team},
  year={2024},
  url={https://github.com/your-username/wann-gpt}
}
```

## References

- Gaier, A., & Ha, D. (2019). Weight Agnostic Neural Networks. *Advances in Neural Information Processing Systems*, 32.
- Vaswani, A., et al. (2017). Attention is All You Need. *Advances in Neural Information Processing Systems*, 30.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Blog*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🚀 **Getting Started Checklist**

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run demo: `python run_demo.py`
3. ✅ Try classification: `python examples/comprehensive_benchmark.py --task classification --preset debug`
4. ✅ Explore configs: Check `examples/comprehensive_benchmark.py --demo-config`
5. ✅ View results: Open generated plots in `./results/` directory

**Ready to evolve some weight-agnostic transformers! 🎯** 