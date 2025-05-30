# Model Analysis Guide

After running `example_hybrid_evolution.py`, you'll have an evolved hybrid model that combines a frozen GPT-2 backbone with evolved sparse output heads. This guide shows you how to examine and test your evolved model.

## Quick Start

### 1. Run the Example Evolution (if you haven't already)
```bash
python example_hybrid_evolution.py
```

This creates the `head_evolution_complex/` directory with your evolved model.

### 2. Quick Model Testing
```bash
python test_evolved_model.py
```

This script provides:
- **Quick evaluation** on test data
- **Optimal weight finding** for best performance  
- **Individual text prediction** examples
- **Model deployment wrapper**

### 3. Comprehensive Analysis
```bash
python examine_evolved_model.py
```

This script provides:
- **Architecture analysis** (sparsity, connections, complexity)
- **Weight sensitivity curves** showing performance across different shared weights
- **Connection pattern visualizations** 
- **Comprehensive benchmarking** with detailed metrics
- **Ensemble performance testing**

## Understanding Your Evolved Model

### Architecture Components

Your evolved hybrid model has these key components:

1. **Frozen GPT-2 Backbone**: Pretrained transformer layers (12 layers, 768 dim)
2. **Evolved LM Head**: Sparse connections for language modeling 
3. **Evolved Classifier**: Sparse connections for classification
4. **Shared Weight**: Single parameter that scales all evolved connections

### Key Metrics

- **Fitness**: Average accuracy across multiple shared weight values
- **Sparsity**: Fraction of connections that are pruned (0.0 = dense, 1.0 = fully sparse)
- **Complexity**: Total number of active connections
- **Weight Sensitivity**: How much performance varies with shared weight changes

## Detailed Usage

### Using the EvolvedModelTester Class

```python
from test_evolved_model import EvolvedModelTester

# Load your evolved model
tester = EvolvedModelTester("./head_evolution_complex")

# Get model information
info = tester.get_model_info()
print(f"Model complexity: {info['architecture']['total_complexity']:,}")
print(f"Fitness: {info['genome_fitness']['classification']:.4f}")

# Test on your own texts
texts = ["I love this movie!", "This film was terrible"]
results = tester.predict(texts)

for text, pred, probs in zip(texts, results['predictions'], results['probabilities']):
    confidence = max(probs)
    sentiment = "positive" if pred == 1 else "negative"
    print(f"'{text}' -> {sentiment} (confidence: {confidence:.3f})")
```

### Understanding Weight Sensitivity

The evolved model uses a single shared weight parameter. Different weights can dramatically affect performance:

- **Negative weights** (-2.0 to -0.5): Often work well for classification
- **Small weights** (-0.5 to 0.5): May provide stable performance  
- **Positive weights** (0.5 to 2.0): Traditional neural network range
- **Zero weight**: Disables the evolved connections entirely

### Interpreting Visualizations

The analysis scripts generate several plots:

#### Weight Sensitivity Curve (`weight_sensitivity.png`)
- Shows accuracy vs shared weight value
- Reveals optimal weight for your model
- Indicates robustness (flat curve = robust, spiky = sensitive)

#### Connection Patterns (`*_connections.png`)  
- Blue/red heatmaps showing which connections exist
- Dense regions = important feature interactions
- Sparse regions = pruned/unnecessary connections

#### Architecture Summary (`model_analysis_results/architecture_summary.png`)
- Overall model structure and parameters
- Layer activity patterns
- Complexity comparisons

## Advanced Analysis

### Benchmarking Against Baselines

```python
from wann_gpt import WannBenchmarkSuite, load_classification_data

# Load your model and data
tester = EvolvedModelTester()
_, test_loader, _ = load_classification_data("imdb", subset_size=500)

# Run comprehensive benchmark
benchmark_suite = WannBenchmarkSuite()
results = benchmark_suite.comprehensive_benchmark(
    model=tester.model,
    test_loader=test_loader,
    task_type="classification"
)

print(f"Robustness score: {results.robustness_score:.4f}")
print(f"Ensemble improvement: {results.ensemble_improvement:.4f}")
```

### Testing Different Shared Weights

```python
# Test specific weight values
test_weights = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

for weight in test_weights:
    tester.model.set_shared_weight(weight)
    results = tester.evaluate_on_dataset(test_loader)
    print(f"Weight {weight:4.1f}: Accuracy {results['accuracy']:.4f}")
```

### Ensemble Prediction

```python
# Use multiple weights for ensemble prediction
weight_samples = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
ensemble_results = tester.evaluator.ensemble_evaluate(
    tester.model, test_loader, weight_samples=weight_samples
)

print(f"Single weight accuracy: {results['accuracy']:.4f}")
print(f"Ensemble accuracy: {ensemble_results['ensemble_accuracy']:.4f}")
print(f"Improvement: +{ensemble_results['ensemble_accuracy'] - results['accuracy']:.4f}")
```

## Output Files and Directories

After running the analysis scripts, you'll find:

### Generated Files
- `weight_sensitivity.png` - Weight sensitivity analysis
- `lm_head_connections.png` - Language model head connections  
- `classifier_connections.png` - Classification head connections

### Generated Directories
- `model_analysis_results/` - Comprehensive benchmark results
  - `architecture_summary.png` - Model architecture overview
  - `weight_sensitivity_classification.png` - Detailed sensitivity analysis
  - `ensemble_comparison_classification.png` - Single vs ensemble performance
  - `benchmark_results.json` - Detailed numerical results

### Evolution Directory (`head_evolution_complex/`)
- `best_head_genome.json` - Evolved model specification
- `checkpoint_gen_*.pkl` - Evolution checkpoints

## Deployment

### For Production Use

1. **Find optimal weight** using `set_optimal_weight()`
2. **Validate on hold-out data** to ensure generalization
3. **Save the model state**:

```python
# Save complete model state
torch.save({
    'model_state_dict': tester.model.state_dict(),
    'optimal_weight': optimal_weight,
    'genome_data': tester.genome.__dict__,
    'architecture_info': tester.model.get_architecture_info()
}, 'evolved_model_for_deployment.pt')
```

4. **Load for inference**:

```python
# Load for inference
checkpoint = torch.load('evolved_model_for_deployment.pt')
tester.model.load_state_dict(checkpoint['model_state_dict'])
tester.model.set_shared_weight(checkpoint['optimal_weight'])
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Run `example_hybrid_evolution.py` first
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Poor performance**: Try different shared weights or re-run evolution
4. **Plots not showing**: Install matplotlib: `pip install matplotlib`

### Performance Tips

- **Use GPU** if available for faster evaluation
- **Reduce subset_size** in data loading for quicker testing  
- **Cache optimal weights** to avoid recomputation
- **Use ensemble prediction** for best accuracy

## Next Steps

1. **Compare with baselines** using the benchmarking suite
2. **Try different datasets** by modifying the data loading
3. **Experiment with weights** to understand sensitivity
4. **Deploy the model** using the EvolvedModelTester wrapper
5. **Evolve new models** with different hyperparameters

The evolved model combines the power of pretrained GPT-2 with efficiently evolved sparse output heads, providing a good balance of performance and efficiency for classification tasks. 