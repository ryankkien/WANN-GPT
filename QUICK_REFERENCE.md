# Quick Reference: Examining & Testing Evolved Models

## After Running `example_hybrid_evolution.py`

You'll have these new files/directories:
- `head_evolution_complex/` - Contains your evolved model
  - `best_head_genome.json` - Model architecture specification  
  - `checkpoint_gen_*.pkl` - Evolution checkpoints
  - **`complete_hybrid_model.pt` - Complete model with GPT-2 weights (NEW!)**

## 🚀 Quick Commands

### 1. Simple Testing & Deployment
```bash
python test_evolved_model.py
```
**What it does:**
- Loads your evolved model
- Finds optimal shared weight automatically
- Tests on sample data
- Shows individual text predictions
- Provides deployment-ready wrapper class

### 2. Complete Model Loading (NEW!)
```bash
python load_complete_model.py
```
**What it does:**
- Loads complete model with GPT-2 weights included
- No need to reconstruct or download GPT-2 again
- Optimal weight automatically set
- Ready for immediate inference

### 3. Comprehensive Analysis  
```bash
python examine_evolved_model.py
```
**What it does:**
- Detailed architecture analysis
- Weight sensitivity curves with plots
- Connection pattern visualizations
- Comprehensive benchmarking suite
- Ensemble performance testing
- Saves detailed plots and reports

## 📊 Understanding Your Model

### Key Components
- **GPT-2 Backbone**: Frozen pretrained transformer (768 dim, 12 layers)
- **Evolved Heads**: Sparse output layers for classification/generation
- **Shared Weight**: Single parameter that controls all evolved connections

### Important Metrics
- **Fitness**: ~0.57 = average accuracy across different shared weights
- **Sparsity**: ~0.38 = 38% of connections pruned for efficiency  
- **Complexity**: ~24M connections (much sparser than full dense model)

### Shared Weight Behavior
- **Zero weight** (0.0): Often works well, disables some connections
- **Negative weights** (-2.0 to -1.0): Can provide good performance
- **Positive weights** (0.5 to 2.0): Traditional neural network range
- **Weight sensitivity**: How much performance changes with weight

## 💡 Key Usage Patterns

### Quick Model Loading (Reconstructed)
```python
from test_evolved_model import EvolvedModelTester

tester = EvolvedModelTester()  # Loads from ./head_evolution_complex/
info = tester.get_model_info()
print(f"Model fitness: {info['genome_fitness']['classification']:.4f}")
```

### Complete Model Loading (NEW - Includes GPT-2!)
```python
from load_complete_model import CompleteModelLoader

loader = CompleteModelLoader()  # Loads complete model with GPT-2 weights
results = loader.predict(["I love this movie!"])
print(f"Prediction: {results['predictions'][0]}")
```

### Text Classification
```python
texts = ["I love this movie!", "This film was boring"]
results = loader.predict(texts)  # or tester.predict(texts)

for text, pred, probs in zip(texts, results['predictions'], results['probabilities']):
    sentiment = "positive" if pred == 1 else "negative"
    confidence = max(probs)
    print(f"'{text}' -> {sentiment} (confidence: {confidence:.3f})")
```

### Finding Optimal Weight
```python
# Automatically finds best weight for your data
optimal_weight = tester.set_optimal_weight(test_loader)
print(f"Best weight: {optimal_weight:.2f}")
```

### Ensemble Prediction (Better Accuracy)
```python
ensemble_results = tester.evaluator.ensemble_evaluate(
    tester.model, test_loader, 
    weight_samples=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
)
print(f"Ensemble accuracy: {ensemble_results['ensemble_accuracy']:.4f}")
```

## 📁 Generated Files

After running `example_hybrid_evolution.py`:

### **Complete Model (NEW!)**
- `head_evolution_complex/complete_hybrid_model.pt` - **Full model with GPT-2 weights**

### Analysis Plots
- `weight_sensitivity.png` - Performance vs shared weight curve
- `lm_head_connections.png` - Language model head connection pattern
- `classifier_connections.png` - Classification head connections
  
### Reports  
- `model_analysis_results/` - Comprehensive benchmark folder
  - `architecture_summary.png` - Model structure overview
  - `benchmark_results.json` - Detailed performance metrics

## 🔧 Common Tasks

### Complete Model Loading (Easiest)
```python
from load_complete_model import CompleteModelLoader
loader = CompleteModelLoader()
results = loader.predict("This movie is amazing!")
```

### Change Shared Weight
```python
loader.set_weight(1.5)  # or tester.model.set_shared_weight(1.5)
```

### Evaluate on Your Data
```python
results = tester.evaluate_on_dataset(your_test_loader)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Save Model for Production (Already Done!)
The complete model is already saved at:
`./head_evolution_complex/complete_hybrid_model.pt`

### Load Complete Model Anywhere
```python
import torch
from load_complete_model import CompleteModelLoader

# Load your evolved model with GPT-2 weights included
loader = CompleteModelLoader("path/to/complete_hybrid_model.pt")
predictions = loader.predict(["Your text here"])
```

## ⚡ Performance Tips

1. **Use GPU** if available (automatically detected)
2. **Use CompleteModelLoader** for fastest loading (no GPT-2 download)
3. **Find optimal weight** for your specific dataset
4. **Use ensemble prediction** for best accuracy
5. **Monitor weight sensitivity** - robust models have flat sensitivity curves
6. **Check sparsity** - higher sparsity = more efficient model

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError` | Run `example_hybrid_evolution.py` first |
| Poor accuracy | Try different shared weights or re-run evolution |
| CUDA out of memory | Reduce batch size or force CPU usage |
| Plots not showing | Install matplotlib: `pip install matplotlib` |
| Slow loading | Use `CompleteModelLoader` instead of reconstructing |

## 📈 What Makes a Good Evolved Model

- **High fitness** (>0.6): Performs well across different shared weights
- **Moderate sparsity** (0.3-0.7): Good balance of efficiency and performance  
- **Low weight sensitivity**: Performance doesn't vary wildly with weight changes
- **Good ensemble improvement**: Multiple weights work together effectively

## 🎯 Next Steps

1. **Use CompleteModelLoader** for fastest deployment
2. **Test on your specific data** by modifying the data loading
3. **Compare with baselines** using the benchmarking suite
4. **Experiment with different weights** for your use case
5. **Re-run evolution** with different hyperparameters if needed

Your evolved model combines GPT-2's language understanding with efficiently learned sparse heads for classification! 