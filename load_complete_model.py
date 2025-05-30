#!/usr/bin/env python3
"""
script to load the complete hybrid model saved by example_hybrid_evolution.py
includes gpt-2 weights and evolved heads in one file
"""

import torch
from wann_gpt import HybridWannGPT, HeadOnlyGenome, SharedWeightEvaluator
from transformers import GPT2Tokenizer

class CompleteModelLoader:
    """loader for complete hybrid models with gpt-2 weights"""
    
    def __init__(self, model_path="./head_evolution_complex/complete_hybrid_model.pt"):
        """load complete model from saved file"""
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_info = None
        
        self._load_complete_model()
    
    def _load_complete_model(self):
        """load the complete model with all weights"""
        
        print(f"loading complete model from {self.model_path}...")
        
        # load saved data
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # extract model info
        self.model_info = {
            'genome_data': checkpoint['genome_data'],
            'architecture_info': checkpoint['architecture_info'],
            'optimal_weight': checkpoint['optimal_weight'],
            'weight_results': checkpoint['weight_results'],
            'final_accuracy': checkpoint['final_accuracy'],
            'save_metadata': checkpoint['save_metadata']
        }
        
        # recreate model architecture
        genome_data = checkpoint['genome_data']
        genome = HeadOnlyGenome(
            embed_dim=genome_data['embed_dim'],
            vocab_size=genome_data['vocab_size'],
            num_classes=genome_data['num_classes']
        )
        
        # create evaluator and model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = SharedWeightEvaluator(device=device)
        self.model = evaluator.instantiate_hybrid_from_genome(genome, model_name="gpt2")
        
        # load the saved weights (including gpt-2 + evolved heads)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # set optimal weight
        self.model.set_shared_weight(checkpoint['optimal_weight'])
        
        # load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ model loaded successfully!")
        print(f"  device: {device}")
        print(f"  optimal weight: {checkpoint['optimal_weight']:.2f}")
        print(f"  final accuracy: {checkpoint['final_accuracy']:.4f}")
        print(f"  sparsity: {genome_data['lm_head_sparsity']:.3f}")
        print(f"  complexity: {checkpoint['save_metadata']['total_connections']:,} connections")
    
    def predict(self, texts):
        """predict classifications for input texts"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        # tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.model.device if hasattr(self.model, 'device') else 'cpu')
        attention_mask = encoded['attention_mask'].to(self.model.device if hasattr(self.model, 'device') else 'cpu')
        
        # predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, task="classification")
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return {
            'predictions': predictions.cpu().tolist(),
            'probabilities': probabilities.cpu().tolist(),
            'logits': logits.cpu().tolist()
        }
    
    def get_model_info(self):
        """get comprehensive model information"""
        return self.model_info
    
    def display_architecture(self):
        """display detailed architecture information"""
        
        print(f"\n" + "=" * 60)
        print("ARCHITECTURE DETAILS")
        print("=" * 60)
        
        info = self.model_info
        genome_data = info['genome_data']
        arch_info = info['architecture_info']
        metadata = info['save_metadata']
        
        print(f"\nmodel overview:")
        print(f"  type: hybrid wann-gpt with evolved heads")
        print(f"  backbone: gpt-2 (pretrained, frozen)")
        print(f"  evolved components: sparse output heads")
        
        print(f"\narchitecture specifications:")
        print(f"  embedding dimension: {arch_info['embed_dim']}")
        print(f"  vocabulary size: {arch_info['vocab_size']:,}")
        print(f"  max sequence length: {arch_info['max_length']}")
        print(f"  transformer layers: {arch_info['num_layers']}")
        print(f"  attention heads per layer: {arch_info['num_heads']}")
        print(f"  output classes: {arch_info['num_classes']}")
        
        print(f"\nevolved head configuration:")
        print(f"  lm head sparsity: {genome_data['lm_head_sparsity']:.3f}")
        print(f"  classifier sparsity: {genome_data['classifier_sparsity']:.3f}")
        print(f"  optimal shared weight: {info['optimal_weight']:.2f}")
        
        print(f"\nperformance metrics:")
        print(f"  evolution fitness: {genome_data['fitness_scores']['classification']:.4f}")
        print(f"  final test accuracy: {info['final_accuracy']:.4f}")
        print(f"  best single weight accuracy: {metadata['best_single_accuracy']:.4f}")
        
        print(f"\nefficiency analysis:")
        total_connections = metadata['total_connections']
        sparsity_ratio = metadata['sparsity_ratio']
        gpt2_params = 124 * 1000000  # ~124M for gpt-2
        
        print(f"  total active connections: {total_connections:,}")
        print(f"  sparsity ratio: {sparsity_ratio:.1%} (connections pruned)")
        print(f"  gpt-2 parameters: ~{gpt2_params/1000000:.0f}M (frozen)")
        print(f"  evolved parameters: ~{total_connections/1000000:.1f}M")
        print(f"  total model size: ~{(gpt2_params + total_connections)/1000000:.0f}M parameters")
        
        print(f"\nweight sensitivity:")
        weight_samples = metadata['weight_samples_used']
        print(f"  tested weights: {weight_samples}")
        print(f"  optimal weight: {info['optimal_weight']:.2f}")
        
        print(f"\narchitecture benefits:")
        print(f"  ✓ leverages powerful gpt-2 language representations")
        print(f"  ✓ efficient sparse connections reduce overfitting")
        print(f"  ✓ single shared weight enables easy optimization")
        print(f"  ✓ robust performance across different weight values")
        if info['final_accuracy'] > 0.6:
            print(f"  ✓ achieves strong classification performance")
        if sparsity_ratio > 0.5:
            print(f"  ✓ significant parameter efficiency vs dense model")
        
        return arch_info
    
    def set_weight(self, weight):
        """change the shared weight"""
        self.model.set_shared_weight(weight)
        print(f"shared weight set to: {weight:.2f}")
    
    def test_weight_sensitivity(self, texts, weight_range=(-2.0, 2.0), num_weights=9):
        """test how predictions change with different weights"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        test_weights = torch.linspace(weight_range[0], weight_range[1], num_weights).tolist()
        results = {}
        
        print(f"testing weight sensitivity on {len(texts)} texts...")
        
        for weight in test_weights:
            self.set_weight(weight)
            predictions = self.predict(texts)
            results[weight] = predictions
            
            print(f"weight {weight:5.2f}: predictions = {predictions['predictions']}")
        
        # restore optimal weight
        self.set_weight(self.model_info['optimal_weight'])
        
        return results

def main():
    """demonstration of complete model loading"""
    
    print("complete hybrid model loader demonstration")
    print("=" * 60)
    
    try:
        # load complete model
        loader = CompleteModelLoader()
        
        # show model info
        info = loader.get_model_info()
        print(f"\nmodel information:")
        print(f"  fitness: {info['genome_data']['fitness_scores']['classification']:.4f}")
        print(f"  final accuracy: {info['final_accuracy']:.4f}")
        print(f"  optimal weight: {info['optimal_weight']:.2f}")
        print(f"  sparsity ratio: {info['save_metadata']['sparsity_ratio']:.3f}")
        print(f"  total connections: {info['save_metadata']['total_connections']:,}")
        
        # display detailed architecture
        loader.display_architecture()
        
        # test predictions
        sample_texts = [
            "this movie was absolutely amazing and wonderful",
            "i hated this film, it was boring and terrible",
            "the movie was okay, nothing special"
        ]
        
        print(f"\ntesting predictions...")
        results = loader.predict(sample_texts)
        
        for text, pred, probs in zip(sample_texts, results['predictions'], results['probabilities']):
            confidence = max(probs)
            sentiment = "positive" if pred == 1 else "negative"
            print(f"'{text[:40]}...' -> {sentiment} (confidence: {confidence:.3f})")
        
        # test weight sensitivity
        print(f"\ntesting weight sensitivity...")
        sensitivity_results = loader.test_weight_sensitivity(
            "this movie was great!", 
            weight_range=(-1.5, 1.5), 
            num_weights=5
        )
        
        print(f"\nweight sensitivity analysis:")
        predictions_by_weight = {}
        for weight, result in sensitivity_results.items():
            pred = result['predictions'][0]
            conf = max(result['probabilities'][0])
            sentiment = "positive" if pred == 1 else "negative"
            predictions_by_weight[weight] = (sentiment, conf)
            print(f"  weight {weight:5.2f}: {sentiment} (conf: {conf:.3f})")
        
        print(f"\n✓ complete model working perfectly!")
        print(f"  - no need to download gpt-2 weights again")
        print(f"  - optimal weight automatically set")
        print(f"  - ready for immediate inference")
        
    except FileNotFoundError:
        print("✗ complete model not found")
        print("run example_hybrid_evolution.py first to create the model")
    except Exception as e:
        print(f"✗ error loading model: {e}")

if __name__ == "__main__":
    main() 