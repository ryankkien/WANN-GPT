#!/usr/bin/env python3
"""
simple script to quickly test evolved model on new data
useful for deployment and quick evaluation
"""

import torch
import json
from pathlib import Path
from wann_gpt import (
    HybridWannGPT, HeadOnlyGenome, SharedWeightEvaluator, 
    load_classification_data
)

class EvolvedModelTester:
    """simple wrapper for testing evolved models"""
    
    def __init__(self, evolution_dir="./head_evolution_complex"):
        """load evolved model from evolution directory"""
        
        self.evolution_path = Path(evolution_dir)
        self.model = None
        self.genome = None
        self.evaluator = None
        
        self._load_model()
    
    def _load_model(self):
        """load the evolved model"""
        
        # load genome data
        genome_file = self.evolution_path / "best_head_genome.json"
        with open(genome_file, 'r') as f:
            genome_data = json.load(f)
        
        # create genome
        self.genome = HeadOnlyGenome(
            embed_dim=genome_data["embed_dim"],
            vocab_size=genome_data["vocab_size"], 
            num_classes=genome_data["num_classes"]
        )
        
        # set properties
        self.genome.lm_head_sparsity = genome_data["lm_head_sparsity"]
        self.genome.classifier_sparsity = genome_data["classifier_sparsity"]
        self.genome.fitness_scores = genome_data["fitness_scores"]
        
        # instantiate model
        self.evaluator = SharedWeightEvaluator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.evaluator.instantiate_hybrid_from_genome(
            self.genome, model_name="gpt2"
        )
        
        print(f"loaded evolved model:")
        print(f"  fitness: {self.genome.fitness_scores.get('classification', 0.0):.4f}")
        print(f"  complexity: {self.model.get_total_complexity():,}")
        print(f"  device: {self.evaluator.device}")
    
    def set_optimal_weight(self, test_loader, num_weights=7):
        """find and set optimal shared weight"""
        
        print("finding optimal weight...")
        
        # test range of weights
        test_weights = torch.linspace(-2.0, 2.0, num_weights).tolist()
        best_weight = 1.0
        best_accuracy = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for weight in test_weights:
                self.model.set_shared_weight(weight)
                
                correct = 0
                total = 0
                
                # test on small subset
                for i, batch in enumerate(test_loader):
                    if i >= 5:  # only test on few batches
                        break
                        
                    input_ids = batch['input_ids'].to(self.evaluator.device)
                    labels = batch['labels'].to(self.evaluator.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.evaluator.device)
                    
                    logits = self.model(input_ids, attention_mask, task="classification")
                    predictions = torch.argmax(logits, dim=-1)
                    
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                
                accuracy = correct / total if total > 0 else 0.0
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weight = weight
                
                print(f"  weight {weight:5.2f}: accuracy = {accuracy:.4f}")
        
        # set best weight
        self.model.set_shared_weight(best_weight)
        print(f"set optimal weight: {best_weight:.2f} (accuracy: {best_accuracy:.4f})")
        
        return best_weight
    
    def predict(self, texts, tokenizer=None):
        """predict classifications for text inputs"""
        
        if tokenizer is None:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # tokenize texts
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=1024, 
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.evaluator.device)
        attention_mask = encoded['attention_mask'].to(self.evaluator.device)
        
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
    
    def evaluate_on_dataset(self, test_loader):
        """evaluate model on test dataset"""
        
        print("evaluating on test dataset...")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.evaluator.device)
                labels = batch['labels'].to(self.evaluator.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.evaluator.device)
                
                logits = self.model(input_ids, attention_mask, task="classification")
                predictions = torch.argmax(logits, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        accuracy = correct / total
        
        # calculate per-class accuracy if binary
        if len(set(all_labels)) == 2:
            class_0_correct = sum(1 for p, l in zip(all_predictions, all_labels) if l == 0 and p == l)
            class_1_correct = sum(1 for p, l in zip(all_predictions, all_labels) if l == 1 and p == l)
            class_0_total = sum(1 for l in all_labels if l == 0)
            class_1_total = sum(1 for l in all_labels if l == 1)
            
            class_0_acc = class_0_correct / class_0_total if class_0_total > 0 else 0.0
            class_1_acc = class_1_correct / class_1_total if class_1_total > 0 else 0.0
            
            print(f"overall accuracy: {accuracy:.4f}")
            print(f"class 0 accuracy: {class_0_acc:.4f} ({class_0_correct}/{class_0_total})")
            print(f"class 1 accuracy: {class_1_acc:.4f} ({class_1_correct}/{class_1_total})")
        else:
            print(f"accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def get_model_info(self):
        """get model information"""
        
        arch_info = self.model.get_architecture_info()
        
        return {
            'architecture': arch_info,
            'genome_fitness': self.genome.fitness_scores,
            'sparsity': {
                'lm_head': self.genome.lm_head_sparsity,
                'classifier': self.genome.classifier_sparsity
            }
        }

def main():
    """demonstration of evolved model testing"""
    
    print("evolved model tester demonstration")
    print("=" * 50)
    
    # load evolved model
    try:
        tester = EvolvedModelTester()
    except FileNotFoundError:
        print("error: no evolved model found")
        print("run example_hybrid_evolution.py first")
        return
    
    # load test data
    print("\nloading test data...")
    _, test_loader, num_classes = load_classification_data(
        dataset_name="imdb",
        vocab_size=50257,
        max_length=1024,
        batch_size=16,
        subset_size=100  # small for quick testing
    )
    
    # optimize weight
    optimal_weight = tester.set_optimal_weight(test_loader)
    
    # evaluate model
    print(f"\nevaluating model...")
    results = tester.evaluate_on_dataset(test_loader)
    
    # test on individual texts
    print(f"\ntesting individual predictions...")
    sample_texts = [
        "this movie was absolutely terrible and boring",
        "i loved this film, it was amazing and entertaining",
        "the movie was okay, nothing special but not bad either"
    ]
    
    predictions = tester.predict(sample_texts)
    
    for text, pred, probs in zip(sample_texts, predictions['predictions'], predictions['probabilities']):
        confidence = max(probs)
        sentiment = "positive" if pred == 1 else "negative"
        print(f"text: {text[:50]}...")
        print(f"prediction: {sentiment} (confidence: {confidence:.3f})")
        print()
    
    # show model info
    print("model information:")
    info = tester.get_model_info()
    print(f"  total complexity: {info['architecture']['total_complexity']:,}")
    print(f"  shared weight: {info['architecture']['shared_weight']:.2f}")
    print(f"  fitness: {info['genome_fitness']['classification']:.4f}")
    print(f"  lm head sparsity: {info['sparsity']['lm_head']:.3f}")
    print(f"  classifier sparsity: {info['sparsity']['classifier']:.3f}")
    
    print(f"\nmodel is ready for deployment!")
    print(f"use EvolvedModelTester() to load and test your model")

if __name__ == "__main__":
    main() 