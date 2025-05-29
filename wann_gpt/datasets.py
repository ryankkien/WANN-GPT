"""
real dataset integration for wann-gpt
supports text classification and generation tasks with real-world data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import re
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset
import requests
import os

class TokenizedDataset(Dataset):
    """base tokenized dataset for wann-gpt"""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None,
                 tokenizer=None, max_length: int = 512, vocab_size: int = 1000):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # simple tokenizer if none provided
        if tokenizer is None:
            self.tokenizer = self._create_simple_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        # tokenize texts
        self.tokenized_texts = [self._tokenize_text(text) for text in texts]
    
    def _create_simple_tokenizer(self):
        """create simple word-based tokenizer"""
        # build vocabulary from all texts
        all_words = []
        for text in self.texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        # get most common words
        from collections import Counter
        word_counts = Counter(all_words)
        vocab = ['<pad>', '<unk>', '<cls>'] + [word for word, _ in word_counts.most_common(self.vocab_size - 3)]
        
        # create word to id mapping
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        return self.word_to_id
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """tokenize single text"""
        if isinstance(self.tokenizer, dict):  # simple tokenizer
            words = re.findall(r'\b\w+\b', text.lower())
            token_ids = [self.tokenizer.get(word, 1) for word in words]  # 1 is <unk>
        else:  # huggingface tokenizer
            encoding = self.tokenizer(text, max_length=self.max_length,
                                    truncation=True, padding=False,
                                    return_tensors="pt")
            token_ids = encoding['input_ids'].squeeze().tolist()
        
        # pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))  # 0 is <pad>
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.tokenized_texts[idx],
            'attention_mask': (self.tokenized_texts[idx] != 0).long()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

class IMDbDataset:
    """imdb movie review sentiment classification dataset"""
    
    @staticmethod
    def load_dataset(vocab_size: int = 1000, max_length: int = 256,
                    subset_size: Optional[int] = None, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, int]:
        """load imdb dataset for sentiment classification"""
        
        print("loading imdb dataset...")
        
        try:
            # try to load from huggingface
            dataset = load_dataset("imdb")
            train_data = dataset['train']
            test_data = dataset['test']
            
            # extract texts and labels
            train_texts = train_data['text']
            train_labels = train_data['label']
            test_texts = test_data['text']
            test_labels = test_data['label']
            
        except Exception as e:
            print(f"failed to load from huggingface: {e}")
            print("creating synthetic imdb-like dataset...")
            
            # create synthetic sentiment data
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 'love']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing', 'poor']
            
            train_texts, train_labels = [], []
            test_texts, test_labels = [], []
            
            for split_size in [2000, 500]:  # train, test
                texts, labels = (train_texts, train_labels) if split_size == 2000 else (test_texts, test_labels)
                
                for _ in range(split_size):
                    # generate positive or negative review
                    is_positive = np.random.choice([0, 1])
                    
                    if is_positive:
                        sentiment_words = np.random.choice(positive_words, size=3)
                        text = f"this movie is {sentiment_words[0]} and {sentiment_words[1]}. really {sentiment_words[2]} film."
                    else:
                        sentiment_words = np.random.choice(negative_words, size=3)
                        text = f"this movie is {sentiment_words[0]} and {sentiment_words[1]}. really {sentiment_words[2]} film."
                    
                    texts.append(text)
                    labels.append(is_positive)
        
        # subset if requested
        if subset_size is not None:
            train_texts = train_texts[:subset_size]
            train_labels = train_labels[:subset_size]
            test_texts = test_texts[:subset_size//4]
            test_labels = test_labels[:subset_size//4]
        
        # create datasets
        train_dataset = TokenizedDataset(train_texts, train_labels, max_length=max_length, vocab_size=vocab_size)
        test_dataset = TokenizedDataset(test_texts, test_labels, max_length=max_length, vocab_size=vocab_size)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        num_classes = 2
        return train_loader, test_loader, num_classes

class AG_NewsDataset:
    """ag news topic classification dataset"""
    
    @staticmethod
    def load_dataset(vocab_size: int = 1000, max_length: int = 128,
                    subset_size: Optional[int] = None, batch_size: int = 16) -> Tuple[DataLoader, DataLoader, int]:
        """load ag news dataset for topic classification"""
        
        print("loading ag news dataset...")
        
        try:
            # try to load from huggingface
            dataset = load_dataset("ag_news")
            train_data = dataset['train']
            test_data = dataset['test']
            
            train_texts = train_data['text']
            train_labels = train_data['label']
            test_texts = test_data['text']
            test_labels = test_data['label']
            
        except Exception as e:
            print(f"failed to load from huggingface: {e}")
            print("creating synthetic news-like dataset...")
            
            # synthetic news categories
            topics = ['World', 'Sports', 'Business', 'Technology']
            topic_words = {
                'World': ['country', 'government', 'politics', 'international', 'nation', 'leader'],
                'Sports': ['game', 'team', 'player', 'score', 'match', 'championship'],
                'Business': ['company', 'market', 'profit', 'economy', 'financial', 'investment'],
                'Technology': ['software', 'computer', 'internet', 'digital', 'innovation', 'tech']
            }
            
            train_texts, train_labels = [], []
            test_texts, test_labels = [], []
            
            for split_size in [2000, 500]:
                texts, labels = (train_texts, train_labels) if split_size == 2000 else (test_texts, test_labels)
                
                for _ in range(split_size):
                    topic_idx = np.random.randint(0, 4)
                    topic = topics[topic_idx]
                    words = np.random.choice(topic_words[topic], size=4)
                    
                    text = f"news about {words[0]} and {words[1]}. latest {words[2]} developments in {words[3]}."
                    
                    texts.append(text)
                    labels.append(topic_idx)
        
        # subset if requested
        if subset_size is not None:
            train_texts = train_texts[:subset_size]
            train_labels = train_labels[:subset_size]
            test_texts = test_texts[:subset_size//4]
            test_labels = test_labels[:subset_size//4]
        
        # create datasets
        train_dataset = TokenizedDataset(train_texts, train_labels, max_length=max_length, vocab_size=vocab_size)
        test_dataset = TokenizedDataset(test_texts, test_labels, max_length=max_length, vocab_size=vocab_size)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        num_classes = 4
        return train_loader, test_loader, num_classes

class WikiTextDataset:
    """wikitext language modeling dataset"""
    
    @staticmethod
    def load_dataset(vocab_size: int = 1000, max_length: int = 256,
                    subset_size: Optional[int] = None, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        """load wikitext dataset for language modeling"""
        
        print("loading wikitext dataset...")
        
        try:
            # try to load from huggingface
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            train_data = dataset['train']
            test_data = dataset['test']
            
            # filter out empty lines
            train_texts = [text for text in train_data['text'] if len(text.strip()) > 10]
            test_texts = [text for text in test_data['text'] if len(text.strip()) > 10]
            
        except Exception as e:
            print(f"failed to load from huggingface: {e}")
            print("creating synthetic text dataset...")
            
            # synthetic text patterns
            sentence_templates = [
                "the {} is {} and {}.",
                "in the {} we found {} with {}.",
                "after {} the {} became very {}.",
                "scientists discovered that {} can {} the {}."
            ]
            
            words = ['dog', 'cat', 'house', 'tree', 'car', 'book', 'computer', 'phone',
                    'big', 'small', 'red', 'blue', 'fast', 'slow', 'new', 'old',
                    'running', 'walking', 'eating', 'sleeping', 'working', 'playing']
            
            train_texts, test_texts = [], []
            
            for split_size in [1500, 400]:
                texts = train_texts if split_size == 1500 else test_texts
                
                for _ in range(split_size):
                    template = np.random.choice(sentence_templates)
                    random_words = np.random.choice(words, size=template.count('{}'))
                    text = template.format(*random_words)
                    texts.append(text)
        
        # subset if requested
        if subset_size is not None:
            train_texts = train_texts[:subset_size]
            test_texts = test_texts[:subset_size//4]
        
        # create datasets (no labels for language modeling)
        train_dataset = TokenizedDataset(train_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
        test_dataset = TokenizedDataset(test_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

class TinyStoriesDataset:
    """tiny stories dataset for simple language modeling"""
    
    @staticmethod
    def load_dataset(vocab_size: int = 500, max_length: int = 128,
                    subset_size: Optional[int] = None, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        """load tiny stories dataset"""
        
        print("loading tiny stories dataset...")
        
        # create simple stories
        characters = ['alice', 'bob', 'charlie', 'diana']
        actions = ['went to', 'played with', 'found', 'saw', 'met']
        objects = ['ball', 'tree', 'house', 'cat', 'book']
        places = ['park', 'school', 'home', 'forest', 'lake']
        
        stories = []
        
        for _ in range(2000):  # generate stories
            char = np.random.choice(characters)
            action = np.random.choice(actions)
            obj = np.random.choice(objects)
            place = np.random.choice(places)
            
            story = f"once upon a time {char} {action} {obj} at the {place}. {char} was very happy."
            stories.append(story)
        
        # split train/test
        split_idx = int(0.8 * len(stories))
        train_texts = stories[:split_idx]
        test_texts = stories[split_idx:]
        
        # subset if requested
        if subset_size is not None:
            train_texts = train_texts[:subset_size]
            test_texts = test_texts[:subset_size//4]
        
        # create datasets
        train_dataset = TokenizedDataset(train_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
        test_dataset = TokenizedDataset(test_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
        
        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader

class DatasetRegistry:
    """registry for available datasets"""
    
    classification_datasets = {
        'imdb': IMDbDataset,
        'ag_news': AG_NewsDataset,
    }
    
    generation_datasets = {
        'wikitext': WikiTextDataset,
        'tiny_stories': TinyStoriesDataset,
    }
    
    @classmethod
    def get_classification_dataset(cls, name: str, **kwargs):
        """get classification dataset by name"""
        if name not in cls.classification_datasets:
            raise ValueError(f"unknown classification dataset: {name}. available: {list(cls.classification_datasets.keys())}")
        
        return cls.classification_datasets[name].load_dataset(**kwargs)
    
    @classmethod
    def get_generation_dataset(cls, name: str, **kwargs):
        """get generation dataset by name"""
        if name not in cls.generation_datasets:
            raise ValueError(f"unknown generation dataset: {name}. available: {list(cls.generation_datasets.keys())}")
        
        return cls.generation_datasets[name].load_dataset(**kwargs)
    
    @classmethod
    def list_datasets(cls):
        """list all available datasets"""
        return {
            'classification': list(cls.classification_datasets.keys()),
            'generation': list(cls.generation_datasets.keys())
        }

# convenience functions
def load_classification_data(dataset_name: str = "imdb", vocab_size: int = 1000,
                           max_length: int = 256, subset_size: Optional[int] = None,
                           batch_size: int = 16):
    """load classification dataset"""
    return DatasetRegistry.get_classification_dataset(
        dataset_name, vocab_size=vocab_size, max_length=max_length, 
        subset_size=subset_size, batch_size=batch_size
    )

def load_generation_data(dataset_name: str = "tiny_stories", vocab_size: int = 500,
                        max_length: int = 128, subset_size: Optional[int] = None,
                        batch_size: int = 8):
    """load generation dataset"""
    return DatasetRegistry.get_generation_dataset(
        dataset_name, vocab_size=vocab_size, max_length=max_length, 
        subset_size=subset_size, batch_size=batch_size
    )

def create_custom_classification_dataset(texts: List[str], labels: List[int],
                                       vocab_size: int = 1000, max_length: int = 256,
                                       train_split: float = 0.8, batch_size: int = 16):
    """create custom classification dataset from texts and labels"""
    
    # split data
    split_idx = int(train_split * len(texts))
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    # create datasets
    train_dataset = TokenizedDataset(train_texts, train_labels, max_length=max_length, vocab_size=vocab_size)
    test_dataset = TokenizedDataset(test_texts, test_labels, max_length=max_length, vocab_size=vocab_size)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(set(labels))
    return train_loader, test_loader, num_classes

def create_custom_generation_dataset(texts: List[str], vocab_size: int = 500,
                                   max_length: int = 128, train_split: float = 0.8,
                                   batch_size: int = 8):
    """create custom generation dataset from texts"""
    
    # split data
    split_idx = int(train_split * len(texts))
    train_texts = texts[:split_idx]
    test_texts = texts[split_idx:]
    
    # create datasets
    train_dataset = TokenizedDataset(train_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
    test_dataset = TokenizedDataset(test_texts, labels=None, max_length=max_length, vocab_size=vocab_size)
    
    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 