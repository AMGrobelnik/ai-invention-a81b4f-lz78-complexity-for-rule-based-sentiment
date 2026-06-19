#!/usr/bin/env python3
"""Quick test script to validate method.py implementation."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path.cwd()))

from method import lz78_complexity, load_dataset, vader_predict, textblob_predict
import numpy as np

print("Testing LZ78 complexity...")
test_texts = [
    "abababab",     # Repetitive -> LOW complexity
    "abcdefgh",     # Random -> HIGH complexity
    "I love this!", # Positive sentiment
    "I hate this."  # Negative sentiment
]
for text in test_texts:
    c = lz78_complexity(text)
    print(f"{text[:20]}... -> {c:.4f}")

# Verify: repetitive < random
assert lz78_complexity("abababab") < lz78_complexity("abcdefgh")
print("✓ LZ78 complexity test passed!")

# Test with mini dataset (small sample)
print("\nTesting with small sample from IMDB dataset...")
data = load_dataset('imdb', max_examples=100)  # Load only 100 examples for testing
test_data = data['test'][:50]  # Use 50 test examples
test_texts = [item['text'] for item in test_data]
test_labels = [item['label'] for item in test_data]

print(f"Loaded {len(test_texts)} test examples")

# Compute complexities
complexities = [lz78_complexity(t) for t in test_texts]
pos_c = [c for c, l in zip(complexities, test_labels) if l == 'positive']
neg_c = [c for c, l in zip(complexities, test_labels) if l == 'negative']

print(f"Complexity stats:")
print(f"  Positive: mean={np.mean(pos_c):.4f}, std={np.std(pos_c):.4f}")
print(f"  Negative: mean={np.mean(neg_c):.4f}, std={np.std(neg_c):.4f}")

# Test VADER
print("\nTesting VADER...")
vader_preds = vader_predict(test_texts)
print(f"VADER predictions: {len(vader_preds)} examples")

# Test TextBlob
print("\nTesting TextBlob...")
tb_preds = textblob_predict(test_texts)
print(f"TextBlob predictions: {len(tb_preds)} examples")

print("\n✓ All basic tests passed!")
