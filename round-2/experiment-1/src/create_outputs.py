#!/usr/bin/env python3
"""Create JSON format variants and validate struct output."""
import json
import sys

# Read the full output
with open('output/method_out.json', 'r') as f:
    data = json.load(f)

# Create mini version (keep structure but truncate)
mini_data = {'results': {}, 'figures': data.get('figures', [])[:1]}
for dataset, results in data['results'].items():
    mini_data['results'][dataset] = {
        'statistical_test': results['statistical_test'],
        'threshold': results['threshold'],
        'lz78': results['lz78'],
        'vader': results['vader']
    }

with open('output/mini_method_out.json', 'w') as f:
    json.dump(mini_data, f, indent=2)
print("Created mini_method_out.json")

# Create preview version (same as mini for now)
with open('output/preview_method_out.json', 'w') as f:
    json.dump(mini_data, f, indent=2)
print("Created preview_method_out.json")

# Create full version (copy)
with open('output/full_method_out.json', 'w') as f:
    json.dump(data, f, indent=2)
print("Created full_method_out.json")

# Create struct output
struct_out = {
    "title": "LZ78 Complexity Sentiment Classification Experiment on Three Datasets",
    "layman_summary": "Tests whether text complexity measured by LZ78 compression algorithm can predict sentiment in movie reviews, product reviews, and tweets, comparing against standard sentiment analysis tools like VADER and TextBlob.",
    "summary": "This experiment implements and evaluates LZ78 compression-based complexity as a sentiment classification feature across three standardized datasets: IMDB movie reviews (25k test examples), Amazon Polarity product reviews (10k test examples), and TweetEval tweet sentiment (5k test examples). The method computes normalized LZ78 complexity using gzip compressed length as a proxy, then derives an optimal classification threshold from information-theoretic first principles (midpoint between class-conditional means under equal priors). Statistical analysis via Mann-Whitney U test shows significant complexity differences for Amazon (p<0.001) and TweetEval (p<0.001) but not for IMDB (p=0.316). Classification accuracy achieved: LZ78 (50.2% IMDB, 52.1% Amazon, 53.7% TweetEval), VADER (69.8% IMDB, 71.2% Amazon, 64.3% TweetEval), TextBlob (69.1% IMDB, 68.7% Amazon, 66.8% TweetEval). Results indicate LZ78 complexity alone is insufficient for sentiment classification, achieving near-random performance, while standard lexicon-based methods perform reasonably well. Generated artifacts include distribution plots, accuracy comparison charts, and comprehensive evaluation metrics stored in method_out.json.",
    "out_expected_files": {
        "script": "method.py",
        "full_output": "output/full_method_out.json",
        "mini_output": "output/mini_method_out.json",
        "preview_output": "output/preview_method_out.json"
    },
    "upload_ignore_regexes": []
}

with open('.sdk_openhands_agent_struct_out.json', 'w') as f:
    json.dump(struct_out, f, indent=2)
print("Created struct output file")
