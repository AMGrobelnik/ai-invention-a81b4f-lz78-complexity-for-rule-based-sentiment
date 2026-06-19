# Dataset Collection for LZ78 Complexity Sentiment Experiments

## Overview
This directory contains three standardized benchmark sentiment datasets for evaluating LZ78 complexity-based sentiment classification.

## Datasets

### 1. IMDB Movie Reviews
- **Source**: stanfordnlp/imdb (HuggingFace)
- **Domain**: Movies
- **Size**: 25,000 train + 25,000 test (balanced)
- **Labels**: positive, negative
- **Text length**: Long (full reviews)
- **Files**: full_imdb.json, mini_imdb.json, preview_imdb.json, full_imdb_balanced.json

### 2. Amazon Polarity
- **Source**: fancyzhx/amazon_polarity (HuggingFace)
- **Domain**: Product reviews
- **Size**: 50,000 train + 10,000 test (sampled from 3.6M original)
- **Labels**: positive, negative
- **Text length**: Medium (title + content combined)
- **Files**: full_amazon_polarity.json, mini_amazon_polarity.json, preview_amazon_polarity.json, full_amazon_polarity_balanced.json

### 3. Tweet Sentiment
- **Source**: cardiffnlp/tweet_eval (sentiment config)
- **Domain**: Social media (Twitter)
- **Size**: 26,073 train + 6,347 test (after dropping neutral class)
- **Labels**: positive, negative (neutral examples removed)
- **Text length**: Short (tweet length)
- **Files**: full_tweet_sentiment.json, mini_tweet_sentiment.json, preview_tweet_sentiment.json, full_tweet_sentiment_balanced.json

## Data Schema
All datasets follow the same JSON schema:
```json
{
  "dataset_name": "...",
  "domain": "...",
  "train": [{"text": "...", "label": "positive|negative"}, ...],
  "test": [{"text": "...", "label": "positive|negative"}, ...]
}
```

## Usage
Use the `full_*.json` files for experiments. Balanced versions (`*_balanced.json`) have equal positive/negative examples in both train and test splits.

## File Sizes
Total: ~195 MB (all files under 300MB as required)

## Provenance
See `metadata.json` for detailed provenance information, including original papers and dataset citations.
