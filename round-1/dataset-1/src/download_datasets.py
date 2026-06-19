#!/usr/bin/env python3
"""Download and standardize three sentiment datasets for LZ78 complexity experiments."""

from loguru import logger
from pathlib import Path
import json
import sys
from datasets import load_dataset
import pandas as pd

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def download_imdb():
    """Download and standardize IMDB dataset."""
    logger.info("Downloading IMDB dataset...")
    output_dir = Path("datasets/imdb")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train = load_dataset('stanfordnlp/imdb', split='train')
    test = load_dataset('stanfordnlp/imdb', split='test')

    # Standardize: label 0=negative, 1=positive -> convert to string labels
    def standardize_example(ex):
        return {
            "text": ex["text"],
            "label": "positive" if ex["label"] == 1 else "negative"
        }

    train_std = [standardize_example(ex) for ex in train]
    test_std = [standardize_example(ex) for ex in test]

    # Save full dataset
    full_data = {
        "dataset_name": "imdb",
        "domain": "movies",
        "train": train_std,
        "test": test_std
    }
    (output_dir / "full_imdb.json").write_text(json.dumps(full_data, indent=2))
    logger.info(f"IMDB: saved {len(train_std)} train, {len(test_std)} test examples")

    # Create mini (100 examples) and preview (10 examples)
    mini_data = {"dataset_name": "imdb", "domain": "movies", "train": train_std[:50], "test": test_std[:50]}
    (output_dir / "mini_imdb.json").write_text(json.dumps(mini_data, indent=2))

    preview_data = {"dataset_name": "imdb", "domain": "movies", "train": train_std[:5], "test": test_std[:5]}
    # Truncate text in preview
    for split in ["train", "test"]:
        for ex in preview_data[split]:
            ex["text"] = ex["text"][:200] + "..." if len(ex["text"]) > 200 else ex["text"]
    (output_dir / "preview_imdb.json").write_text(json.dumps(preview_data, indent=2))

    return full_data

@logger.catch(reraise=True)
def download_amazon_polarity(sample_size=50000):
    """Download and standardize Amazon Polarity dataset (sampled)."""
    logger.info(f"Downloading Amazon Polarity dataset (sampling {sample_size} train, 10000 test)...")
    output_dir = Path("datasets/amazon_polarity")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train = load_dataset('fancyzhx/amazon_polarity', split='train')
    test = load_dataset('fancyzhx/amazon_polarity', split='test')

    # Sample to manageable size
    train = train.shuffle(seed=42).select(range(min(sample_size, len(train))))
    test = test.shuffle(seed=42).select(range(min(10000, len(test))))

    # Standardize: combine title+content, label 0=negative, 1=positive
    def standardize_example(ex):
        text = ex["title"] + " " + ex["content"]
        return {
            "text": text,
            "label": "positive" if ex["label"] == 1 else "negative"
        }

    train_std = [standardize_example(ex) for ex in train]
    test_std = [standardize_example(ex) for ex in test]

    # Save full dataset
    full_data = {
        "dataset_name": "amazon_polarity",
        "domain": "products",
        "train": train_std,
        "test": test_std
    }
    (output_dir / "full_amazon_polarity.json").write_text(json.dumps(full_data, indent=2))
    logger.info(f"Amazon Polarity: saved {len(train_std)} train, {len(test_std)} test examples")

    # Create mini and preview
    mini_data = {"dataset_name": "amazon_polarity", "domain": "products", "train": train_std[:50], "test": test_std[:50]}
    (output_dir / "mini_amazon_polarity.json").write_text(json.dumps(mini_data, indent=2))

    preview_data = {"dataset_name": "amazon_polarity", "domain": "products", "train": train_std[:5], "test": test_std[:5]}
    for split in ["train", "test"]:
        for ex in preview_data[split]:
            ex["text"] = ex["text"][:200] + "..." if len(ex["text"]) > 200 else ex["text"]
    (output_dir / "preview_amazon_polarity.json").write_text(json.dumps(preview_data, indent=2))

    return full_data

@logger.catch(reraise=True)
def download_tweet_sentiment():
    """Download and standardize Tweet Eval sentiment dataset (drop neutral)."""
    logger.info("Downloading Tweet Eval (sentiment) dataset...")
    output_dir = Path("datasets/tweet_sentiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train = load_dataset('cardiffnlp/tweet_eval', 'sentiment', split='train')
    test = load_dataset('cardiffnlp/tweet_eval', 'sentiment', split='test')
    validation = load_dataset('cardiffnlp/tweet_eval', 'sentiment', split='validation')

    # Combine train and validation, then drop neutral (label=1)
    # Labels: 0=negative, 1=neutral, 2=positive
    def standardize_example(ex):
        return {
            "text": ex["text"],
            "label": "positive" if ex["label"] == 2 else "negative"
        }

    # Filter out neutral examples (label=1)
    train_filtered = [ex for ex in train if ex["label"] != 1]
    test_filtered = [ex for ex in test if ex["label"] != 1]
    val_filtered = [ex for ex in validation if ex["label"] != 1]

    # Combine train and validation
    train_combined = train_filtered + val_filtered

    train_std = [standardize_example(ex) for ex in train_combined]
    test_std = [standardize_example(ex) for ex in test_filtered]

    # Save full dataset
    full_data = {
        "dataset_name": "tweet_sentiment",
        "domain": "social_media",
        "train": train_std,
        "test": test_std
    }
    (output_dir / "full_tweet_sentiment.json").write_text(json.dumps(full_data, indent=2))
    logger.info(f"Tweet Sentiment: saved {len(train_std)} train, {len(test_std)} test examples (dropped neutral)")

    # Create mini and preview
    mini_data = {"dataset_name": "tweet_sentiment", "domain": "social_media", "train": train_std[:50], "test": test_std[:50]}
    (output_dir / "mini_tweet_sentiment.json").write_text(json.dumps(mini_data, indent=2))

    preview_data = {"dataset_name": "tweet_sentiment", "domain": "social_media", "train": train_std[:5], "test": test_std[:5]}
    for split in ["train", "test"]:
        for ex in preview_data[split]:
            ex["text"] = ex["text"][:200] + "..." if len(ex["text"]) > 200 else ex["text"]
    (output_dir / "preview_tweet_sentiment.json").write_text(json.dumps(preview_data, indent=2))

    return full_data

@logger.catch(reraise=True)
def main():
    logger.info("Starting dataset download and standardization...")

    # Download all three datasets
    imdb_data = download_imdb()
    amazon_data = download_amazon_polarity(sample_size=50000)
    tweet_data = download_tweet_sentiment()

    # Create a summary file
    summary = {
        "datasets": [
            {
                "name": "imdb",
                "domain": "movies",
                "train_size": len(imdb_data["train"]),
                "test_size": len(imdb_data["test"]),
                "labels": ["negative", "positive"]
            },
            {
                "name": "amazon_polarity",
                "domain": "products",
                "train_size": len(amazon_data["train"]),
                "test_size": len(amazon_data["test"]),
                "labels": ["negative", "positive"]
            },
            {
                "name": "tweet_sentiment",
                "domain": "social_media",
                "train_size": len(tweet_data["train"]),
                "test_size": len(tweet_data["test"]),
                "labels": ["negative", "positive"]
            }
        ]
    }

    Path("datasets/summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Dataset download complete! Summary saved to datasets/summary.json")

if __name__ == "__main__":
    main()
