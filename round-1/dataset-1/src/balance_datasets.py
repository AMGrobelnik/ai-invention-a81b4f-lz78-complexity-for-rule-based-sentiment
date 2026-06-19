#!/usr/bin/env python3
"""Create balanced versions of datasets for fair evaluation."""

from loguru import logger
from pathlib import Path
import json
import sys
import random

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

@logger.catch(reraise=True)
def balance_dataset(input_path, output_path, seed=42):
    """Balance dataset by subsampling the larger class to match the smaller."""
    data = json.loads(Path(input_path).read_text())

    for split in ["train", "test"]:
        examples = data[split]
        pos_examples = [ex for ex in examples if ex["label"] == "positive"]
        neg_examples = [ex for ex in examples if ex["label"] == "negative"]

        min_size = min(len(pos_examples), len(neg_examples))
        random.seed(seed)
        pos_balanced = random.sample(pos_examples, min_size)
        neg_balanced = random.sample(neg_examples, min_size)

        balanced = pos_balanced + neg_balanced
        random.seed(seed)
        random.shuffle(balanced)
        data[split] = balanced

        logger.info(f"{split}: {len(pos_examples)} positive, {len(neg_examples)} negative -> {min_size*2} balanced")

    Path(output_path).write_text(json.dumps(data, indent=2))
    logger.info(f"Saved balanced dataset to {output_path}")

@logger.catch(reraise=True)
def main():
    # Balance tweet_sentiment dataset (most imbalanced)
    logger.info("Balancing tweet_sentiment dataset...")
    balance_dataset(
        "datasets/tweet_sentiment/full_tweet_sentiment.json",
        "datasets/tweet_sentiment/full_tweet_sentiment_balanced.json"
    )

    # Also create balanced versions of train splits for IMDB and Amazon
    for name in ["imdb", "amazon_polarity"]:
        logger.info(f"Balancing {name} dataset...")
        balance_dataset(
            f"datasets/{name}/full_{name}.json",
            f"datasets/{name}/full_{name}_balanced.json"
        )

if __name__ == "__main__":
    main()
