#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "pathlib",
# ]
# ///

"""Load standardized datasets and convert to exp_sel_data_out.json schema."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def main():
    """Load datasets and create standardized output."""
    logger.info("Loading standardized datasets...")

    datasets_output = []

    # Load each dataset with reduced sample sizes to keep total under 100MB
    dataset_configs = [
        ('imdb', 10000, 5000),  # 10k train, 5k test
        ('amazon_polarity', 15000, 5000),  # 15k train, 5k test
        ('tweet_sentiment', 8000, 3000)  # 8k train, 3k test
    ]

    for name, train_size, test_size in dataset_configs:
        # Use balanced version for fair evaluation
        filepath = Path(f"datasets/{name}/full_{name}_balanced.json")

        if not filepath.exists():
            logger.warning(f"Balanced version not found, using regular: {filepath}")
            filepath = Path(f"datasets/{name}/full_{name}.json")

        data = json.loads(filepath.read_text())
        logger.info(f"Loaded {name}: {len(data['train'])} train, {len(data['test'])} test")

        # Convert to exp_sel_data_out schema
        examples = []

        # Add train examples (sample if needed)
        train_examples = data['train'][:train_size]
        for i, ex in enumerate(train_examples):
            examples.append({
                "input": ex["text"],
                "output": ex["label"],
                "metadata_split": "train",
                "metadata_row_index": i
            })

        # Add test examples (sample if needed)
        test_examples = data['test'][:test_size]
        for i, ex in enumerate(test_examples):
            examples.append({
                "input": ex["text"],
                "output": ex["label"],
                "metadata_split": "test",
                "metadata_row_index": i
            })

        datasets_output.append({
            "dataset": name,
            "examples": examples
        })

    # Create output in exp_sel_data_out format
    output = {"datasets": datasets_output}

    # Save full output
    output_path = Path("full_data_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(datasets_output)} datasets to {output_path}")

    # Print summary
    for ds in datasets_output:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")

if __name__ == "__main__":
    main()
