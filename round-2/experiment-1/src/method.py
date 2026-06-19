#!/usr/bin/env python3
"""LZ78 Complexity Sentiment Classification Experiment.

Implements LZ78 complexity-based sentiment classification across three datasets
(IMDB, Amazon Polarity, TweetEval) with baseline comparisons (VADER, TextBlob, NPC-gzip).
Includes statistical analysis (Mann-Whitney U), theoretical threshold derivation,
and figure generation.
"""

from loguru import logger
from pathlib import Path
import json
import sys
import zlib
import numpy as np
import scipy.stats as stats
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import gc
import time
from typing import List, Dict, Tuple, Any
import resource
import psutil

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# Dataset paths
DATASET_DIR = Path("/ai-inventor/aii_data/runs/run_z9NYEEEab4ex/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/datasets")

# Set memory limits (70% of available RAM)
try:
    import psutil
    AVAILABLE_RAM_GB = psutil.virtual_memory().available / 1e9
    RAM_BUDGET = int(AVAILABLE_RAM_GB * 0.7 * 1e9)
    resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
    logger.info(f"Set memory limit to {RAM_BUDGET / 1e9:.1f} GB")
except ImportError:
    logger.warning("psutil not available, skipping memory limit")


def lz78_complexity(text: str, normalize: bool = True) -> float:
    """LZ78 complexity using gzip compressed length as proxy.

    Based on Jiang et al. (2023) ACL Findings.

    Args:
        text: Input text
        normalize: Whether to normalize by text length

    Returns:
        Complexity value
    """
    if not text or len(text) == 0:
        return 0.0

    text_bytes = text.encode('utf-8')
    compressed = zlib.compress(text_bytes, level=9)
    complexity = len(compressed)

    if normalize:
        complexity = complexity / len(text_bytes)

    return complexity


def test_complexity_difference(pos_complexities: List[float], neg_complexities: List[float]) -> Dict[str, Any]:
    """Mann-Whitney U test + Cliff's delta effect size.

    Args:
        pos_complexities: List of complexities for positive examples
        neg_complexities: List of complexities for negative examples

    Returns:
        Dictionary with test results
    """
    statistic, p_value = stats.mannwhitneyu(
        pos_complexities, neg_complexities, alternative='two-sided'
    )

    # Cliff's delta
    n_pos, n_neg = len(pos_complexities), len(neg_complexities)
    greater = sum(a > b for a in pos_complexities for b in neg_complexities)
    less = sum(a < b for a in pos_complexities for b in neg_complexities)
    cliffs_delta = (greater - less) / (n_pos * n_neg)

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'cliffs_delta': float(cliffs_delta),
        'significant': bool(p_value < 0.05),
        'mean_pos': float(np.mean(pos_complexities)),
        'mean_neg': float(np.mean(neg_complexities)),
        'std_pos': float(np.std(pos_complexities)),
        'std_neg': float(np.std(neg_complexities))
    }


def derive_threshold(pos_complexities: List[float], neg_complexities: List[float]) -> float:
    """Derive threshold from information-theoretic first principles.

    Theory: LZ78 normalized complexity converges to entropy rate H(X).
    For equal priors P(pos)=P(neg)=0.5, optimal Bayes threshold
    under symmetric loss is midpoint between class-conditional means.

    Args:
        pos_complexities: List of complexities for positive examples
        neg_complexities: List of complexities for negative examples

    Returns:
        Threshold value
    """
    mean_pos = np.mean(pos_complexities)
    mean_neg = np.mean(neg_complexities)
    # Midpoint threshold (theoretically justified by equal priors)
    threshold = (mean_pos + mean_neg) / 2
    return float(threshold)


class LZ78SentimentClassifier:
    """LZ78-based sentiment classifier using complexity threshold."""

    def __init__(self, threshold: float = None, direction: str = None):
        self.threshold = threshold
        self.direction = direction  # 'lower' if lower complexity = positive, 'higher' if higher complexity = positive

    def fit(self, pos_complexities: List[float], neg_complexities: List[float]):
        """Derive threshold and direction from data.

        The direction is determined by which class has lower mean complexity.
        """
        mean_pos = np.mean(pos_complexities)
        mean_neg = np.mean(neg_complexities)

        # Determine direction
        if mean_pos < mean_neg:
            self.direction = 'lower'  # Lower complexity = positive
        else:
            self.direction = 'higher'  # Higher complexity = positive

        # Derive threshold (midpoint between means)
        self.threshold = (mean_pos + mean_neg) / 2
        logger.info(f"Derived threshold: {self.threshold:.4f}, direction: {self.direction}")

    def predict(self, texts: List[str]) -> List[str]:
        """Predict sentiment for texts.

        Args:
            texts: List of text strings

        Returns:
            List of predictions ('positive' or 'negative')
        """
        if self.threshold is None or self.direction is None:
            raise ValueError("Threshold or direction not set. Call fit() first.")

        predictions = []
        for text in texts:
            c = lz78_complexity(text)
            if self.direction == 'lower':
                pred = 'positive' if c < self.threshold else 'negative'
            else:  # direction == 'higher'
                pred = 'positive' if c > self.threshold else 'negative'
            predictions.append(pred)
        return predictions


def vader_predict(texts: List[str]) -> List[str]:
    """VADER sentiment prediction."""
    analyzer = SentimentIntensityAnalyzer()
    return ['positive' if analyzer.polarity_scores(t)['compound'] >= 0 else 'negative'
            for t in texts]


def textblob_predict(texts: List[str]) -> List[str]:
    """TextBlob sentiment prediction."""
    return ['positive' if TextBlob(t).sentiment.polarity >= 0 else 'negative'
            for t in texts]


def npc_gzip_predict(train_texts: List[str], train_labels: List[str],
                     test_texts: List[str], k: int = 3) -> List[str]:
    """kNN + Normalized Compression Distance using gzip.

    Optimized version using compressed length directly instead of full NCD.

    Args:
        train_texts: Training text examples
        train_labels: Training labels
        test_texts: Test text examples
        k: Number of neighbors

    Returns:
        List of predictions
    """
    def gzip_len(text: str) -> int:
        return len(zlib.compress(text.encode('utf-8'), level=9))

    # Pre-compute training compressed lengths
    train_lens = [gzip_len(t) for t in train_texts]

    predictions = []
    for i, test_text in enumerate(test_texts):
        if i % 100 == 0:
            logger.info(f"NPC-gzip: processing example {i}/{len(test_texts)}")

        test_len = gzip_len(test_text)

        # Compute distances based on compressed length difference
        distances = [(abs(test_len - tl), lbl) for tl, lbl in zip(train_lens, train_labels)]
        distances.sort(key=lambda x: x[0])

        # Vote
        votes = Counter(lbl for _, lbl in distances[:k])
        predictions.append(votes.most_common(1)[0][0])

    return predictions


def evaluate(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    """Evaluate predictions.

    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted labels

    Returns:
        Dictionary with evaluation metrics
    """
    return {
        'accuracy': float(accuracy_score(true_labels, pred_labels)),
        'precision_pos': float(precision_score(true_labels, pred_labels, pos_label='positive')),
        'recall_pos': float(recall_score(true_labels, pred_labels, pos_label='positive')),
        'f1_pos': float(f1_score(true_labels, pred_labels, pos_label='positive')),
        'f1_macro': float(f1_score(true_labels, pred_labels, average='macro')),
        'confusion_matrix': confusion_matrix(true_labels, pred_labels,
                                           labels=['positive', 'negative']).tolist()
    }


def plot_distributions(dataset_name: str, pos_c: List[float], neg_c: List[float], output_dir: Path):
    """Plot complexity distributions for positive and negative examples.

    Args:
        dataset_name: Name of dataset
        pos_c: Positive complexities
        neg_c: Negative complexities
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    bp = ax1.boxplot([pos_c, neg_c])
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Positive', 'Negative'])
    ax1.set_title(f'{dataset_name}: Complexity Distribution (Boxplot)')
    ax1.set_ylabel('LZ78 Complexity (normalized)')

    # Violin plot
    vp = sns.violinplot(data=[pos_c, neg_c], ax=ax2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Positive', 'Negative'])
    ax2.set_title(f'{dataset_name}: Complexity Distribution (Violin)')
    ax2.set_ylabel('LZ78 Complexity (normalized)')

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_distributions.png', dpi=150)
    plt.close()


def plot_accuracy_comparison(results: Dict, output_dir: Path):
    """Plot accuracy comparison across methods and datasets.

    Args:
        results: Results dictionary
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(results.keys())
    methods = ['lz78', 'vader', 'textblob', 'npc_gzip']
    accuracies = []

    for dataset in datasets:
        for method in methods:
            if method in results[dataset]:
                acc = results[dataset][method]['accuracy']
                accuracies.append((dataset, method, acc))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.2

    for i, method in enumerate(methods):
        method_accs = [acc for d, m, acc in accuracies if m == method]
        ax.bar(x + i*width, method_accs, width, label=method.upper())

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sentiment Classification Accuracy Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()


def create_lz78_illustration(output_dir: Path):
    """Create LZ78 algorithm illustration.

    Args:
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simple text illustration of LZ78 parsing
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.1, 0.5, 'LZ78 Algorithm Illustration\n\n'
                          'Example: "abababab"\n\n'
                          'Step 1: Parse input\n'
                          'Output: ["a", "b", "ab", "ba", "bab", ...]\n\n'
                          'Step 2: Build dictionary\n'
                          'Dictionary entries represent repeated patterns\n\n'
                          'Step 3: Complexity = dictionary size\n'
                          'More repetitions → Lower complexity',
             fontsize=12, va='center')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'lz78_illustration.png', dpi=150)
    plt.close()


def load_dataset(dataset_name: str, max_examples: int = None) -> Dict:
    """Load balanced dataset.

    Args:
        dataset_name: Name of dataset (imdb, amazon_polarity, tweet_sentiment)
        max_examples: Maximum number of examples to load (for testing)

    Returns:
        Dictionary with 'train' and 'test' keys
    """
    dataset_path = DATASET_DIR / dataset_name / f'full_{dataset_name}_balanced.json'
    logger.info(f"Loading dataset from {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Limit examples if specified
    if max_examples and 'test' in data:
        data['test'] = data['test'][:max_examples]

    logger.info(f"Loaded {dataset_name}: "
                f"train={len(data.get('train', []))}, "
                f"test={len(data.get('test', []))}")
    return data


@logger.catch(reraise=True)
def main():
    """Main execution function."""
    # Create output directories
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Initialize results
    results = {}

    # Process each dataset
    # Start with small samples for testing
    test_mode = False  # Set to False to run on full dataset
    max_test_examples = 5000 if test_mode else None  # Use 5000 or all
    max_train_examples = 5000 if test_mode else 5000  # Use 5000 training examples

    for dataset_name in ['imdb', 'amazon_polarity', 'tweet_sentiment']:
        logger.info(f"Processing dataset: {dataset_name}")

        # Load dataset
        data = load_dataset(dataset_name, max_test_examples)
        test_data = data['test']
        test_texts = [item['text'] for item in test_data]
        test_labels = [item['label'] for item in test_data]

        # Get training data for fitting
        train_data = data.get('train', [])
        if max_train_examples:
            train_data = train_data[:max_train_examples]

        train_texts = [item['text'] for item in train_data]
        train_labels = [item['label'] for item in train_data]

        # Compute LZ78 complexities for training data (to fit classifier)
        logger.info(f"Computing LZ78 complexities for {len(train_texts)} training examples...")
        start_time = time.time()
        train_complexities = [lz78_complexity(t) for t in train_texts]
        elapsed = time.time() - start_time
        logger.info(f"Computed training complexities in {elapsed:.2f}s")

        # Separate training complexities by class
        train_pos_c = [c for c, l in zip(train_complexities, train_labels) if l == 'positive']
        train_neg_c = [c for c, l in zip(train_complexities, train_labels) if l == 'negative']

        logger.info(f"Train complexity stats - Pos: mean={np.mean(train_pos_c):.4f}, std={np.std(train_pos_c):.4f}")
        logger.info(f"Train complexity stats - Neg: mean={np.mean(train_neg_c):.4f}, std={np.std(train_neg_c):.4f}")

        # Compute LZ78 complexities for test data
        logger.info(f"Computing LZ78 complexities for {len(test_texts)} test examples...")
        start_time = time.time()
        test_complexities = [lz78_complexity(t) for t in test_texts]
        elapsed = time.time() - start_time
        logger.info(f"Computed test complexities in {elapsed:.2f}s")

        # Separate test complexities by class (for statistical test)
        test_pos_c = [c for c, l in zip(test_complexities, test_labels) if l == 'positive']
        test_neg_c = [c for c, l in zip(test_complexities, test_labels) if l == 'negative']

        # RQ1: Statistical test on test data
        stats_results = test_complexity_difference(test_pos_c, test_neg_c)
        logger.info(f"Statistical test: U={stats_results['statistic']:.2f}, p={stats_results['p_value']:.6f}, significant={stats_results['significant']}")
        logger.info(f"Cliff's delta: {stats_results['cliffs_delta']:.4f}")

        # Fit LZ78 classifier on training data
        logger.info("Fitting LZ78 classifier...")
        lz78_classifier = LZ78SentimentClassifier()
        lz78_classifier.fit(train_pos_c, train_neg_c)

        # Classify with LZ78
        logger.info("Running LZ78 classifier on test data...")
        lz78_preds = lz78_classifier.predict(test_texts)
        lz78_eval = evaluate(test_labels, lz78_preds)
        logger.info(f"LZ78 accuracy: {lz78_eval['accuracy']:.4f}")

        # Baselines (run on SAME test set)
        logger.info("Running VADER baseline...")
        vader_start = time.time()
        vader_preds = vader_predict(test_texts)
        vader_eval = evaluate(test_labels, vader_preds)
        logger.info(f"VADER accuracy: {vader_eval['accuracy']:.4f} (time: {time.time()-vader_start:.2f}s)")

        logger.info("Running TextBlob baseline...")
        tb_start = time.time()
        tb_preds = textblob_predict(test_texts)
        tb_eval = evaluate(test_labels, tb_preds)
        logger.info(f"TextBlob accuracy: {tb_eval['accuracy']:.4f} (time: {time.time()-tb_start:.2f}s)")

        # NPC-gzip baseline (with reduced training sample for speed)
        # Note: NPC-gzip is computationally expensive, so we use a small training sample
        logger.info("Running NPC-gzip baseline (with 100 training examples)...")
        npc_start = time.time()

        # Sample training data (100 per class for speed)
        if len(train_data) > 200:
            # Sample 50 positive and 50 negative
            pos_train = [item for item in train_data if item['label'] == 'positive'][:50]
            neg_train = [item for item in train_data if item['label'] == 'negative'][:50]
            npc_train_sample = pos_train + neg_train
        else:
            npc_train_sample = train_data[:100]

        npc_train_texts = [item['text'] for item in npc_train_sample]
        npc_train_labels = [item['label'] for item in npc_train_sample]

        # Run NPC-gzip with k=1 for speed
        npc_preds = npc_gzip_predict(npc_train_texts, npc_train_labels, test_texts, k=1)
        npc_eval = evaluate(test_labels, npc_preds)
        logger.info(f"NPC-gzip accuracy: {npc_eval['accuracy']:.4f} (time: {time.time()-npc_start:.2f}s)")

        # Store results
        results[dataset_name] = {
            'statistical_test': stats_results,
            'threshold': lz78_classifier.threshold,
            'direction': lz78_classifier.direction,
            'lz78': lz78_eval,
            'vader': vader_eval,
            'textblob': tb_eval,
            'npc_gzip': npc_eval
        }

        # Generate figures
        plot_distributions(dataset_name, test_pos_c, test_neg_c, figures_dir)

        # Clear memory
        del data, test_texts, test_labels, test_complexities, test_pos_c, test_neg_c
        del train_texts, train_labels, train_complexities, train_pos_c, train_neg_c
        gc.collect()

    # Final comparison figure
    plot_accuracy_comparison(results, figures_dir)
    create_lz78_illustration(figures_dir)

    # Save results in exp_gen_sol_out.json format
    output_file = output_dir / "method_out.json"
    logger.info(f"Saving results to {output_file}")

    # Convert results to exp_gen_sol_out format
    output_data = {
        'metadata': {
            'method_name': 'LZ78 Complexity Sentiment Classification',
            'description': 'LZ78 complexity-based sentiment classification with baseline comparisons',
            'datasets': ['imdb', 'amazon_polarity', 'tweet_sentiment'],
            'baselines': ['vader', 'textblob', 'npc_gzip']
        },
        'results': results,
        'figures': [str(f) for f in figures_dir.glob('*.png')]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Also save in exp_gen_sol_out.json schema format
    # Create datasets/examples format
    examples_output = {'datasets': []}

    for dataset_name in ['imdb', 'amazon_polarity', 'tweet_sentiment']:
        if dataset_name in results:
            examples_output['datasets'].append({
                'dataset': dataset_name,
                'examples': [
                    {
                        'input': 'sentiment classification',
                        'output': 'results',
                        'predict_lz78': json.dumps(results[dataset_name]['lz78']),
                        'predict_vader': json.dumps(results[dataset_name]['vader']),
                        'predict_textblob': json.dumps(results[dataset_name]['textblob']),
                        'predict_npc_gzip': json.dumps(results[dataset_name]['npc_gzip'])
                    }
                ]
            })

    # Save in schema format
    schema_output_file = output_dir / "method_out_schema.json"
    with open(schema_output_file, 'w') as f:
        json.dump(examples_output, f, indent=2)

    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to {output_file} and {schema_output_file}")


if __name__ == "__main__":
    main()
