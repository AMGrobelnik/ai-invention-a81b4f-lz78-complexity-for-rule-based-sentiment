#!/usr/bin/env python3
"""Convert method_out.json to exp_gen_sol_out.json schema format."""
import json
from pathlib import Path

# Load current results
with open('output/method_out.json') as f:
    results = json.load(f)

# Convert to exp_gen_sol_out schema format
output_data = {
    'datasets': []
}

# For each dataset, create examples with predictions
for dataset_name in ['imdb', 'amazon_polarity', 'tweet_sentiment']:
    if dataset_name in results['results']:
        r = results['results'][dataset_name]
        
        # Create a summary example for each dataset
        example = {
            'input': f'Sentiment classification on {dataset_name}',
            'output': 'Experiment results',
            'metadata_dataset': dataset_name,
            'metadata_test_size': str(sum(sum(r["lz78"]["confusion_matrix"][i]) for i in range(2))),
            'metadata_p_value': str(r['statistical_test']['p_value']),
            'metadata_significant': str(r['statistical_test']['significant']),
            'metadata_cliffs_delta': str(r['statistical_test']['cliffs_delta']),
            'metadata_threshold': str(r['threshold']),
            'metadata_direction': r['direction'],
            'predict_lz78_accuracy': str(r['lz78']['accuracy']),
            'predict_lz78_f1_macro': str(r['lz78']['f1_macro']),
            'predict_vader_accuracy': str(r['vader']['accuracy']),
            'predict_vader_f1_macro': str(r['vader']['f1_macro']),
            'predict_textblob_accuracy': str(r['textblob']['accuracy']),
            'predict_textblob_f1_macro': str(r['textblob']['f1_macro']),
            'predict_npc_gzip_accuracy': str(r['npc_gzip']['accuracy']),
            'predict_npc_gzip_f1_macro': str(r['npc_gzip']['f1_macro'])
        }
        
        output_data['datasets'].append({
            'dataset': dataset_name,
            'examples': [example]
        })

# Save in schema format
output_file = Path('output/method_out_schema_formatted.json')
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Saved formatted output to {output_file}")
