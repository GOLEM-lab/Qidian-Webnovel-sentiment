#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for fine-tuned XLM-RoBERTa model on golden labels
Evaluates the model and provides language-specific statistics (EN and ZH)
"""

import sys
import os
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score
import torch
from torch.utils.data import DataLoader
import json

# Configuration
MODEL_PATH = '/scratch/p315327/finetuned_xlm_roberta_base/model'
GOLDEN_LABEL_FILE = '/scratch/p315327/finetuned_xlm_roberta_base/data/annotator_goldenlabel.csv'
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_LABELS = 3  # 0=negative, 1=neutral, 2=positive
REVERSE_LABEL_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Load tokenizer and model
print("="*60, flush=True)
print("Loading tokenizer and model...", flush=True)
print(f"Model path: {MODEL_PATH}", flush=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Tokenizer loaded", flush=True)
except Exception as e:
    print(f"ERROR loading tokenizer: {e}", flush=True)
    raise

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS
    )
    print("✓ Model loaded", flush=True)
except Exception as e:
    print(f"ERROR loading model: {e}", flush=True)
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✓ Model moved to device: {device}", flush=True)
print("="*60, flush=True)

# Load golden label file
print("\n" + "="*60, flush=True)
print("Loading golden label evaluation file...", flush=True)
print(f"File: {GOLDEN_LABEL_FILE}", flush=True)

if not os.path.exists(GOLDEN_LABEL_FILE):
    raise FileNotFoundError(
        f"Golden label file not found: {GOLDEN_LABEL_FILE}\n"
        f"Please ensure the file exists with columns: 'text', 'label', 'language'"
    )

try:
    golden_df = pd.read_csv(GOLDEN_LABEL_FILE, sep=None, engine='python', encoding='utf-8')
    print(f"✓ Loaded {len(golden_df)} samples", flush=True)
except Exception as e:
    print(f"Error loading golden label file: {e}", flush=True)
    # Try with different encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            golden_df = pd.read_csv(GOLDEN_LABEL_FILE, sep=None, engine='python', encoding=encoding)
            print(f"✓ Loaded {len(golden_df)} samples with encoding '{encoding}'", flush=True)
            break
        except:
            continue
    else:
        raise ValueError(f"Could not read golden label file with any encoding: {GOLDEN_LABEL_FILE}")

# Verify required columns exist
required_cols = ['text', 'label', 'language']
missing_cols = [col for col in required_cols if col not in golden_df.columns]
if missing_cols:
    raise ValueError(
        f"Golden label file must have columns: {required_cols}. "
        f"Missing: {missing_cols}. Found: {golden_df.columns.tolist()}"
    )

# Clean the golden label data
initial_len = len(golden_df)
golden_df = golden_df.dropna(subset=['text', 'label', 'language'])
if len(golden_df) < initial_len:
    print(f"Warning: Removed {initial_len - len(golden_df)} rows with missing values", flush=True)

# Verify language values
unique_languages = golden_df['language'].unique()
print(f"Languages found: {unique_languages}", flush=True)
if not all(lang in ['EN', 'ZH'] for lang in unique_languages):
    print(f"Warning: Expected languages 'EN' and 'ZH', but found: {unique_languages}", flush=True)

# Convert labels to int (handle both string and integer labels)
print(f"\nUnique label values found: {golden_df['label'].unique()}", flush=True)
try:
    # Try to convert directly - if it fails, labels are strings
    test_convert = golden_df['label'].astype(int)
    golden_df['label'] = test_convert
    print("Labels are already numeric", flush=True)
except (ValueError, TypeError):
    # Labels are strings, need to map them
    print("Labels are strings, mapping to integers...", flush=True)
    # Map string labels to integers
    label_mapping = {
        'negative': 0, 'Negative': 0, 'NEGATIVE': 0,
        'neutral': 1, 'Neutral': 1, 'NEUTRAL': 1,
        'positive': 2, 'Positive': 2, 'POSITIVE': 2
    }
    # Map labels
    golden_df['label'] = golden_df['label'].map(label_mapping)
    # Check for unmapped labels
    if golden_df['label'].isna().any():
        unmapped = golden_df[golden_df['label'].isna()]['label'].unique()
        raise ValueError(f"Could not map some labels to integers: {unmapped}")
    golden_df['label'] = golden_df['label'].astype(int)
    print(f"Applied label mapping. New unique values: {golden_df['label'].unique()}", flush=True)

# Verify labels are in expected range
if not all(golden_df['label'].isin([0, 1, 2])):
    raise ValueError(f"Labels must be 0, 1, or 2. Found: {golden_df['label'].unique()}")

# Print statistics
print(f"\nGolden label distribution (overall): {golden_df['label'].value_counts().sort_index().to_dict()}", flush=True)
for lang in ['EN', 'ZH']:
    lang_df = golden_df[golden_df['language'] == lang]
    if len(lang_df) > 0:
        print(f"Golden label distribution ({lang}): {lang_df['label'].value_counts().sort_index().to_dict()}", flush=True)
        print(f"  {lang} samples: {len(lang_df)}", flush=True)
print("="*60, flush=True)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

# Prepare dataset
print("\nPreparing dataset for evaluation...", flush=True)
golden_dataset = Dataset.from_pandas(golden_df[['text', 'label', 'language']])
golden_dataset = golden_dataset.map(tokenize_function, batched=True)
golden_dataset = golden_dataset.rename_column("label", "labels")
golden_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
print("✓ Dataset prepared", flush=True)

# Evaluation function
def evaluate_dataset(dataset, model, tokenizer, device, batch_size=BATCH_SIZE):
    """Evaluate model on a dataset and return predictions and labels"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    return all_predictions, all_labels

def compute_metrics(predictions, labels):
    """Compute evaluation metrics including per-class metrics"""
    # Overall metrics
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'macro_f1': f1_score(labels, predictions, average='macro'),
        'macro_recall': recall_score(labels, predictions, average='macro'),
        'macro_precision': precision_score(labels, predictions, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(labels, predictions)
    }
    
    # Per-class metrics
    per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, label_name in enumerate(['negative', 'neutral', 'positive']):
        metrics['per_class'][label_name] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int(np.sum(labels == i))
        }
    
    return metrics

def print_per_class_metrics(metrics, indent="  "):
    """Print per-class metrics in a readable format"""
    if 'per_class' not in metrics:
        return
    
    print(f"{indent}Per-Class Metrics:", flush=True)
    for label_name in ['negative', 'neutral', 'positive']:
        if label_name in metrics['per_class']:
            class_metrics = metrics['per_class'][label_name]
            print(f"{indent}  {label_name.capitalize()} (class {['negative', 'neutral', 'positive'].index(label_name)}):", flush=True)
            print(f"{indent}    Precision: {class_metrics['precision']:.4f}", flush=True)
            print(f"{indent}    Recall:    {class_metrics['recall']:.4f}", flush=True)
            print(f"{indent}    F1-Score:  {class_metrics['f1']:.4f}", flush=True)
            print(f"{indent}    Support:   {class_metrics['support']}", flush=True)

# Evaluate on full golden label dataset
print("\n" + "="*60, flush=True)
print("GOLDEN LABEL EVALUATION RESULTS", flush=True)
print("="*60, flush=True)

print("\nEvaluating on full dataset...", flush=True)
all_predictions, all_labels = evaluate_dataset(golden_dataset, model, tokenizer, device)
golden_results = compute_metrics(all_predictions, all_labels)
golden_results['num_samples'] = len(golden_df)

print("\nOverall Golden Label Metrics:", flush=True)
for key, value in golden_results.items():
    if key == 'per_class':
        continue
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}", flush=True)
    else:
        print(f"  {key}: {value}", flush=True)

# Print per-class metrics
print_per_class_metrics(golden_results)

# Compute language-specific metrics
print("\n" + "-"*60, flush=True)
print("Language-Specific Evaluation", flush=True)
print("-"*60, flush=True)

language_results = {}
for lang in ['EN', 'ZH']:
    lang_df = golden_df[golden_df['language'] == lang].copy()
    if len(lang_df) == 0:
        print(f"\n{lang} Language: No samples found", flush=True)
        language_results[lang] = None
        continue
    
    print(f"\n{lang} Language Results:", flush=True)
    print(f"  Number of samples: {len(lang_df)}", flush=True)
    
    # Create separate dataset for this language
    lang_dataset = Dataset.from_pandas(lang_df[['text', 'label']])
    lang_dataset = lang_dataset.map(tokenize_function, batched=True)
    lang_dataset = lang_dataset.rename_column("label", "labels")
    lang_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Evaluate on this language subset
    lang_predictions, lang_labels = evaluate_dataset(lang_dataset, model, tokenizer, device)
    
    # Compute metrics
    lang_metrics = compute_metrics(lang_predictions, lang_labels)
    lang_metrics['num_samples'] = len(lang_df)
    language_results[lang] = lang_metrics
    
    for key, value in lang_metrics.items():
        if key == 'per_class':
            continue
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)
    
    # Print per-class metrics
    print_per_class_metrics(lang_metrics)

print("\n" + "="*60, flush=True)

# Save evaluation results to a file
results_summary = {
    'golden_label_overall': golden_results,
    'golden_label_by_language': language_results,
    'model_path': MODEL_PATH,
    'num_labels': NUM_LABELS,
    'golden_label_samples': len(golden_df),
    'golden_label_samples_by_language': {
        lang: len(golden_df[golden_df['language'] == lang]) 
        for lang in ['EN', 'ZH']
    }
}

results_file = './evaluation_results.json'
os.makedirs(os.path.dirname(results_file) if os.path.dirname(results_file) else '.', exist_ok=True)
try:
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n{'='*60}", flush=True)
    print(f"Evaluation results saved to: {results_file}", flush=True)
    print(f"{'='*60}", flush=True)
except Exception as e:
    print(f"Warning: Could not save results to file: {e}", flush=True)

print("\nEvaluation completed!", flush=True)

