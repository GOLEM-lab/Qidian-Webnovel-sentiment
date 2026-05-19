#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentiment prediction script using fine-tuned XLM-RoBERTa model§
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from tqdm import tqdm
import os
import sys
import logging
from pathlib import Path

# Configuration
MODEL_PATH = './model'  # Relative path to model folder
MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32  # Adjust based on your memory

# Label mapping (same as training)
LABEL_MAPPING = {0: 'negative', 1: 'neutral', 2: 'positive'}

def setup_logging(log_file=None):
    """Setup logging to both console and file"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )
    return logging.getLogger(__name__)

def load_model(model_path, logger):
    """Load the fine-tuned model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    model_path_abs = os.path.abspath(model_path)
    logger.info(f"Model path: {model_path_abs}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path_abs}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("Using CPU")
    
    logger.info(f"Model loaded successfully on device: {device}")
    return model, tokenizer, device

def predict_batch(texts, model, tokenizer, device, max_length=MAX_LENGTH):
    """Predict sentiment for a batch of texts"""
    # Convert to strings and handle None values
    texts = [str(text) if text is not None else "" for text in texts]
    
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)
    
    # Predict
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            probabilities = torch.softmax(outputs.logits, dim=1)
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(f"GPU out of memory. Try reducing batch size. Error: {e}")
            raise
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()

def read_csv_robust(input_file, logger):
    """Read CSV file with multiple fallback strategies"""
    logger.info(f"Reading input file: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Try different encodings and separators
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    separators = [None, ',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                if sep is None:
                    df = pd.read_csv(input_file, sep=None, engine='python', encoding=encoding)
                else:
                    df = pd.read_csv(input_file, sep=sep, encoding=encoding)
                logger.info(f"Successfully read CSV with encoding={encoding}, sep={sep}")
                return df
            except Exception as e:
                continue
    
    raise ValueError(f"Could not read CSV file with any encoding/separator combination: {input_file}")

def predict_sentiment(input_file, output_file, text_column='text', model_path=MODEL_PATH, 
                     batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_LENGTH, log_file=None):
    """
    Predict sentiment for all texts in a CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save predictions
        text_column: Name of the column containing text
        model_path: Path to the model directory
        batch_size: Batch size for prediction
        max_length: Maximum sequence length
        log_file: Optional path to log file
    """
    # Setup logging
    logger = setup_logging(log_file)
    
    try:
        # Load model
        model, tokenizer, device = load_model(model_path, logger)
        
        # Read input data
        df = read_csv_robust(input_file, logger)
        
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check if text column exists (case-insensitive)
        text_col_lower = {col.lower(): col for col in df.columns}
        if text_column.lower() in text_col_lower:
            text_column = text_col_lower[text_column.lower()]
            logger.info(f"Using text column: '{text_column}'")
        elif text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}\n"
                f"Please specify the correct column name using --text_column argument."
            )
        
        # Remove rows with missing text
        initial_len = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].astype(str).str.strip() != '']
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with missing or empty text")
        
        if len(df) == 0:
            raise ValueError("No valid text data found in the input file")
        
        texts = df[text_column].astype(str).tolist()
        
        # Predict in batches
        logger.info(f"Predicting sentiment for {len(texts)} texts (batch size: {batch_size})...")
        all_predictions = []
        all_probabilities = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]
            try:
                predictions, probabilities = predict_batch(batch_texts, model, tokenizer, device, max_length)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"GPU out of memory at batch {i//batch_size + 1}. "
                               f"Try reducing batch size (current: {batch_size})")
                raise
        
        # Add predictions to dataframe
        df['predicted_label'] = all_predictions
        df['predicted_sentiment'] = [LABEL_MAPPING[pred] for pred in all_predictions]
        df['confidence'] = [max(prob) for prob in all_probabilities]
        df['prob_negative'] = [prob[0] for prob in all_probabilities]
        df['prob_neutral'] = [prob[1] for prob in all_probabilities]
        df['prob_positive'] = [prob[2] for prob in all_probabilities]
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Save results
        logger.info(f"Saving predictions to: {output_file}")
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total predictions: {len(df)}")
        logger.info("\nSentiment distribution:")
        sentiment_counts = df['predicted_sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\nAverage confidence: {df['confidence'].mean():.4f}")
        logger.info(f"Min confidence: {df['confidence'].min():.4f}")
        logger.info(f"Max confidence: {df['confidence'].max():.4f}")
        
        logger.info("\nDone! ✓")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict sentiment using fine-tuned XLM-RoBERTa model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_sentiment.py -i data.csv -o results.csv
  python predict_sentiment.py -i data.csv -o results.csv --text_column review --model_path ./my_model
  python predict_sentiment.py -i data.csv -o results.csv --batch_size 16 --log_file run.log

CSV Format:
  The input CSV file must contain a column with text data. By default, the script
  looks for a column named 'text', but you can specify a different column name.
  
  Required column:
    - text (or custom name specified with --text_column)
  
  Optional columns (will be preserved in output):
    - Any other columns you want to keep in the output
  
  Output columns added:
    - predicted_label: Numeric label (0=negative, 1=neutral, 2=positive)
    - predicted_sentiment: Text label (negative/neutral/positive)
    - confidence: Confidence score (0-1)
    - prob_negative: Probability of negative sentiment
    - prob_neutral: Probability of neutral sentiment
    - prob_positive: Probability of positive sentiment
        """
    )
    parser.add_argument('--input', '-i', required=True, 
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', required=True, 
                       help='Output CSV file path')
    parser.add_argument('--text_column', '-t', default='text', 
                       help='Name of text column in CSV (default: text)')
    parser.add_argument('--model_path', '-m', default=MODEL_PATH,
                       help=f'Path to model directory (default: {MODEL_PATH})')
    parser.add_argument('--batch_size', '-b', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for prediction (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH,
                       help=f'Maximum sequence length (default: {MAX_LENGTH})')
    parser.add_argument('--log_file', '-l', default=None,
                       help='Optional log file path')
    
    args = parser.parse_args()
    
    predict_sentiment(
        args.input, 
        args.output, 
        args.text_column,
        args.model_path,
        args.batch_size,
        args.max_length,
        args.log_file
    )
