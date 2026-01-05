#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch sentiment prediction script for qidian reviews
Processes all CSV files in data/qidian_reviews/ folder
Adds sentiment predictions in a new column: predict_result_finetune_xlm_roberta_base

This script imports and reuses functions from predict_sentiment.py
"""

import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# Import functions from predict_sentiment.py
# Assuming predict_sentiment.py is in the same directory
import importlib.util

script_dir = os.path.dirname(os.path.abspath(__file__))
predict_sentiment_path = os.path.join(script_dir, 'predict_sentiment.py')

if not os.path.exists(predict_sentiment_path):
    raise FileNotFoundError(
        f"predict_sentiment.py not found in {script_dir}. "
        f"Please ensure predict_sentiment.py is in the same directory as batch_predict_sentiment.py"
    )

# Load the module from file path
spec = importlib.util.spec_from_file_location("predict_sentiment", predict_sentiment_path)
predict_sentiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_sentiment)

# Import the functions and constants we need
setup_logging = predict_sentiment.setup_logging
load_model = predict_sentiment.load_model
predict_batch = predict_sentiment.predict_batch
read_csv_robust = predict_sentiment.read_csv_robust
LABEL_MAPPING = predict_sentiment.LABEL_MAPPING
MODEL_PATH = predict_sentiment.MODEL_PATH
MAX_LENGTH = predict_sentiment.MAX_LENGTH
DEFAULT_BATCH_SIZE = predict_sentiment.DEFAULT_BATCH_SIZE

# Output column name
OUTPUT_COLUMN = 'refined_predict_result_finetune_xlm_roberta_base'

def process_single_file(csv_file, model, tokenizer, device, text_column='text', 
                        batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_LENGTH, logger=None):
    """
    Process a single CSV file: add sentiment predictions and save back
    
    Args:
        csv_file: Path to CSV file
        model: Loaded model
        tokenizer: Loaded tokenizer
        device: Device (CPU/GPU)
        text_column: Name of text column (default: 'text')
        batch_size: Batch size for prediction
        max_length: Maximum sequence length
        logger: Logger instance
    """
    try:
        # Read CSV file
        df = read_csv_robust(csv_file, logger)
        
        if logger:
            logger.info(f"Processing: {os.path.basename(csv_file)} ({len(df)} rows)")
        
        # Check if text column exists (case-insensitive)
        text_col_lower = {col.lower(): col for col in df.columns}
        if text_column.lower() in text_col_lower:
            text_column_actual = text_col_lower[text_column.lower()]
        elif text_column in df.columns:
            text_column_actual = text_column
        else:
            if logger:
                logger.warning(f"Column '{text_column}' not found in {csv_file}. Available: {df.columns.tolist()}")
            return False
        
        # Check if prediction column already exists
        if OUTPUT_COLUMN in df.columns:
            if logger:
                logger.info(f"Column '{OUTPUT_COLUMN}' already exists. Skipping prediction.")
            return True
        
        # Remove rows with missing text
        initial_len = len(df)
        df = df.dropna(subset=[text_column_actual])
        df = df[df[text_column_actual].astype(str).str.strip() != '']
        if len(df) < initial_len:
            if logger:
                logger.warning(f"Removed {initial_len - len(df)} rows with missing or empty text")
        
        if len(df) == 0:
            if logger:
                logger.warning(f"No valid text data found in {csv_file}")
            # Still add the column with NaN values
            df_full = read_csv_robust(csv_file, logger)
            df_full[OUTPUT_COLUMN] = None
            df_full.to_csv(csv_file, index=False, encoding='utf-8')
            return True
        
        texts = df[text_column_actual].astype(str).tolist()
        
        # Predict in batches
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                predictions, _ = predict_batch(batch_texts, model, tokenizer, device, max_length)
                # Convert numeric predictions to sentiment labels
                sentiment_labels = [LABEL_MAPPING[pred] for pred in predictions]
                all_predictions.extend(sentiment_labels)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if logger:
                        logger.error(f"GPU out of memory at batch {i//batch_size + 1}. "
                                   f"Try reducing batch size (current: {batch_size})")
                raise
        
        # Add predictions to dataframe
        df[OUTPUT_COLUMN] = all_predictions
        
        # If we removed rows, merge back with original dataframe
        if initial_len > len(df):
            # Re-read original to get full dataframe
            df_full = read_csv_robust(csv_file, logger)
            # Map predictions to original rows
            df_full[OUTPUT_COLUMN] = None
            df_full.loc[df.index, OUTPUT_COLUMN] = df[OUTPUT_COLUMN]
            df = df_full
        
        # Save back to the same file
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        if logger:
            logger.info(f"✓ Saved predictions to {os.path.basename(csv_file)}")
            # Print summary
            valid_predictions = df[OUTPUT_COLUMN].dropna()
            if len(valid_predictions) > 0:
                sentiment_counts = valid_predictions.value_counts()
                logger.info(f"  Sentiment distribution: {dict(sentiment_counts)}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing {csv_file}: {e}", exc_info=True)
        return False

def batch_predict_sentiment(data_folder, model_path=MODEL_PATH, text_column='text',
                           batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_LENGTH, log_file=None):
    """
    Batch process all CSV files in the data folder
    
    Args:
        data_folder: Path to folder containing CSV files
        model_path: Path to the model directory
        text_column: Name of the text column (default: 'text')
        batch_size: Batch size for prediction
        max_length: Maximum sequence length
        log_file: Optional path to log file
    """
    # Setup logging
    logger = setup_logging(log_file)
    
    try:
        # Load model once (reuse for all files)
        model, tokenizer, device = load_model(model_path, logger)
        
        # Find all CSV files
        data_path = Path(data_folder)
        if not data_path.exists():
            raise FileNotFoundError(f"Data folder not found: {data_path.absolute()}")
        
        csv_files = list(data_path.glob("*.csv"))
        
        if len(csv_files) == 0:
            logger.warning(f"No CSV files found in {data_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        logger.info("="*60)
        
        # Process each file
        success_count = 0
        failed_files = []
        
        for csv_file in tqdm(csv_files, desc="Processing files"):
            success = process_single_file(
                csv_file, model, tokenizer, device, text_column, 
                batch_size, max_length, logger
            )
            if success:
                success_count += 1
            else:
                failed_files.append(csv_file.name)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files: {len(csv_files)}")
        logger.info(f"Successfully processed: {success_count}")
        logger.info(f"Failed: {len(failed_files)}")
        
        if failed_files:
            logger.warning(f"Failed files: {', '.join(failed_files)}")
        
        logger.info("\nDone! ✓")
        
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Default data folder for qidian reviews
    DEFAULT_DATA_FOLDER = 'data/WebNovel free content'
    
    parser = argparse.ArgumentParser(
        description='Batch predict sentiment for all CSV files in data folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_predict_sentiment.py
  python batch_predict_sentiment.py --data_folder data/qidian_reviews
  python batch_predict_sentiment.py --data_folder data/qidian_reviews --model_path ./model
  python batch_predict_sentiment.py --data_folder data/qidian_reviews --batch_size 16 --log_file batch.log

The script will:
  1. Load the sentiment prediction model once
  2. Process all CSV files (*.csv) in the data folder
  3. Add column 'refined_predict_result_finetune_xlm_roberta_base' with sentiment predictions
  4. Save results back to the same files
        """
    )
    parser.add_argument('--data_folder', '-d', default=DEFAULT_DATA_FOLDER,
                       help=f'Path to folder containing CSV files (default: {DEFAULT_DATA_FOLDER})')
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
    
    batch_predict_sentiment(
        args.data_folder,
        args.model_path,
        args.text_column,
        args.batch_size,
        args.max_length,
        args.log_file
    )

