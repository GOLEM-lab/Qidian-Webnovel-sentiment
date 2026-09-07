#!/usr/bin/env python3
"""
Batch sentiment inference for webnovel comments and paragraphs.

Comments  → fpianz/sentiment-webnovel-comments-xlmr-en-zh  (finetuned, maxlen=256)
Paragraphs → cardiffnlp/twitter-xlm-roberta-base-sentiment (zero-shot, maxlen=128)

Input files are the four CSVs staged by stage_september_data.py into
--data_dir: qidian_comments.csv, webnovel_comments.csv, qidian_paragraphs.csv,
webnovel_paragraphs.csv. Each is scored in place.

Adds columns to each CSV: sentiment, valence, p_neg, p_neu, p_pos
  - sentiment: argmax class label (negative / neutral / positive)
  - valence:   p(positive) - p(negative), continuous score in [-1, +1]
  - p_neg/p_neu/p_pos: softmax probabilities

Usage:
  python3 run_inference_webnovel.py --data_dir data/inference/webnovel
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ["negative", "neutral", "positive"]
# tried in order when --text_column is not given, so the staged files (text) and
# the september-data files themselves (content / en_text / zh_text) both work
TEXT_COLUMNS = ["text", "content", "en_text", "zh_text"]
EXPECTED_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class TextDataset(Dataset):
    """Minimal dataset for batched tokenisation."""

    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def collate_fn(self, batch):
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def verify_label_mapping(model, model_name):
    """Check model's id2label matches our assumed order."""
    id2label = model.config.id2label
    normalised = {int(k): v.lower() for k, v in id2label.items()}
    if normalised != EXPECTED_ID2LABEL:
        print(f"ERROR: unexpected label mapping for {model_name}")
        print(f"  Expected: {EXPECTED_ID2LABEL}")
        print(f"  Got:      {normalised}")
        sys.exit(1)
    print(f"  Label mapping verified: {normalised}")


@torch.no_grad()
def run_inference(model, tokenizer, texts, max_length, batch_size, device):
    """Run batched inference, return (N, 3) softmax probabilities."""
    dataset = TextDataset(texts, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )

    all_probs = []
    model.eval()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast():
            logits = model(**batch).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)

        if (i + 1) % 200 == 0:
            done = min((i + 1) * batch_size, len(texts))
            print(f"    {done:,} / {len(texts):,}  ({100 * done / len(texts):.1f}%)",
                  flush=True)

    return np.vstack(all_probs)


def resolve_text_column(df, requested, fname):
    """Pick the column holding the text to score."""
    if requested:
        if requested not in df.columns:
            print(f"ERROR: {fname} has no column '{requested}'"
                  f" (columns: {list(df.columns)})")
            sys.exit(1)
        return requested
    for c in TEXT_COLUMNS:
        if c in df.columns:
            return c
    print(f"ERROR: {fname} has none of {TEXT_COLUMNS} (columns: {list(df.columns)})")
    sys.exit(1)


def process_file(filepath, model, tokenizer, max_length, batch_size, device,
                 text_column=None):
    """Load CSV, run inference, write back with sentiment columns."""
    fname = os.path.basename(filepath)
    print(f"\n  --- {fname} ---")

    # dtype=str / keep_default_na=False keeps every id column exactly as written.
    # Without it pandas turns an integer column holding blanks into float64 and
    # writes ids back as "30.0", undoing the normalisation done upstream.
    df = pd.read_csv(filepath, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    col = resolve_text_column(df, text_column, fname)
    print(f"    Rows: {len(df):,}   text column: {col}")

    texts = df[col].tolist()

    t0 = time.time()
    probs = run_inference(model, tokenizer, texts, max_length, batch_size, device)
    elapsed = time.time() - t0
    rate = len(texts) / elapsed
    print(f"    Done: {elapsed:.1f}s  ({rate:,.0f} texts/s)")

    df["sentiment"] = [LABELS[p] for p in probs.argmax(axis=1)]
    df["valence"] = (probs[:, 2] - probs[:, 0]).round(6)
    df["p_neg"] = probs[:, 0].round(6)
    df["p_neu"] = probs[:, 1].round(6)
    df["p_pos"] = probs[:, 2].round(6)

    # Quick distribution check
    counts = df["sentiment"].value_counts()
    mean_v = df["valence"].mean()
    print(f"    Distribution: {dict(counts)}")
    print(f"    Mean valence: {mean_v:+.4f}")

    df.to_csv(filepath, index=False)
    print(f"    Saved: {filepath}")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Batch sentiment inference for webnovel data"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing the CSV files")
    parser.add_argument("--comment_model",
                        default="fpianz/sentiment-webnovel-comments-xlmr-en-zh",
                        help="HF model for comments (finetuned)")
    parser.add_argument("--paragraph_model",
                        default="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                        help="HF model for paragraphs (zero-shot)")
    parser.add_argument("--comment_maxlen", type=int, default=256)
    parser.add_argument("--paragraph_maxlen", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--text_column", default=None,
                        help="column to score; default: first of "
                             + "/".join(TEXT_COLUMNS) + " present")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    comment_files = [
        os.path.join(args.data_dir, f)
        for f in ["qidian_comments.csv", "webnovel_comments.csv"]
    ]
    paragraph_files = [
        os.path.join(args.data_dir, f)
        for f in ["qidian_paragraphs.csv", "webnovel_paragraphs.csv"]
    ]

    total = 0
    t_start = time.time()

    # ================================================================
    # COMMENTS — finetuned model
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"COMMENTS — {args.comment_model}")
    print(f"  maxlen={args.comment_maxlen}  batch_size={args.batch_size}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(args.comment_model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.comment_model)
    verify_label_mapping(model, args.comment_model)
    model.to(device)

    for fpath in comment_files:
        if os.path.exists(fpath):
            total += process_file(
                fpath, model, tokenizer,
                args.comment_maxlen, args.batch_size, device,
                args.text_column,
            )
        else:
            print(f"  SKIP (not found): {fpath}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================
    # PARAGRAPHS — zero-shot base model
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"PARAGRAPHS — {args.paragraph_model}")
    print(f"  maxlen={args.paragraph_maxlen}  batch_size={args.batch_size}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(args.paragraph_model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.paragraph_model)
    verify_label_mapping(model, args.paragraph_model)
    model.to(device)

    for fpath in paragraph_files:
        if os.path.exists(fpath):
            total += process_file(
                fpath, model, tokenizer,
                args.paragraph_maxlen, args.batch_size, device,
                args.text_column,
            )
        else:
            print(f"  SKIP (not found): {fpath}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — {total:,} texts in {elapsed:.0f}s ({total / elapsed:,.0f} texts/s)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
