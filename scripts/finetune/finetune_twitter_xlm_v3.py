#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter search with 5-fold stratified CV for XLM-RoBERTa sentiment classification (v3)

Runs a grid of configurations:
  - Training mode: human-only vs curriculum (synthetic pre-train → human fine-tune)
  - Frozen layers: 0, 3, 6
  - Learning rate: 2e-5, 3e-5, 5e-5
  - Max length: 128, 256

Uses stratified 5-fold CV on ALL human data (stratified by language × label).
Reports mean ± std for each metric across folds.

Run with: python -u finetune_twitter_xlm_v3.py
Override configs via CLI: python -u finetune_twitter_xlm_v3.py --configs 0 1 5
After CV, the best config (by overall macro_f1_mean) is trained once more and saved under
results_cv/best_model_final/ (disable with --no-final-save).

Data requirements:
  - human_annotated.csv: columns 'text', 'label' (0/1/2), 'language' (EN/ZH)
  - syn_data_combined.csv: columns 'text', 'label' (0/1/2)
"""

import sys
import os
import argparse
import json
import traceback
from copy import deepcopy
from itertools import product as iterproduct
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, "reconfigure") else None

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# =============================================================================
# FIXED CONFIGURATION (shared across all experiments)
# =============================================================================
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
NUM_LABELS = 3
REVERSE_LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
HIDDEN_DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
USE_CLASS_WEIGHTS = True
EARLY_STOPPING_PATIENCE = 3
EPOCHS = 10
EPOCHS_PHASE1 = 3  # For curriculum pre-training
LEARNING_RATE_PHASE2_FACTOR = 0.25  # Phase 2 LR = base LR × this factor
N_FOLDS = 5
SEED = 42

# Final train (after CV): holdout fraction for validation / early stopping, then save best weights
FINAL_VAL_FRACTION = 0.1
FINAL_MODEL_DIR = "./results_cv/best_model_final"
RESULTS_DIR = "./results_cv"

def _get_scratch_base(fallback_dir: str) -> str:
    """
    Pick a scratch directory for large intermediate artifacts.
    Preference: SLURM_TMPDIR (HPC) → TMPDIR (macOS/Linux) → fallback_dir.
    """
    return os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or fallback_dir

# --- Data paths ---
DATA_DIR = "data"
HUMAN_DATA_FILE = "human_annotated.csv"
SYNTHETIC_DATA_FILE = "syn_data_combined.csv"

# =============================================================================
# EXPERIMENT GRID
# =============================================================================
# Each config is (mode, freeze_layers, learning_rate, max_length)
# Pruned to ~15 configs to keep runtime reasonable

CONFIGS = [
    # --- Human-only experiments ---
    # Vary freeze layers with best-guess LR
    {"id": 0,  "mode": "human_only", "freeze": 0, "lr": 2e-5, "maxlen": 128, "desc": "human, no-freeze, lr2e-5, len128"},
    {"id": 1,  "mode": "human_only", "freeze": 3, "lr": 2e-5, "maxlen": 128, "desc": "human, freeze3, lr2e-5, len128"},
    {"id": 2,  "mode": "human_only", "freeze": 6, "lr": 2e-5, "maxlen": 128, "desc": "human, freeze6, lr2e-5, len128"},
    # Vary LR with no freezing
    {"id": 3,  "mode": "human_only", "freeze": 0, "lr": 3e-5, "maxlen": 128, "desc": "human, no-freeze, lr3e-5, len128"},
    {"id": 4,  "mode": "human_only", "freeze": 0, "lr": 5e-5, "maxlen": 128, "desc": "human, no-freeze, lr5e-5, len128"},
    # Max length 256 comparison (best freeze/lr combo repeated)
    {"id": 5,  "mode": "human_only", "freeze": 0, "lr": 2e-5, "maxlen": 256, "desc": "human, no-freeze, lr2e-5, len256"},
    {"id": 6,  "mode": "human_only", "freeze": 0, "lr": 3e-5, "maxlen": 256, "desc": "human, no-freeze, lr3e-5, len256"},
    # Freeze 3 with higher LR
    {"id": 7,  "mode": "human_only", "freeze": 3, "lr": 3e-5, "maxlen": 128, "desc": "human, freeze3, lr3e-5, len128"},
    {"id": 8,  "mode": "human_only", "freeze": 3, "lr": 5e-5, "maxlen": 128, "desc": "human, freeze3, lr5e-5, len128"},
    # --- Curriculum experiments ---
    # Best human-only combos repeated with curriculum
    {"id": 9,  "mode": "curriculum", "freeze": 0, "lr": 2e-5, "maxlen": 128, "desc": "curric, no-freeze, lr2e-5, len128"},
    {"id": 10, "mode": "curriculum", "freeze": 0, "lr": 3e-5, "maxlen": 128, "desc": "curric, no-freeze, lr3e-5, len128"},
    {"id": 11, "mode": "curriculum", "freeze": 3, "lr": 2e-5, "maxlen": 128, "desc": "curric, freeze3, lr2e-5, len128"},
    {"id": 12, "mode": "curriculum", "freeze": 3, "lr": 3e-5, "maxlen": 128, "desc": "curric, freeze3, lr3e-5, len128"},
    {"id": 13, "mode": "curriculum", "freeze": 0, "lr": 2e-5, "maxlen": 256, "desc": "curric, no-freeze, lr2e-5, len256"},
    {"id": 14, "mode": "curriculum", "freeze": 0, "lr": 3e-5, "maxlen": 256, "desc": "curric, no-freeze, lr3e-5, len256"},
]

set_seed(SEED)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_csv_robust(filepath, required_columns):
    """Load CSV with encoding fallback."""
    print(f"Loading: {filepath}")
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(filepath, sep=None, encoding=encoding, engine="python")
            print(f"  Loaded {len(df)} rows (encoding: {encoding})")
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not read {filepath}")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {df.columns.tolist()}")

    initial = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) < initial:
        print(f"  Dropped {initial - len(df)} rows with NaN")
    return df


def resample_synthetic_to_match(synthetic_df, target_dist, n_samples, seed):
    """Resample synthetic data to match target label distribution."""
    rng = np.random.RandomState(seed)
    parts = []
    for label, frac in target_dist.items():
        n = int(round(n_samples * frac))
        pool = synthetic_df[synthetic_df["label"] == label]
        if len(pool) == 0:
            continue
        sampled = pool.sample(n=n, replace=(n > len(pool)), random_state=rng)
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# TOKENIZATION
# =============================================================================
def create_tokenized_dataset(df, tokenizer, max_length):
    """DataFrame → tokenized HF Dataset."""
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    def tok_fn(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=max_length
        )

    ds = ds.map(tok_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# =============================================================================
# MODEL
# =============================================================================
def load_model(model_name, freeze_layers=0):
    """Load model with dropout and optional layer freezing."""
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = HIDDEN_DROPOUT
    config.attention_probs_dropout_prob = ATTENTION_DROPOUT
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"
    
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
    )

    if freeze_layers > 0:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        n_layers = len(model.roberta.encoder.layer)
        for layer in model.roberta.encoder.layer[: min(freeze_layers, n_layers)]:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,}/{total:,} trainable (frozen layers: {freeze_layers})")
    return model


# =============================================================================
# TRAINER
# =============================================================================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
        "kappa": cohen_kappa_score(labels, preds),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            w = torch.tensor(
                [class_weights[i] for i in range(NUM_LABELS)], dtype=torch.float32
            )
            if torch.cuda.is_available():
                w = w.cuda()
            self.loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ModelOnlyCheckpointTrainer(WeightedTrainer):
    """
    Trainer that checkpoints ONLY model weights/config (no optimizer/scheduler/RNG).

    This avoids flaky filesystem / huge-state serialization errors while still enabling
    `load_best_model_at_end` (best checkpoint selection across epochs).
    """

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model
        if hasattr(model, "module"):
            model = model.module
        model.save_pretrained(output_dir, safe_serialization=False)
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            tok.save_pretrained(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        # Intentionally skip optimizer/scheduler serialization.
        return

    def _save_rng_state(self, output_dir):
        # Skip RNG state serialization as well (optional, but reduces IO).
        return


def get_class_weights(train_df):
    weights = compute_class_weight(
        "balanced", classes=np.array(list(range(NUM_LABELS))), y=train_df["label"].values
    )
    return {i: float(w) for i, w in enumerate(weights)}


def make_training_args(
    output_dir,
    lr,
    epochs,
    max_length,
    load_best=True,
    run_name="",
    save_strategy="epoch",
):
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        load_best_model_at_end=load_best,
        metric_for_best_model="macro_f1" if load_best else None,
        greater_is_better=True if load_best else None,
        save_total_limit=1,
        save_only_model=True,
        learning_rate=lr,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        logging_first_step=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=0,
    )


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_on_test(model, tokenizer, test_df, max_length, device):
    """Evaluate model, return dict with overall + per-language metrics."""
    results = {}

    subsets = [("overall", test_df)]
    for lang in ["EN", "ZH"]:
        lang_df = test_df[test_df["language"] == lang]
        if len(lang_df) > 0:
            subsets.append((lang, lang_df))

    for name, sub_df in subsets:
        ds = create_tokenized_dataset(sub_df, tokenizer, max_length)
        loader = DataLoader(ds, batch_size=BATCH_SIZE)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                preds = torch.argmax(model(**inputs).logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        results[name] = {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
            "macro_recall": float(recall_score(all_labels, all_preds, average="macro")),
            "kappa": float(cohen_kappa_score(all_labels, all_preds)),
            "n_samples": len(sub_df),
        }

        # Per-class metrics
        for avg in ["macro"]:
            for label_idx in range(NUM_LABELS):
                mask = all_labels == label_idx
                if mask.sum() > 0:
                    results[name][f"f1_class{label_idx}"] = float(
                        f1_score(all_labels == label_idx, all_preds == label_idx)
                    )

    return results


# =============================================================================
# SINGLE FOLD TRAINING
# =============================================================================
def train_single_fold(
    config, fold_idx, train_df, val_df, synthetic_df, tokenizer, device
):
    """Train one fold for one config. Returns test metrics dict."""
    mode = config["mode"]
    freeze = config["freeze"]
    lr = config["lr"]
    maxlen = config["maxlen"]
    cid = config["id"]

    # Use scratch for checkpoints to avoid home directory quota issues.
    scratch_base = _get_scratch_base(RESULTS_DIR)
    fold_dir = os.path.join(scratch_base, f"xlmr_cv_c{cid}_fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    # Compute label dist from train fold
    label_counts = train_df["label"].value_counts().sort_index()
    label_dist = (label_counts / label_counts.sum()).to_dict()
    class_weights = get_class_weights(train_df) if USE_CLASS_WEIGHTS else None

    if mode == "curriculum":
        # === Phase 1: pre-train on synthetic ===
        model = load_model(MODEL_NAME, freeze_layers=freeze)
        model.to(device)

        # Resample synthetic to match human label dist
        n_syn = len(synthetic_df)
        syn_resampled = resample_synthetic_to_match(synthetic_df, label_dist, n_syn, SEED + fold_idx)
        syn_ds = create_tokenized_dataset(syn_resampled, tokenizer, maxlen)

        # Use a small portion of human val for monitoring phase 1
        val_ds = create_tokenized_dataset(val_df, tokenizer, maxlen)

        syn_weights = get_class_weights(syn_resampled) if USE_CLASS_WEIGHTS else None

        args_p1 = make_training_args(
            output_dir=os.path.join(fold_dir, "phase1"),
            lr=lr,
            epochs=EPOCHS_PHASE1,
            max_length=maxlen,
            load_best=False,
            run_name=f"c{cid}_f{fold_idx}_p1",
        )

        trainer_p1 = WeightedTrainer(
            class_weights=syn_weights,
            model=model,
            args=args_p1,
            train_dataset=syn_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer_p1.train()

        # === Phase 2: fine-tune on human ===
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True

        human_train_ds = create_tokenized_dataset(train_df, tokenizer, maxlen)

        lr_p2 = lr * LEARNING_RATE_PHASE2_FACTOR
        args_p2 = make_training_args(
            output_dir=os.path.join(fold_dir, "phase2"),
            lr=lr_p2,
            epochs=EPOCHS,
            max_length=maxlen,
            load_best=True,
            run_name=f"c{cid}_f{fold_idx}_p2",
        )

        trainer_p2 = ModelOnlyCheckpointTrainer(
            class_weights=class_weights,
            model=model,
            args=args_p2,
            train_dataset=human_train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer_p2.train()
        best_ckpt = trainer_p2.state.best_model_checkpoint

    else:
        # === Human-only training ===
        model = load_model(MODEL_NAME, freeze_layers=freeze)
        model.to(device)

        train_ds = create_tokenized_dataset(train_df, tokenizer, maxlen)
        val_ds = create_tokenized_dataset(val_df, tokenizer, maxlen)

        args = make_training_args(
            output_dir=fold_dir,
            lr=lr,
            epochs=EPOCHS,
            max_length=maxlen,
            load_best=True,
            run_name=f"c{cid}_f{fold_idx}",
        )

        trainer = ModelOnlyCheckpointTrainer(
            class_weights=class_weights,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )
        trainer.train()
        best_ckpt = trainer.state.best_model_checkpoint

    # === Evaluate on held-out val fold ===
    model.to(device)
    results = evaluate_on_test(model, tokenizer, val_df, maxlen, device)

    # Remove scratch checkpoints/workspace to avoid quota/disk issues.
    import shutil
    shutil.rmtree(fold_dir, ignore_errors=True)

    return results


def pick_best_config(all_results):
    """Return CONFIG dict with highest overall macro_f1_mean among successful CV runs."""
    valid = [
        (cid, agg)
        for cid, agg in all_results.items()
        if "error" not in agg and agg.get("overall", {}).get("macro_f1_mean") is not None
    ]
    if not valid:
        return None
    best_id, _ = max(valid, key=lambda x: x[1]["overall"]["macro_f1_mean"])
    return next(c for c in CONFIGS if c["id"] == best_id)


def train_final_and_save(
    config, human_df, synthetic_df, tokenizer, device, save_dir=FINAL_MODEL_DIR
):
    """
    Single training run using the winning hyperparameters: ~90% human train / 10% val
    (stratified by language × label when possible), then save model + tokenizer to save_dir.
    Trainer checkpoints go to save_dir/_trainer_tmp and are removed after save.
    """
    import shutil

    mode = config["mode"]
    freeze = config["freeze"]
    lr = config["lr"]
    maxlen = config["maxlen"]
    cid = config["id"]

    # Prefer node-local scratch for checkpoints (much more reliable than shared FS).
    base_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or save_dir
    workspace = os.path.join(base_tmp, f"xlmr_final_c{cid}_trainer")
    export_tmp_dir = os.path.join(base_tmp, f"xlmr_final_c{cid}_export")
    for p in [workspace, export_tmp_dir]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)

    df = human_df.copy()
    df["_strat"] = df["language"].astype(str) + "_" + df["label"].astype(str)
    try:
        train_df, val_df = train_test_split(
            df,
            test_size=FINAL_VAL_FRACTION,
            random_state=SEED,
            stratify=df["_strat"],
        )
    except ValueError:
        train_df, val_df = train_test_split(
            df, test_size=FINAL_VAL_FRACTION, random_state=SEED
        )
    train_df = train_df.drop(columns=["_strat"])
    val_df = val_df.drop(columns=["_strat"])

    label_counts = train_df["label"].value_counts().sort_index()
    label_dist = (label_counts / label_counts.sum()).to_dict()
    class_weights = get_class_weights(train_df) if USE_CLASS_WEIGHTS else None

    print(f"\n{'='*70}")
    print(f"FINAL TRAIN + SAVE — config {cid}: {config['desc']}")
    print(f"{'='*70}")
    print(f"  Train: {len(train_df)}, Val (holdout): {len(val_df)}")
    print(f"  Saving to: {os.path.abspath(save_dir)}")

    if mode == "curriculum":
        model = load_model(MODEL_NAME, freeze_layers=freeze)
        model.to(device)

        n_syn = len(synthetic_df)
        syn_resampled = resample_synthetic_to_match(
            synthetic_df, label_dist, n_syn, SEED
        )
        syn_ds = create_tokenized_dataset(syn_resampled, tokenizer, maxlen)
        val_ds = create_tokenized_dataset(val_df, tokenizer, maxlen)
        syn_weights = get_class_weights(syn_resampled) if USE_CLASS_WEIGHTS else None

        args_p1 = make_training_args(
            output_dir=os.path.join(workspace, "phase1"),
            lr=lr,
            epochs=EPOCHS_PHASE1,
            max_length=maxlen,
            load_best=False,
            run_name=f"final_c{cid}_p1",
            save_strategy="no",
        )
        trainer_p1 = WeightedTrainer(
            class_weights=syn_weights,
            model=model,
            args=args_p1,
            train_dataset=syn_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer_p1.train()

        for param in model.parameters():
            param.requires_grad = True

        human_train_ds = create_tokenized_dataset(train_df, tokenizer, maxlen)
        lr_p2 = lr * LEARNING_RATE_PHASE2_FACTOR
        args_p2 = make_training_args(
            output_dir=os.path.join(workspace, "phase2"),
            lr=lr_p2,
            epochs=EPOCHS,
            max_length=maxlen,
            load_best=True,
            run_name=f"final_c{cid}_p2",
            save_strategy="epoch",
        )
        trainer_p2 = ModelOnlyCheckpointTrainer(
            class_weights=class_weights,
            model=model,
            args=args_p2,
            train_dataset=human_train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )
        trainer_p2.train()
        best_ckpt = trainer_p2.state.best_model_checkpoint
    else:
        model = load_model(MODEL_NAME, freeze_layers=freeze)
        model.to(device)
        train_ds = create_tokenized_dataset(train_df, tokenizer, maxlen)
        val_ds = create_tokenized_dataset(val_df, tokenizer, maxlen)
        args = make_training_args(
            output_dir=workspace,
            lr=lr,
            epochs=EPOCHS,
            max_length=maxlen,
            load_best=True,
            run_name=f"final_c{cid}",
            save_strategy="epoch",
        )
        trainer = ModelOnlyCheckpointTrainer(
            class_weights=class_weights,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
        )
        trainer.train()
        best_ckpt = trainer.state.best_model_checkpoint

    model.to(device)
    val_metrics = evaluate_on_test(model, tokenizer, val_df, maxlen, device)

    # Export the best checkpoint (one directory) as the final artifact.
    if not best_ckpt or not os.path.exists(best_ckpt):
        raise RuntimeError("Best checkpoint path not found; cannot export final model.")

    shutil.rmtree(export_tmp_dir, ignore_errors=True)
    shutil.copytree(best_ckpt, export_tmp_dir)
    # Ensure tokenizer is present even if checkpoint save skipped it.
    tokenizer.save_pretrained(export_tmp_dir)

    meta = {
        "config": config,
        "save_dir": os.path.abspath(save_dir),
        "val_holdout_metrics": val_metrics,
        "train_n": len(train_df),
        "val_n": len(val_df),
    }
    with open(os.path.join(export_tmp_dir, "final_train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Move the exported artifact into place (atomic swap on same filesystem).
    # If save_dir is on a different filesystem than export_tmp_dir, fall back to copy.
    old_backup_dir = f"{save_dir}_old"
    if os.path.exists(old_backup_dir):
        shutil.rmtree(old_backup_dir, ignore_errors=True)
    if os.path.exists(save_dir):
        os.replace(save_dir, old_backup_dir)
    try:
        os.replace(export_tmp_dir, save_dir)
    except OSError:
        shutil.copytree(export_tmp_dir, save_dir)
        shutil.rmtree(export_tmp_dir, ignore_errors=True)
    if os.path.exists(old_backup_dir):
        shutil.rmtree(old_backup_dir, ignore_errors=True)
    shutil.rmtree(workspace, ignore_errors=True)
    print(f"  Saved model + tokenizer to {os.path.abspath(save_dir)}")
    print(f"  Val holdout macro_f1: {val_metrics.get('overall', {}).get('macro_f1', 0):.4f}")


# =============================================================================
# RUN CONFIG WITH K-FOLD CV
# =============================================================================
def run_config_cv(config, human_df, synthetic_df, tokenizer, device):
    """Run 5-fold CV for a single config. Returns aggregated metrics."""
    cid = config["id"]
    desc = config["desc"]

    print(f"\n{'#'*70}")
    print(f"# CONFIG {cid}: {desc}")
    print(f"{'#'*70}")

    # Stratified K-fold by language × label
    human_df = human_df.copy()
    human_df["_strat"] = human_df["language"] + "_" + human_df["label"].astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(human_df, human_df["_strat"])
    ):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        train_fold = human_df.iloc[train_idx].drop(columns=["_strat"])
        val_fold = human_df.iloc[val_idx].drop(columns=["_strat"])

        print(f"  Train: {len(train_fold)}, Val: {len(val_fold)}")
        print(f"  Train labels: {train_fold['label'].value_counts().sort_index().to_dict()}")
        print(f"  Val labels: {val_fold['label'].value_counts().sort_index().to_dict()}")

        try:
            results = train_single_fold(
                config, fold_idx, train_fold, val_fold, synthetic_df, tokenizer, device
            )
            fold_results.append(results)

            # Print fold results
            o = results.get("overall", {})
            en = results.get("EN", {})
            zh = results.get("ZH", {})
            print(
                f"  Fold {fold_idx+1}: "
                f"F1={o.get('macro_f1',0):.4f}, "
                f"Acc={o.get('accuracy',0):.4f}, "
                f"EN_F1={en.get('macro_f1',0):.4f}, "
                f"ZH_F1={zh.get('macro_f1',0):.4f}"
            )
        except Exception as e:
            print(f"  ERROR in fold {fold_idx+1}: {e}")
            traceback.print_exc()
            fold_results.append(None)

    # Aggregate across folds
    valid_folds = [r for r in fold_results if r is not None]
    if not valid_folds:
        return {"error": "All folds failed"}

    aggregated = {}
    for subset in ["overall", "EN", "ZH"]:
        subset_results = [r[subset] for r in valid_folds if subset in r]
        if not subset_results:
            continue
        metrics = {}
        for key in ["accuracy", "macro_f1", "macro_recall", "kappa"]:
            values = [r[key] for r in subset_results if key in r]
            if values:
                metrics[f"{key}_mean"] = float(np.mean(values))
                metrics[f"{key}_std"] = float(np.std(values))
        metrics["n_folds"] = len(subset_results)
        aggregated[subset] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONFIG {cid} SUMMARY ({len(valid_folds)}/{N_FOLDS} folds)")
    print(f"{'='*60}")
    for subset in ["overall", "EN", "ZH"]:
        if subset in aggregated:
            m = aggregated[subset]
            print(
                f"  {subset}: "
                f"F1={m.get('macro_f1_mean',0):.4f}±{m.get('macro_f1_std',0):.4f}, "
                f"Acc={m.get('accuracy_mean',0):.4f}±{m.get('accuracy_std',0):.4f}, "
                f"Kappa={m.get('kappa_mean',0):.4f}±{m.get('kappa_std',0):.4f}"
            )

    # Save config results
    config_dir = f"./results_cv/config_{cid}"
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "cv_results.json"), "w") as f:
        json.dump(
            {"config": config, "aggregated": aggregated, "per_fold": fold_results},
            f,
            indent=2,
        )

    return aggregated


# =============================================================================
# FINAL COMPARISON
# =============================================================================
def print_final_comparison(all_results):
    """Print a ranked comparison table of all configs."""
    print(f"\n{'='*100}")
    print("FINAL COMPARISON — ALL CONFIGS (5-fold CV, mean ± std)")
    print(f"{'='*100}")

    header = (
        f"{'ID':>3} {'Description':<42} "
        f"{'F1':>12} {'Acc':>12} {'Kappa':>12} "
        f"{'F1_EN':>12} {'F1_ZH':>12}"
    )
    print(header)
    print("-" * 100)

    # Sort by overall macro F1 mean
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("overall", {}).get("macro_f1_mean", 0),
        reverse=True,
    )

    for cid, agg in sorted_results:
        if "error" in agg:
            print(f"{cid:>3} ERROR: {agg['error']}")
            continue

        o = agg.get("overall", {})
        en = agg.get("EN", {})
        zh = agg.get("ZH", {})

        config = next(c for c in CONFIGS if c["id"] == cid)

        def fmt(mean_key, std_key, d):
            m = d.get(mean_key, 0)
            s = d.get(std_key, 0)
            return f"{m:.4f}±{s:.4f}"

        row = (
            f"{cid:>3} {config['desc']:<42} "
            f"{fmt('macro_f1_mean', 'macro_f1_std', o):>12} "
            f"{fmt('accuracy_mean', 'accuracy_std', o):>12} "
            f"{fmt('kappa_mean', 'kappa_std', o):>12} "
            f"{fmt('macro_f1_mean', 'macro_f1_std', en):>12} "
            f"{fmt('macro_f1_mean', 'macro_f1_std', zh):>12}"
        )
        print(row)

    print("-" * 100)

    # Best config
    if sorted_results:
        best_id, best_agg = sorted_results[0]
        best_f1 = best_agg.get("overall", {}).get("macro_f1_mean", 0)
        best_std = best_agg.get("overall", {}).get("macro_f1_std", 0)
        best_config = next(c for c in CONFIGS if c["id"] == best_id)
        print(f"\nBest: Config {best_id} ({best_config['desc']})")
        print(f"  Macro F1 = {best_f1:.4f} ± {best_std:.4f}")

    # EN-ZH balance analysis
    print(f"\nEN-ZH balance (|F1_EN - F1_ZH|):")
    for cid, agg in sorted_results[:5]:
        if "error" in agg:
            continue
        en_f1 = agg.get("EN", {}).get("macro_f1_mean", 0)
        zh_f1 = agg.get("ZH", {}).get("macro_f1_mean", 0)
        gap = abs(en_f1 - zh_f1)
        config = next(c for c in CONFIGS if c["id"] == cid)
        print(f"  Config {cid} ({config['desc']}): gap = {gap:.4f}")

    # Save final comparison
    comparison_path = "./results_cv/final_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(
            {cid: agg for cid, agg in all_results.items()},
            f,
            indent=2,
        )
    print(f"\nComparison saved to {comparison_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search with 5-fold CV")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=int,
        default=None,
        help="Config IDs to run (default: all). E.g. --configs 0 1 3 9",
    )
    parser.add_argument(
        "--no-final-save",
        action="store_true",
        help="Skip final train+save after CV (only metrics / comparison JSON).",
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print("XLM-RoBERTa Sentiment — Hyperparameter Search v3 (5-fold CV)")
    print(f"{'='*70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Seed: {SEED}")
    print(f"Folds: {N_FOLDS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Load data ---
    human_df = load_csv_robust(
        os.path.join(DATA_DIR, HUMAN_DATA_FILE), ["text", "label", "language"]
    )
    human_df["label"] = human_df["label"].astype(int)

    synthetic_df = load_csv_robust(
        os.path.join(DATA_DIR, SYNTHETIC_DATA_FILE), ["text", "label"]
    )
    synthetic_df["label"] = synthetic_df["label"].astype(int)

    print(f"\nHuman data: {len(human_df)}")
    print(f"  Labels: {human_df['label'].value_counts().sort_index().to_dict()}")
    for lang in ["EN", "ZH"]:
        sub = human_df[human_df["language"] == lang]
        print(f"  {lang}: {len(sub)} — {sub['label'].value_counts().sort_index().to_dict()}")

    print(f"\nSynthetic data: {len(synthetic_df)}")
    print(f"  Labels: {synthetic_df['label'].value_counts().sort_index().to_dict()}")

    # --- Select configs to run ---
    if args.configs is not None:
        configs_to_run = [c for c in CONFIGS if c["id"] in args.configs]
        if not configs_to_run:
            print(f"ERROR: No valid config IDs in {args.configs}")
            print(f"Available: {[c['id'] for c in CONFIGS]}")
            return
    else:
        configs_to_run = CONFIGS

    print(f"\nRunning {len(configs_to_run)} configs × {N_FOLDS} folds = {len(configs_to_run) * N_FOLDS} training runs")
    print(f"Configs: {[c['id'] for c in configs_to_run]}")

    # --- Run experiments ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for config in configs_to_run:
        try:
            agg = run_config_cv(config, human_df, synthetic_df, tokenizer, device)
            all_results[config["id"]] = agg
        except Exception as e:
            print(f"\nFATAL ERROR in config {config['id']}: {e}")
            traceback.print_exc()
            all_results[config["id"]] = {"error": str(e)}

    # --- Final comparison ---
    if all_results:
        print_final_comparison(all_results)

    # --- Best config: one full train + save (Option A) ---
    if all_results and not args.no_final_save:
        best_cfg = pick_best_config(all_results)
        if best_cfg is None:
            print("\nSkipping final save: no successful CV results with overall macro_f1.")
        else:
            try:
                train_final_and_save(
                    best_cfg,
                    human_df,
                    synthetic_df,
                    tokenizer,
                    device,
                    save_dir=FINAL_MODEL_DIR,
                )
            except Exception as e:
                print(f"\nERROR in final train/save: {e}")
                traceback.print_exc()

    print(f"\n{'='*70}")
    print("All experiments complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
