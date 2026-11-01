"""
FinGuard Phase 2: Financial Domain NLI Fine-Tuning Script.

Fine-tunes 'cross-encoder/nli-distilroberta-base' on financial claim-evidence pairs.
Premise: Financial evidence passage.
Hypothesis: Financial claim.
Target Labels:
  0 = Contradiction
  1 = Entailment
  2 = Neutral

Features:
- Offline environment resilience (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1).
- Strict random seed fixation for reproducibility (SEED=42).
- Class-balanced weighted CrossEntropyLoss to counter class distribution skew.
- Validation evaluation at every epoch calculating:
  - Loss, Accuracy, Macro-F1, Weighted-F1, per-class P/R/F1.
- Early stopping tracking validation Macro-F1 with configurable patience.
- Complete artifact persistence:
  - Model weights and configuration (save_pretrained)
  - Tokenizer assets (save_pretrained)
  - Training history and best checkpoint metadata (JSON).
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple

# Force offline mode before transformers imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_financial_nli")

LABEL_MAPPING = {0: "contradiction", 1: "entailment", 2: "neutral"}
LABEL_TO_ID = {"contradiction": 0, "entailment": 1, "neutral": 2}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FinancialNLIDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_length: int = 160):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        premise = item["premise"]
        hypothesis = item["claim"]
        label = int(item["label"])

        encoded = self.tokenizer(
            premise,
            hypothesis,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def balance_dataset(records: List[Dict[str, Any]], max_per_class: int = 1200, seed: int = 42) -> List[Dict[str, Any]]:
    """Samples up to max_per_class items per class, prioritizing high-confidence non-weak pairs."""
    by_class = {0: [], 1: [], 2: []}
    for r in records:
        lbl = int(r["label"])
        by_class[lbl].append(r)

    rng = random.Random(seed)
    balanced = []
    for lbl, items in by_class.items():
        # Sort so high-confidence pairs come first
        items.sort(key=lambda x: (not x.get("is_weakly_supervised", False), x.get("confidence", 0.5)), reverse=True)
        if len(items) > max_per_class:
            selected = items[:max_per_class]
        else:
            selected = items
        balanced.extend(selected)
        logger.info("Class %d (%s): %d samples selected (from %d total)", lbl, LABEL_MAPPING[lbl], len(selected), len(items))

    rng.shuffle(balanced)
    return balanced

def evaluate(model, dataloader, device, criterion=None) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    num_samples = len(all_labels)
    avg_loss = (total_loss / num_samples) if criterion else 0.0
    acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc) * 100.0, 2),
        "macro_f1": round(float(macro_f1) * 100.0, 2),
        "weighted_f1": round(float(weighted_f1) * 100.0, 2),
        "macro_precision": round(float(macro_p) * 100.0, 2),
        "macro_recall": round(float(macro_r) * 100.0, 2),
        "per_class": {
            "contradiction": {
                "precision": round(float(per_class_p[0]) * 100.0, 2),
                "recall": round(float(per_class_r[0]) * 100.0, 2),
                "f1": round(float(per_class_f1[0]) * 100.0, 2),
                "support": int(per_class_supp[0])
            },
            "entailment": {
                "precision": round(float(per_class_p[1]) * 100.0, 2),
                "recall": round(float(per_class_r[1]) * 100.0, 2),
                "f1": round(float(per_class_f1[1]) * 100.0, 2),
                "support": int(per_class_supp[1])
            },
            "neutral": {
                "precision": round(float(per_class_p[2]) * 100.0, 2),
                "recall": round(float(per_class_r[2]) * 100.0, 2),
                "f1": round(float(per_class_f1[2]) * 100.0, 2),
                "support": int(per_class_supp[2])
            }
        },
        "confusion_matrix": cm,
        "sample_count": num_samples
    }

def main():
    parser = argparse.ArgumentParser(description="Train Financial Domain NLI Model")
    parser.add_argument("--base_model", default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--data_dir", default="phase2/data/financial_nli")
    parser.add_argument("--output_dir", default="models/finetuned_nli")
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--balance_train", action="store_true", default=True)
    parser.add_argument("--max_samples_per_class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training using device: %s", device)

    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "validation.jsonl")
    logger.info("Raw loaded: %d train records, %d val records", len(train_records), len(val_records))

    if args.balance_train:
        train_records = balance_dataset(train_records, max_per_class=args.max_samples_per_class, seed=args.seed)
        logger.info("Class-balanced training set size: %d records", len(train_records))

    # 2. Tokenizer and Model
    logger.info("Loading pretrained model and tokenizer: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, local_files_only=True)

    # Explicitly configure label mappings
    model.config.id2label = LABEL_MAPPING
    model.config.label2id = LABEL_TO_ID
    model.to(device)

    # 3. DataLoaders
    train_dataset = FinancialNLIDataset(train_records, tokenizer, max_length=args.max_length)
    val_dataset = FinancialNLIDataset(val_records, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Class Weights for Loss
    train_labels = [r["label"] for r in train_records]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    weights = [total_samples / (3.0 * class_counts.get(i, 1)) for i in range(3)]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    logger.info("Computed class weights: %s", [round(w, 4) for w in weights])

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 5. Baseline Pretrained Zero-Shot Evaluation (Epoch 0)
    logger.info("Evaluating zero-shot baseline on validation set before fine-tuning...")
    zero_shot_metrics = evaluate(model, val_loader, device, criterion)
    logger.info("Zero-Shot Baseline -> Val Acc: %.2f%%, Macro-F1: %.2f%%, Weighted-F1: %.2f%%",
                zero_shot_metrics["accuracy"], zero_shot_metrics["macro_f1"], zero_shot_metrics["weighted_f1"])
    logger.info("Zero-Shot Per-Class: Contradiction F1=%.2f%%, Entailment F1=%.2f%%, Neutral F1=%.2f%%",
                zero_shot_metrics["per_class"]["contradiction"]["f1"],
                zero_shot_metrics["per_class"]["entailment"]["f1"],
                zero_shot_metrics["per_class"]["neutral"]["f1"])

    # 6. Training Loop with Early Stopping
    training_history = []
    best_macro_f1 = zero_shot_metrics["macro_f1"]
    best_epoch = 0
    patience_counter = 0

    logger.info("Beginning fine-tuning for up to %d epochs...", args.epochs)
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batch_count = len(train_loader)

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == batch_count:
                logger.info("Epoch %d/%d [Batch %d/%d] - Batch Loss: %.4f",
                            epoch, args.epochs, step, batch_count, loss.item())

        epoch_loss = running_loss / batch_count
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device, criterion)
        logger.info("Epoch %d Complete (%.1fs) - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val Macro-F1: %.2f%%",
                    epoch, epoch_time, epoch_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["macro_f1"])
        logger.info("Val Per-Class: Contra F1=%.2f%%, Entail F1=%.2f%%, Neutral F1=%.2f%%",
                    val_metrics["per_class"]["contradiction"]["f1"],
                    val_metrics["per_class"]["entailment"]["f1"],
                    val_metrics["per_class"]["neutral"]["f1"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "epoch_duration_seconds": round(epoch_time, 2),
            "val_metrics": val_metrics
        }
        training_history.append(epoch_record)

        # Check for improvement
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            logger.info("New best validation Macro-F1: %.2f%%! Saving best checkpoint to %s", best_macro_f1, output_dir)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            logger.info("No improvement in Macro-F1 (best was %.2f%% at epoch %d). Patience: %d/%d",
                        best_macro_f1, best_epoch, patience_counter, args.patience)
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    total_training_time = time.time() - start_total_time
    logger.info("Training complete in %.2f minutes. Best Epoch: %d with Val Macro-F1: %.2f%%",
                total_training_time / 60.0, best_epoch, best_macro_f1)

    # Save training metadata & history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_checkpoint": args.base_model,
            "total_epochs_run": len(training_history),
            "total_training_seconds": round(total_training_time, 2),
            "zero_shot_baseline": zero_shot_metrics,
            "training_history": training_history,
            "best_checkpoint": {
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "class_weights": [round(w, 4) for w in weights]
            }
        }, f, indent=2)
    logger.info("Saved training history to %s", history_path)

    config_path = output_dir / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Saved training configuration to %s", config_path)

if __name__ == "__main__":
    main()

