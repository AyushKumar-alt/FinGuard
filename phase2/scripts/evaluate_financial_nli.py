"""
FinGuard Phase 2: Pair-Level NLI Model Evaluation Script.

Evaluates and compares:
  Model A: Original Pretrained Zero-Shot Cross-Encoder ('cross-encoder/nli-distilroberta-base')
  Model B: Financially Fine-Tuned Cross-Encoder ('models/finetuned_nli')

Evaluated on the exact same held-out validation pairs to isolate the pure effect of fine-tuning.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Ensure offline operation
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_financial_nli")

LABEL_MAPPING = {0: "contradiction", 1: "entailment", 2: "neutral"}

class NLIInferenceDataset(Dataset):
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

def evaluate_model_on_dataset(model_path: str, records: List[Dict[str, Any]], batch_size: int = 16, max_length: int = 160) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model from %s on %s...", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    dataset = NLIInferenceDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    return {
        "model_path": str(model_path),
        "sample_count": len(all_labels),
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
        "confusion_matrix": cm
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare NLI Models")
    parser.add_argument("--eval_file", default="phase2/data/financial_nli/validation.jsonl")
    parser.add_argument("--zero_shot_model", default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--finetuned_model", default="models/finetuned_nli")
    parser.add_argument("--output_file", default="phase2/results/financial_nli/pair_level_comparison.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=160)
    args = parser.parse_args()

    eval_path = ROOT / args.eval_file
    out_path = ROOT / args.output_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(eval_path)
    logger.info("Loaded %d evaluation pairs from %s", len(records), eval_path)

    # 1. Evaluate Zero-Shot
    logger.info("=== Evaluating Model A: Pretrained Zero-Shot ===")
    zero_shot_results = evaluate_model_on_dataset(
        args.zero_shot_model, records, batch_size=args.batch_size, max_length=args.max_length
    )

    # 2. Evaluate Fine-Tuned
    logger.info("=== Evaluating Model B: Financial Fine-Tuned ===")
    finetuned_results = evaluate_model_on_dataset(
        args.finetuned_model, records, batch_size=args.batch_size, max_length=args.max_length
    )

    # Compute Deltas
    deltas = {
        "accuracy_delta": round(finetuned_results["accuracy"] - zero_shot_results["accuracy"], 2),
        "macro_f1_delta": round(finetuned_results["macro_f1"] - zero_shot_results["macro_f1"], 2),
        "weighted_f1_delta": round(finetuned_results["weighted_f1"] - zero_shot_results["weighted_f1"], 2),
        "contradiction_f1_delta": round(
            finetuned_results["per_class"]["contradiction"]["f1"] - zero_shot_results["per_class"]["contradiction"]["f1"], 2
        ),
        "entailment_f1_delta": round(
            finetuned_results["per_class"]["entailment"]["f1"] - zero_shot_results["per_class"]["entailment"]["f1"], 2
        ),
        "neutral_f1_delta": round(
            finetuned_results["per_class"]["neutral"]["f1"] - zero_shot_results["per_class"]["neutral"]["f1"], 2
        )
    }

    comparison = {
        "eval_dataset": str(args.eval_file),
        "pair_count": len(records),
        "model_a_zero_shot": zero_shot_results,
        "model_b_finetuned": finetuned_results,
        "deltas": deltas
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved comparison results to %s", out_path)

    print("\n" + "=" * 70)
    print("FINANCIAL NLI PAIR-LEVEL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Zero-Shot':<12} | {'Fine-Tuned':<12} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {zero_shot_results['accuracy']:>10.2f}% | {finetuned_results['accuracy']:>10.2f}% | {deltas['accuracy_delta']:>+9.2f}%")
    print(f"{'Macro-F1':<25} | {zero_shot_results['macro_f1']:>10.2f}% | {finetuned_results['macro_f1']:>10.2f}% | {deltas['macro_f1_delta']:>+9.2f}%")
    print(f"{'Weighted-F1':<25} | {zero_shot_results['weighted_f1']:>10.2f}% | {finetuned_results['weighted_f1']:>10.2f}% | {deltas['weighted_f1_delta']:>+9.2f}%")
    print(f"{'Contradiction F1':<25} | {zero_shot_results['per_class']['contradiction']['f1']:>10.2f}% | {finetuned_results['per_class']['contradiction']['f1']:>10.2f}% | {deltas['contradiction_f1_delta']:>+9.2f}%")
    print(f"{'Entailment F1':<25} | {zero_shot_results['per_class']['entailment']['f1']:>10.2f}% | {finetuned_results['per_class']['entailment']['f1']:>10.2f}% | {deltas['entailment_f1_delta']:>+9.2f}%")
    print(f"{'Neutral F1':<25} | {zero_shot_results['per_class']['neutral']['f1']:>10.2f}% | {finetuned_results['per_class']['neutral']['f1']:>10.2f}% | {deltas['neutral_f1_delta']:>+9.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()

