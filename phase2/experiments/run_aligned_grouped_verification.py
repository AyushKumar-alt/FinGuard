"""Run the controlled source-grouped FinGuard verification experiment."""

import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from src.claim_understanding import extract_financial_entities

OUTPUT_DIR = ROOT / "outputs" / "experiments" / "aligned_grouped_verification"
CORPUS_PATH = OUTPUT_DIR / "retrieval_corpus_train_val.json"
LABELS = ["false", "true", "NEI"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
FAILURE_ID = len(LABELS)
SEED = 42
C_CANDIDATES = [0.1, 0.5, 1.0, 2.0, 5.0]
CSV_FIELDS = [
    "claim_id",
    "source_url",
    "claim",
    "gold_label",
    "system",
    "prediction",
    "confidence",
    "status",
    "error",
    "is_abstention",
    "is_uncertain",
    "is_conflicting",
    "selected_hyperparameter_c",
    "retrieval_candidates",
    "reranked_candidates",
    "selected_evidence",
    "supporting_evidence",
    "contradicting_evidence",
    "neutral_evidence",
    "verifier_engine",
    "component_status",
    "execution_id",
    "execution_reused",
]


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aligned_grouped_verification")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "experiment.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalize_url(value):
    if not isinstance(value, str) or not value.strip():
        return ""
    value = value.strip()
    if "://" not in value and not value.startswith("//"):
        value = "//" + value
    parsed = urlsplit(value)
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", parsed.path)
    if not host:
        return value.rstrip("/").casefold()
    return f"{host}{path}".casefold()


def read_splits():
    train = pd.read_csv(ROOT / "data" / "grouped_train.csv")
    validation = pd.read_csv(ROOT / "data" / "grouped_val.csv")
    test = pd.read_csv(ROOT / "data" / "grouped_test.csv")
    return train, validation, test


def validate_splits(train, validation, test):
    if len(test) != 675:
        raise ValueError(f"Expected exactly 675 grouped test claims, found {len(test)}.")
    if test["id"].duplicated().any():
        raise ValueError("Grouped test contains duplicate claim IDs.")

    splits = {"train": train, "validation": validation, "test": test}
    ids = {name: set(frame["id"].astype(int)) for name, frame in splits.items()}
    urls = {
        name: {normalize_url(url) for url in frame["url"].dropna().astype(str)}
        for name, frame in splits.items()
    }
    claims = {
        name: {claim.strip().casefold() for claim in frame["claim"].dropna().astype(str)}
        for name, frame in splits.items()
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if ids[left] & ids[right]:
            raise ValueError(f"Claim ID leakage between {left} and {right}.")
        if urls[left] & urls[right]:
            raise ValueError(f"Source URL leakage between {left} and {right}.")
        if claims[left] & claims[right]:
            raise ValueError(f"Duplicate claim text between {left} and {right}.")

    for frame_name, frame in splits.items():
        if not set(frame["label"].astype(str)).issubset(LABEL_TO_ID):
            raise ValueError(f"Unexpected gold label in {frame_name}.")
        expected = frame["label"].map(LABEL_TO_ID).astype(int)
        if not np.array_equal(expected.to_numpy(), frame["label_id"].astype(int).to_numpy()):
            raise ValueError(f"label_id does not match label in {frame_name}.")

    return {
        "counts": {name: len(frame) for name, frame in splits.items()},
        "unique_source_urls": {name: len(urls[name]) for name in splits},
        "label_counts": {name: frame["label"].value_counts().to_dict() for name, frame in splits.items()},
        "test_ids": ids["test"],
        "train_val_ids": ids["train"] | ids["validation"],
        "test_urls": urls["test"],
    }


def build_leakage_safe_corpus(train_val_ids, test_ids, test_urls, logger):
    """Keep only training/validation evidence and exclude held-out source URLs."""
    with open(ROOT / "finfact.json", "r", encoding="utf-8") as handle:
        raw_records = json.load(handle)

    test_source_urls = set(test_urls)
    for record_id in test_ids:
        record = raw_records[record_id]
        for evidence in record.get("evidence", []) or []:
            if isinstance(evidence, dict):
                test_source_urls.update(
                    normalize_url(href)
                    for href in evidence.get("hrefs", []) or []
                    if normalize_url(href)
                )

    safe_records = []
    excluded_evidence = 0
    kept_evidence = 0
    included_ids = set()
    for record_id in sorted(train_val_ids):
        record = raw_records[record_id]
        safe_evidence = []
        for evidence in record.get("evidence", []) or []:
            if not isinstance(evidence, dict):
                continue
            sentence = evidence.get("sentence", "")
            if not isinstance(sentence, str) or len(sentence.split()) < 4:
                continue
            hrefs = [
                href for href in evidence.get("hrefs", []) or []
                if isinstance(href, str)
                and href.strip()
                and normalize_url(href) not in test_source_urls
            ]
            original_hrefs = [
                href for href in evidence.get("hrefs", []) or []
                if isinstance(href, str) and href.strip()
            ]
            if len(hrefs) != len(original_hrefs):
                excluded_evidence += 1
                continue
            safe_evidence.append({"sentence": sentence.strip(), "hrefs": hrefs})
        if safe_evidence:
            safe_records.append({"claim": str(record.get("claim", "")), "evidence": safe_evidence})
            included_ids.add(record_id)
            kept_evidence += len(safe_evidence)

    if included_ids & test_ids:
        raise ValueError("Leakage-safe corpus unexpectedly includes a grouped test record.")
    corpus_source_urls = {
        normalize_url(href)
        for record in safe_records
        for evidence in record["evidence"]
        for href in evidence["hrefs"]
        if normalize_url(href)
    }
    leaked_sources = corpus_source_urls & test_source_urls
    if leaked_sources:
        raise ValueError(f"Found {len(leaked_sources)} test source URLs in retrieval corpus.")
    if not safe_records:
        raise ValueError("No training/validation evidence remains after source filtering.")

    corpus_payload = json.dumps(safe_records, ensure_ascii=False, separators=(",", ":"))
    CORPUS_PATH.write_text(corpus_payload, encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest()
    logger.info(
        "Isolated corpus: %d train/validation records, %d evidence sentences, "
        "%d unique source URLs; excluded %d test-linked evidence entries; "
        "test-source URL overlap=0; sha256=%s",
        len(safe_records),
        kept_evidence,
        len(corpus_source_urls),
        excluded_evidence,
        corpus_hash,
    )
    return {
        "path": str(CORPUS_PATH.relative_to(ROOT)),
        "sha256": corpus_hash,
        "training_validation_records_with_evidence": len(safe_records),
        "evidence_sentences_before_pipeline_deduplication": kept_evidence,
        "unique_citation_urls": len(corpus_source_urls),
        "excluded_test_linked_evidence_entries": excluded_evidence,
        "test_linked_source_url_count": len(test_source_urls),
        "test_source_overlap": 0,
        "record_ids": sorted(included_ids),
    }


def train_claim_only_baseline(train, validation, test, logger):
    x_train = train["claim"].fillna("").astype(str)
    y_train = train["label_id"].astype(int)
    x_validation = validation["claim"].fillna("").astype(str)
    y_validation = validation["label_id"].astype(int)
    x_test = test["claim"].fillna("").astype(str)
    validation_results = {}
    best_model = None
    best_c = None
    best_macro_f1 = -1.0
    for c_value in C_CANDIDATES:
        candidate = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10000,
                        ngram_range=(1, 2),
                        stop_words="english",
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        candidate.fit(x_train, y_train)
        validation_predictions = candidate.predict(x_validation)
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_validation,
            validation_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
        accuracy = accuracy_score(y_validation, validation_predictions)
        validation_results[str(c_value)] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        }
        logger.info("System A validation C=%s accuracy=%.4f macro_f1=%.4f", c_value, accuracy, macro_f1)
        if macro_f1 > best_macro_f1:
            best_model = candidate
            best_c = c_value
            best_macro_f1 = float(macro_f1)

    predictions = best_model.predict(x_test).astype(int)
    probabilities = best_model.predict_proba(x_test)
    prediction_rows = []
    for row, prediction, probabilities_row in zip(test.itertuples(index=False), predictions, probabilities):
        prediction_rows.append(
            {
                "claim_id": int(row.id),
                "source_url": row.url,
                "claim": row.claim,
                "gold_label": row.label,
                "system": "A",
                "prediction": ID_TO_LABEL[int(prediction)],
                "confidence": float(probabilities_row[int(prediction)]),
                "status": "ok",
                "error": "",
                "is_abstention": bool(prediction == LABEL_TO_ID["NEI"]),
                "is_uncertain": bool(prediction == LABEL_TO_ID["NEI"]),
                "is_conflicting": False,
                "selected_hyperparameter_c": best_c,
                "retrieval_candidates": "[]",
                "reranked_candidates": "[]",
                "selected_evidence": "[]",
                "supporting_evidence": "[]",
                "contradicting_evidence": "[]",
                "neutral_evidence": "[]",
                "verifier_engine": "TF-IDF + Logistic Regression (claim-only)",
                "component_status": "{}",
                "execution_id": f"A-{int(row.id)}",
                "execution_reused": False,
            }
        )
    logger.info("System A selected C=%s on grouped validation macro-F1 %.4f", best_c, best_macro_f1)
    return prediction_rows, {
        "text_input": "claim only",
        "validation_metric": "macro_f1",
        "candidate_c_values": C_CANDIDATES,
        "selected_c": best_c,
        "validation_results": validation_results,
        "training_rows": len(train),
        "validation_rows": len(validation),
    }


def construct_default_pipeline(corpus_info, logger):
    import src.pipeline as pipeline_module

    def experiment_dense_retriever(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(is_available=False)

    pipeline_module.PretrainedDenseRetriever = experiment_dense_retriever
    try:
        pipeline = pipeline_module.FinGuardPipeline(
            json_path=str(CORPUS_PATH),
            load_heavy_models=True,
        )
    finally:
        from src.dense_retrieval import PretrainedDenseRetriever
        pipeline_module.PretrainedDenseRetriever = PretrainedDenseRetriever

    if pipeline.reranker is None or not pipeline.reranker.is_available:
        raise RuntimeError("Required pretrained cross-encoder reranker is unavailable.")
    if not pipeline.verifier.has_transformer:
        raise RuntimeError("Required pretrained NLI verifier is unavailable.")

    logger.info(
        "Actual FinGuardPipeline initialized with NLI=%s, reranker=%s; "
        "isolated corpus=%s; unused dense indexing skipped on BM25 path",
        pipeline.verifier.nli_model_name,
        pipeline.reranker.model_name,
        corpus_info["path"],
    )
    return pipeline


def nli_records(result):
    records = {}
    for category, key in [
        ("supporting", "supporting_evidence"),
        ("contradicting", "contradicting_evidence"),
        ("neutral", "neutral_evidence"),
    ]:
        for evidence in result.get(key, []) or []:
            scores = {
                "entailment": evidence.get("entailment_score"),
                "contradiction": evidence.get("contradiction_score"),
                "neutral": evidence.get("neutral_score"),
            }
            valid_scores = {key: value for key, value in scores.items() if isinstance(value, (int, float))}
            predicted = max(valid_scores, key=valid_scores.get) if valid_scores else None
            records[str(evidence.get("doc_id"))] = {
                "threshold_group": category,
                "predicted_nli_label": predicted,
                **scores,
            }
    return records


def run_evidence_systems(test, corpus_info, logger):
    try:
        pipeline = construct_default_pipeline(corpus_info, logger)
    except Exception as error:
        message = f"pipeline_initialization: {type(error).__name__}: {error}"
        logger.exception("Pipeline initialization failed; no alternate implementation will be used.")
        return [
            make_pipeline_failure(row, message, "B", f"B-{int(row.id)}")
            for row in test.itertuples(index=False)
        ] + [
            make_pipeline_failure(row, message, "C", f"C-{int(row.id)}")
            for row in test.itertuples(index=False)
        ], {"initialization_error": message, "completed_calls": 0}

    original_bm25 = pipeline.retriever.retrieve_bm25
    original_rerank = pipeline.reranker.rerank
    capture = {}

    def captured_bm25(query, top_k=10):
        ranked = original_bm25(query, top_k=top_k)
        capture["retrieved"] = list(ranked)
        return ranked

    def captured_rerank(claim, candidate_docs, top_k=5, batch_size=32):
        ranked = original_rerank(
            claim,
            candidate_docs,
            top_k=len(candidate_docs),
            batch_size=batch_size,
        )
        capture["reranked"] = [dict(document) for document in ranked]
        return ranked[:top_k]

    pipeline.retriever.retrieve_bm25 = captured_bm25
    pipeline.reranker.rerank = captured_rerank
    prediction_rows = []
    call_failures = []
    for position, row in enumerate(test.itertuples(index=False), start=1):
        capture.clear()
        execution_id = f"pipeline-{int(row.id)}"
        try:
            # Use run() with its actual production defaults; no gold fields are passed.
            result = pipeline.run(row.claim)
            retrieved_rows = []
            for doc_id, score in capture.get("retrieved", []):
                if 0 <= doc_id < len(pipeline.corpus):
                    doc = pipeline.corpus[doc_id]
                    retrieved_rows.append(
                        {
                            "doc_id": int(doc_id),
                            "bm25_score": float(score),
                            "text": doc.get("text", ""),
                            "source_urls": doc.get("source_urls", []),
                            "parent_claim_ids": doc.get("parent_claim_ids", []),
                        }
                    )

            reranked_rows = capture.get("reranked", [])
            nli_by_doc = nli_records(result)
            selected_rows = []
            for document in result.get("selected_evidence_docs", []) or []:
                selected = dict(document)
                selected["nli"] = nli_by_doc.get(str(selected.get("doc_id")))
                selected_rows.append(selected)

            actual_selected = [item.get("doc_id") for item in selected_rows]
            traced_selected = [item.get("doc_id") for item in reranked_rows[:4]]
            if actual_selected != traced_selected:
                raise RuntimeError(
                    f"Instrumentation changed selected rerank order: "
                    f"pipeline={actual_selected}, trace={traced_selected}"
                )
            component_status = result.get("component_status", {})
            if not component_status.get("cross_encoder_reranker") or not component_status.get("nli_verifier"):
                raise RuntimeError(f"Required component inactive at inference: {component_status}")
            if result.get("verifier_engine", "").startswith("TF-IDF"):
                raise RuntimeError("Pipeline used baseline fallback instead of the pretrained NLI model.")

            common = {
                "claim_id": int(row.id),
                "source_url": row.url,
                "claim": row.claim,
                "gold_label": row.label,
                "prediction": result.get("verdict"),
                "confidence": result.get("confidence"),
                "status": "ok",
                "error": "",
                "is_abstention": result.get("verdict") == "NEI",
                "is_uncertain": bool(result.get("is_uncertain")),
                "is_conflicting": bool(result.get("is_conflicting")),
                "selected_hyperparameter_c": "",
                "retrieval_candidates": json.dumps(retrieved_rows, ensure_ascii=False),
                "reranked_candidates": json.dumps(reranked_rows, ensure_ascii=False),
                "selected_evidence": json.dumps(selected_rows, ensure_ascii=False),
                "supporting_evidence": json.dumps(result.get("supporting_evidence", []), ensure_ascii=False),
                "contradicting_evidence": json.dumps(result.get("contradicting_evidence", []), ensure_ascii=False),
                "neutral_evidence": json.dumps(result.get("neutral_evidence", []), ensure_ascii=False),
                "verifier_engine": result.get("verifier_engine"),
                "component_status": json.dumps(component_status, sort_keys=True),
                "execution_id": execution_id,
                "execution_reused": False,
            }
            prediction_rows.append({"system": "B", **common})
            prediction_rows.append(
                {**common, "system": "C", "execution_id": execution_id, "execution_reused": True}
            )
        except Exception as error:
            message = f"pipeline_run: {type(error).__name__}: {error}"
            logger.exception(
                "Evidence pipeline failed for test claim %d/%d (id=%s).",
                position,
                len(test),
                row.id,
            )
            call_failures.append({"claim_id": int(row.id), "error": message})
            prediction_rows.append(make_pipeline_failure(row, message, "B", execution_id))
            prediction_rows.append(
                make_pipeline_failure(row, message, "C", execution_id, execution_reused=True)
            )
        if position % 10 == 0 or position == len(test):
            logger.info("Evidence pipeline progress: %d/%d claims", position, len(test))
            write_predictions(prediction_rows, OUTPUT_DIR / "per_claim_predictions.partial.csv")

    return prediction_rows, {
        "initialization_error": None,
        "completed_calls": len(test) - len(call_failures),
        "call_failures": call_failures,
        "retrieval_mode": "bm25",
        "top_k_retrieve": 15,
        "top_k_rerank": 4,
        "query_expansion": True,
        "reranker_model": pipeline.reranker.model_name,
        "nli_model": pipeline.verifier.nli_model_name,
        "actual_pipeline_method": "FinGuardPipeline.run",
    }


def make_pipeline_failure(row, message, system, execution_id, execution_reused=False):
    return {
        "claim_id": int(row.id),
        "source_url": row.url,
        "claim": row.claim,
        "gold_label": row.label,
        "system": system,
        "prediction": "",
        "confidence": "",
        "status": "failed",
        "error": message,
        "is_abstention": False,
        "is_uncertain": False,
        "is_conflicting": False,
        "selected_hyperparameter_c": "",
        "retrieval_candidates": "[]",
        "reranked_candidates": "[]",
        "selected_evidence": "[]",
        "supporting_evidence": "[]",
        "contradicting_evidence": "[]",
        "neutral_evidence": "[]",
        "verifier_engine": "",
        "component_status": "{}",
        "execution_id": execution_id,
        "execution_reused": execution_reused,
    }


def write_predictions(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def system_metrics(rows):
    gold = [LABEL_TO_ID[row["gold_label"]] for row in rows]
    predicted = [
        LABEL_TO_ID[row["prediction"]]
        if row.get("status") == "ok" and row.get("prediction") in LABEL_TO_ID
        else FAILURE_ID
        for row in rows
    ]
    success = [prediction != FAILURE_ID for prediction in predicted]
    accuracy = float(np.mean([actual == guess for actual, guess in zip(gold, predicted)]))
    selective_accuracy = (
        float(np.mean([gold[i] == predicted[i] for i in range(len(rows)) if success[i]]))
        if any(success)
        else None
    )
    p, r, f, support = precision_recall_fscore_support(
        gold,
        predicted,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    _, _, f_weighted, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(gold, predicted, labels=[0, 1, 2, FAILURE_ID])
    class_metrics = {
        LABELS[index]: {
            "precision": float(p[index]),
            "recall": float(r[index]),
            "f1": float(f[index]),
            "support": int(support[index]),
        }
        for index in range(3)
    }
    prediction_distribution = Counter(
        row["prediction"] if row.get("status") == "ok" else "FAILURE"
        for row in rows
    )
    return {
        "n_test_claims": len(rows),
        "successful_predictions": int(sum(success)),
        "failed_predictions": int(len(rows) - sum(success)),
        "accuracy": accuracy,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "weighted_f1": float(f_weighted),
        "per_class": class_metrics,
        "confusion_matrix": {
            "labels": LABELS + ["FAILURE"],
            "rows_gold_columns_prediction": matrix.tolist(),
        },
        "prediction_distribution": {
            label: int(prediction_distribution.get(label, 0))
            for label in LABELS + ["FAILURE"]
        },
        "abstentions_or_nei_outputs": int(prediction_distribution.get("NEI", 0)),
        "coverage": float(sum(success) / len(rows)) if rows else 0.0,
        "selective_accuracy": selective_accuracy,
    }


def parse_json_field(row, name):
    try:
        return json.loads(row.get(name) or "[]")
    except (TypeError, json.JSONDecodeError):
        return []


def evidence_metrics(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    denominator = len(rows)
    support_count = sum(bool(parse_json_field(row, "supporting_evidence")) for row in successful)
    contradict_count = sum(bool(parse_json_field(row, "contradicting_evidence")) for row in successful)
    conflict_count = sum(bool(row.get("is_conflicting")) for row in successful)
    usable_count = sum(
        any(
            isinstance(item.get("text"), str) and len(item["text"].split()) >= 3
            for item in parse_json_field(row, "selected_evidence")
        )
        for row in successful
    )
    return {
        "denominator_all_675_claims": denominator,
        "successful_evidence_pipeline_calls": len(successful),
        "retrieval_success_rate_at_least_one_usable_selected_passage": usable_count / denominator,
        "claims_with_at_least_one_supporting_passage": support_count / denominator,
        "claims_with_at_least_one_contradicting_passage": contradict_count / denominator,
        "claims_with_conflicting_evidence": conflict_count / denominator,
        "claims_with_no_usable_evidence": (denominator - usable_count) / denominator,
        "failed_calls_included_as_no_usable_evidence": denominator - len(successful),
        "interpretation": (
            "Retrieval success is operational usability of a selected source-backed passage, "
            "not relevance against test gold evidence. Support/contradiction/conflict use "
            "the verifier's existing per-sentence threshold groups."
        ),
    }


def write_confusion_matrices(metrics_by_system):
    directory = OUTPUT_DIR / "confusion_matrices"
    directory.mkdir(parents=True, exist_ok=True)
    for system, metrics in metrics_by_system.items():
        matrix = metrics["confusion_matrix"]
        path = directory / f"system_{system.lower()}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["gold\\predicted"] + matrix["labels"])
            for label, values in zip(matrix["labels"], matrix["rows_gold_columns_prediction"]):
                writer.writerow([label] + values)


def category_for_error(row):
    if row["system"] == "A":
        return (
            "Other",
            "Claim-only classifier error; evidence-stage attribution is not available for System A.",
        )
    selected = parse_json_field(row, "selected_evidence")
    supporting = parse_json_field(row, "supporting_evidence")
    contradicting = parse_json_field(row, "contradicting_evidence")
    neutral = parse_json_field(row, "neutral_evidence")
    if row.get("status") != "ok":
        return "Other", "Pipeline call failed; no prediction was substituted."
    if not any(
        isinstance(item.get("text"), str) and len(item["text"].split()) >= 3
        for item in selected
    ):
        return "Insufficient evidence", "No selected passage met verifier's 3-token minimum."

    claim_terms = {
        token for token in re.findall(r"[a-z0-9]+", row.get("claim", "").casefold())
        if len(token) > 2 and token not in ENGLISH_STOP_WORDS
    }
    passages = [item.get("text", "") for item in selected if isinstance(item.get("text"), str)]
    best_overlap = max(
        (
            len(
                claim_terms
                & {
                    token for token in re.findall(r"[a-z0-9]+", passage.casefold())
                    if len(token) > 2 and token not in ENGLISH_STOP_WORDS
                }
            )
            / max(len(claim_terms), 1)
            for passage in passages
        ),
        default=0.0,
    )
    if best_overlap < 0.20:
        return (
            "Irrelevant retrieval",
            f"No selected passage shared 20% of claim content terms (best overlap={best_overlap:.3f}).",
        )
    if bool(supporting) and bool(contradicting):
        return "Contradictory evidence", "Both existing support and contradiction threshold groups are non-empty."

    passage_text = " ".join(passages).casefold()
    parsed = extract_financial_entities(row.get("claim", ""))
    periods = parsed.get("periods", [])
    if periods and not any(str(period).casefold() in passage_text for period in periods):
        return "Wrong fiscal period", "Claim period is absent from all selected passages."
    amounts = parsed.get("monetary_amounts", []) + parsed.get("percentages", [])
    if amounts:
        claim_numbers = {
            re.sub(r"[^\d.]", "", value)
            for value in amounts
            if re.sub(r"[^\d.]", "", value)
        }
        passage_numbers = {
            re.sub(r"[^\d.]", "", value)
            for value in re.findall(r"\d[\d,.]*(?:\s*%|\s*(?:percent|bps))?", passage_text)
            if re.sub(r"[^\d.]", "", value)
        }
        if claim_numbers and not claim_numbers.intersection(passage_numbers):
            return "Wrong amount or percentage", "Claimed numeric value does not appear in selected passages."
    entities = parsed.get("organizations", [])
    if entities and not any(str(entity).casefold() in passage_text for entity in entities):
        return "Wrong entity", "Extracted claim organization/entity is absent from selected passages."
    metrics = parsed.get("financial_metrics", [])
    if metrics and not any(str(metric).casefold() in passage_text for metric in metrics):
        return "Wrong financial metric", "Extracted claim financial metric is absent from selected passages."
    negations = parsed.get("negations", [])
    if negations and not any(str(negation).casefold() in passage_text for negation in negations):
        return "Negation or polarity error", "Claim contains a detected negation absent from selected passages."

    if not supporting and not contradicting and neutral:
        return "NLI neutral collapse", "Every usable selected sentence was assigned to the neutral group."
    if row.get("prediction") == "NEI" and (supporting or contradicting):
        return "Aggregation or threshold error", "Evidence crossed a 0.50 group threshold but final verdict remained NEI."
    return "Other", "No single automated stage diagnostic established a specific cause."


def build_error_analysis(rows):
    errors = []
    for row in rows:
        if row.get("status") == "ok" and row.get("prediction") == row.get("gold_label"):
            continue
        category, rationale = category_for_error(row)
        selected = parse_json_field(row, "selected_evidence")
        errors.append(
            {
                "claim_id": row["claim_id"],
                "system": row["system"],
                "gold_label": row["gold_label"],
                "prediction": row["prediction"],
                "status": row["status"],
                "primary_category": category,
                "category_basis": rationale,
                "claim": row["claim"],
                "selected_evidence": json.dumps(selected, ensure_ascii=False),
                "supporting_evidence": row.get("supporting_evidence", "[]"),
                "contradicting_evidence": row.get("contradicting_evidence", "[]"),
                "neutral_evidence": row.get("neutral_evidence", "[]"),
                "error": row.get("error", ""),
                "manual_review": "",
            }
        )
    fields = list(errors[0]) if errors else [
        "claim_id", "system", "gold_label", "prediction", "status",
        "primary_category", "category_basis", "claim", "selected_evidence",
        "supporting_evidence", "contradicting_evidence", "neutral_evidence",
        "error", "manual_review",
    ]
    with open(OUTPUT_DIR / "error_analysis.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)
    return errors


def main():
    os.chdir(ROOT)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    logger = setup_logging()
    logger.info("Starting aligned grouped benchmark with seed=%d", SEED)
    train, validation, test = read_splits()
    split_info = validate_splits(train, validation, test)
    logger.info(
        "Validated exact split sizes train=%d validation=%d test=%d; "
        "test source URL overlaps=0, ID overlaps=0, duplicate-claim overlaps=0",
        len(train),
        len(validation),
        len(test),
    )

    corpus_info = build_leakage_safe_corpus(
        split_info["train_val_ids"],
        split_info["test_ids"],
        split_info["test_urls"],
        logger,
    )

    baseline_rows, baseline_tuning = train_claim_only_baseline(train, validation, test, logger)
    evidence_rows, evidence_run_info = run_evidence_systems(test, corpus_info, logger)
    all_rows = baseline_rows + evidence_rows
    expected_rows = len(test) * 3
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} per-system rows; found {len(all_rows)}.")
    for system in ["A", "B", "C"]:
        system_rows = [row for row in all_rows if row["system"] == system]
        if len(system_rows) != len(test):
            raise RuntimeError(f"System {system} has {len(system_rows)} rows, expected {len(test)}.")
        if {row["claim_id"] for row in system_rows} != split_info["test_ids"]:
            raise RuntimeError(f"System {system} did not preserve the exact test population.")

    write_predictions(all_rows, OUTPUT_DIR / "per_claim_predictions.csv")
    partial_path = OUTPUT_DIR / "per_claim_predictions.partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    metrics_by_system = {}
    for system in ["A", "B", "C"]:
        system_rows = [row for row in all_rows if row["system"] == system]
        metrics_by_system[system] = system_metrics(system_rows)
        if system in ("B", "C"):
            metrics_by_system[system]["evidence"] = evidence_metrics(system_rows)
    write_confusion_matrices(metrics_by_system)
    errors = build_error_analysis(all_rows)
    error_counts = {
        system: sum(row["system"] == system for row in errors)
        for system in ["A", "B", "C"]
    }

    systems_equivalent = all(
        row_b["claim_id"] == row_c["claim_id"]
        and row_b["prediction"] == row_c["prediction"]
        and row_b["confidence"] == row_c["confidence"]
        and row_b["selected_evidence"] == row_c["selected_evidence"]
        and row_b["error"] == row_c["error"]
        for row_b, row_c in zip(
            [row for row in evidence_rows if row["system"] == "B"],
            [row for row in evidence_rows if row["system"] == "C"],
        )
    )
    results = {
        "experiment": "Aligned source-grouped FinGuard verification benchmark",
        "seed": SEED,
        "split": {
            key: value
            for key, value in split_info.items()
            if key not in {"test_ids", "train_val_ids", "test_urls"}
        },
        "corpus": corpus_info,
        "system_a_tuning": baseline_tuning,
        "evidence_pipeline_execution": evidence_run_info,
        "systems_functional_equivalence": {
            "system_b_and_c_identical": systems_equivalent,
            "trace": (
                "System B specifies BM25, query expansion on, reranker on, NLI on, top_k_retrieve=15, "
                "top_k_rerank=4. FinGuardPipeline.run() declares exactly those defaults and executes "
                "the same BM25 branch, reranker, and verifier in that order. System C invokes "
                "pipeline.run(claim) without overrides. Therefore one actual default pipeline call "
                "per claim is reused for both labels; it is not represented as an independent model."
            ),
            "reused_prediction_rows": sum(row["system"] == "C" for row in evidence_rows),
        },
        "metrics": metrics_by_system,
        "error_analysis": {
            "incorrect_or_failed_prediction_rows": len(errors),
            "incorrect_rows_by_system": error_counts,
            "primary_category_counts": {
                system: dict(Counter(row["primary_category"] for row in errors if row["system"] == system))
                for system in ["A", "B", "C"]
            },
            "limitations": (
                "Category assignments are rule-based triage from captured evidence groups, not "
                "independent human adjudications; the CSV retains evidence and rule basis for review."
            ),
        },
    }
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    logger.info("Saved predictions for %d claim-system pairs.", len(all_rows))
    logger.info("System A accuracy=%.4f macro-F1=%.4f", metrics_by_system["A"]["accuracy"], metrics_by_system["A"]["macro_f1"])
    logger.info("System B accuracy=%.4f macro-F1=%.4f", metrics_by_system["B"]["accuracy"], metrics_by_system["B"]["macro_f1"])
    logger.info("System C accuracy=%.4f macro-F1=%.4f", metrics_by_system["C"]["accuracy"], metrics_by_system["C"]["macro_f1"])
    logger.info("B/C functional equivalence=%s; error rows=%s", systems_equivalent, error_counts)


if __name__ == "__main__":
    main()
