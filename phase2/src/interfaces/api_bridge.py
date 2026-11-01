"""
FinGuard Python API Bridge.
Provides both CLI and HTTP interfaces for the Next.js frontend to interact
with FinGuard's retrieval, claim understanding, and verification engines.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_pipeline_instance = None

def get_pipeline(load_heavy_models: bool = True):
    global _pipeline_instance
    if _pipeline_instance is None:
        from src.pipeline import FinGuardPipeline
        json_path = os.path.join(REPO_ROOT, "finfact.json")
        _pipeline_instance = FinGuardPipeline(json_path=json_path, load_heavy_models=load_heavy_models)
    return _pipeline_instance

def handle_verify(payload: Dict[str, Any]) -> Dict[str, Any]:
    claim = payload.get("claim", "")
    retrieval_mode = payload.get("retrieval_mode", "bm25")
    use_query_expansion = payload.get("use_query_expansion", True)
    use_reranker = payload.get("use_reranker", True)
    use_nli = payload.get("use_nli", True)
    top_k_retrieve = int(payload.get("top_k_retrieve", 15))
    top_k_rerank = int(payload.get("top_k_rerank", 4))

    pipeline = get_pipeline(load_heavy_models=True)
    result = pipeline.run(
        claim=claim,
        retrieval_mode=retrieval_mode,
        use_query_expansion=use_query_expansion,
        use_reranker=use_reranker,
        use_nli=use_nli,
        top_k_retrieve=top_k_retrieve,
        top_k_rerank=top_k_rerank
    )
    return result

def handle_status() -> Dict[str, Any]:
    embeddings_file = os.path.join(REPO_ROOT, "outputs", "dense_evidence_embeddings.npz")
    index_file = os.path.join(REPO_ROOT, "outputs", "positional_index.json")
    baseline_file = os.path.join(REPO_ROOT, "outputs", "best_baseline_model.pkl")
    dataset_file = os.path.join(REPO_ROOT, "finfact.json")

    has_embeddings = os.path.exists(embeddings_file)
    has_index = os.path.exists(index_file)
    has_baseline = os.path.exists(baseline_file)
    has_dataset = os.path.exists(dataset_file)

    dense_size_mb = round(os.path.getsize(embeddings_file) / (1024 * 1024), 2) if has_embeddings else 0
    dataset_size_mb = round(os.path.getsize(dataset_file) / (1024 * 1024), 2) if has_dataset else 0

    return {
        "status": "online",
        "pipeline": "FinGuard 2.0 Research Platform",
        "total_claims": 3369,
        "corpus_units": 12941,
        "dense_embeddings_cached": has_embeddings,
        "dense_embeddings_size_mb": dense_size_mb,
        "positional_index_cached": has_index,
        "calibrated_baseline_available": has_baseline,
        "dataset_available": has_dataset,
        "dataset_size_mb": dataset_size_mb,
        "models": {
            "dense_retriever": "sentence-transformers/all-MiniLM-L6-v2",
            "cross_encoder_reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "nli_verifier": "cross-encoder/nli-distilroberta-base",
            "baseline_verifier": "Calibrated Balanced Logistic Regression (TF-IDF)"
        }
    }

def main():
    parser = argparse.ArgumentParser(description="FinGuard API Bridge")
    parser.add_argument("command", choices=["verify", "status", "serve"], help="Command to execute")
    parser.add_argument("--file", type=str, default=None, help="Path to JSON file containing payload")
    parser.add_argument("--input", type=str, default=None, help="JSON input string")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP server mode")
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(handle_status(), indent=2))
        return

    if args.command == "verify":
        payload = {}
        if args.file and os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        elif args.input:
            payload = json.loads(args.input)
        else:
            # Read from stdin if available
            stdin_data = sys.stdin.read().strip()
            if stdin_data:
                payload = json.loads(stdin_data)

        result = handle_verify(payload)
        print(json.dumps(result, indent=2))
        return

    if args.command == "serve":
        # Lightweight standard library HTTP server
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class FinGuardHTTPHandler(BaseHTTPRequestHandler):
            def _send_cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                self.send_response(200)
                self._send_cors()
                self.end_headers()

            def do_GET(self):
                self._send_cors()
                if self.path in ("/status", "/api/status"):
                    res = handle_status()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                self._send_cors()
                if self.path in ("/verify", "/api/verify"):
                    content_length = int(self.headers.get("Content-Length", 0))
                    post_data = self.rfile.read(content_length)
                    try:
                        payload = json.loads(post_data.decode("utf-8"))
                        res = handle_verify(payload)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(res).encode("utf-8"))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

        server = HTTPServer(("0.0.0.0", args.port), FinGuardHTTPHandler)
        print(f"FinGuard API Bridge serving on port {args.port}...", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down API server.")
            server.server_close()

if __name__ == "__main__":
    main()
