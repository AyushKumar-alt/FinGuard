# 🛡️ FinGuard Research Platform — Frontend Web Interface

Interactive, evidence-grounded financial claim verification workspace and multi-strategy retrieval laboratory. Built with **Next.js 14 (App Router)**, **TypeScript**, and **Tailwind CSS**.

---

## 🚀 Quick Start

### 1. Start the Python inference backend
Keep this running in a terminal from the repository root. The Python service owns query construction, retrieval, reranking, NLI, and verdict aggregation:
```bash
# From the repository root
python -m pip install -r requirements.txt
python -m src.api_bridge serve --port 8000
```

### 2. Install and start the frontend
```bash
# In a second terminal, from the repository root
npm --prefix frontend ci
npm --prefix frontend run dev
```
Open **[http://localhost:3000](http://localhost:3000)**. The browser UI sends the claim to the Next.js `/api/verify` adapter, which forwards it to `PYTHON_API_URL` (default `http://127.0.0.1:8000`). No inference runs in browser code.

To use a different backend address, set `PYTHON_API_URL` in the frontend server environment before starting Next.js.

---

## 🛠️ Build & Production

```bash
# From the repository root
npm --prefix frontend run build

# Start production server on port 3000
npm --prefix frontend run start
```

---

## 📁 Key Routes & Pages

| Route | Name | Description |
|---|---|---|
| `/` | **Overview Dashboard** | Real-time status cards, system gauges, and quick verification input. |
| `/verify` | **Claim Verification** | Multi-strategy evidence retrieval, NLI conflict resolution, and verdict provenance. |
| `/investigations` | **Investigations & History** | Searchable audit trail of verified claims and evidence snapshots. |
| `/evidence` | **Evidence Explorer** | Corpus browser with Boolean query engine and positional snippet matches. |
| `/retrieval-lab` | **Retrieval Lab** | Benchmarking 8 retrieval configurations (BM25, TF-IDF, RRF, Hybrid, Pipeline 1 Lemmatized). |
| `/evaluation` | **Verification & Evaluation**| Confusion matrices, ablation studies, and verifier comparisons. |
| `/gold-benchmark` | **Gold Benchmark** | Fact-checker gold reference relevance and SHA-256 verifiable datasets. |
| `/review-queue` | **Human Review Queue** | Adjudication interface for auditing borderline evidence and annotator notes. |
| `/reports` | **Reports & Exports** | Downloadable benchmark evaluations and reproducible JSON manifests. |
| `/settings` | **System Settings** | Model cache status, environment parameters, and theme configuration. |

---

## API architecture

```text
Browser UI (Next.js)
  -> Next.js POST /api/phase2/verify or /api/verify (validation and HTTP proxy)
  -> Python API bridge POST /api/phase2/verify or /verify (port 8000)
  -> FinGuardPipeline.run() and configured Phase 2 components
  <- actual verdict, evidence with publisher/title/URLs, query diagnostics, and 8-stage transparency
```

Health check:
```text
Browser / Monitoring
  -> GET /api/health
  -> Python backend GET /api/health (status, loaded models, corpus statistics)
```

The UI uses the backend defaults: BM25 retrieval with configured finance-aware multi-query expansion and Weighted RRF when multiple queries are generated, followed by reranking and NLI when available. It does not expose experimental retrieval toggles. If the Python service is unavailable, the API reports an error; it does not invent an NEI prediction or substitute archived labels.
