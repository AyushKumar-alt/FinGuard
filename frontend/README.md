# 🛡️ FinGuard Research Platform — Frontend Web Interface

Interactive, evidence-grounded financial claim verification workspace and multi-strategy retrieval laboratory. Built with **Next.js 14 (App Router)**, **TypeScript**, and **Tailwind CSS**.

---

## 🚀 Quick Start

### 1. (Optional) Start Python Backend Daemon
For live real-time claim verification using the full FinGuard neural & lexical pipeline:
```bash
# From repository root
python -m src.api_bridge serve --port 8000
```

### 2. Install & Start Development Server
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already installed)
npm install

# Start Next.js development server
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 🛠️ Build & Production

```bash
# Production build
npm run build

# Start production server
npm start
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

## 🔌 API Architecture & Multi-Tier Fallback

The verification API (`/api/verify`) uses a 4-tier graceful fallback mechanism:
1. **Tier 1: HTTP Daemon** — Queries `http://127.0.0.1:8000/verify` (fastest when models are pre-loaded).
2. **Tier 2: Subprocess Execution** — Invokes `python -m src.api_bridge verify --file <temp>` directly.
3. **Tier 3: Lexical Corpus Fallback** — Searches canonical `finfact.json` for high-overlap evidence.
4. **Tier 4: Honest NEI Verdict** — Returns `Not Enough Information` if no sufficient evidence is found.

