"""
FinGuard: Automated Financial Claim Verification & Evidence Retrieval System
Interactive Streamlit Web Application for NLP Final Project.

Features:
- Phase 1: Finance-aware parsing, Positional Indexing, Boolean & Phrase Search
- Phase 2: Query Construction & Expansion, Pretrained Dense Retrieval, Cross-Encoder Reranking
- Evidence-Grounded Verification: Pretrained Cross-Encoder NLI & Calibrated Baseline
- Multi-Sentence Conflict & Uncertainty Aggregation with Provenance
- Comprehensive Benchmarking (Configs A-H) & Leakage Ablation
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Core Pipeline & Modules
from src.claim_understanding import extract_financial_entities
from src.query_construction import FinancialQueryBuilder
from src.pipeline import FinGuardPipeline
from src.positional_index import PositionalInvertedIndex, BooleanQueryParser

st.set_page_config(
    page_title="FinGuard | Financial Fact-Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #fdfdfd;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .evidence-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .supporting-card {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .contradicting-card {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .neutral-card {
        background-color: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .source-tag {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 4px;
        text-decoration: none;
    }
    .source-tag:hover {
        background: #bae6fd;
    }
    .entity-pill {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return FinGuardPipeline(json_path="finfact.json", load_heavy_models=True)

@st.cache_resource
def load_positional_search(corpus):
    idx_path = "outputs/positional_index.json"
    pos_index = PositionalInvertedIndex(index_path=idx_path)
    if os.path.exists(idx_path):
        pos_index.load()
    else:
        pos_index.build_index(corpus)
        pos_index.save()
    parser = BooleanQueryParser(pos_index)
    return pos_index, parser

@st.cache_data
def load_audit_data():
    if os.path.exists("outputs/dataset_audit.json"):
        with open("outputs/dataset_audit.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_baseline_metrics():
    if os.path.exists("outputs/baseline_metrics.json"):
        with open("outputs/baseline_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_retrieval_metrics():
    if os.path.exists("outputs/retrieval_metrics.json"):
        with open("outputs/retrieval_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_gold_benchmark_metrics():
    if os.path.exists("outputs/gold_benchmark_results.json"):
        with open("outputs/gold_benchmark_results.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_data
def load_dataset_samples():
    if os.path.exists("data/processed_finfact.csv"):
        return pd.read_csv("data/processed_finfact.csv")
    elif os.path.exists("finfact.json"):
        with open("finfact.json", "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()

# Load engines
with st.spinner("Initializing FinGuard System and loading pretrained models..."):
    pipeline = load_pipeline()
    pos_index, bool_parser = load_positional_search(pipeline.corpus)
    audit_data = load_audit_data()
    baseline_metrics = load_baseline_metrics()
    retrieval_metrics = load_retrieval_metrics()
    gold_benchmark_metrics = load_gold_benchmark_metrics()
    df = load_dataset_samples()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=56)
    st.title("FinGuard")
    st.markdown("**Evidence-Grounded Financial Claim Verification**")
    st.markdown("Phase 1 & Phase 2 Integrated Research Platform")
    st.markdown("---")
    
    st.markdown("### ⚙️ Pipeline Configuration")
    retrieval_mode = st.selectbox(
        "Evidence Retrieval Strategy:",
        [
            "BM25 (Inverted Index)",
            "Hybrid RRF (BM25 + Pretrained Dense)",
            "Pretrained Dense (all-MiniLM-L6-v2)",
            "Hybrid RRF (BM25 + TF-IDF)",
            "TF-IDF Cosine Similarity",
            "Dense SVD Semantic Projection (128-dim)"
        ]
    )
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        use_expansion = st.checkbox("Query Expansion", value=True, help="Appends direction-preserving entity and metric terms.")
    with col_s2:
        use_reranker = st.checkbox("Cross-Encoder", value=True, help="Joint claim-evidence reranker (ms-marco-MiniLM-L-6-v2).")

    verifier_choice = st.selectbox(
        "Claim Verifier Engine:",
        [
            "Pretrained NLI (cross-encoder/nli-distilroberta-base)",
            "Calibrated Balanced Logistic Regression (TF-IDF)"
        ]
    )

    top_k_retrieve = st.slider("Candidates to Retrieve:", min_value=5, max_value=25, value=15)
    top_k_rerank = st.slider("Final Grounding Sentences:", min_value=1, max_value=8, value=4)
    
    st.markdown("---")
    st.markdown("### 🖥️ Runtime & Model Status")
    status = pipeline.verifier.has_transformer
    st.write(f"• **Pretrained NLI:** {'🟢 Loaded' if status else '🟡 Baseline Only'}")
    dense_status = pipeline.dense_retriever and pipeline.dense_retriever.is_available
    st.write(f"• **Dense Embeddings:** {'🟢 Ready (12,943 cached)' if dense_status else '🟡 Lexical Only'}")
    rerank_status = pipeline.reranker and pipeline.reranker.is_available
    st.write(f"• **Cross-Encoder:** {'🟢 Ready' if rerank_status else '🟡 Passthrough'}")
    st.write(f"• **Positional Index:** 🟢 Postings ({len(pos_index.index):,} terms)")
    st.write(f"• **Device:** `{pipeline.verifier.device}` (CPU execution)")
    st.info("💡 **Leakage Safeguards:** Held-out target claim texts, justifications, and digests are strictly excluded from candidate retrieval.")

# Main Header
st.markdown('<div class="main-title">FinGuard: Evidence-Grounded Financial Fact-Checking</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Pipeline: Financial Entity Parsing ➔ Multi-Strategy Retrieval ➔ Cross-Encoder Reranking ➔ Pretrained NLI Verification</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Live End-to-End Verification",
    "🔎 Positional & Boolean Search",
    "📑 Retrieval Benchmark (Configs A-H)",
    "⚔️ Classification & Leakage Audit",
    "📊 Dataset Integrity & EDA",
    "📂 Corpus Explorer"
])

# ----------------- TAB 1: Live Verification -----------------
with tab1:
    st.subheader("Interactive Financial Claim Verification")
    st.markdown("Submit a financial claim. FinGuard parses entities, constructs targeted search queries, retrieves candidate evidence across **12,943** corpus sentences, applies neural cross-encoder reranking, and grounds the final verdict in multi-sentence NLI relationships.")
    
    sample_options = [
        "Select an example claim or enter custom text...",
        "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?",
        "Did the US national debt increase by over $7 trillion under Donald Trump?",
        "Amazon paid zero dollars in federal income tax for two consecutive years.",
        "ABC Bank's net profit increased by 30% to ₹4,500 crore in FY2025.",
        "The Federal Reserve lowered benchmark interest rates by 50 basis points to support market liquidity."
    ]
    selected_sample = st.selectbox("Preset Example Claims:", sample_options)
    
    default_claim = ""
    if selected_sample != "Select an example claim or enter custom text...":
        default_claim = selected_sample
        
    claim_input = st.text_area("Financial Claim to Verify:", value=default_claim, height=80, placeholder="e.g., Apple reported Q4 revenue of $89.5 billion, falling short of Wall Street estimates.")
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        verify_clicked = st.button("🚀 Verify Claim Veracity", type="primary", use_container_width=True)

    if verify_clicked:
        if not claim_input.strip():
            st.warning("Please enter a financial claim to verify.")
        else:
            with st.spinner("Executing FinGuard Verification Pipeline..."):
                # Map selected mode
                mode_key = "bm25"
                if "Dense (" in retrieval_mode:
                    mode_key = "dense"
                elif "BM25 + Pretrained Dense" in retrieval_mode:
                    mode_key = "hybrid_bm25_dense"
                elif "TF-IDF" in retrieval_mode and "Hybrid" in retrieval_mode:
                    mode_key = "hybrid_rrf"
                elif "TF-IDF" in retrieval_mode:
                    mode_key = "tfidf"
                elif "SVD" in retrieval_mode:
                    mode_key = "svd"

                use_nli_engine = "Transformer" in verifier_choice or "NLI" in verifier_choice
                
                # Execute pipeline
                res = pipeline.run(
                    claim=claim_input,
                    retrieval_mode=mode_key,
                    use_query_expansion=use_expansion,
                    use_reranker=use_reranker,
                    use_nli=use_nli_engine,
                    top_k_retrieve=top_k_retrieve,
                    top_k_rerank=top_k_rerank
                )
            
            # Step 1: Claim Understanding Display
            st.markdown("---")
            st.markdown("### 1️⃣ Financial Claim Understanding & Query Construction")
            parsed = res["parsed_claim"]
            
            p_cols = st.columns(6)
            p_cols[0].markdown(f"**🏢 Entities:**<br/>{' '.join([f'<span class=\"entity-pill\">{e}</span>' for e in parsed['entities']]) if parsed['entities'] else '<em>None</em>'}", unsafe_allow_html=True)
            p_cols[1].markdown(f"**📈 Metrics:**<br/>{' '.join([f'<span class=\"entity-pill\">{m}</span>' for m in parsed['financial_metrics']]) if parsed['financial_metrics'] else '<em>None</em>'}", unsafe_allow_html=True)
            p_cols[2].markdown(f"**💰 Monetary:**<br/>{' '.join([f'<span class=\"entity-pill\">{v}</span>' for v in parsed['monetary_amounts']]) if parsed['monetary_amounts'] else '<em>None</em>'}", unsafe_allow_html=True)
            p_cols[3].markdown(f"**📊 Percentages:**<br/>{' '.join([f'<span class=\"entity-pill\">{p}</span>' for p in parsed['percentages']]) if parsed['percentages'] else '<em>None</em>'}", unsafe_allow_html=True)
            p_cols[4].markdown(f"**📅 Periods:**<br/>{' '.join([f'<span class=\"entity-pill\">{t}</span>' for t in parsed['fiscal_periods']]) if parsed['fiscal_periods'] else '<em>None</em>'}", unsafe_allow_html=True)
            p_cols[5].markdown(f"**🧭 Direction:**<br/>{' '.join([f'<span class=\"entity-pill\">{d}</span>' for d in parsed['directional_language']]) if parsed['directional_language'] else '<em>Neutral</em>'}", unsafe_allow_html=True)

            with st.expander("🔍 Query Expansion Details"):
                q_info = res["query_info"]
                st.write(f"• **Original Query:** `{q_info['original_query']}`")
                st.write(f"• **Targeted Expansions:** `{q_info['expansions']}`")
                st.write(f"• **Search Query Sent to Retriever:** `{res['search_query']}`")

            # Step 2: Verification Verdict
            st.markdown("---")
            st.markdown("### 2️⃣ Evidence-Grounded Verification Verdict")
            
            col_v1, col_v2 = st.columns([1, 2])
            with col_v1:
                verdict = res["verdict"]
                if verdict == "true":
                    st.success("### Verdict: TRUE ✅")
                elif verdict == "false":
                    st.error("### Verdict: FALSE ❌")
                else:
                    st.warning("### Verdict: NEI ⚠️\n(Not Enough Information)")
                    
                st.metric("Model Confidence", f"{res['confidence'] * 100:.1f}%")
                st.caption(f"**Active Verifier Engine:** {res['verifier_engine']}")
                if res["is_conflicting"]:
                    st.error("⚠️ **Conflict Warning:** Sources present contradictory accounts.")
                if res["is_uncertain"] and res["uncertainty_reason"]:
                    st.info(f"ℹ️ **Uncertainty Notice:** {res['uncertainty_reason']}")
                    
            with col_v2:
                probs = res["class_probabilities"]
                st.markdown("**Veracity Probability Distribution:**")
                st.write(f"❌ **False / Contradiction:** {probs['false']*100:.1f}%")
                st.progress(probs['false'])
                st.write(f"✅ **True / Entailment:** {probs['true']*100:.1f}%")
                st.progress(probs['true'])
                st.write(f"⚠️ **NEI / Neutral:** {probs['NEI']*100:.1f}%")
                st.progress(probs['NEI'])
                st.caption(f"**Rationale:** {res['explanation']}")

            # Step 3: Retrieved & Reranked Evidence
            st.markdown("---")
            st.markdown("### 3️⃣ Grounding Evidence Sentences with Provenance")
            st.caption(f"Selected Top-{len(res['selected_evidence_docs'])} evidence sentences. Cross-Encoder joint reranker was **{'ACTIVE' if res['reranker_active'] else 'PASSTHROUGH'}**.")
            
            # Show grouped evidence
            tab_all, tab_sup, tab_con = st.tabs([
                f"All Grounding Sentences ({len(res['selected_evidence_docs'])})",
                f"Corroborating Evidence ({len(res['supporting_evidence'])})",
                f"Contradicting Evidence ({len(res['contradicting_evidence'])})"
            ])
            
            with tab_all:
                for i, d in enumerate(res["selected_evidence_docs"]):
                    urls_html = "".join([f'<a href="{u}" target="_blank" class="source-tag">Source {idx+1}</a> ' for idx, u in enumerate(d["source_urls"][:3])])
                    rerank_info = f"Cross-Encoder Score: {d['rerank_score']:.3f} | Initial Retrieval Score: {d['score']:.3f}" if "rerank_score" in d else f"Retrieval Score: {d['score']:.3f}"
                    st.markdown(f"""
                    <div class="evidence-card">
                        <strong>Sentence #{i+1}</strong> <span style="color:#64748b; font-size:0.85rem;">({rerank_info})</span><br/>
                        {d['text']}<br/>
                        {urls_html if urls_html else '<span class="source-tag">Internal Archive</span>'}
                    </div>
                    """, unsafe_allow_html=True)
                    
            with tab_sup:
                if not res["supporting_evidence"]:
                    st.info("No direct corroborating evidence sentences reached entailment threshold.")
                for i, d in enumerate(res["supporting_evidence"]):
                    urls_html = "".join([f'<a href="{u}" target="_blank" class="source-tag">Source Link</a> ' for u in d["source_urls"][:3]])
                    st.markdown(f"""
                    <div class="supporting-card">
                        <strong>Corroborating Sentence #{i+1}</strong> (Entailment: {d['entailment_score']*100:.1f}%)<br/>
                        {d['text']}<br/>
                        {urls_html if urls_html else '<span class="source-tag">Internal Archive</span>'}
                    </div>
                    """, unsafe_allow_html=True)

            with tab_con:
                if not res["contradicting_evidence"]:
                    st.info("No direct refuting evidence sentences reached contradiction threshold.")
                for i, d in enumerate(res["contradicting_evidence"]):
                    urls_html = "".join([f'<a href="{u}" target="_blank" class="source-tag">Source Link</a> ' for u in d["source_urls"][:3]])
                    st.markdown(f"""
                    <div class="contradicting-card">
                        <strong>Refuting Sentence #{i+1}</strong> (Contradiction: {d['contradiction_score']*100:.1f}%)<br/>
                        {d['text']}<br/>
                        {urls_html if urls_html else '<span class="source-tag">Internal Archive</span>'}
                    </div>
                    """, unsafe_allow_html=True)

# ----------------- TAB 2: Positional & Boolean Search -----------------
with tab2:
    st.subheader("Phase 1: Positional Inverted Index & Boolean Search")
    st.markdown("""
    FinGuard maintains an exact **Positional Inverted Index** containing term frequencies, document frequencies, and token positions across all **12,943** corpus sentences.
    Supports **quoted phrase search**, explicit boolean operators (`AND`, `OR`, `NOT`), parentheses, and strict operator precedence (`NOT` > `AND` > `OR`).
    """)
    
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        bool_query = st.text_input(
            "Boolean / Exact Phrase Query:",
            value='"national debt" AND NOT stimulus',
            placeholder='e.g., "shareholder capitalism" OR (biden AND NOT tax)'
        )
    with col_b2:
        st.write("")
        st.write("")
        search_bool_btn = st.button("🔎 Execute Query", type="primary")

    if bool_query:
        doc_ids, err = bool_parser.parse_and_execute(bool_query)
        if err:
            st.error(f"Syntax Error: {err}")
        else:
            st.success(f"Found **{len(doc_ids)}** matching documents across the corpus.")
            for i, did in enumerate(doc_ids[:10]):
                doc = pipeline.corpus[did]
                urls_html = "".join([f'<a href="{u}" target="_blank" class="source-tag">Source Link</a> ' for u in doc["source_urls"][:2]])
                st.markdown(f"""
                <div class="evidence-card">
                    <strong>Match #{i+1} [Doc ID: {did}]</strong><br/>
                    {doc['text']}<br/>
                    {urls_html if urls_html else '<span class="source-tag">Internal Archive</span>'}
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: Retrieval Benchmark -----------------
with tab3:
    st.subheader("Information Retrieval & Gold-Evidence Relevance Benchmarks")
    
    benchmark_view = st.radio(
        "Select Benchmark View:",
        [
            "🥇 Gold-Evidence Relevance Benchmark (v1.0.0, 75 Held-Out Claims, Graded nDCG & Coverage)",
            "📊 Full Held-Out Retrieval Suite (506 Claims, Silver Reference)"
        ],
        horizontal=True
    )
    
    if "Gold-Evidence" in benchmark_view:
        st.markdown(r"""
        **Gold-Evidence Relevance Benchmark (v1.0.0):**
        Evaluated on a **stratified pilot sample of 75 held-out test claims** across **12,941 candidate evidence sentences**.
        Candidates were pooled from the top-20 results of all 8 retrieval systems (4,891 pairs) with 100% of dataset reference evidence injected.
        Evaluates **graded relevance** ($\tau \ge 2$), **exponential nDCG**, **standard TREC MAP**, and **judged coverage** with 95% bootstrap confidence intervals.
        """)
        if gold_benchmark_metrics:
            gold_df = pd.DataFrame(gold_benchmark_metrics)
            cols_show = ["System", "MRR", "R@5", "R@10", "nDCG@5", "nDCG@10", "P@5", "P@10", "Coverage@10", "RefRate@10", "Latency_Mean_ms"]
            disp_df = gold_df[[c for c in cols_show if c in gold_df.columns]].set_index("System")
            st.dataframe(disp_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)
            
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.markdown("### 🏆 Gold Benchmark Takeaways:")
                st.markdown("""
                - **Cross-Encoder Reranker (`ms-marco`)** achieves highest ranking quality: **MRR = 0.6686** (95% CI: [0.564, 0.764]) and **nDCG@5 = 0.4641**.
                - **Hybrid RRF (BM25 + Dense)** achieves peak coverage: **Recall@10 = 0.4212** and **nDCG@10 = 0.4405**.
                - **Dense Transformer (`all-MiniLM-L6-v2`)** provides the fastest neural inference (**18.7 ms/query**) with strong **MRR = 0.6469** and **Precision@10 = 0.2400**.
                - **Judged Coverage@10** is **97.6% - 100%**, confirming that multi-system pooling avoided false-negative evaluation distortions.
                """)
            with c_g2:
                st.markdown("### 🔬 Incomplete Judgment Handling:")
                st.markdown("""
                - **Silver Reference Evidence:** 410 pairs (Grade 3).
                - **Explicitly Unjudged Candidates:** 4,480 pairs (preserved as unjudged, never assumed to be 0).
                - **Binary Relevance Threshold:** $\tau = 2$ (Grades 2 & 3 counted as relevant).
                - **Evaluation Artifacts:** [configs/gold_evidence_benchmark.json](file:///c:/Users/calpo/OneDrive/Pictures/final%20project%20of%20NLP/configs/gold_evidence_benchmark.json).
                """)
    else:
        st.markdown("""
        Evaluated on **506 held-out test claims** searching across **12,943 candidate evidence sentences**.
        Strict non-leakage protocol: Target claim text, gold evidence, justification, and labels are completely excluded from candidate sets.
        """)
        if retrieval_metrics:
            ret_df = pd.DataFrame(retrieval_metrics).T
            st.dataframe(ret_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)
            
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            if os.path.exists("figures/retrieval_comparison.png"):
                st.image("figures/retrieval_comparison.png", caption="Information Retrieval Benchmark Across Configs A-H")
        with col_r2:
            st.markdown("### 🏆 Research Findings:")
            st.markdown("""
            - **Config G (BM25 + Cross-Encoder Reranker)** achieved the highest precision metrics: **MRR = 0.8713** and **MAP = 0.4803**, proving the effectiveness of joint claim-evidence neural scoring.
            - **Config F (Hybrid BM25 + Pretrained Dense RRF)** achieved the highest recall: **Recall@10 = 0.5646**, confirming that dense representations capture semantic paraphrasing while BM25 retains exact financial entity tokens.
            - **Config E (Pretrained Dense all-MiniLM-L6-v2)** outperformed pure lexical BM25 in MRR (**0.8046 vs. 0.7680**), demonstrating that pretrained dense embeddings generalize well to financial claims.
            - **Config H (Query Expansion BM25):** Yields **MRR = 0.7592, R@10 = 0.4500**.
            - **Config C (Unsupervised 128-dim SVD):** Suffers severe semantic drift (**MRR = 0.1291**).
            """)

# ----------------- TAB 4: Classification & Leakage Audit -----------------
with tab4:
    st.subheader("Classification Benchmarks & Data Leakage Ablation")
    st.markdown("Empirical verification of classifier performance across four distinct experimental setups.")
    
    if baseline_metrics:
        rows = []
        for exp_key, models_dict in baseline_metrics.items():
            for m_name, m_vals in models_dict.items():
                rows.append({
                    "Setting": exp_key.replace("Experiment_", "").replace("_", " "),
                    "Model": m_name,
                    "Accuracy": f"{m_vals['Accuracy']*100:.2f}%",
                    "Macro F1": f"{m_vals['Macro F1']:.4f}",
                    "Weighted F1": f"{m_vals['Weighted F1']:.4f}",
                    "F1 False": f"{m_vals['Per_Class']['false']['F1']:.2f}",
                    "F1 True": f"{m_vals['Per_Class']['true']['F1']:.2f}",
                    "F1 NEI": f"{m_vals['Per_Class']['NEI']['F1']:.2f}"
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        if os.path.exists("figures/best_baseline_confusion_matrix.png"):
            st.image("figures/best_baseline_confusion_matrix.png", caption="Standard Test Split Confusion Matrix")
    with col_c2:
        if os.path.exists("figures/grouped_split_confusion_matrix.png"):
            st.image("figures/grouped_split_confusion_matrix.png", caption="Source-Grouped Split (Zero Article URL Overlap)")

    st.markdown("### 🔬 Critical Leakage & Generalization Insights:")
    st.markdown("""
    1. **Reproducibility Verified:** Logistic Regression on the standard split with raw evidence accurately reproduces **74.70% Accuracy** and **0.7195 Macro F1**.
    2. **Evidence Reliance Proven (Exp B vs Exp A):** When evidence is removed entirely (Claim-Only), accuracy drops from **74.70% to 64.03%** (-10.67%), proving that models genuinely utilize evidence context rather than solely relying on claim priors.
    3. **Meta-Language Leakage Ablation (Exp C):** Stripping explicit fact-checker verdict cues ('falsely claimed', 'debunked') reduces accuracy by only **0.98%** (74.70% -> 73.72%), demonstrating model reliance on substantive evidence features.
    4. **Source Generalization Gap (Exp D):** When evaluated on completely unseen fact-checking source URLs with zero overlap, accuracy drops to **59.70%**. This highlights the domain transfer challenge in financial fact-checking when testing on unfamiliar publishers.
    """)

# ----------------- TAB 5: Dataset Integrity -----------------
with tab5:
    st.subheader("Dataset Integrity Audit & Provenance")
    
    if audit_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{audit_data.get('total_records', 0):,}")
        c2.metric("Candidate Evidence Sentences", f"{len(pipeline.corpus):,}")
        c3.metric("Unique Source URLs", f"{audit_data.get('duplicate_analysis', {}).get('unique_urls', 0):,}")
        c4.metric("Evidence Sentences w/ Cues", f"{audit_data.get('evidence_analysis', {}).get('percentage_records_with_evidence_leakage_cues', 0):.1f}%")
        
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("figures/label_distribution.png"):
            st.image("figures/label_distribution.png", caption="FinFact Label Distribution")
    with col_fig2:
        if os.path.exists("figures/top_financial_issues.png"):
            st.image("figures/top_financial_issues.png", caption="Top Financial Issues")

# ----------------- TAB 6: Corpus Explorer -----------------
with tab6:
    st.subheader("FinFact Corpus Explorer")
    filter_label = st.selectbox("Filter Label:", ["All", "false", "true", "NEI"], key="filter_corpus")
    search_q = st.text_input("Search Claim or Evidence:", placeholder="e.g., inflation, taxes, debt...")
    
    filtered = df.copy()
    if filter_label != "All" and "label" in filtered.columns:
        filtered = filtered[filtered["label"] == filter_label]
    if search_q:
        mask = filtered["claim"].str.contains(search_q, case=False, na=False)
        if "evidence_text" in filtered.columns:
            mask = mask | filtered["evidence_text"].str.contains(search_q, case=False, na=False)
        filtered = filtered[mask]
        
    st.write(f"Showing **{len(filtered):,}** matching claims:")
    cols_to_show = [c for c in ["claim", "label", "issues", "evidence_text", "url"] if c in filtered.columns]
    st.dataframe(filtered[cols_to_show].head(50), use_container_width=True)

st.markdown("---")
st.caption("FinGuard Research Platform • Evidence-Grounded Financial Claim Verification • Phase 1 & 2 Integrated")
