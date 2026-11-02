"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  ShieldCheck,
  Sparkles,
  Sliders,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Tag,
  Layers,
  Search,
  Cpu,
  BarChart2,
  CheckCircle2,
  Info,
} from "lucide-react";
import { VerdictBadge } from "@/components/ui/VerdictBadge";
import { EvidenceCard } from "@/components/ui/EvidenceCard";
import { LoadingState } from "@/components/ui/LoadingState";
import { verifyClaim } from "@/lib/api";
import { CandidateEvidence, ClaimVerificationResponse } from "@/lib/types";

const SAMPLE_CLAIMS = [
  "Amazon paid zero dollars in federal corporate income tax for two consecutive years.",
  "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?",
  "Did the US national debt increase by over $7 trillion under Donald Trump?",
  "ABC Bank's net profit increased by 30% to ₹4,500 crore in FY2025.",
  "Social Security trust fund is completely empty and will cease all payments next month.",
];

function VerifyContent() {
  const searchParams = useSearchParams();
  const initialClaim = searchParams.get("claim") || "";

  const [claim, setClaim] = useState(initialClaim);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClaimVerificationResponse | null>(null);

  const [showTechnicalDetails, setShowTechnicalDetails] = useState(true);

  // Auto-verify if claim came from query param
  useEffect(() => {
    if (initialClaim && !result && !loading) {
      handleVerify(initialClaim);
    }
  }, [initialClaim]);

  const handleVerify = async (claimToVerify?: string) => {
    const text = (claimToVerify || claim).trim();
    if (!text) {
      setError("Please enter a non-empty financial claim to verify.");
      return;
    }

    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const response = await verifyClaim({ claim: text });
      setResult(response);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Verification failed. Check whether the Python API is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  };

  const nliActive = Boolean(result?.verifier_engine?.includes("NLI"));
  const evidenceByDocId = new Map<
    number,
    { item: CandidateEvidence; category: "supporting" | "contradicting" | "neutral" }
  >();
  if (result && nliActive) {
    for (const [category, items] of [
      ["supporting", result.supporting_evidence || []],
      ["contradicting", result.contradicting_evidence || []],
      ["neutral", result.neutral_evidence || []],
    ] as const) {
      for (const item of items) {
        if (typeof item.doc_id === "number") evidenceByDocId.set(item.doc_id, { item, category });
      }
    }
  }

  const selectedEvidence = (result?.selected_evidence_docs ?? []).map((document, index) => {
    const nli = typeof document.doc_id === "number" ? evidenceByDocId.get(document.doc_id) : undefined;
    return {
      ...document,
      publisher: document.publisher || nli?.item.publisher,
      title: document.title || nli?.item.title,
      entailment_score: document.entailment_score ?? nli?.item.entailment_score,
      contradiction_score: document.contradiction_score ?? nli?.item.contradiction_score,
      neutral_score: document.neutral_score ?? nli?.item.neutral_score,
      assigned_category: (document.assigned_category || nli?.category || "retrieved") as CandidateEvidence["assigned_category"],
      rank: index + 1,
    };
  });

  return (
    <div className="space-y-6 pt-2">
      {/* Workspace Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4" />
            FinGuard Phase 2 Verification Platform
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Financial Claim Verification &amp; Evidence Grounding
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1 max-w-3xl">
          Enter a real-world financial statement to execute the complete Phase 2 verification pipeline: claim entity extraction, finance-aware multi-query expansion, BM25 + Weighted RRF retrieval, Cross-Encoder reranking, and NLI verification over the FinFact corpus.
        </p>
      </div>

      {/* Main Submission Panel */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleVerify();
          }}
          className="space-y-4"
        >
          <div className="relative">
            <label className="block text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider mb-2">
              Financial Claim Input
            </label>
            <textarea
              rows={3}
              value={claim}
              onChange={(e) => setClaim(e.target.value)}
              placeholder="e.g. Amazon reported a 20% increase in revenue in FY2025 compared with FY2024."
              className="w-full p-4 text-sm bg-[#F8F7FD] dark:bg-[#211E3E] text-[#171638] dark:text-white rounded-2xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all resize-none placeholder:text-[#85839A]"
            />
          </div>

          {/* Sample Chips */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-bold text-[#85839A]">Representative benchmark claims:</span>
            {SAMPLE_CLAIMS.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setClaim(sample);
                  handleVerify(sample);
                }}
                className="text-[11px] px-2.5 py-1 rounded-xl bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary hover:text-white text-primary font-medium transition-colors text-left truncate max-w-xs"
              >
                {sample.slice(0, 38)}...
              </button>
            ))}
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-2 border-t border-[#E9E7F2] dark:border-[#2D294F]">
            <span className="text-[11px] text-[#85839A]">
              Phase 2 Production Pipeline · Closed FinFact Evidence Corpus (12,941 sentences)
            </span>
            <button
              type="submit"
              disabled={loading}
              className="px-6 py-2.5 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white text-xs font-bold shadow-md hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:pointer-events-none flex items-center gap-2"
            >
              {loading ? (
                <>Verifying Claim...</>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-emerald-300" />
                  <span>Verify Claim</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-400 text-xs flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <div>
            <h4 className="font-bold">Verification Notice</h4>
            <p className="mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && <LoadingState message="Executing FinGuard Phase 2 Pipeline (Retrieval, Reranking &amp; NLI Verification)..." />}

      {/* Verification Results Panel */}
      {result && !loading && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-200">
          {/* Main Verdict Banner Card */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="space-y-2.5 max-w-3xl">
              <div className="flex flex-wrap items-center gap-3">
                <VerdictBadge verdict={result.verdict} size="lg" />
                <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-[#F4F3FA] dark:bg-[#28244D] text-[#615F79] dark:text-[#B5B2CD] border border-[#E9E7F2] dark:border-[#2D294F]">
                  Engine: {result.verifier_engine || "FinGuard Verifier"}
                </span>
              </div>

              <h2 className="text-base md:text-lg font-bold text-[#171638] dark:text-white leading-snug">
                &ldquo;{result.claim}&rdquo;
              </h2>

              <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
                {result.explanation}
              </p>

              {result.is_uncertain && result.uncertainty_reason && (
                <p className="text-xs font-medium text-amber-700 dark:text-amber-300 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Uncertainty: {result.uncertainty_reason}</span>
                </p>
              )}

              {/* Raw Model Scores (labeled accurately as model score, not calibrated confidence) */}
              <div className="pt-2 border-t border-[#E9E7F2] dark:border-[#2D294F] flex flex-wrap items-center gap-3 text-xs">
                <span className="text-[#85839A]">Raw Model Score:</span>
                <span className="font-mono font-bold text-[#171638] dark:text-white bg-[#F8F7FD] dark:bg-[#211E3E] px-2 py-0.5 rounded border border-[#E9E7F2] dark:border-[#2D294F]">
                  {result.confidence.toFixed(4)}
                </span>
                <span className="text-[10px] text-[#85839A] italic">
                  (Heuristic model output, uncalibrated)
                </span>
                {result.class_probabilities && (
                  <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    <span className="text-[#85839A]">Class scores:</span>
                    <span>False: <strong>{Number(result.class_probabilities.false || 0).toFixed(3)}</strong></span>
                    <span>·</span>
                    <span>True: <strong>{Number(result.class_probabilities.true || 0).toFixed(3)}</strong></span>
                    <span>·</span>
                    <span>NEI: <strong>{Number(result.class_probabilities.NEI || 0).toFixed(3)}</strong></span>
                  </div>
                )}
              </div>
            </div>

            <div className="shrink-0 border-t md:border-t-0 md:border-l border-[#E9E7F2] dark:border-[#2D294F] pt-4 md:pt-0 md:pl-6 text-right">
              <span className="text-[10px] uppercase font-bold text-[#85839A] block">
                {nliActive ? "NLI Stance Breakdown" : "Selected Evidence"}
              </span>
              <span className="text-base font-extrabold text-[#171638] dark:text-white">
                {nliActive
                  ? `${result.supporting_evidence.length} support · ${result.contradicting_evidence.length} contradict · ${result.neutral_evidence.length} neutral`
                  : `${selectedEvidence.length} passages`}
              </span>
              <span className="text-[11px] text-[#85839A] block mt-1">
                Candidate pool: {result.total_candidates_retrieved ?? selectedEvidence.length}
              </span>
            </div>
          </div>

          {/* Contradiction / Conflict Warning Alert */}
          {result.is_conflicting && (
            <div className="p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-300 text-xs flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 text-amber-600" />
              <div>
                <h4 className="font-bold">Multi-Source Evidence Conflict Detected</h4>
                <p className="mt-0.5">
                  The backend identified retrieved passages supporting and contradicting the claim across independent sources. Following FinGuard aggregation protocol, the final verdict is gated to NEI.
                </p>
              </div>
            </div>
          )}

          {/* Pipeline Transparency: 8 Processing Stages */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary" />
                <h3 className="text-sm font-bold uppercase tracking-wider text-[#171638] dark:text-white">
                  Pipeline Transparency: 8 Processing Stages
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                className="text-xs font-semibold text-primary hover:text-primary-hover flex items-center gap-1 transition-colors"
              >
                <span>{showTechnicalDetails ? "Collapse stages" : "Expand all 8 stages"}</span>
                {showTechnicalDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </div>

            {showTechnicalDetails && (
              <div className="space-y-4 text-xs">
                {/* Stage 1 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">1</span>
                      Claim Processing &amp; Financial Entity Extraction
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">Completed</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2 pt-1 text-[11px]">
                    <div>
                      <span className="text-[#85839A] block">Organizations:</span>
                      <strong className="text-[#171638] dark:text-white">
                        {result.parsed_claim?.organizations?.join(", ") || "None"}
                      </strong>
                    </div>
                    <div>
                      <span className="text-[#85839A] block">Financial Metrics:</span>
                      <strong className="text-[#171638] dark:text-white">
                        {result.parsed_claim?.financial_metrics?.join(", ") || "None"}
                      </strong>
                    </div>
                    <div>
                      <span className="text-[#85839A] block">Monetary / Percents:</span>
                      <strong className="text-emerald-600 dark:text-emerald-400 font-mono">
                        {[...(result.parsed_claim?.monetary_amounts || []), ...(result.parsed_claim?.percentages || [])].join(", ") || "None"}
                      </strong>
                    </div>
                    <div>
                      <span className="text-[#85839A] block">Periods &amp; Directions:</span>
                      <strong className="text-primary">
                        {[...(result.parsed_claim?.periods || []), ...(result.parsed_claim?.directional_terms || []).map(d => d.term)].join(", ") || "Unspecified"}
                      </strong>
                    </div>
                  </div>
                </div>

                {/* Stage 2 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">2</span>
                      Financial Query Construction
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">Formulated</span>
                  </div>
                  <div className="pt-1">
                    <span className="text-[#85839A] text-[11px] block">Search Query Formulation:</span>
                    <code className="text-primary font-mono text-[11px] break-words">
                      {result.search_query || result.query_info?.original_query || result.claim}
                    </code>
                  </div>
                </div>

                {/* Stage 3 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">3</span>
                      Finance-Aware Synonym &amp; Multi-Query Expansion
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                      {result.query_info?.multi_queries?.length ? `${result.query_info.multi_queries.length} queries generated` : "Base query"}
                    </span>
                  </div>
                  {result.query_info?.multi_queries && result.query_info.multi_queries.length > 0 ? (
                    <div className="mt-2 space-y-1.5">
                      {result.query_info.multi_queries.map((q, idx) => (
                        <div key={idx} className="flex flex-wrap items-center justify-between gap-2 p-2 rounded-xl bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] text-[11px]">
                          <div className="flex items-center gap-2 max-w-xl truncate">
                            <span className="px-2 py-0.5 rounded bg-[#F4F3FA] dark:bg-[#28244D] font-bold text-primary text-[10px]">
                              {q.group}
                            </span>
                            <span className="font-mono text-[#171638] dark:text-white truncate">{q.query}</span>
                          </div>
                          <div className="flex items-center gap-3 text-[#85839A] shrink-0 font-mono text-[10px]">
                            <span>weight={q.weight}</span>
                            <span>{q.purpose}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[11px] text-[#85839A] pt-1">Single financial search query evaluated.</p>
                  )}
                </div>

                {/* Stage 4 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">4</span>
                      BM25 Retrieval &amp; Weighted Reciprocal Rank Fusion
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400 font-mono">
                      {result.retrieval_engine || "bm25_weighted_rrf"}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    Retrieved <strong>{result.total_candidates_retrieved ?? selectedEvidence.length}</strong> candidate evidence passages from the 12,943-sentence FinFact inverted index using Weighted RRF fusion (k=60).
                  </p>
                </div>

                {/* Stage 5 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">5</span>
                      Cross-Encoder Joint Reranking
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                      {result.reranker_active ? "cross-encoder/ms-marco-MiniLM-L-6-v2" : "Passthrough"}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    Evaluated retrieved candidate pairs jointly with neural cross-attention and selected top <strong>{selectedEvidence.length}</strong> highest-relevance evidence passages for verification.
                  </p>
                </div>

                {/* Stage 6 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">6</span>
                      NLI Verification
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                      {result.verifier_engine || "Pretrained NLI"}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    Scored each premise-hypothesis pair independently into <em>entailment</em>, <em>contradiction</em>, and <em>neutral</em> logits. Classified {result.supporting_evidence?.length ?? 0} supporting, {result.contradicting_evidence?.length ?? 0} contradicting, and {result.neutral_evidence?.length ?? 0} neutral passages.
                  </p>
                </div>

                {/* Stage 7 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">7</span>
                      Multi-Sentence Evidence Aggregation
                    </span>
                    <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
                      {result.is_conflicting ? "Conflict Gated" : "Aggregation Complete"}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    Applied FinGuard multi-sentence aggregation protocol: checked cross-source contradiction/support concordance, evaluated dominant stance score against decision threshold (0.60), and gated conflicting evidence to NEI.
                  </p>
                </div>

                {/* Stage 8 */}
                <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-full bg-primary text-white text-[10px] flex items-center justify-center font-bold">8</span>
                      Final Decision Verdict
                    </span>
                    <VerdictBadge verdict={result.verdict} size="sm" />
                  </div>
                  <p className="text-[11px] text-[#615F79] dark:text-[#B5B2CD]">
                    Verdict display: <strong>{result.verdict_display || result.verdict}</strong> · {result.explanation}
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Evidence returned by the backend, in post-reranking order */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-[#171638] dark:text-white flex items-center gap-2">
                <Layers className="w-4 h-4 text-primary" />
                <span>Grounded Evidence Passages ({selectedEvidence.length})</span>
              </h3>
              <span className="text-xs text-[#85839A]">
                Ranked by Cross-Encoder relevance score
              </span>
            </div>

            {selectedEvidence.length > 0 ? (
              <div className="space-y-3">
                {selectedEvidence.map((evidence, index) => (
                  <EvidenceCard
                    key={evidence.doc_id ?? index}
                    evidence={evidence}
                    category={nliActive ? evidence.assigned_category : "retrieved"}
                    rank={index + 1}
                  />
                ))}
              </div>
            ) : (
              <div className="p-8 text-center bg-white dark:bg-[#18162E] rounded-3xl border border-[#E9E7F2] dark:border-[#2D294F] text-xs text-[#85839A]">
                The backend returned no usable evidence passages for this claim.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function VerifyPage() {
  return (
    <Suspense fallback={<LoadingState message="Loading Verification Workspace..." />}>
      <VerifyContent />
    </Suspense>
  );
}
