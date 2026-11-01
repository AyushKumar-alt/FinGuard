"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  ShieldCheck,
  Search,
  Sparkles,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Tag,
  Clock,
  Layers,
  ArrowRight,
  Info,
} from "lucide-react";
import { VerdictBadge } from "@/components/ui/VerdictBadge";
import { EvidenceCard } from "@/components/ui/EvidenceCard";
import { LoadingState } from "@/components/ui/LoadingState";
import { verifyClaim } from "@/lib/api";
import { ClaimVerificationResponse } from "@/lib/types";

const SAMPLE_CLAIMS = [
  "Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?",
  "Amazon paid zero dollars in federal corporate income tax for two consecutive years.",
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

  // Advanced Controls
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [retrievalMode, setRetrievalMode] = useState("bm25");
  const [useQueryExpansion, setUseQueryExpansion] = useState(true);
  const [useReranker, setUseReranker] = useState(true);
  const [useNli, setUseNli] = useState(true);
  const [topKRetrieve, setTopKRetrieve] = useState(15);
  const [topKRerank, setTopKRerank] = useState(4);

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
    setLoading(true);

    try {
      const response = await verifyClaim({
        claim: text,
        retrieval_mode: retrievalMode,
        use_query_expansion: useQueryExpansion,
        use_reranker: useReranker,
        use_nli: useNli,
        top_k_retrieve: topKRetrieve,
        top_k_rerank: topKRerank,
      });
      setResult(response);
    } catch (err: any) {
      setError(err.message || "Failed to verify claim. Check if API backend is reachable.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 pt-2">
      {/* Workspace Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4" />
            Verification Workspace
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Verify Financial Claims
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
          Ground financial assertions against 12,941 candidate evidence sentences with neural cross-encoder reranking.
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
            <span className="text-[11px] font-bold text-[#85839A]">Representative FinFact claims:</span>
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
                {sample.slice(0, 36)}...
              </button>
            ))}
          </div>

          {/* Advanced Pipeline Toggles */}
          <div className="pt-2 border-t border-[#E9E7F2] dark:border-[#2D294F]">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs font-bold text-[#85839A] hover:text-[#171638] dark:hover:text-white flex items-center gap-1.5 transition-colors"
            >
              <Sliders className="w-3.5 h-3.5 text-primary" />
              <span>Advanced Pipeline Controls</span>
              {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>

            {showAdvanced && (
              <div className="mt-3 p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div>
                  <label className="block font-bold text-[#171638] dark:text-white mb-1">
                    Retrieval Strategy
                  </label>
                  <select
                    value={retrievalMode}
                    onChange={(e) => setRetrievalMode(e.target.value)}
                    className="w-full p-2 bg-white dark:bg-[#18162E] text-[#171638] dark:text-white rounded-xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="bm25">Config A: BM25 (Inverted Index)</option>
                    <option value="dense">Config E: Pretrained Dense (MiniLM)</option>
                    <option value="hybrid_dense">Config F: Hybrid RRF (BM25 + Dense)</option>
                    <option value="tfidf">Config B: TF-IDF Cosine</option>
                    <option value="svd">Config C: Dense SVD Projection</option>
                  </select>
                </div>

                <div>
                  <label className="block font-bold text-[#171638] dark:text-white mb-1">
                    Verifier Model
                  </label>
                  <select
                    value={useNli ? "nli" : "baseline"}
                    onChange={(e) => setUseNli(e.target.value === "nli")}
                    className="w-full p-2 bg-white dark:bg-[#18162E] text-[#171638] dark:text-white rounded-xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="nli">Pretrained NLI (cross-encoder/nli-distilroberta)</option>
                    <option value="baseline">Calibrated Logistic Regression (In-Domain)</option>
                  </select>
                </div>

                <div className="flex flex-col justify-center space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer font-medium text-[#171638] dark:text-white">
                    <input
                      type="checkbox"
                      checked={useQueryExpansion}
                      onChange={(e) => setUseQueryExpansion(e.target.checked)}
                      className="rounded text-primary focus:ring-primary"
                    />
                    <span>Query Expansion</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer font-medium text-[#171638] dark:text-white">
                    <input
                      type="checkbox"
                      checked={useReranker}
                      onChange={(e) => setUseReranker(e.target.checked)}
                      className="rounded text-primary focus:ring-primary"
                    />
                    <span>Cross-Encoder Reranker</span>
                  </label>
                </div>

                <div>
                  <div className="flex justify-between font-bold text-[#171638] dark:text-white mb-1">
                    <span>Candidates (Top-K)</span>
                    <span className="text-primary font-mono">{topKRetrieve}</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={25}
                    value={topKRetrieve}
                    onChange={(e) => setTopKRetrieve(parseInt(e.target.value, 10))}
                    className="w-full accent-primary"
                  />
                  <div className="text-[10px] text-[#85839A] mt-1">Grounded passages: {topKRerank}</div>
                </div>
              </div>
            )}
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-2">
            <span className="text-[11px] text-[#85839A]">
              Leakage boundary: target claim justifications are strictly masked from candidates.
            </span>
            <button
              type="submit"
              disabled={loading || !claim.trim()}
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
      {loading && <LoadingState message="Executing FinGuard End-to-End Pipeline..." />}

      {/* Verification Results Panel */}
      {result && !loading && (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-2 duration-200">
          {/* Main Verdict Banner Card */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div className="space-y-2 max-w-3xl">
              <div className="flex items-center gap-3">
                <VerdictBadge verdict={result.verdict} size="lg" />
                <span className="text-xs font-bold text-[#85839A]">
                  Confidence: <strong className="text-[#171638] dark:text-white">{(result.confidence * 100).toFixed(1)}%</strong>
                </span>
                {result.latency_ms && (
                  <span className="text-xs font-mono text-[#85839A] bg-[#F4F3FA] dark:bg-[#28244D] px-2.5 py-0.5 rounded-full">
                    {result.latency_ms.toFixed(0)} ms
                  </span>
                )}
              </div>

              <h2 className="text-base md:text-lg font-bold text-[#171638] dark:text-white leading-snug">
                &ldquo;{result.claim}&rdquo;
              </h2>

              <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
                {result.explanation}
              </p>
            </div>

            {/* Quick summary badges */}
            <div className="flex flex-row md:flex-col items-center md:items-end gap-2 shrink-0 self-stretch justify-between md:justify-center border-t md:border-t-0 md:border-l border-[#E9E7F2] dark:border-[#2D294F] pt-4 md:pt-0 md:pl-6">
              <div className="text-right">
                <span className="text-[10px] uppercase font-bold text-[#85839A] block">Supporting</span>
                <span className="text-base font-extrabold text-[#239B79]">
                  {result.supporting_evidence.length} passages
                </span>
              </div>
              <div className="text-right">
                <span className="text-[10px] uppercase font-bold text-[#85839A] block">Contradicting</span>
                <span className="text-base font-extrabold text-[#D95465]">
                  {result.contradicting_evidence.length} passages
                </span>
              </div>
            </div>
          </div>

          {/* Contradiction / Conflict Warning Alert */}
          {result.is_conflicting && (
            <div className="p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-300 text-xs flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 shrink-0 text-amber-600" />
              <div>
                <h4 className="font-bold">Multi-Source Evidence Conflict Detected</h4>
                <p className="mt-0.5">
                  The candidate evidence contains mutually opposing statements across different sources. FinGuard safely gates this claim to NEI rather than hallucinating a false consensus.
                </p>
              </div>
            </div>
          )}

          {/* Grounded Evidence Breakdown Sections */}
          <div className="space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-[#171638] dark:text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-primary" />
              <span>Grounded Evidence Passages</span>
            </h3>

            {/* Supporting Evidence */}
            {result.supporting_evidence.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center gap-2 text-xs font-bold text-[#239B79]">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Supporting Evidence ({result.supporting_evidence.length})</span>
                </div>
                <div className="space-y-2.5">
                  {result.supporting_evidence.map((ev, i) => (
                    <EvidenceCard key={i} evidence={ev} category="supporting" rank={i + 1} />
                  ))}
                </div>
              </div>
            )}

            {/* Contradicting Evidence */}
            {result.contradicting_evidence.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center gap-2 text-xs font-bold text-[#D95465]">
                  <XCircle className="w-4 h-4" />
                  <span>Contradicting Evidence ({result.contradicting_evidence.length})</span>
                </div>
                <div className="space-y-2.5">
                  {result.contradicting_evidence.map((ev, i) => (
                    <EvidenceCard key={i} evidence={ev} category="contradicting" rank={i + 1} />
                  ))}
                </div>
              </div>
            )}

            {/* Neutral / Retrieved Background */}
            {result.neutral_evidence.length > 0 && (
              <div className="space-y-2.5">
                <div className="flex items-center gap-2 text-xs font-bold text-[#7065E8]">
                  <HelpCircle className="w-4 h-4" />
                  <span>Retrieved Neutral / Contextual Evidence ({result.neutral_evidence.length})</span>
                </div>
                <div className="space-y-2.5">
                  {result.neutral_evidence.map((ev, i) => (
                    <EvidenceCard key={i} evidence={ev} category="neutral" rank={i + 1} />
                  ))}
                </div>
              </div>
            )}

            {result.supporting_evidence.length === 0 &&
              result.contradicting_evidence.length === 0 &&
              result.neutral_evidence.length === 0 && (
                <div className="p-8 text-center bg-white dark:bg-[#18162E] rounded-3xl border border-[#E9E7F2] dark:border-[#2D294F] text-xs text-[#85839A]">
                  No directly grounding evidence sentences met the relevance threshold for this query.
                </div>
              )}
          </div>

          {/* Investigation Details: Parsed Financial Entities & Query Expansions */}
          {result.parsed_claim && (
            <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
              <h3 className="text-sm font-bold text-[#171638] dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                <Tag className="w-4 h-4 text-primary" />
                <span>Financial Claim Understanding Breakdown</span>
              </h3>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div className="p-3 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <span className="text-[10px] font-bold text-[#85839A] uppercase block mb-1">
                    Corporate Entities
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {result.parsed_claim.companies?.length ? (
                      result.parsed_claim.companies.map((c, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-md bg-white dark:bg-[#18162E] font-semibold text-[#171638] dark:text-white border border-[#E9E7F2] dark:border-[#2D294F]">
                          {c}
                        </span>
                      ))
                    ) : (
                      <span className="text-[#85839A] italic">None identified</span>
                    )}
                  </div>
                </div>

                <div className="p-3 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <span className="text-[10px] font-bold text-[#85839A] uppercase block mb-1">
                    Financial Metrics
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {result.parsed_claim.metrics?.length ? (
                      result.parsed_claim.metrics.map((m, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-md bg-white dark:bg-[#18162E] font-semibold text-[#171638] dark:text-white border border-[#E9E7F2] dark:border-[#2D294F]">
                          {m}
                        </span>
                      ))
                    ) : (
                      <span className="text-[#85839A] italic">None identified</span>
                    )}
                  </div>
                </div>

                <div className="p-3 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <span className="text-[10px] font-bold text-[#85839A] uppercase block mb-1">
                    Percentages &amp; Currencies
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {[...(result.parsed_claim.percentages || []), ...(result.parsed_claim.currencies || [])].length ? (
                      [...(result.parsed_claim.percentages || []), ...(result.parsed_claim.currencies || [])].map((v, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-md bg-white dark:bg-[#18162E] font-mono text-emerald-600 dark:text-emerald-400 font-semibold border border-[#E9E7F2] dark:border-[#2D294F]">
                          {v}
                        </span>
                      ))
                    ) : (
                      <span className="text-[#85839A] italic">None</span>
                    )}
                  </div>
                </div>

                <div className="p-3 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
                  <span className="text-[10px] font-bold text-[#85839A] uppercase block mb-1">
                    Fiscal Periods &amp; Direction
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {[...(result.parsed_claim.fiscal_periods || []), ...(result.parsed_claim.directional_keywords || [])].length ? (
                      [...(result.parsed_claim.fiscal_periods || []), ...(result.parsed_claim.directional_keywords || [])].map((d, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-md bg-white dark:bg-[#18162E] font-semibold text-primary border border-[#E9E7F2] dark:border-[#2D294F]">
                          {d}
                        </span>
                      ))
                    ) : (
                      <span className="text-[#85839A] italic">Unspecified</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Query Expansion Info */}
              {result.query_info && (
                <div className="mt-4 pt-3 border-t border-[#E9E7F2] dark:border-[#2D294F] flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-bold text-[#85839A]">Expanded Search Query:</span>
                  <code className="px-2 py-1 rounded-md bg-[#F4F3FA] dark:bg-[#28244D] text-primary font-mono text-[11px]">
                    {result.query_info.combined_query}
                  </code>
                </div>
              )}
            </div>
          )}
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
