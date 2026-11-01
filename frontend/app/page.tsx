"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ShieldCheck,
  Search,
  Database,
  Cpu,
  BarChart3,
  ArrowRight,
  Plus,
  FileCheck,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  TrendingUp,
  Layers,
  Sparkles,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { CircularProgress } from "@/components/ui/CircularProgress";
import { VerdictBadge } from "@/components/ui/VerdictBadge";
import { fetchSystemStatus, fetchInvestigations } from "@/lib/api";
import { SystemStatus } from "@/lib/types";

export default function OverviewPage() {
  const router = useRouter();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [investigations, setInvestigations] = useState<any[]>([]);
  const [quickClaim, setQuickClaim] = useState("");

  useEffect(() => {
    fetchSystemStatus().then(setStatus).catch(console.error);
    fetchInvestigations().then(setInvestigations).catch(console.error);
  }, []);

  const handleQuickSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (quickClaim.trim()) {
      router.push(`/verify?claim=${encodeURIComponent(quickClaim.trim())}`);
    }
  };

  return (
    <div className="space-y-6 pt-2">
      {/* Hero Greeting Section (Directly inspired by screenshot header) */}
      <div className="flex flex-col xl:flex-row items-start xl:items-center justify-between gap-6 pb-2">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              Evidence-Grounded Fact-Checking
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl lg:text-4xl font-extrabold text-[#171638] dark:text-white tracking-tight">
            Financial Claim Intelligence
          </h1>
          <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1 max-w-xl">
            Verify financial assertions, retrieve grounded evidence, and inspect neural cross-encoder reranking across the FinFact benchmark.
          </p>
        </div>

        {/* Top Feature Shortcut Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full xl:w-auto">
          {/* Quick Add / New Claim Tile */}
          <Link
            href="/verify"
            className="h-28 rounded-3xl bg-white dark:bg-[#18162E] border-2 border-dashed border-[#E9E7F2] dark:border-[#2D294F] hover:border-primary flex flex-col items-center justify-center p-3 text-center transition-all group hover:scale-[1.02] shadow-sm"
          >
            <div className="w-9 h-9 rounded-2xl bg-[#EEECFC] dark:bg-[#2B2564] text-primary flex items-center justify-center mb-1.5 group-hover:bg-primary group-hover:text-white transition-colors">
              <Plus className="w-5 h-5" />
            </div>
            <span className="text-xs font-bold text-[#171638] dark:text-white">Verify Claim</span>
            <span className="text-[10px] text-[#85839A]">New inquiry</span>
          </Link>

          {/* Feature 1 */}
          <Link
            href="/retrieval-lab"
            className="h-28 rounded-3xl bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] p-3 flex flex-col justify-between hover:shadow-card-hover transition-all hover:scale-[1.02]"
          >
            <div className="w-8 h-8 rounded-xl bg-purple-50 dark:bg-purple-950/40 text-primary flex items-center justify-center">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white">Retrieval Lab</h4>
              <p className="text-[10px] text-[#85839A]">8 configurations</p>
            </div>
          </Link>

          {/* Feature 2 */}
          <Link
            href="/gold-benchmark"
            className="h-28 rounded-3xl bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] p-3 flex flex-col justify-between hover:shadow-card-hover transition-all hover:scale-[1.02]"
          >
            <div className="w-8 h-8 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 flex items-center justify-center">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white">Benchmark v2</h4>
              <p className="text-[10px] text-[#85839A]">97.6% labor reduction</p>
            </div>
          </Link>

          {/* Feature 3 */}
          <Link
            href="/evidence"
            className="h-28 rounded-3xl bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] p-3 flex flex-col justify-between hover:shadow-card-hover transition-all hover:scale-[1.02]"
          >
            <div className="w-8 h-8 rounded-xl bg-amber-50 dark:bg-amber-950/40 text-amber-600 flex items-center justify-center">
              <Search className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white">Corpus Search</h4>
              <p className="text-[10px] text-[#85839A]">12,941 sentences</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="FinFact Corpus Units"
          value={status?.corpus_units?.toLocaleString() || "12,941"}
          subtitle="Deduplicated canonical sentences indexed"
          icon={Layers}
          badge="Inverted Index"
          badgeColor="purple"
        />
        <StatCard
          title="Best Retrieval MRR"
          value="0.8713"
          subtitle="Config G: BM25 + Cross-Encoder Reranker"
          icon={TrendingUp}
          badge="High Precision"
          badgeColor="emerald"
        />
        <StatCard
          title="Baseline Test Accuracy"
          value="74.51%"
          subtitle="Calibrated Logistic Regression on evidence"
          icon={ShieldCheck}
          badge="Macro F1: 0.718"
          badgeColor="purple"
        />
        <StatCard
          title="Review Effort Saved"
          value="97.6%"
          subtitle="4,374 concordant pairs auto-adjudicated"
          icon={FileCheck}
          badge="Benchmark v2"
          badgeColor="emerald"
        />
      </div>

      {/* Main Dashboard Layout: Left 2 Cols (Investigation input + Recent claims), Right 1 Col (Widgets & Callout) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (Span 2) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Quick Investigation Input Card */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-base font-bold text-[#171638] dark:text-white">
                  Start an Investigation
                </h3>
                <p className="text-xs text-[#85839A]">
                  Type a financial claim or pick a representative test example below.
                </p>
              </div>
              <span className="text-[11px] font-bold text-primary bg-[#EEECFC] dark:bg-[#2B2564] px-2.5 py-1 rounded-full">
                Interactive Verifier
              </span>
            </div>

            <form onSubmit={handleQuickSubmit} className="space-y-3">
              <div className="relative">
                <textarea
                  rows={3}
                  value={quickClaim}
                  onChange={(e) => setQuickClaim(e.target.value)}
                  placeholder="e.g. Did the US national debt increase by over $7 trillion under Donald Trump?"
                  className="w-full p-4 text-sm bg-[#F8F7FD] dark:bg-[#211E3E] text-[#171638] dark:text-white rounded-2xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary transition-all resize-none placeholder:text-[#85839A]"
                />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                <div className="flex flex-wrap items-center gap-1.5 text-xs text-[#85839A]">
                  <span className="text-[11px] font-semibold">Try sample:</span>
                  <button
                    type="button"
                    onClick={() => setQuickClaim("Amazon paid zero dollars in federal income tax for two consecutive years.")}
                    className="text-[11px] px-2 py-0.5 rounded-lg bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary/10 text-primary font-medium transition-colors"
                  >
                    Amazon 0$ tax
                  </button>
                  <button
                    type="button"
                    onClick={() => setQuickClaim("Did Biden Call for an 'End to Shareholder Capitalism' in July 2020?")}
                    className="text-[11px] px-2 py-0.5 rounded-lg bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary/10 text-primary font-medium transition-colors"
                  >
                    Biden Capitalism
                  </button>
                  <button
                    type="button"
                    onClick={() => setQuickClaim("Did the US national debt increase by over $7 trillion under Donald Trump?")}
                    className="text-[11px] px-2 py-0.5 rounded-lg bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary/10 text-primary font-medium transition-colors"
                  >
                    US Debt $7T
                  </button>
                </div>

                <button
                  type="submit"
                  disabled={!quickClaim.trim()}
                  className="px-5 py-2.5 rounded-full bg-[#171638] dark:bg-white text-white dark:text-[#171638] text-xs font-bold shadow-md hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:pointer-events-none flex items-center gap-1.5"
                >
                  <span>Verify with Pipeline</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </form>
          </div>

          {/* Recent Investigations Table Card */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-[#171638] dark:text-white">
                  Benchmark Claims & Investigations
                </h3>
                <p className="text-xs text-[#85839A]">
                  Curated financial inquiries evaluated across ground-truth journalistic sources.
                </p>
              </div>
              <Link
                href="/investigations"
                className="text-xs font-bold text-primary hover:text-primary-hover flex items-center gap-1"
              >
                <span>View all</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#E9E7F2] dark:border-[#2D294F] text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                    <th className="py-2.5 px-3">Claim Assertion</th>
                    <th className="py-2.5 px-3">Verdict</th>
                    <th className="py-2.5 px-3 hidden sm:table-cell">Source</th>
                    <th className="py-2.5 px-3 hidden md:table-cell">Evidence</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E9E7F2] dark:divide-[#2D294F] text-xs">
                  {investigations.slice(0, 5).map((item) => (
                    <tr key={item.id} className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E] transition-colors">
                      <td className="py-3 px-3 font-medium text-[#171638] dark:text-white max-w-xs truncate">
                        {item.claim}
                      </td>
                      <td className="py-3 px-3">
                        <VerdictBadge verdict={item.verdict} size="sm" />
                      </td>
                      <td className="py-3 px-3 text-[#85839A] hidden sm:table-cell">
                        {item.author}
                      </td>
                      <td className="py-3 px-3 text-[#85839A] hidden md:table-cell font-mono">
                        {item.evidence_count} sentences
                      </td>
                      <td className="py-3 px-3 text-right">
                        <Link
                          href={`/verify?claim=${encodeURIComponent(item.claim)}`}
                          className="px-2.5 py-1 rounded-full bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary hover:text-white text-primary font-bold text-[11px] transition-colors"
                        >
                          Verify
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right Column (Span 1): Donut gauges, Promo Callout, Benchmark Summary */}
        <div className="space-y-6">
          {/* Circular Donut Gauges (Matching screenshot's "90% Marketing", "65% Typography") */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
            <CircularProgress
              percentage={97.6}
              label="Human Labor Saved"
              category="Automation v2"
              note="4,374 non-relevant pairs verified by dual-assessor pipeline"
              color="emerald"
            />
            <CircularProgress
              percentage={74.5}
              label="Classifier Accuracy"
              category="Supervised Model"
              note="Calibrated balanced Logistic Regression on gold evidence"
              color="purple"
            />
          </div>

          {/* Saturated Purple Promo / Callout Card (Inspired directly by the "Go premium!" card in the screenshot) */}
          <div className="bg-[#5448C8] dark:bg-[#3B3294] text-white rounded-3xl p-6 shadow-xl relative overflow-hidden flex flex-col justify-between min-h-[220px]">
            {/* Background decorative glow */}
            <div className="absolute -top-12 -right-12 w-36 h-36 rounded-full bg-white/10 blur-xl pointer-events-none" />

            <div>
              <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-300">
                Independent Audit Complete
              </span>
              <h3 className="text-xl font-extrabold mt-1 tracking-tight">
                Benchmark v2.0 Released
              </h3>
              <p className="text-xs text-white/80 mt-2 leading-relaxed">
                Buckley &amp; Voorhees (2004) bpref metric implemented to ensure unbiased evaluation across incomplete judgments.
              </p>
            </div>

            <div className="mt-5 pt-3 border-t border-white/15 flex items-center justify-between">
              <span className="text-[11px] font-mono text-white/70">
                SHA-256 Verified
              </span>
              <Link
                href="/gold-benchmark"
                className="px-4 py-2 rounded-full bg-white text-[#5448C8] hover:bg-gray-100 text-xs font-extrabold shadow-md transition-transform hover:scale-105"
              >
                Inspect Audit
              </Link>
            </div>
          </div>

          {/* Model Status Card */}
          <div className="bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
            <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider mb-3">
              Active Engines &amp; Safeguards
            </h4>
            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between py-1 border-b border-[#E9E7F2] dark:border-[#2D294F]">
                <span className="text-[#85839A]">Dense Retrieval</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> all-MiniLM-L6-v2
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-[#E9E7F2] dark:border-[#2D294F]">
                <span className="text-[#85839A]">Cross-Encoder</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> ms-marco-MiniLM
                </span>
              </div>
              <div className="flex items-center justify-between py-1 border-b border-[#E9E7F2] dark:border-[#2D294F]">
                <span className="text-[#85839A]">NLI Verifier</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" /> nli-distilroberta
                </span>
              </div>
              <div className="flex items-center justify-between py-1">
                <span className="text-[#85839A]">Leakage Safeguard</span>
                <span className="font-bold text-primary">Zero Claim Leakage</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

