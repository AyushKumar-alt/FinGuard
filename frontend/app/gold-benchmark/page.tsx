"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Database,
  CheckCircle2,
  FileCheck,
  ShieldCheck,
  ArrowRight,
  Lock,
  Layers,
  Sparkles,
  Info,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { CircularProgress } from "@/components/ui/CircularProgress";
import { fetchGoldBenchmark } from "@/lib/api";
import { GoldBenchmarkOverview } from "@/lib/types";

export default function GoldBenchmarkPage() {
  const [data, setData] = useState<{ v1: GoldBenchmarkOverview; v2: GoldBenchmarkOverview } | null>(null);
  const [activeVersion, setActiveVersion] = useState<"v2" | "v1">("v2");

  useEffect(() => {
    fetchGoldBenchmark().then(setData).catch(console.error);
  }, []);

  const current = activeVersion === "v2" ? data?.v2 : data?.v1;

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
              <Database className="w-4 h-4" />
              Benchmark Governance &amp; Versioning
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
            Gold-Evidence Relevance Benchmark
          </h1>
          <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
            Formal data contracts, candidate pooling depth-20, dual-assessor automation, and cryptographic SHA-256 manifests.
          </p>
        </div>

        {/* Version Switcher Pill */}
        <div className="flex items-center p-1 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] shadow-sm">
          <button
            onClick={() => setActiveVersion("v2")}
            className={`px-4 py-1.5 rounded-full text-xs font-bold transition-colors ${
              activeVersion === "v2"
                ? "bg-[#171638] text-white dark:bg-white dark:text-[#171638] shadow-sm"
                : "text-[#85839A] hover:text-[#171638] dark:hover:text-white"
            }`}
          >
            Benchmark v2.0 (Active)
          </button>
          <button
            onClick={() => setActiveVersion("v1")}
            className={`px-4 py-1.5 rounded-full text-xs font-bold transition-colors ${
              activeVersion === "v1"
                ? "bg-[#171638] text-white dark:bg-white dark:text-[#171638] shadow-sm"
                : "text-[#85839A] hover:text-[#171638] dark:hover:text-white"
            }`}
          >
            Benchmark v1.0 (Frozen)
          </button>
        </div>
      </div>

      {/* KPI Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Candidate Pairs in Pool"
          value="4,891"
          subtitle="Depth 20 across 8 architectures (3,839 units)"
          icon={Layers}
          badge="75 Pilot Claims"
          badgeColor="purple"
        />
        <StatCard
          title="Concordant Non-relevant"
          value="4,374"
          subtitle="Both neural & rubric rejected (Grade 0)"
          icon={CheckCircle2}
          badge="89.4% Automated"
          badgeColor="emerald"
        />
        <StatCard
          title="Human Review Queue"
          value="107"
          subtitle="Prioritized borderline / discordant cases"
          icon={FileCheck}
          badge="97.6% Saved"
          badgeColor="emerald"
        />
        <StatCard
          title="Judged Coverage@10"
          value={activeVersion === "v2" ? "93.73%" : "19.47%"}
          subtitle={activeVersion === "v2" ? "Definitive judged coverage" : "Severe incompleteness"}
          icon={ShieldCheck}
          badge={activeVersion === "v2" ? "Validated" : "Incomplete"}
          badgeColor={activeVersion === "v2" ? "emerald" : "amber"}
        />
      </div>

      {/* Version Comparison Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-[#171638] dark:text-white">
              Benchmark Candidate Inventory &amp; Distribution
            </h3>
            <p className="text-xs text-[#85839A]">
              Before (v1.0.0) vs. After (v2.0.0 Audited &amp; Automated) candidate breakdown.
            </p>
          </div>
          <Link
            href="/review-queue"
            className="px-4 py-2 rounded-full bg-[#171638] text-white text-xs font-bold shadow-sm hover:scale-105 transition-transform flex items-center gap-1.5"
          >
            <span>Open Human Review Queue</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#E9E7F2] dark:border-[#2D294F] text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                <th className="py-3 px-3">Category / Metric</th>
                <th className="py-3 px-3">Benchmark v1.0.0</th>
                <th className="py-3 px-3">Benchmark v2.0.0</th>
                <th className="py-3 px-3">Scientific Rationale</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E9E7F2] dark:divide-[#2D294F] text-xs">
              <tr className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E]">
                <td className="py-3 px-3 font-semibold text-[#171638] dark:text-white">Total Pooled Candidate Pairs</td>
                <td className="py-3 px-3 font-mono">4,891</td>
                <td className="py-3 px-3 font-mono font-bold text-primary">4,891</td>
                <td className="py-3 px-3 text-[#85839A]">Pool identically preserved (depth-20 across 8 systems)</td>
              </tr>
              <tr className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E]">
                <td className="py-3 px-3 font-semibold text-[#171638] dark:text-white">Silver Reference Pairs (Grade 3)</td>
                <td className="py-3 px-3 font-mono">410 (8.38%)</td>
                <td className="py-3 px-3 font-mono font-bold text-emerald-600">410 (8.38%)</td>
                <td className="py-3 px-3 text-[#85839A]">Preserved verbatim with source URLs and IDs intact</td>
              </tr>
              <tr className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E]">
                <td className="py-3 px-3 font-semibold text-[#171638] dark:text-white">Concordant Non-relevant (Grade 0)</td>
                <td className="py-3 px-3 font-mono text-[#85839A]">0 (0.0%)</td>
                <td className="py-3 px-3 font-mono font-bold text-emerald-600">4,374 (89.43%)</td>
                <td className="py-3 px-3 text-[#85839A]">Dual-assessor consensus (cross-encoder score &lt; 0.15 + 0 rubric)</td>
              </tr>
              <tr className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E]">
                <td className="py-3 px-3 font-semibold text-[#171638] dark:text-white">Unjudged Candidates in Queue</td>
                <td className="py-3 px-3 font-mono text-rose-500 font-bold">4,480 (91.60%)</td>
                <td className="py-3 px-3 font-mono font-bold text-primary">107 (2.19%)</td>
                <td className="py-3 px-3 text-[#85839A]">Isolated for targeted human review (97.6% effort reduction)</td>
              </tr>
              <tr className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E]">
                <td className="py-3 px-3 font-semibold text-[#171638] dark:text-white">Judged Coverage@10</td>
                <td className="py-3 px-3 font-mono text-amber-500">19.47%</td>
                <td className="py-3 px-3 font-mono font-bold text-emerald-600">93.73%</td>
                <td className="py-3 px-3 text-[#85839A]">Corrected denominator bug; true definitive coverage reported</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Buckley & Voorhees Incomplete Information Sensitivity Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
          <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-primary" />
            <span>Incomplete Judgment Sensitivity (bpref)</span>
          </h4>
          <p className="text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
            When 91.6% of candidates were unjudged in v1, Cranfield MRR treated novel retrieved passages as non-relevant. Under Buckley &amp; Voorhees (2004) <code className="text-primary font-mono font-bold">bpref</code>, unjudged passages are completely discarded rather than assumed to be negative:
          </p>
          <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] font-mono text-[11px] text-[#171638] dark:text-white">
            bpref = (1/R) * Σ [ 1 - min(n_r, R) / R ]
          </div>
          <p className="text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
            In Benchmark v2, Cross-Encoder Reranker achieves highest bpref (0.3396) followed by Hybrid RRF (0.3296), confirming architecture rankings remain robust across both complete and incomplete evaluation regimes.
          </p>
        </div>

        {/* Cryptographic Manifest Checksums */}
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
          <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider flex items-center gap-2">
            <Lock className="w-4 h-4 text-emerald-500" />
            <span>Cryptographic Checksum Verification</span>
          </h4>
          <p className="text-xs text-[#85839A] leading-relaxed">
            Benchmark versions are strictly immutable. Artifact hashes are verified byte-for-byte using SHA-256:
          </p>
          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
              <span className="font-bold block text-[#171638] dark:text-white text-[11px]">Benchmark v1.0.0 Manifest</span>
              <code className="text-[10px] text-[#85839A] break-all font-mono">
                70d04c10bb47b42aa15c7a52ca1b9ff78a4ea9e530045d6f1bfd0b497b7b952b
              </code>
            </div>
            <div className="p-2.5 rounded-xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
              <span className="font-bold block text-[#171638] dark:text-white text-[11px]">Benchmark v2.0.0 Manifest</span>
              <code className="text-[10px] text-primary break-all font-mono">
                683b5675c2e3dddfd7d91cb61d9cf38fc6f272c72b226e6ef10ca4eb54d39f70
              </code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

