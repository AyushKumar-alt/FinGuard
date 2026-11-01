"use client";

import React, { useEffect, useState } from "react";
import {
  Cpu,
  TrendingUp,
  Clock,
  Layers,
  Sparkles,
  ArrowRight,
  Info,
  CheckCircle2,
  Sliders,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { fetchRetrievalMetrics } from "@/lib/api";
import { RetrievalMetric } from "@/lib/types";

export default function RetrievalLabPage() {
  const [metrics, setMetrics] = useState<RetrievalMetric[]>([]);
  const [selectedSort, setSelectedSort] = useState<keyof RetrievalMetric>("mrr");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    fetchRetrievalMetrics().then(setMetrics).catch(console.error);
  }, []);

  const sortedMetrics = [...metrics].sort((a, b) => {
    const valA = (a[selectedSort] as number) ?? 0;
    const valB = (b[selectedSort] as number) ?? 0;
    return sortAsc ? valA - valB : valB - valA;
  });

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <Cpu className="w-4 h-4" />
            IR Research Benchmarking
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Retrieval Lab: 8-Way Architecture Benchmark
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1 max-w-3xl">
          Controlled comparative evaluation of 8 information retrieval architectures over 506 held-out test claims across 12,943 candidate sentences with strict target-claim non-leakage masking.
        </p>
      </div>

      {/* Top Benchmark Summary Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Best MRR (Config G)"
          value="0.8713"
          subtitle="BM25 + Cross-Encoder Reranker"
          icon={TrendingUp}
          badge="High Precision"
          badgeColor="purple"
        />
        <StatCard
          title="Best Recall@10 (Config F)"
          value="0.5646"
          subtitle="Hybrid RRF (BM25 + Pretrained Dense)"
          icon={Layers}
          badge="Max Coverage"
          badgeColor="emerald"
        />
        <StatCard
          title="Fastest Dense Search"
          value="18.7 ms"
          subtitle="all-MiniLM-L6-v2 Cached Embeddings"
          icon={Clock}
          badge="Normalized Inner Prod"
          badgeColor="slate"
        />
        <StatCard
          title="Evaluated Pool"
          value="506 Claims"
          subtitle="Held-out standard test split (12,943 docs)"
          icon={Cpu}
          badge="Zero Leakage"
          badgeColor="purple"
        />
      </div>

      {/* Main Benchmark Table Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-[#171638] dark:text-white">
              Controlled IR Benchmark Performance Matrix
            </h3>
            <p className="text-xs text-[#85839A]">
              Click column headers to sort architectures by any metric.
            </p>
          </div>

          <div className="text-xs text-[#85839A] flex items-center gap-2">
            <span>Ranking criterion:</span>
            <span className="font-bold text-primary uppercase font-mono">
              {String(selectedSort)}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#E9E7F2] dark:border-[#2D294F] text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                <th className="py-3 px-3">Config</th>
                <th className="py-3 px-3">Architecture &amp; Model</th>
                <th
                  onClick={() => {
                    setSelectedSort("mrr");
                    setSortAsc(selectedSort === "mrr" ? !sortAsc : false);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  MRR {selectedSort === "mrr" && (sortAsc ? "↑" : "↓")}
                </th>
                <th
                  onClick={() => {
                    setSelectedSort("recall_at_5");
                    setSortAsc(selectedSort === "recall_at_5" ? !sortAsc : false);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  Recall@5 {selectedSort === "recall_at_5" && (sortAsc ? "↑" : "↓")}
                </th>
                <th
                  onClick={() => {
                    setSelectedSort("recall_at_10");
                    setSortAsc(selectedSort === "recall_at_10" ? !sortAsc : false);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  Recall@10 {selectedSort === "recall_at_10" && (sortAsc ? "↑" : "↓")}
                </th>
                <th
                  onClick={() => {
                    setSelectedSort("map");
                    setSortAsc(selectedSort === "map" ? !sortAsc : false);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  MAP {selectedSort === "map" && (sortAsc ? "↑" : "↓")}
                </th>
                <th
                  onClick={() => {
                    setSelectedSort("precision_at_5");
                    setSortAsc(selectedSort === "precision_at_5" ? !sortAsc : false);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  P@5 {selectedSort === "precision_at_5" && (sortAsc ? "↑" : "↓")}
                </th>
                <th
                  onClick={() => {
                    setSelectedSort("latency_ms");
                    setSortAsc(selectedSort === "latency_ms" ? !sortAsc : true);
                  }}
                  className="py-3 px-3 cursor-pointer hover:text-primary transition-colors"
                >
                  Latency {selectedSort === "latency_ms" && (sortAsc ? "↑" : "↓")}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E9E7F2] dark:divide-[#2D294F] text-xs">
              {sortedMetrics.map((row) => (
                <tr
                  key={row.config}
                  className={`hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E] transition-colors ${
                    row.is_best_mrr
                      ? "bg-purple-50/50 dark:bg-purple-950/20"
                      : row.is_best_recall
                      ? "bg-emerald-50/50 dark:bg-emerald-950/20"
                      : ""
                  }`}
                >
                  <td className="py-3.5 px-3 font-mono font-bold text-primary">
                    {row.config}
                  </td>
                  <td className="py-3.5 px-3">
                    <div className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span>{row.name}</span>
                      {row.is_best_mrr && (
                        <span className="text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-[#7065E8] text-white">
                          Best MRR
                        </span>
                      )}
                      {row.is_best_recall && (
                        <span className="text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-[#239B79] text-white">
                          Best Recall
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-[#85839A] font-normal">{row.strategy}</div>
                  </td>
                  <td className="py-3.5 px-3 font-mono font-bold text-[#171638] dark:text-white">
                    {row.mrr.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A] dark:text-[#A09EB7]">
                    {row.recall_at_5.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono font-bold text-[#171638] dark:text-white">
                    {row.recall_at_10.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A] dark:text-[#A09EB7]">
                    {row.map.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A] dark:text-[#A09EB7]">
                    {row.precision_at_5.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A] dark:text-[#A09EB7]">
                    {row.latency_ms ? `${row.latency_ms.toFixed(1)} ms` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scientific Insights & Limitations Disclaimer */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
          <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            <span>Key Information Retrieval Insights</span>
          </h4>
          <ul className="space-y-2 text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
            <li>
              • <strong className="text-[#171638] dark:text-white">Cross-Encoder Maximizes Top-Rank Precision:</strong> Config G achieves the highest MRR (0.8713) and MAP (0.4803) through all-to-all cross-attention between claim and evidence tokens.
            </li>
            <li>
              • <strong className="text-[#171638] dark:text-white">Dense-Lexical Hybrid Maximizes Recall:</strong> Config F achieves Recall@10 of 0.5646, effectively bridging lexical exact-match sensitivity with embedding paraphrase capture.
            </li>
            <li>
              • <strong className="text-[#171638] dark:text-white">Direction Preservation:</strong> Query expansion (Config H) improves recall without polarity reversals, ensuring assertions are not inverted.
            </li>
          </ul>
        </div>

        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
          <h4 className="text-xs font-bold text-[#171638] dark:text-white uppercase tracking-wider flex items-center gap-1.5">
            <Info className="w-4 h-4 text-primary" />
            <span>Scientific Evaluation Limitations</span>
          </h4>
          <p className="text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
            Traditional Cranfield evaluation treats unjudged retrieved candidates as non-relevant (Grade 0). In collections with incomplete annotations, this artificially penalizes retrievers that discover novel, previously unannotated evidence.
          </p>
          <p className="text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
            FinGuard Benchmark v2 repairs this via dual-assessor automation and Buckley &amp; Voorhees (2004) <code className="text-primary font-mono font-bold">bpref</code> evaluation.
          </p>
        </div>
      </div>
    </div>
  );
}

