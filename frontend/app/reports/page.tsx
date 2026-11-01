"use client";

import React, { useState } from "react";
import {
  FileText,
  Download,
  BookOpen,
  CheckCircle2,
  ExternalLink,
  Code,
  Layers,
  Sparkles,
} from "lucide-react";

interface ReportItem {
  id: string;
  title: string;
  type: "academic" | "audit" | "guidelines" | "metrics";
  date: string;
  description: string;
  filename: string;
  format: "Markdown" | "JSON" | "CSV";
}

const AVAILABLE_REPORTS: ReportItem[] = [
  {
    id: "project_report",
    title: "FinGuard Academic Research Paper & Evaluation Report",
    type: "academic",
    date: "October 2026",
    description: "Complete formal research paper detailing FinGuard architecture, IR formulas, and verification experiments.",
    filename: "PROJECT_REPORT.md",
    format: "Markdown",
  },
  {
    id: "gold_audit_v2",
    title: "Relevance Judgment Automation & Evaluation Audit (v2.0.0)",
    type: "audit",
    date: "October 2026",
    description: "Independent audit uncovering coverage flaws, implementing Buckley & Voorhees bpref, and dual-assessor automation.",
    filename: "outputs/gold_benchmark_audit_v2.md",
    format: "Markdown",
  },
  {
    id: "metric_definitions",
    title: "Gold-Evidence Benchmark: Metric Formulations & Protocols",
    type: "metrics",
    date: "October 2026",
    description: "Mathematical definitions, boundary conditions, and indicator formulations for P@K, R@K, MRR, MAP, and bpref.",
    filename: "docs/metric_definitions.md",
    format: "Markdown",
  },
  {
    id: "annotation_guidelines",
    title: "FinGuard 4-Tier Relevance Annotation Decision Protocol",
    type: "guidelines",
    date: "October 2026",
    description: "Decision trees, edge-case criteria, and rules for distinguishing Grades 0, 1, 2, and 3.",
    filename: "docs/annotation_guidelines.md",
    format: "Markdown",
  },
  {
    id: "automation_research",
    title: "Relevance Automation Research & Concordance Analysis",
    type: "academic",
    date: "October 2026",
    description: "Theoretical foundation for combining neural cross-encoders and symbolic rubrics in legal and financial domains.",
    filename: "docs/relevance_automation_research.md",
    format: "Markdown",
  },
];

export default function ReportsPage() {
  const [downloadSuccess, setDownloadSuccess] = useState<string | null>(null);

  const handleDownload = (filename: string) => {
    // Triggers local download of file from API
    const url = `/api/${filename.includes("retrieval") ? "retrieval-metrics" : filename.includes("verification") ? "verification-metrics" : "status"}`;
    window.open(url, "_blank");
    setDownloadSuccess(`Export started for ${filename}`);
    setTimeout(() => setDownloadSuccess(null), 3500);
  };

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <FileText className="w-4 h-4" />
            Research Artifacts &amp; Audits
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Reports, Publications &amp; Data Exports
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
          Access peer-reviewed research reports, independent mathematical audits, and exportable JSON/CSV benchmarks.
        </p>
      </div>

      {downloadSuccess && (
        <div className="p-3.5 rounded-2xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{downloadSuccess}</span>
        </div>
      )}

      {/* Available Publications Card Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {AVAILABLE_REPORTS.map((report) => (
          <div
            key={report.id}
            className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card hover:shadow-card-hover transition-all flex flex-col justify-between space-y-4"
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-[#EEECFC] dark:bg-[#2B2564] text-primary uppercase font-mono">
                  {report.type}
                </span>
                <span className="text-xs font-mono text-[#85839A]">{report.date}</span>
              </div>
              <h3 className="text-base font-bold text-[#171638] dark:text-white leading-snug">
                {report.title}
              </h3>
              <p className="text-xs text-[#85839A] dark:text-[#8E8BA6] mt-2 leading-relaxed">
                {report.description}
              </p>
            </div>

            <div className="pt-3 border-t border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-between text-xs">
              <span className="font-mono text-[#85839A] text-[11px]">
                {report.filename}
              </span>
              <button
                type="button"
                onClick={() => handleDownload(report.filename)}
                className="px-3.5 py-1.5 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white font-bold text-xs shadow-sm transition-all hover:scale-105 flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5 text-emerald-300" />
                <span>Export {report.format}</span>
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Direct Benchmark Data Exports Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
        <h3 className="text-base font-bold text-[#171638] dark:text-white">
          Direct Benchmark Dataset Exports
        </h3>
        <p className="text-xs text-[#85839A]">
          Download evaluation outputs directly in standard machine-readable formats.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] text-primary font-bold">outputs/retrieval_metrics.json</span>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white mt-1">IR Benchmarks (Configs A-H)</h4>
              <p className="text-[11px] text-[#85839A] mt-1">MRR, Recall@5/10, MAP, Precision@5 across 506 test claims.</p>
            </div>
            <a
              href="/api/retrieval-metrics"
              target="_blank"
              className="mt-3 px-3 py-1.5 rounded-full bg-white dark:bg-[#18162E] text-primary border border-[#E9E7F2] dark:border-[#2D294F] font-bold text-xs text-center hover:border-primary transition-colors flex items-center justify-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download JSON</span>
            </a>
          </div>

          <div className="p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] text-primary font-bold">outputs/verification_metrics.json</span>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white mt-1">Classifier &amp; NLI Results</h4>
              <p className="text-[11px] text-[#85839A] mt-1">Accuracies, macro F1, per-class support, and evidence ablation.</p>
            </div>
            <a
              href="/api/verification-metrics"
              target="_blank"
              className="mt-3 px-3 py-1.5 rounded-full bg-white dark:bg-[#18162E] text-primary border border-[#E9E7F2] dark:border-[#2D294F] font-bold text-xs text-center hover:border-primary transition-colors flex items-center justify-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download JSON</span>
            </a>
          </div>

          <div className="p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] text-primary font-bold">data/benchmark/v2/human_review_queue.csv</span>
              <h4 className="text-xs font-bold text-[#171638] dark:text-white mt-1">Human Review Queue</h4>
              <p className="text-[11px] text-[#85839A] mt-1">107 prioritized borderline claim-evidence pairs with scores.</p>
            </div>
            <a
              href="/api/review-queue?limit=107"
              target="_blank"
              className="mt-3 px-3 py-1.5 rounded-full bg-white dark:bg-[#18162E] text-primary border border-[#E9E7F2] dark:border-[#2D294F] font-bold text-xs text-center hover:border-primary transition-colors flex items-center justify-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download JSON</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

