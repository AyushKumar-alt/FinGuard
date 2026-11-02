"use client";

import React, { useState } from "react";
import { CandidateEvidence } from "@/lib/types";
import { ExternalLink, ChevronDown, ChevronUp, ShieldCheck, Tag } from "lucide-react";

interface EvidenceCardProps {
  evidence: CandidateEvidence;
  category?: "supporting" | "contradicting" | "neutral" | "contextual" | "retrieved";
  rank?: number;
}

export function EvidenceCard({
  evidence,
  category = "neutral",
  rank,
}: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const categoryStyles = {
    supporting: {
      border: "border-l-4 border-l-[#239B79] border-t border-r border-b border-[#E9E7F2] dark:border-[#2D294F]",
      pill: "bg-emerald-50 text-[#239B79] dark:bg-emerald-950/40 dark:text-emerald-400",
      label: "Entails (Supports Claim)",
    },
    contradicting: {
      border: "border-l-4 border-l-[#D95465] border-t border-r border-b border-[#E9E7F2] dark:border-[#2D294F]",
      pill: "bg-rose-50 text-[#D95465] dark:bg-rose-950/40 dark:text-rose-400",
      label: "Refutes (Contradicts Claim)",
    },
    neutral: {
      border: "border-l-4 border-l-[#7065E8] border-t border-r border-b border-[#E9E7F2] dark:border-[#2D294F]",
      pill: "bg-[#EEECFC] text-[#7065E8] dark:bg-[#2B2564] dark:text-[#A59DFF]",
      label: "NLI neutral",
    },
    contextual: {
      border: "border-l-4 border-l-[#D9A441] border-t border-r border-b border-[#E9E7F2] dark:border-[#2D294F]",
      pill: "bg-amber-50 text-[#D9A441] dark:bg-amber-950/40 dark:text-amber-400",
      label: "Contextual Background",
    },
    retrieved: {
      border: "border-l-4 border-l-slate-400 border-t border-r border-b border-[#E9E7F2] dark:border-[#2D294F]",
      pill: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-200",
      label: "Retrieved passage · no NLI stance",
    },
  };

  const style = categoryStyles[category] || categoryStyles.neutral;
  const isLong = (evidence.text || "").length > 200;
  const displayText = isLong && !expanded ? evidence.text.slice(0, 200) + "..." : evidence.text;

  return (
    <div
      className={`bg-white dark:bg-[#18162E] rounded-2xl p-4 shadow-sm hover:shadow-card transition-all ${style.border}`}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
        <div className="flex flex-wrap items-center gap-2">
          {rank !== undefined && (
            <span className="w-6 h-6 rounded-full bg-[#171638] dark:bg-white text-white dark:text-[#171638] text-[11px] font-extrabold flex items-center justify-center">
              #{rank}
            </span>
          )}
          <span className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full ${style.pill}`}>
            {style.label}
          </span>
          {evidence.publisher && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-md bg-[#EEECFC] dark:bg-[#2B2564] text-primary border border-primary/20">
              {evidence.publisher}
            </span>
          )}
          {evidence.doc_id !== undefined && (
            <span className="text-[10px] font-mono text-[#85839A] dark:text-[#8E8BA6] flex items-center gap-1">
              <Tag className="w-3 h-3" />
              <span>doc_{evidence.doc_id}</span>
            </span>
          )}
        </div>

        {/* Scores: BM25 / Retrieval and Reranker */}
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          {evidence.score !== undefined && (
            <span className="text-[#85839A] dark:text-[#8E8BA6]">
              Retrieval score: <strong className="text-[#171638] dark:text-white font-mono">{evidence.score.toFixed(4)}</strong>
            </span>
          )}
          {evidence.rerank_score !== undefined && (
            <span className="px-2 py-0.5 rounded-md bg-[#F4F3FA] dark:bg-[#28244D] text-[#7065E8] dark:text-[#A59DFF] font-semibold text-[10px]">
              Cross-encoder score: {evidence.rerank_score.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      {evidence.title && (
        <div className="mb-2 text-xs font-semibold text-[#615F79] dark:text-[#B5B2CD] italic truncate">
          &ldquo;{evidence.title}&rdquo;
        </div>
      )}

      {(evidence.entailment_score !== undefined ||
        evidence.contradiction_score !== undefined ||
        evidence.neutral_score !== undefined) && (
        <div className="mt-3 flex flex-wrap gap-2 text-[10px] text-[#615F79] dark:text-[#B5B2CD]">
          <span className="font-bold uppercase tracking-wide">NLI output scores</span>
          {evidence.entailment_score !== undefined && <span>Entailment {evidence.entailment_score.toFixed(4)}</span>}
          {evidence.contradiction_score !== undefined && <span>Contradiction {evidence.contradiction_score.toFixed(4)}</span>}
          {evidence.neutral_score !== undefined && <span>Neutral {evidence.neutral_score.toFixed(4)}</span>}
          <span className="italic">not calibrated confidence</span>
        </div>
      )}

      {/* Evidence passage body */}
      <p className="text-sm text-[#171638] dark:text-gray-100 leading-relaxed font-normal">
        &ldquo;{displayText}&rdquo;
      </p>

      {isLong && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-xs font-semibold text-primary hover:text-primary-hover flex items-center gap-1 transition-colors"
        >
          {expanded ? (
            <>
              Show less <ChevronUp className="w-3.5 h-3.5" />
            </>
          ) : (
            <>
              Read full passage <ChevronDown className="w-3.5 h-3.5" />
            </>
          )}
        </button>
      )}

      {/* Source Provenance Links */}
      {evidence.source_urls && evidence.source_urls.length > 0 && (
        <div className="mt-3 pt-2.5 border-t border-[#E9E7F2] dark:border-[#2D294F] flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[#85839A] dark:text-[#8E8BA6]">
            Sources:
          </span>
          {evidence.source_urls.slice(0, 3).map((url, idx) => {
            let domain = "source";
            try {
              domain = new URL(url).hostname.replace("www.", "");
            } catch (_) {}
            return (
              <a
                key={idx}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline hover:text-primary-hover font-medium bg-[#F8F7FD] dark:bg-[#211E3E] px-2 py-0.5 rounded-lg border border-[#E9E7F2] dark:border-[#2D294F]"
              >
                <span>{domain}</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}
