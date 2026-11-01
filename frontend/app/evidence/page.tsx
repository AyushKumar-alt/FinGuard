"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Search,
  Tag,
  ExternalLink,
  Layers,
  ArrowRight,
  Filter,
  CheckCircle2,
  FileText,
  Code,
} from "lucide-react";
import { searchEvidenceCorpus } from "@/lib/api";

export default function EvidencePage() {
  const [evidenceList, setEvidenceList] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [totalCount, setTotalCount] = useState(12941);
  const [searchMode, setSearchMode] = useState<"standard" | "boolean">("standard");

  useEffect(() => {
    executeSearch("");
  }, []);

  const executeSearch = async (searchTerm: string) => {
    setLoading(true);
    try {
      const results = await searchEvidenceCorpus(searchTerm, 30);
      setEvidenceList(results);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    executeSearch(query);
  };

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <Layers className="w-4 h-4" />
            Candidate Corpus
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Evidence Explorer
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
          Search and inspect 12,941 canonical deduplicated evidence sentences extracted from FinFact fact-checking investigations.
        </p>
      </div>

      {/* Search Bar */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
        <form onSubmit={handleSearch} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-[#85839A] absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                searchMode === "boolean"
                  ? 'Boolean query: ("income tax" AND revenue) NOT deduction'
                  : "Search evidence passages by keywords, company, or monetary amount..."
              }
              className="w-full h-11 pl-10 pr-4 text-xs md:text-sm bg-[#F8F7FD] dark:bg-[#211E3E] text-[#171638] dark:text-white rounded-2xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary placeholder:text-[#85839A]"
            />
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSearchMode(searchMode === "standard" ? "boolean" : "standard")}
              className={`h-11 px-4 rounded-2xl text-xs font-bold border transition-colors flex items-center gap-1.5 ${
                searchMode === "boolean"
                  ? "bg-primary text-white border-primary"
                  : "bg-white dark:bg-[#18162E] text-[#85839A] border-[#E9E7F2] dark:border-[#2D294F] hover:border-primary"
              }`}
            >
              <Code className="w-3.5 h-3.5" />
              <span>Boolean Mode</span>
            </button>

            <button
              type="submit"
              disabled={loading}
              className="h-11 px-6 rounded-2xl bg-[#171638] hover:bg-[#2A2855] text-white text-xs font-bold shadow-md transition-all disabled:opacity-50 flex items-center gap-1.5"
            >
              <span>Search Corpus</span>
            </button>
          </div>
        </form>

        {/* Quick query chips */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs text-[#85839A] pt-1">
          <span className="text-[11px] font-semibold">Quick filters:</span>
          {["corporate tax", "national debt", "shareholder capitalism", "revenue growth", "inflation rate"].map((tag, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setQuery(tag);
                executeSearch(tag);
              }}
              className="text-[11px] px-2.5 py-0.5 rounded-lg bg-[#F4F3FA] dark:bg-[#28244D] hover:bg-primary/10 text-primary font-medium transition-colors"
            >
              {tag}
            </button>
          ))}
        </div>
      </div>

      {/* Results List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <span className="text-xs font-bold text-[#85839A]">
            Showing <strong className="text-[#171638] dark:text-white">{evidenceList.length}</strong> matching candidate passages (out of 12,941 total units)
          </span>
          <span className="text-[11px] font-mono text-[#85839A]">
            Non-leakage masking active
          </span>
        </div>

        {evidenceList.map((item, idx) => (
          <div
            key={idx}
            className="bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card hover:shadow-card-hover transition-all space-y-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full bg-[#EEECFC] dark:bg-[#2B2564] text-primary text-[10px] font-bold font-mono">
                  {item.canonical_id || `ev_${String(item.doc_id).padStart(5, "0")}`}
                </span>
                <span className="text-xs text-[#85839A] flex items-center gap-1 font-medium">
                  <span>Parent Claim #{item.parent_claim_id}</span>
                </span>
              </div>

              <span className="text-[11px] text-[#85839A] font-mono">
                {item.word_count || 24} words
              </span>
            </div>

            <p className="text-sm text-[#171638] dark:text-white font-normal leading-relaxed">
              &ldquo;{item.text}&rdquo;
            </p>

            {item.claim_text && (
              <div className="p-3 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] text-xs text-[#85839A] flex items-start gap-2">
                <Tag className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold text-[#171638] dark:text-white">Associated Claim: </span>
                  <span>{item.claim_text}</span>
                </div>
                <Link
                  href={`/verify?claim=${encodeURIComponent(item.claim_text)}`}
                  className="text-primary hover:underline font-bold text-[11px] shrink-0 inline-flex items-center gap-1"
                >
                  <span>Verify</span>
                  <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
            )}

            {/* Source citations */}
            {item.source_urls && item.source_urls.length > 0 && (
              <div className="pt-2 border-t border-[#E9E7F2] dark:border-[#2D294F] flex flex-wrap items-center gap-2 text-xs">
                <span className="text-[10px] font-bold uppercase text-[#85839A]">Sources:</span>
                {item.source_urls.slice(0, 3).map((url: string, i: number) => {
                  let host = "link";
                  try {
                    host = new URL(url).hostname.replace("www.", "");
                  } catch (_) {}
                  return (
                    <a
                      key={i}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline bg-[#F4F3FA] dark:bg-[#28244D] px-2 py-0.5 rounded-lg border border-[#E9E7F2] dark:border-[#2D294F]"
                    >
                      <span>{host}</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

