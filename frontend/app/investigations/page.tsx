"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import {
  Clock,
  Search,
  Filter,
  ExternalLink,
  ShieldCheck,
  Tag,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import { VerdictBadge } from "@/components/ui/VerdictBadge";
import { fetchInvestigations } from "@/lib/api";

export default function InvestigationsPage() {
  const [investigations, setInvestigations] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [filterVerdict, setFilterVerdict] = useState<string>("all");
  const [selectedItem, setSelectedItem] = useState<any | null>(null);

  useEffect(() => {
    fetchInvestigations().then(setInvestigations).catch(console.error);
  }, []);

  const filtered = investigations.filter((item) => {
    const matchText =
      item.claim.toLowerCase().includes(search.toLowerCase()) ||
      item.author.toLowerCase().includes(search.toLowerCase()) ||
      (item.issues || []).some((tag: string) => tag.toLowerCase().includes(search.toLowerCase()));

    const matchVerdict = filterVerdict === "all" || item.verdict.toLowerCase() === filterVerdict.toLowerCase();

    return matchText && matchVerdict;
  });

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
              <Clock className="w-4 h-4" />
              Investigations Archive
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
            Claim Inquiries &amp; Benchmark History
          </h1>
          <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
            Browse verified inquiries, ground-truth verdicts, and journalistic source citations from FinFact.
          </p>
        </div>

        <Link
          href="/verify"
          className="px-4 py-2 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white text-xs font-bold shadow-md transition-all hover:scale-105 flex items-center gap-1.5"
        >
          <Sparkles className="w-3.5 h-3.5 text-emerald-300" />
          <span>New Investigation</span>
        </Link>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-4 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-[#85839A] absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search claims, topics, or authors..."
            className="w-full h-9 pl-9 pr-3 text-xs bg-[#F8F7FD] dark:bg-[#211E3E] text-[#171638] dark:text-white rounded-full border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-[#85839A]"
          />
        </div>

        {/* Verdict Filters */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto overflow-x-auto w-full sm:w-auto">
          <span className="text-[11px] font-bold text-[#85839A] mr-1 hidden md:inline">Verdict:</span>
          {["all", "true", "false", "NEI"].map((v) => (
            <button
              key={v}
              onClick={() => setFilterVerdict(v)}
              className={`px-3 py-1 text-xs font-bold rounded-full transition-colors whitespace-nowrap ${
                filterVerdict === v
                  ? "bg-[#171638] text-white dark:bg-white dark:text-[#171638]"
                  : "bg-[#F4F3FA] dark:bg-[#28244D] text-[#85839A] hover:text-[#171638] dark:hover:text-white"
              }`}
            >
              {v === "all" ? "All Inquiries" : v.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card">
        <div className="flex items-center justify-between mb-4">
          <span className="text-xs font-bold text-[#85839A]">
            Showing <strong className="text-[#171638] dark:text-white">{filtered.length}</strong> inquiries
          </span>
          <span className="text-[11px] text-[#85839A]">
            Dataset: FinFact $N=3,369$
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#E9E7F2] dark:border-[#2D294F] text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                <th className="py-3 px-3">ID</th>
                <th className="py-3 px-3">Claim Assertion</th>
                <th className="py-3 px-3">Verdict</th>
                <th className="py-3 px-3 hidden sm:table-cell">Source Organization</th>
                <th className="py-3 px-3 hidden md:table-cell">Date</th>
                <th className="py-3 px-3 hidden lg:table-cell">Topics</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E9E7F2] dark:divide-[#2D294F] text-xs">
              {filtered.map((item) => (
                <tr key={item.id} className="hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E] transition-colors">
                  <td className="py-3 px-3 font-mono text-[11px] text-primary font-bold">
                    {item.id}
                  </td>
                  <td className="py-3 px-3 font-medium text-[#171638] dark:text-white max-w-sm">
                    {item.claim}
                  </td>
                  <td className="py-3 px-3">
                    <VerdictBadge verdict={item.verdict} size="sm" />
                  </td>
                  <td className="py-3 px-3 text-[#85839A] hidden sm:table-cell">
                    {item.author}
                  </td>
                  <td className="py-3 px-3 text-[#85839A] hidden md:table-cell font-mono">
                    {item.posted}
                  </td>
                  <td className="py-3 px-3 hidden lg:table-cell">
                    <div className="flex flex-wrap gap-1">
                      {(item.issues || []).slice(0, 2).map((tag: string, i: number) => (
                        <span key={i} className="text-[10px] px-1.5 py-0.5 rounded-md bg-[#F4F3FA] dark:bg-[#28244D] text-[#85839A]">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-3 px-3 text-right">
                    <Link
                      href={`/verify?claim=${encodeURIComponent(item.claim)}`}
                      className="px-3 py-1 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white text-[11px] font-bold shadow-sm transition-all inline-flex items-center gap-1"
                    >
                      <span>Verify</span>
                      <ArrowRight className="w-3 h-3" />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

