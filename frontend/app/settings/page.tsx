"use client";

import React, { useEffect, useState } from "react";
import {
  Settings,
  ShieldCheck,
  CheckCircle2,
  HardDrive,
  Cpu,
  Layers,
  Sparkles,
  Terminal,
  RefreshCw,
} from "lucide-react";
import { useTheme } from "@/components/layout/ThemeProvider";
import { fetchSystemStatus } from "@/lib/api";
import { SystemStatus } from "@/lib/types";

export default function SettingsPage() {
  const { theme, toggleTheme } = useTheme();
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadStatus = async () => {
    setRefreshing(true);
    try {
      const data = await fetchSystemStatus();
      setStatus(data);
    } catch (e) {
      console.error(e);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
              <Settings className="w-4 h-4" />
              Runtime Configuration &amp; Status
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
            Settings &amp; System Health
          </h1>
          <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
            Real-time diagnostics, local caching integrity, neural model allocations, and interface preferences.
          </p>
        </div>

        <button
          onClick={loadStatus}
          disabled={refreshing}
          className="px-4 py-2 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] text-xs font-bold text-[#171638] dark:text-white shadow-sm hover:border-primary transition-all flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-primary ${refreshing ? "animate-spin" : ""}`} />
          <span>Refresh Diagnostics</span>
        </button>
      </div>

      {/* Diagnostics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Model Architecture & Weights Card */}
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
          <h3 className="text-base font-bold text-[#171638] dark:text-white flex items-center gap-2">
            <Cpu className="w-4 h-4 text-primary" />
            <span>Pretrained Neural Models</span>
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-start justify-between gap-2">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Dense Bi-Encoder Retriever</span>
                <span className="font-mono text-[11px] text-[#85839A]">sentence-transformers/all-MiniLM-L6-v2</span>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 font-bold text-[10px]">
                384-dim Dense
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-start justify-between gap-2">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Cross-Encoder Joint Reranker</span>
                <span className="font-mono text-[11px] text-[#85839A]">cross-encoder/ms-marco-MiniLM-L-6-v2</span>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 font-bold text-[10px]">
                Cross-Attention
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-start justify-between gap-2">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Pretrained NLI Verifier</span>
                <span className="font-mono text-[11px] text-[#85839A]">cross-encoder/nli-distilroberta-base</span>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 font-bold text-[10px]">
                id2label
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-start justify-between gap-2">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Supervised In-Domain Baseline</span>
                <span className="font-mono text-[11px] text-[#85839A]">Calibrated Balanced Logistic Regression</span>
              </div>
              <span className="px-2 py-0.5 rounded-full bg-purple-50 dark:bg-purple-950/40 text-primary font-bold text-[10px]">
                74.51% Acc
              </span>
            </div>
          </div>
        </div>

        {/* Local Caches & Artifacts Integrity */}
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
          <h3 className="text-base font-bold text-[#171638] dark:text-white flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-emerald-500" />
            <span>Artifact &amp; Cache Storage Integrity</span>
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-between">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Dense Embeddings Cache (.npz)</span>
                <span className="font-mono text-[11px] text-[#85839A]">outputs/dense_evidence_embeddings.npz</span>
              </div>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {status?.dense_embeddings_size_mb || 17.58} MB
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-between">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">FinFact Benchmark Dataset</span>
                <span className="font-mono text-[11px] text-[#85839A]">finfact.json (3,369 records)</span>
              </div>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {status?.dataset_size_mb || 28.58} MB
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-between">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Calibrated Classifier Weights</span>
                <span className="font-mono text-[11px] text-[#85839A]">outputs/best_baseline_model.pkl</span>
              </div>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                Ready
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] flex items-center justify-between">
              <div>
                <span className="font-bold text-[#171638] dark:text-white block">Corpus Scope</span>
                <span className="text-[11px] text-[#85839A]">Canonical candidate evidence units</span>
              </div>
              <span className="font-mono font-bold text-primary">
                12,941 units
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Interface Preferences Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
        <h3 className="text-base font-bold text-[#171638] dark:text-white">
          Appearance &amp; Theme Preferences
        </h3>

        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F]">
          <div>
            <h4 className="text-xs font-bold text-[#171638] dark:text-white">Theme Palette</h4>
            <p className="text-[11px] text-[#85839A] mt-0.5">
              Switch between the modern soft lavender palette and the high-contrast dark theme.
            </p>
          </div>

          <button
            onClick={toggleTheme}
            className="px-4 py-2 rounded-full bg-white dark:bg-[#18162E] border border-[#E9E7F2] dark:border-[#2D294F] text-xs font-bold text-[#171638] dark:text-white shadow-sm hover:border-primary transition-colors"
          >
            Switch to {theme === "light" ? "Dark Theme" : "Light Theme"}
          </button>
        </div>
      </div>

      {/* Local Startup & Command Guide */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3">
        <h3 className="text-base font-bold text-[#171638] dark:text-white flex items-center gap-2">
          <Terminal className="w-4 h-4 text-primary" />
          <span>Local Development &amp; Daemon Commands</span>
        </h3>
        <p className="text-xs text-[#85839A] leading-relaxed">
          The Next.js frontend connects directly to local precomputed benchmark outputs. To optionally start the live background Python inference daemon on port 8000:
        </p>
        <div className="p-3 rounded-2xl bg-[#171638] text-white font-mono text-xs overflow-x-auto">
          python -m src.api_bridge serve --port 8000
        </div>
        <p className="text-[11px] text-[#85839A]">
          To launch the Next.js frontend web interface: <code className="font-mono text-primary font-bold">npm run dev</code>
        </p>
      </div>
    </div>
  );
}

