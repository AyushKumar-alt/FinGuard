"use client";

import React, { useEffect, useState } from "react";
import {
  BarChart3,
  ShieldCheck,
  TrendingDown,
  Info,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Sparkles,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { fetchVerificationMetrics } from "@/lib/api";
import { VerifierMetric, BaselineAblationResult } from "@/lib/types";

export default function EvaluationPage() {
  const [data, setData] = useState<{
    verifiers: VerifierMetric[];
    ablation: BaselineAblationResult[];
  } | null>(null);

  useEffect(() => {
    fetchVerificationMetrics().then(setData).catch(console.error);
  }, []);

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
            <BarChart3 className="w-4 h-4" />
            Model &amp; Verifier Governance
          </span>
        </div>
        <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
          Claim Verification &amp; Ablation Evaluation
        </h1>
        <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1 max-w-3xl">
          Comparative assessment of supervised in-domain classifiers, zero-shot pretrained natural language inference (NLI), and feature ablation experiments on FinFact.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Best Verifier Accuracy"
          value="74.51%"
          subtitle="Calibrated Balanced Logistic Regression"
          icon={ShieldCheck}
          badge="In-Domain"
          badgeColor="emerald"
        />
        <StatCard
          title="Macro F1-Score"
          value="0.7178"
          subtitle="Balanced across False, True, and NEI classes"
          icon={BarChart3}
          badge="3-Class FEVER"
          badgeColor="purple"
        />
        <StatCard
          title="Evidence Reliance Drop"
          value="-10.48%"
          subtitle="Accuracy drop when removing evidence context"
          icon={TrendingDown}
          badge="Ablation Proven"
          badgeColor="amber"
        />
        <StatCard
          title="Zero-Shot NLI Gap"
          value="34.00%"
          subtitle="cross-encoder/nli-distilroberta on raw text"
          icon={AlertTriangle}
          badge="Out-of-Domain"
          badgeColor="rose"
        />
      </div>

      {/* Verification Systems Benchmark Table */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
        <div>
          <h3 className="text-base font-bold text-[#171638] dark:text-white">
            Verification Systems Comparison
          </h3>
          <p className="text-xs text-[#85839A]">
            Evaluated on held-out standard test split claims with strict separation of training sources.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#E9E7F2] dark:border-[#2D294F] text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                <th className="py-3 px-3">System</th>
                <th className="py-3 px-3">Verifier Architecture</th>
                <th className="py-3 px-3">Evaluation Context</th>
                <th className="py-3 px-3">Sample Size</th>
                <th className="py-3 px-3">Accuracy</th>
                <th className="py-3 px-3">Macro F1</th>
                <th className="py-3 px-3">Weighted F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E9E7F2] dark:divide-[#2D294F] text-xs">
              {data?.verifiers.map((sys) => (
                <tr
                  key={sys.system_id}
                  className={`hover:bg-[#F8F7FD] dark:hover:bg-[#211E3E] transition-colors ${
                    sys.system_id === 1 ? "bg-purple-50/40 dark:bg-purple-950/20" : ""
                  }`}
                >
                  <td className="py-3.5 px-3 font-mono font-bold text-primary">
                    #{sys.system_id}
                  </td>
                  <td className="py-3.5 px-3">
                    <div className="font-bold text-[#171638] dark:text-white flex items-center gap-1.5">
                      <span>{sys.name}</span>
                      {sys.system_id === 1 && (
                        <span className="text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-[#239B79] text-white">
                          Best Performer
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-[#85839A] font-mono">{sys.model}</div>
                  </td>
                  <td className="py-3.5 px-3 text-[#85839A] max-w-xs">{sys.context}</td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A]">{sys.sample_size}</td>
                  <td className="py-3.5 px-3 font-mono font-bold text-[#171638] dark:text-white">
                    {(sys.accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-3.5 px-3 font-mono font-bold text-primary">
                    {sys.macro_f1.toFixed(4)}
                  </td>
                  <td className="py-3.5 px-3 font-mono text-[#85839A]">
                    {sys.weighted_f1.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Evidence-Reliance Ablation Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-[#171638] dark:text-white">
                Evidence Reliance Ablation Study
              </h3>
              <p className="text-xs text-[#85839A]">
                Testing whether the model grounds on evidence or memorizes claim prior shortcuts.
              </p>
            </div>
            <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400 font-mono">
              -10.48% drop
            </span>
          </div>

          <div className="space-y-3 pt-2">
            {data?.ablation.map((item, i) => (
              <div
                key={i}
                className="p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-xs text-[#171638] dark:text-white">
                    {item.setting}
                  </span>
                  <span className="font-mono text-xs font-bold text-primary">
                    {(item.accuracy * 100).toFixed(2)}% Accuracy
                  </span>
                </div>
                {/* Visual Progress Bar */}
                <div className="w-full h-2.5 rounded-full bg-[#E9E7F2] dark:bg-[#2D294F] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      i === 0 ? "bg-[#239B79]" : "bg-[#D9A441]"
                    }`}
                    style={{ width: `${item.accuracy * 100}%` }}
                  />
                </div>
                <div className="flex justify-between text-[11px] text-[#85839A]">
                  <span>Macro F1: {item.macro_f1.toFixed(4)}</span>
                  <span>Weighted F1: {item.weighted_f1.toFixed(4)}</span>
                </div>
              </div>
            ))}
          </div>

          <p className="text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed pt-1">
            Empirical Proof: Removing evidence context causes an immediate accuracy decline from 74.51% to 64.03%, demonstrating that the classifier actively leverages the retrieved evidence rather than merely exploiting lexical priors in the claim string.
          </p>
        </div>

        {/* The Zero-Shot NLI Gap Explanation */}
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-3 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-[#171638] dark:text-white flex items-center gap-2">
              <Info className="w-4 h-4 text-primary" />
              <span>Understanding the Zero-Shot NLI Gap</span>
            </h3>
            <p className="text-xs text-[#85839A] mt-1">
              Why general-domain pretrained NLI models struggle on journalistic financial fact-checking:
            </p>

            <ul className="space-y-2.5 mt-3 text-xs text-[#85839A] dark:text-[#A09EB7] leading-relaxed">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <span>
                  <strong className="text-[#171638] dark:text-white">Lack of Formal Syllogisms:</strong> Journalistic fact-checking evidence reports background facts rather than formal mathematical proofs, causing pretrained NLI models to assign conservative &ldquo;Neutral&rdquo; scores.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <span>
                  <strong className="text-[#171638] dark:text-white">Domain Vocabulary:</strong> Numerical thresholds, fiscal quarters (`FY2025`), and accounting categories (`loss carryforwards`) require financial domain calibration.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                <span>
                  <strong className="text-[#171638] dark:text-white">Conflict Gating:</strong> FinGuard overcomes this limitation by combining in-domain supervised calibration with multi-sentence conflict aggregation.
                </span>
              </li>
            </ul>
          </div>

          <div className="p-3.5 rounded-2xl bg-[#EEECFC] dark:bg-[#2B2564] text-xs text-[#7065E8] dark:text-[#A59DFF] font-medium">
            Tip: In the Verification Workspace (`/verify`), you can toggle between the Pretrained NLI verifier and the Calibrated Logistic Regression model to compare their reasoning.
          </div>
        </div>
      </div>
    </div>
  );
}

