"use client";

import React, { useEffect, useState } from "react";
import {
  UserCheck,
  CheckCircle2,
  XCircle,
  HelpCircle,
  AlertCircle,
  Tag,
  Save,
  ArrowRight,
  Sparkles,
  Info,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { fetchReviewQueue, submitHumanJudgment } from "@/lib/api";
import { HumanReviewItem } from "@/lib/types";

export default function HumanReviewQueuePage() {
  const [queue, setQueue] = useState<HumanReviewItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedGrade, setSelectedGrade] = useState<number | null>(null);
  const [reviewerNotes, setReviewerNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchReviewQueue(107).then((items) => {
      setQueue(items);
    }).catch(console.error);
  }, []);

  const currentItem = queue[currentIndex];

  const handleGradeSelect = (grade: number) => {
    setSelectedGrade(grade);
  };

  const handleSubmitReview = async () => {
    if (!currentItem || selectedGrade === null) return;

    setSubmitting(true);
    setFeedbackMessage(null);

    try {
      await submitHumanJudgment({
        claim_id: currentItem.claim_id,
        evidence_id: currentItem.evidence_id,
        assigned_grade: selectedGrade,
        reviewer_notes: reviewerNotes,
      });

      setSubmittedCount((prev) => prev + 1);
      setFeedbackMessage(`Grade ${selectedGrade} recorded successfully for ${currentItem.claim_id}!`);

      // Reset selection for next item
      setSelectedGrade(null);
      setReviewerNotes("");

      if (currentIndex < queue.length - 1) {
        setCurrentIndex((prev) => prev + 1);
      }
    } catch (err: any) {
      setFeedbackMessage(`Error saving review: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const progressPercentage = queue.length > 0 ? (submittedCount / queue.length) * 100 : 0;

  return (
    <div className="space-y-6 pt-2">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-widest text-primary flex items-center gap-1.5">
              <UserCheck className="w-4 h-4" />
              Prioritized Review Workflow
            </span>
          </div>
          <h1 className="text-2xl md:text-3xl font-extrabold text-[#171638] dark:text-white tracking-tight">
            Human Review Queue (Benchmark v2)
          </h1>
          <p className="text-xs md:text-sm text-[#85839A] dark:text-[#A09EB7] mt-1">
            Review the 107 high-uncertainty candidate pairs isolated by the dual-assessor automation pipeline.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-[#85839A]">Reviewed in session:</span>
          <span className="px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 font-extrabold text-xs">
            {submittedCount} / {queue.length}
          </span>
        </div>
      </div>

      {/* Progress Bar Card */}
      <div className="bg-white dark:bg-[#18162E] rounded-3xl p-5 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-2">
        <div className="flex justify-between items-center text-xs">
          <span className="font-bold text-[#171638] dark:text-white">
            Targeted Queue Progress
          </span>
          <span className="font-mono text-[#85839A]">
            Candidate {currentIndex + 1} of {queue.length}
          </span>
        </div>
        <div className="w-full h-2 rounded-full bg-[#E9E7F2] dark:bg-[#2D294F] overflow-hidden">
          <div
            className="h-full rounded-full bg-[#7065E8] transition-all duration-300"
            style={{ width: `${Math.max(5, progressPercentage)}%` }}
          />
        </div>
      </div>

      {/* Feedback Message */}
      {feedbackMessage && (
        <div className="p-3.5 rounded-2xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{feedbackMessage}</span>
        </div>
      )}

      {/* Review Adjudication Card */}
      {currentItem ? (
        <div className="bg-white dark:bg-[#18162E] rounded-3xl p-6 border border-[#E9E7F2] dark:border-[#2D294F] shadow-card space-y-6">
          {/* Claim and Meta */}
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-[11px] font-mono text-primary font-bold">
                {currentItem.claim_id} • {currentItem.evidence_id}
              </span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400 font-semibold">
                Reason: {currentItem.uncertainty_reason}
              </span>
            </div>

            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#85839A]">
                Target Financial Claim:
              </span>
              <h2 className="text-base md:text-lg font-bold text-[#171638] dark:text-white mt-0.5">
                &ldquo;{currentItem.claim_text}&rdquo;
              </h2>
            </div>
          </div>

          {/* Evidence Sentence Body */}
          <div className="p-4 rounded-2xl bg-[#F8F7FD] dark:bg-[#211E3E] border border-[#E9E7F2] dark:border-[#2D294F] space-y-2">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-bold uppercase tracking-wider text-[#85839A]">
                Candidate Evidence Passage to Evaluate:
              </span>
              <span className="font-mono text-primary font-semibold">
                Neural Score: {currentItem.cross_encoder_score?.toFixed(3) || "0.220"}
              </span>
            </div>
            <p className="text-sm md:text-base text-[#171638] dark:text-white leading-relaxed font-normal">
              &ldquo;{currentItem.evidence_text}&rdquo;
            </p>
          </div>

          {/* 4-Tier Annotation Decision Rubric */}
          <div className="space-y-3">
            <label className="block text-xs font-bold uppercase tracking-wider text-[#171638] dark:text-white">
              Select Relevance Grade (Annotation Decision):
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {/* Grade 0 */}
              <button
                type="button"
                onClick={() => handleGradeSelect(0)}
                className={`p-3.5 rounded-2xl border text-left transition-all ${
                  selectedGrade === 0
                    ? "border-[#D95465] bg-rose-50/50 dark:bg-rose-950/30 ring-2 ring-[#D95465]"
                    : "border-[#E9E7F2] dark:border-[#2D294F] hover:border-rose-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-[#D95465]">Grade 0</span>
                  <XCircle className="w-4 h-4 text-[#D95465]" />
                </div>
                <h4 className="text-xs font-bold text-[#171638] dark:text-white">Not Relevant</h4>
                <p className="text-[10px] text-[#85839A] mt-0.5">Irrelevant context or spurious keyword match.</p>
              </button>

              {/* Grade 1 */}
              <button
                type="button"
                onClick={() => handleGradeSelect(1)}
                className={`p-3.5 rounded-2xl border text-left transition-all ${
                  selectedGrade === 1
                    ? "border-[#D9A441] bg-amber-50/50 dark:bg-amber-950/30 ring-2 ring-[#D9A441]"
                    : "border-[#E9E7F2] dark:border-[#2D294F] hover:border-amber-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-[#D9A441]">Grade 1</span>
                  <HelpCircle className="w-4 h-4 text-[#D9A441]" />
                </div>
                <h4 className="text-xs font-bold text-[#171638] dark:text-white">Contextual</h4>
                <p className="text-[10px] text-[#85839A] mt-0.5">Mentions entity/topic but does not prove/refute claim.</p>
              </button>

              {/* Grade 2 */}
              <button
                type="button"
                onClick={() => handleGradeSelect(2)}
                className={`p-3.5 rounded-2xl border text-left transition-all ${
                  selectedGrade === 2
                    ? "border-[#7065E8] bg-purple-50/50 dark:bg-purple-950/30 ring-2 ring-[#7065E8]"
                    : "border-[#E9E7F2] dark:border-[#2D294F] hover:border-purple-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-primary">Grade 2</span>
                  <CheckCircle2 className="w-4 h-4 text-primary" />
                </div>
                <h4 className="text-xs font-bold text-[#171638] dark:text-white">Substantial</h4>
                <p className="text-[10px] text-[#85839A] mt-0.5">Contains key factual data directly related to claim.</p>
              </button>

              {/* Grade 3 */}
              <button
                type="button"
                onClick={() => handleGradeSelect(3)}
                className={`p-3.5 rounded-2xl border text-left transition-all ${
                  selectedGrade === 3
                    ? "border-[#239B79] bg-emerald-50/50 dark:bg-emerald-950/30 ring-2 ring-[#239B79]"
                    : "border-[#E9E7F2] dark:border-[#2D294F] hover:border-emerald-300"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-bold text-[#239B79]">Grade 3</span>
                  <Sparkles className="w-4 h-4 text-[#239B79]" />
                </div>
                <h4 className="text-xs font-bold text-[#171638] dark:text-white">Directly Decisive</h4>
                <p className="text-[10px] text-[#85839A] mt-0.5">Fully entails or refutes the target financial claim.</p>
              </button>
            </div>
          </div>

          {/* Notes Input */}
          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-[#85839A] uppercase tracking-wider">
              Reviewer Notes (Optional):
            </label>
            <input
              type="text"
              value={reviewerNotes}
              onChange={(e) => setReviewerNotes(e.target.value)}
              placeholder="e.g. Entity matches but numbers discuss FY2023 instead of FY2025"
              className="w-full p-3 text-xs bg-[#F8F7FD] dark:bg-[#211E3E] text-[#171638] dark:text-white rounded-xl border border-[#E9E7F2] dark:border-[#2D294F] focus:outline-none focus:ring-1 focus:ring-primary"
            />
          </div>

          {/* Action Row */}
          <div className="flex items-center justify-between pt-3 border-t border-[#E9E7F2] dark:border-[#2D294F]">
            <button
              type="button"
              onClick={() => {
                if (currentIndex < queue.length - 1) setCurrentIndex(currentIndex + 1);
              }}
              className="text-xs font-semibold text-[#85839A] hover:text-[#171638] dark:hover:text-white transition-colors"
            >
              Skip Candidate
            </button>

            <button
              type="button"
              disabled={selectedGrade === null || submitting}
              onClick={handleSubmitReview}
              className="px-6 py-2.5 rounded-full bg-[#171638] hover:bg-[#2A2855] text-white text-xs font-bold shadow-md hover:scale-105 active:scale-95 transition-all disabled:opacity-50 disabled:pointer-events-none flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              <span>Submit &amp; Next Candidate</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="p-12 text-center bg-white dark:bg-[#18162E] rounded-3xl border border-[#E9E7F2] dark:border-[#2D294F] text-xs text-[#85839A]">
          Loading human review queue items...
        </div>
      )}
    </div>
  );
}

