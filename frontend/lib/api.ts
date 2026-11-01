import {
  ClaimVerificationRequest,
  ClaimVerificationResponse,
  RetrievalMetric,
  VerifierMetric,
  BaselineAblationResult,
  GoldBenchmarkOverview,
  HumanReviewItem,
  SystemStatus,
} from "./types";

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch("/api/status", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch system status");
  return res.json();
}

export async function verifyClaim(payload: ClaimVerificationRequest): Promise<ClaimVerificationResponse> {
  const res = await fetch("/api/verify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Verification failed with HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchRetrievalMetrics(): Promise<RetrievalMetric[]> {
  const res = await fetch("/api/retrieval-metrics");
  if (!res.ok) throw new Error("Failed to load retrieval metrics");
  return res.json();
}

export async function fetchVerificationMetrics(): Promise<{
  verifiers: VerifierMetric[];
  ablation: BaselineAblationResult[];
}> {
  const res = await fetch("/api/verification-metrics");
  if (!res.ok) throw new Error("Failed to load verification metrics");
  return res.json();
}

export async function fetchGoldBenchmark(): Promise<{
  v1: GoldBenchmarkOverview;
  v2: GoldBenchmarkOverview;
}> {
  const res = await fetch("/api/gold-benchmark");
  if (!res.ok) throw new Error("Failed to load gold benchmark overview");
  return res.json();
}

export async function fetchReviewQueue(limit = 100): Promise<HumanReviewItem[]> {
  const res = await fetch(`/api/review-queue?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to load human review queue");
  return res.json();
}

export async function submitHumanJudgment(payload: {
  claim_id: string;
  evidence_id: string;
  assigned_grade: number;
  reviewer_notes?: string;
}): Promise<{ success: boolean; message: string }> {
  const res = await fetch("/api/review-queue/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to submit judgment");
  return res.json();
}

export async function fetchInvestigations(): Promise<any[]> {
  const res = await fetch("/api/investigations");
  if (!res.ok) throw new Error("Failed to load investigations");
  return res.json();
}

export async function searchEvidenceCorpus(query: string, limit = 20): Promise<any[]> {
  const res = await fetch(`/api/evidence?q=${encodeURIComponent(query)}&limit=${limit}`);
  if (!res.ok) throw new Error("Failed to search evidence corpus");
  return res.json();
}

