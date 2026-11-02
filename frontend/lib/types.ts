export type VerdictType = "true" | "false" | "NEI";

export interface FinancialEntities {
  organizations?: string[];
  financial_metrics?: string[];
  monetary_amounts?: string[];
  periods?: string[];
  directional_terms?: Array<{ term: string; direction: string }>;
  negations?: string[];
  percentages?: string[];
  currencies?: string[];
  comparisons?: string[];
  tokens?: string[];
  companies?: string[];
  metrics?: string[];
  fiscal_periods?: string[];
  directional_keywords?: string[];
  negation_tokens?: string[];
}

export interface QueryInfo {
  original_query?: string;
  expanded_queries?: string[];
  base_query?: string;
  expansions?: string[];
  combined_query?: string;
  multi_queries?: Array<{
    query: string;
    group: string;
    weight: number;
    purpose: string;
  }>;
  query_strings?: string[];
  key_entities?: string[];
  key_metrics?: string[];
  monetary_values?: string[];
  time_periods?: string[];
}

export interface CandidateEvidence {
  doc_id?: number;
  text: string;
  score?: number;
  rerank_score?: number;
  retrieval_score?: number;
  rank?: number;
  source_urls?: string[];
  parent_claim_ids?: number[];
  retrieval_pipeline?: string;
  relevance_grade?: number | null;
  entailment_score?: number;
  contradiction_score?: number;
  neutral_score?: number;
  assigned_category?: "supporting" | "contradicting" | "neutral" | "contextual" | "retrieved";
  publisher?: string;
  title?: string;
}

export interface ClaimVerificationResponse {
  claim: string;
  verdict: VerdictType;
  verdict_display: string;
  confidence: number;
  class_probabilities?: Record<string, number>;
  is_conflicting: boolean;
  is_uncertain: boolean;
  uncertainty_reason: string | null;
  explanation: string;
  parsed_claim?: FinancialEntities;
  query_info?: QueryInfo;
  search_query?: string;
  retrieval_mode?: string;
  retrieval_engine?: string;
  total_candidates_retrieved?: number;
  selected_evidence_docs?: CandidateEvidence[];
  reranker_active?: boolean;
  verifier_engine?: string;
  component_status?: {
    dense_retriever: boolean;
    cross_encoder_reranker: boolean;
    nli_verifier: boolean;
    baseline_verifier: boolean;
  };
  retrieved_candidates?: CandidateEvidence[];
  reranked_candidates?: CandidateEvidence[];
  supporting_evidence: CandidateEvidence[];
  contradicting_evidence: CandidateEvidence[];
  neutral_evidence: CandidateEvidence[];
  latency_ms?: number;
}

export interface ClaimVerificationRequest {
  claim: string;
  retrieval_mode?: string;
  use_query_expansion?: boolean;
  use_reranker?: boolean;
  use_nli?: boolean;
  top_k_retrieve?: number;
  top_k_rerank?: number;
}

export interface RetrievalMetric {
  config: string;
  name: string;
  strategy: string;
  mrr: number;
  recall_at_5: number;
  recall_at_10: number;
  map: number;
  precision_at_5: number;
  precision_at_10?: number;
  ndcg_at_5?: number;
  ndcg_at_10?: number;
  bpref?: number;
  judged_coverage_at_10?: number;
  refrate_at_10?: number;
  latency_ms?: number;
  sample_size?: number;
  is_best_mrr?: boolean;
  is_best_recall?: boolean;
}

export interface VerifierMetric {
  system_id: number;
  name: string;
  model: string;
  context: string;
  sample_size: number;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  per_class?: Record<string, { precision: number; recall: number; f1: number; support: number }>;
}

export interface BaselineAblationResult {
  setting: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  drop_from_full?: number;
}

export interface GoldBenchmarkOverview {
  version: string;
  total_pool_pairs: number;
  unique_evidence_units: number;
  pilot_claims_evaluated: number;
  silver_reference_pairs: number;
  concordant_non_relevant: number;
  unjudged_queue_pairs: number;
  reduction_percentage: number;
  coverage_at_10: number;
  checksum: string;
}

export interface HumanReviewItem {
  claim_id: string;
  evidence_id: string;
  claim_text: string;
  evidence_text: string;
  cross_encoder_score: number;
  rubric_overlap: string[];
  uncertainty_reason: string;
  current_status: "unjudged" | "reviewed" | "silver_reference";
  assigned_grade?: number | null;
  reviewer_notes?: string;
}

export interface CorpusSentence {
  doc_id: number;
  canonical_id: string;
  text: string;
  source_urls: string[];
  parent_claim_ids: number[];
  word_count: number;
}

export interface SystemStatus {
  status: "online" | "degraded" | "offline";
  pipeline: string;
  total_claims: number;
  corpus_units: number;
  dense_embeddings_cached: boolean;
  dense_embeddings_size_mb: number;
  positional_index_cached: boolean;
  calibrated_baseline_available: boolean;
  dataset_available: boolean;
  dataset_size_mb: number;
  models: {
    dense_retriever: string;
    cross_encoder_reranker: string;
    nli_verifier: string;
    baseline_verifier: string;
  };
}
