# FinGuard Gold-Evidence Relevance Benchmark Results

**Benchmark Version:** 1.0.0  
**Evaluation Claims:** 75 held-out claims  
**Relevance Threshold:** Grade >= 2 (Grades 2 & 3 counted as relevant)  
**Candidate Corpus:** 12,941 candidate evidence sentences  

### Retrieval Performance Matrix

| System               |    MRR |    R@5 |   R@10 |   nDCG@5 |   nDCG@10 |   bpref |    P@5 |   P@10 |   Coverage@10 |   RefRate@10 |   Latency_Mean_ms |
|:---------------------|-------:|-------:|-------:|---------:|----------:|--------:|-------:|-------:|--------------:|-------------:|------------------:|
| BM25                 | 0.5661 | 0.3041 | 0.3636 |   0.388  |    0.3719 |  0.4034 | 0.3067 | 0.1933 |        0.1947 |       0.3636 |             51.11 |
| TFIDF                | 0.4653 | 0.2289 | 0.2841 |   0.3032 |    0.2925 |  0.3334 | 0.2427 | 0.1547 |        0.156  |       0.2841 |             30.32 |
| SVD                  | 0.0928 | 0.0424 | 0.0547 |   0.0498 |    0.0504 |  0.0759 | 0.0427 | 0.028  |        0.028  |       0.0547 |             16.53 |
| Dense_Transformer    | 0.6469 | 0.313  | 0.41   |   0.4492 |    0.4375 |  0.4607 | 0.36   | 0.24   |        0.24   |       0.41   |             21.6  |
| Hybrid_BM25_Dense    | 0.6445 | 0.3476 | 0.4212 |   0.4597 |    0.4405 |  0.4795 | 0.3707 | 0.2347 |        0.236  |       0.4212 |             62.97 |
| Hybrid_BM25_TFIDF    | 0.5136 | 0.2546 | 0.3315 |   0.3453 |    0.3406 |  0.3776 | 0.2747 | 0.1853 |        0.1867 |       0.3315 |             87.25 |
| Cross_Encoder_Rerank | 0.6686 | 0.3372 | 0.392  |   0.4641 |    0.4366 |  0.4034 | 0.352  | 0.2133 |        0.2147 |       0.392  |            729.04 |
| Query_Expansion_BM25 | 0.5462 | 0.2775 | 0.3411 |   0.3627 |    0.3534 |  0.3892 | 0.288  | 0.188  |        0.1893 |       0.3411 |             79.23 |

### Metric Explanations
- **MRR:** Mean Reciprocal Rank of first relevant item (Grade >= 2).
- **R@K:** Recall at rank K relative to total relevant items for the claim.
- **nDCG@K:** Normalized Discounted Cumulative Gain using graded gains (2^g - 1).
- **bpref:** Binary Preference measure (Buckley & Voorhees 2004) under incomplete relevance judgments.
- **P@K:** Precision at rank K.
- **Coverage@K:** Fraction of retrieved top-K items having explicit, non-null judgments.
- **RefRate@K:** Fraction of original dataset reference evidence retrieved in top-K.
- **Latency:** End-to-end retrieval time in milliseconds.
