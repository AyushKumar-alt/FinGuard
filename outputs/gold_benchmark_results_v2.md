# FinGuard Gold-Evidence Relevance Benchmark Results

**Benchmark Version:** 1.0.0  
**Evaluation Claims:** 75 held-out claims  
**Relevance Threshold:** Grade >= 2 (Grades 2 & 3 counted as relevant)  
**Candidate Corpus:** 12,941 candidate evidence sentences  

### Retrieval Performance Matrix

| System               |    MRR |    R@5 |   R@10 |   nDCG@5 |   nDCG@10 |   bpref |    P@5 |   P@10 |   Coverage@10 |   RefRate@10 |   Latency_Mean_ms |
|:---------------------|-------:|-------:|-------:|---------:|----------:|--------:|-------:|-------:|--------------:|-------------:|------------------:|
| BM25                 | 0.5661 | 0.3041 | 0.3636 |   0.388  |    0.3719 |  0.2634 | 0.3067 | 0.1933 |        0.9373 |       0.3636 |             45.22 |
| TFIDF                | 0.4653 | 0.2289 | 0.2841 |   0.3032 |    0.2925 |  0.1999 | 0.2427 | 0.1547 |        0.96   |       0.2841 |             30.32 |
| SVD                  | 0.0928 | 0.0424 | 0.0547 |   0.0498 |    0.0504 |  0.034  | 0.0427 | 0.028  |        0.9813 |       0.0547 |             17.55 |
| Dense_Transformer    | 0.6469 | 0.313  | 0.41   |   0.4492 |    0.4375 |  0.3248 | 0.36   | 0.24   |        0.9293 |       0.41   |             18.97 |
| Hybrid_BM25_Dense    | 0.6445 | 0.3476 | 0.4212 |   0.4597 |    0.4405 |  0.3296 | 0.3707 | 0.2347 |        0.9013 |       0.4212 |             64.22 |
| Hybrid_BM25_TFIDF    | 0.5136 | 0.2546 | 0.3315 |   0.3453 |    0.3406 |  0.2457 | 0.2747 | 0.1853 |        0.948  |       0.3315 |             74.52 |
| Cross_Encoder_Rerank | 0.6686 | 0.3372 | 0.392  |   0.4641 |    0.4366 |  0.3396 | 0.352  | 0.2133 |        0.9227 |       0.392  |            516    |
| Query_Expansion_BM25 | 0.5462 | 0.2775 | 0.3411 |   0.3627 |    0.3534 |  0.251  | 0.288  | 0.188  |        0.94   |       0.3411 |             51.58 |

### Metric Explanations
- **MRR:** Mean Reciprocal Rank of first relevant item (Grade >= 2).
- **R@K:** Recall at rank K relative to total relevant items for the claim.
- **nDCG@K:** Normalized Discounted Cumulative Gain using graded gains (2^g - 1).
- **bpref:** Binary Preference measure (Buckley & Voorhees 2004) under incomplete relevance judgments.
- **P@K:** Precision at rank K.
- **Coverage@K:** Fraction of retrieved top-K items having explicit, non-null judgments.
- **RefRate@K:** Fraction of original dataset reference evidence retrieved in top-K.
- **Latency:** End-to-end retrieval time in milliseconds.
