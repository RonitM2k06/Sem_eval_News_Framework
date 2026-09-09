# Data Leakage Audit Report

## Audit Scope & Methodology
- **Audited Partitions**: Official SemEval-2025 Task 10 Train (`data/processed/train.json`, 1,781 articles) $\leftrightarrow$ Gold Development Set (`data/processed/dev.json`, 173 articles).
- **Official Test Set Status**: The official competition test partition (`data/raw/semeval_repo/dataset/test/`) contains only unlabelled raw documents for blind submission; it is completely excluded from training, development tuning, and leakage analysis.

---

## Summary of Findings
- **Exact Text Matches (SHA-256)**: 0 (0.00%)
- **Normalized Text Matches**: 0 (0.00%)
- **High Jaccard Near-Duplicates ($\ge 0.85$)**: 0 (0.00%)
- **Article ID Overlaps**: 0 (0.00%)
- **Observed Contamination**: **0.00% across audited splits**
- **Audit Verdict**: **PASSED (Zero Observed Train ↔ Dev Contamination)**

---

## Detailed Audit Results
1. **Article ID Collision Check**: Checked all 1,781 training IDs against all 173 development IDs. Set intersection size = 0.
2. **Cryptographic Hash Verification**: SHA-256 hashes of whitespace-trimmed texts compared across splits. Overlap count = 0.
3. **Lexical Jaccard Similarity**: Word 3-gram overlap evaluated across all pairwise cross-split article pairs. Maximum observed Jaccard similarity < 0.42 (below 0.85 threshold).
