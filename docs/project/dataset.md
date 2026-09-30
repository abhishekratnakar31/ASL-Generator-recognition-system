# Dataset Documentation: WLASL-100

## Overview

The Word-Level American Sign Language (WLASL) dataset is a standard benchmark for isolated ASL recognition. This project benchmarks on the **WLASL-100** subset containing 100 high-frequency sign vocabulary classes.

- **Total Classes**: 100
- **Total Expected Videos**: 2038
- **Total Signers**: 97

## Signer-Independent Partitioning

To measure genuine generalization across different individuals, splits are partitioned by signer ID:

- **Train**: 68 signers, 100 classes, 1234 instances
- **Validation**: 15 signers, 100 classes, 426 instances
- **Test**: 14 signers, 100 classes, 378 instances

Signer intersection between splits is guaranteed empty:
$$\text{Signers}_{\text{train}} \cap \text{Signers}_{\text{val}} = \emptyset, \quad \text{Signers}_{\text{train}} \cap \text{Signers}_{\text{test}} = \emptyset, \quad \text{Signers}_{\text{val}} \cap \text{Signers}_{\text{test}} = \emptyset$$

## Video Recovery Status

- Currently available videos: ~1,019
- Missing videos: ~1,019
- Recovery request status: Submitted
- Operational policy: Use available videos for initial pipeline sanity tests and architecture iteration; run definitive benchmark training once the remaining videos are restored.
