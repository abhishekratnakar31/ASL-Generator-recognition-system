# Experiment Tracking & Protocols

## Experimentation Principles

1. **Empirical Rigor**: No architecture decision is accepted on intuition; all modifications require quantified comparison against the baseline.
2. **Standard Metrics**: Models are compared on Validation/Test Top-1 Accuracy, Top-5 Accuracy, macro F1, and cross-entropy loss.
3. **Reproducibility**: Experiments must log seed (`seed=42`), exact hyperparameters, and environment specifications.

## Logged Experiments

| Experiment ID | Description | Validation Top-1 | Validation Top-5 | Test Top-1 | Status | Notes |
|---|---|---|---|---|---|---|
| `EXP-00-SMOKE` | ResNet50 + 2-layer LSTM forward/backward smoke test | N/A | N/A | N/A | Completed | Forward pass output `[4, 100]`, initial loss ~4.61, gradients verified. |
| `EXP-01-TRAIN` | ResNet50 (frozen) + 2-layer LSTM training | TBD | TBD | TBD | Next | Initial training with available dataset. |
