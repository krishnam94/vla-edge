# CoRL 2026 Expansion Plan (May 29 deadline)

Based on deep research conducted Apr 6, 2026.

## The Novel Algorithm: ACAM (Adaptive Conformal Action Monitoring)

Combination of 3 principled techniques into one new framework:
1. **Adaptive Conformal Inference (ACI)** - online calibration under shift (Gibbs & Candes NeurIPS 2021)
2. **Learned nonconformity scores** - 52% tighter intervals (LCP 2025, arXiv:2509.21955)
3. **Multi-scale temporal monitoring** - step + chunk + episode level

No prior work combines these for VLA action-space monitoring.

## Priority Implementation Order

| P | Method | Effort | Impact |
|---|--------|--------|--------|
| P0 | ACI online adaptation (~30 lines) | Easy | Critical |
| P0 | Same-env cross-arch (all 3 on LIBERO) | Moderate | Critical |
| P1 | Conformal Risk Control (control clip magnitude) | Moderate | High |
| P1 | AgACI parameter-free (Zaffran ICML 2022) | Easy | High |
| P1 | Causal violation analysis (predict failure from patterns) | Moderate | High |
| P2 | Learned nonconformity scores (LCP) | Hard | High |
| P2 | Weighted CP for non-exchangeability (Barber+ AoS 2023) | Moderate | Medium |
| P3 | Real robot validation | Hard | Critical |

## Key Papers to Build On

- Gibbs & Candes, "Adaptive Conformal Inference" NeurIPS 2021 (arXiv:2106.00170)
- Zaffran et al., "Adaptive Conformal Predictions for Time Series" ICML 2022 (arXiv:2202.07282)
- Barber et al., "CP Beyond Exchangeability" AoS 2023 (arXiv:2202.13415)
- Angelopoulos et al., "Conformal Risk Control" ICLR 2024 (arXiv:2208.02814)
- Lekeufack et al., "Conformal Decision Theory" ICRA 2024 (arXiv:2310.05921)
- LCP, "Learnable Conformal Prediction" 2025 (arXiv:2509.21955)
- SAFE (NeurIPS 2025, arXiv:2506.09937) - closest competitor
- FIPER (NeurIPS 2025, arXiv:2510.09459) - CP-calibrated failure prediction

## Competitor Positioning

| Method | Needs Model Internals? | Training Required? | Formal Guarantees? | Overhead |
|--------|----------------------|-------------------|-------------------|----------|
| SAFE | Yes (features) | Yes (detector) | CP coverage | ~10ms |
| FIPER | Yes (entropy) | No (CP only) | CP coverage | ~5ms |
| Sentinel | Yes (embeddings) | No | None formal | ~50ms |
| **ACAM** | **No (black-box)** | **Optional (LCP)** | **ACI coverage** | **~13us** |

## Decision Point: Apr 20
Decide whether to pursue CoRL based on:
1. Can we get same-env cross-arch working?
2. Can we implement ACI + violation predictiveness in 2 weeks?
3. Do we have time alongside Manning book?
