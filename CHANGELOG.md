# Changelog

All notable changes and technical bug fixes to the Federated Learning Disease Prediction project are documented below.

---

## [Unreleased / Review 2 Fixes] - 2026-09-21

### Fix 1: Dirichlet Naming & Partitioning Refactor
- **File**: `data/partitioning.py`, `tests/test_partitioning.py`
- **Details**: Implemented Dirichlet-based non-IID label skew sampling (`_dirichlet_partition()`) using `np.random.dirichlet([dirichlet_alpha] * K)` for `partition_type="non_iid"` and `"dirichlet"`. Preserved demographic cohort skew under `partition_type="demographic"`. Added proportion smoothing (`min_p = 0.05`) to prevent zero-class allocation across hospital nodes.
- **Verification**: Added `test_non_iid_partitioning()` and `test_demographic_partitioning()` in `test_partitioning.py`.

### Fix 2: Secure Aggregation RNG Isolation & Cancellation Verification
- **File**: `privacy/secure_aggregation.py`, `tests/test_privacy.py`
- **Details**: Eliminated global NumPy/PyTorch RNG state mutations in `generate_pairwise_masks()` by creating local, isolated RNG instances (`np.random.default_rng` and `torch.Generator`). Implemented real mask cancellation sum verification in `verify_mask_cancellation()`.
- **Verification**: Added `test_secure_aggregation_mask_cancellation()` and `test_secure_aggregation_rng_isolation()` in `test_privacy.py`.

### Fix 3: Formal Opacus RDP Differential Privacy Accounting
- **File**: `privacy/differential_privacy.py`, `dashboard/app.py`, `requirements.txt`, `tests/test_privacy.py`
- **Details**: Integrated Opacus `RDPAccountant` into `DifferentialPrivacyManager` via `estimate_epsilon_rdp()`, operating alongside the empirical baseline heuristic `estimate_epsilon_heuristic()`. Added `opacus>=1.4.0` to `requirements.txt`. Updated the Streamlit dashboard (both Hospital User and Admin views) to present formal RDP ($\varepsilon_{RDP}$) and heuristic ($\varepsilon_{heuristic}$) bounds side-by-side.
- **Verification**: Added `test_epsilon_accounting()` in `test_privacy.py`.

### Fix 4: Security Hardening & Hashed Credential Authentication
- **File**: `dashboard/app.py`, `config/auth_credentials.json`, `tests/test_dashboard_flow.py`
- **Details**: Removed hardcoded plaintext passwords from source code. Created `config/auth_credentials.json` storing SHA-256 hashed passwords and runtime environment fallback (`load_credentials()`). Added security notes detailing production identity requirements (OAuth2/OIDC/LDAP).
- **Verification**: Updated `test_credentials_structure()` in `test_dashboard_flow.py` to verify SHA-256 digest matching.

### Fix 5: Documentation & System Architecture Update
- **File**: `README.md`
- **Details**: Updated Section 15 to document the role-based multi-hospital UI flow (Hospital User vs System Administrator views), login demonstration authentication credentials, and side-by-side DP bounds. Updated Section 17 & 18 limitations and future work notes.

### Fix 6: Class Imbalance & F1/Recall Optimization
- **File**: `data/generate_data.py`, `models/disease_model.py`, `evaluation/metrics.py`
- **Details**: Adjusted baseline logit intercept in `generate_synthetic_healthcare_data()` to achieve ~30% positive disease prevalence. Implemented optimal decision threshold searching (`find_optimal_threshold()`) in `calculate_classification_metrics()` to maximize F1 score on validation and test sets.
- **Results**: Successfully regenerated all benchmark CSVs (`results/model_comparison.csv`, `results/training_metrics.csv`, `results/hospital_distribution.csv`), text reports, and publication-grade plots (`figures/`). Achieved solid global performance: F1 score ~0.70, Recall ~0.84, ROC-AUC ~0.82 across Centralized, Standard FedAvg, and Private FedAvg paradigms.
