# Federated Learning for Privacy-Preserving Disease Prediction in Smart Hospitals

**Final-Year Engineering Project & Academic Prototype (Review 2)**

> **DISCLAIMER**: This project is an educational/research prototype using synthetic healthcare data and is not a clinically validated system.

---

## 1. Project Title
**Federated Learning for Privacy-Preserving Disease Prediction in Smart Hospitals**

---

## 2. Abstract
Modern healthcare institutions possess vast cohorts of electronic health records (EHR). However, strict data protection regulations such as HIPAA and GDPR prevent cross-institutional pooling of raw patient medical records. This project presents a privacy-preserving decentralized machine learning system based on **Federated Learning (FL)** using the **Federated Averaging (FedAvg)** algorithm combined with **$L_2$ update clipping** and **Gaussian noise perturbation**. The system simulates 5 smart hospital nodes training a PyTorch Multi-Layer Perceptron (MLP) for binary disease risk prediction without exposing raw patient data to a central authority.

---

## 3. Problem Statement
Centralized machine learning models require aggregating raw patient data into a central server, introducing significant risks of data breach, privacy violation, and regulatory non-compliance. Conversely, isolated local training leads to data starvation and overfitting due to Non-IID (non-independently and identically distributed) demographic skew across individual hospitals. There is a need for a collaborative training paradigm that achieves global predictive performance comparable to centralized models while ensuring raw patient records remain strictly inside local hospital perimeters.

---

## 4. Objectives
1. **Simulate a Multi-Hospital System**: Construct 5 isolated hospital client environments (`Hospital A`, `Hospital B`, `Hospital C`, `Hospital D`, `Hospital E`) holding 5,000 synthetic patient records.
2. **Implement Non-IID Data Partitioning**: Control demographic heterogeneity (age, glucose, blood pressure, BMI, cholesterol skews) across hospitals.
3. **Train PyTorch MLP Models**: Evaluate binary classification models ($32 \rightarrow 16 \rightarrow 1$) across 4 paradigms:
   - Local-Only Models
   - Centralized Baseline Model (Upper-bound reference)
   - Federated Learning (Standard FedAvg)
   - Federated Learning with Privacy (FedAvg + $L_2$ Clip + Gaussian Noise)
4. **Implement Privacy Mechanism**: Apply $L_2$ norm clipping ($C = 1.0$) and Gaussian noise addition ($\sigma = 0.1$) to local parameter updates.
5. **Develop Interactive Review 2 Dashboard**: Build an 8-tab Streamlit dashboard with real-time model inference and publication-grade visualizations.

---

## 5. System Architecture
```
+------------------+
|   Hospital A     |──┐
| (Geriatric Cohort|  |
+------------------+  |
+------------------+  |
|   Hospital B     |──┤
| (Metabolic Cohort|  |
+------------------+  |      Model Updates (Δw)        +--------------------------+
+------------------+  ├───────────────────────────────► |  CENTRAL FEDERATED SERVER|
|   Hospital C     |──┤                                 |                          |
|  (Younger Cohort) |  |   Global Model Broadcast (W)    |  1. Sample-Weighted      |
+------------------+  | ◄───────────────────────────────|     FedAvg Aggregation   |
+------------------+  |                                 |  2. Global Evaluation    |
|   Hospital D     |──┤                                 +--------------------------+
|(Cardio Cohort)   |  |
+------------------+  |
+------------------+  |
|   Hospital E     |──┘
|  (Mixed Cohort)  |
+------------------+
```

---

## 6. Methodology
1. **Dataset Generation**: 5,000 synthetic patient records generated reproducibly with seed 42.
2. **Preprocessing**: Features standard-scaled and categorical attributes one-hot encoded.
3. **Local Training**: Each hospital node executes 3 local epochs of Adam optimization on its isolated dataset.
4. **Update Obfuscation**: Weight update vectors $\Delta w_k = w_k - W^{global}$ are clipped to $\| \Delta w_k \|_2 \le 1.0$ and perturbed with Gaussian noise $\mathcal{N}(0, (0.1 \cdot 1.0)^2 I)$.
5. **Weighted Aggregation**: Central server aggregates protected updates using sample counts $n_k$.
6. **Global Broadcast**: Server updates global parameters $W^{global}$ and broadcasts them for the next round.

---

## 7. Federated Learning Explanation
Federated Learning enables decentralized model training where data remains on local devices/servers. Instead of transmitting raw training data, participating clients compute local model updates and transmit only parameter weights ($\Delta w$) to a central server.

---

## 8. FedAvg Formula
The central server aggregates local weight updates weighted by each hospital's sample size $n_k$:

$$W_{global} = \sum_{k=1}^K \frac{n_k}{N} W_k$$

where:
- $K = 5$ (Number of participating hospitals)
- $n_k$ = Number of training samples at hospital $k$
- $N = \sum_{k=1}^K n_k = 5,000$ (Total training samples)
- $W_k$ = Updated local model parameters from hospital $k$

---

## 9. Privacy Mechanism
- **$L_2$ Gradient/Update Clipping**: Prevents any single client update from exerting excessive influence:
  $$\overline{\Delta w_k} = \Delta w_k \cdot \min\left(1, \frac{C}{\|\Delta w_k\|_2}\right), \quad C = 1.0$$
- **Gaussian Noise Addition**: Adds calibrated noise to clipped updates:
  $$\tilde{\Delta w_k} = \overline{\Delta w_k} + \mathcal{N}(0, \sigma^2 C^2 I), \quad \sigma = 0.1$$
- **Terminology**: Described as *"DP noise + L2 clipping mechanism without formal privacy accounting"*.

---

## 10. Dataset
- **Total Records**: 5,000 synthetic patients
- **Target Variable**: `disease_risk` (0 = Low Risk, 1 = High Risk)
- **Features**: `age`, `gender`, `blood_pressure_sys`, `blood_pressure_dia`, `glucose`, `cholesterol`, `bmi`, `heart_rate`, `smoking_status`, `hypertension`, `diabetes`, `family_history`.

---

## 11. IID vs Non-IID
- **IID**: Uniform random partition of data across hospitals.
- **Non-IID**: Statistical demographic skew across hospitals:
  - **Hospital A**: Geriatric population (age $\ge 50$)
  - **Hospital B**: Metabolic population (high glucose & diabetes)
  - **Hospital C**: Younger demographic (age $< 40$)
  - **Hospital D**: Cardiovascular population (high BP & cholesterol)
  - **Hospital E**: Mixed population

---

## 12. Experimental Setup
- **Framework**: PyTorch
- **Architecture**: MLP ($13 \rightarrow 32 \rightarrow 16 \rightarrow 1$) with Dropout (0.1) & ReLU
- **Optimizer**: Adam ($\text{lr} = 0.001$)
- **FL Communication Rounds**: 10
- **Local Epochs per Round**: 3
- **Batch Size**: 32

---

## 13. Metrics
Evaluated dynamically from actual model predictions:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**
- **Training Time (seconds)**
- **Communication Volume (bytes)**

---

## 14. How to Run

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Execute Complete Pipeline
```powershell
python main.py
```

### Step 3: Run Unit Tests
```powershell
pytest
```

### Step 4: Launch Interactive Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```

---

## 15. Dashboard Explanation & Role-Based UI
The web application features an enterprise multi-role Smart Hospital portal with automated authentication:

### 15.1 Authentication & Role-Based Access
- **Login Screen**: Secure entry point for Smart Hospital personnel and system administrators.
- **Demo Credentials** (loaded via `config/auth_credentials.json` with SHA-256 hash verification):
  - `hospital_a` through `hospital_e` (Password: `hospital123`) $\rightarrow$ Hospital User Role
  - `admin` (Password: `admin123`) $\rightarrow$ System Administrator Role
- **Hospital User View**: Node-level perimeter view focused on single-hospital operations:
  - Node Status & Local Cohort Metadata
  - Local Clinical Disease Inference Console
  - Federated Training Protocol & Round Progress
  - Local Node Privacy Status & Data Perimeter Enforcement
- **System Administrator View**: Multi-hospital Supervision Console:
  - 8 Supervision Views: *Overview*, *Hospital Network*, *Federated Training*, *Model Comparison*, *Disease Prediction*, *Privacy & Security*, *Architecture*, and *Experiment Results*.
  - Dual Differential Privacy Accounting: Displays formal Opacus Rényi DP ($\varepsilon_{RDP}$) and empirical baseline heuristic ($\varepsilon_{heuristic}$) bounds side-by-side.
  - Complete Pipeline Execution & Artifact Reload Triggers.

---

## 16. Output Files
The pipeline automatically generates:

- **`results/`**:
  - `training_metrics.csv`
  - `model_comparison.csv`
  - `hospital_distribution.csv`
  - `predictions.csv`
  - `final_results.txt`
  - `experiment_results.json`
  - `global_disease_model.pkl`

- **`figures/`**:
  - `training_convergence.png`
  - `model_comparison.png`
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `hospital_distribution.png`
  - `privacy_utility_tradeoff.png`

- **`models/`**:
  - `centralized_model.pt`
  - `federated_model.pt`
  - `federated_private_model.pt`

---

## 17. Limitations
1. Uses synthetic patient records rather than real clinical EHR datasets.
2. Formal Rényi Differential Privacy (RDP) accounting via Opacus is provided alongside baseline empirical heuristics. For production deployment, formal DP parameter tuning should be adapted to specific institutional threat models.
3. Secure Aggregation is simulated pairwise masking rather than production cryptographic secret sharing (e.g., Diffie-Hellman Key Exchange / Shamir's Secret Sharing).
4. Demonstration authentication uses SHA-256 hashed configuration entries. Production environments must integrate enterprise identity providers (OAuth2/OIDC, SAML, or LDAP).

---

## 18. Future Enhancement
1. Integration of real MIMIC-III / eICU clinical datasets.
2. Full production integration of Opacus DP-SGD optimizer wrappers during local PyTorch training.
3. Implementation of cryptographic Diffie-Hellman Secure Aggregation protocols.


---

## 19. Viva Questions and Answers

**Q1: What is Federated Learning and why is it needed in healthcare?**  
*A*: Federated Learning is a decentralized machine learning paradigm where models are trained locally across multiple institutions (hospitals) without transmitting raw patient data to a central server. It is essential in healthcare to comply with strict privacy regulations (HIPAA, GDPR) while enabling multi-institutional AI collaboration.

**Q2: How does FedAvg aggregate model updates?**  
*A*: FedAvg computes a sample-count weighted average of client model weight updates: $W_{global} = \sum_{k=1}^K \frac{n_k}{N} W_k$, ensuring hospitals with larger patient cohorts contribute proportionally to the global model update.

**Q3: What is Non-IID data distribution and how does it affect FL?**  
*A*: Non-IID (non-independently and identically distributed) data means individual hospitals possess patient populations with distinct demographic or clinical distributions (e.g., one hospital has older patients while another has more diabetic patients). Non-IID data causes weight divergence during local training, making FedAvg convergence more challenging compared to IID data.

**Q4: How does $L_2$ clipping and Gaussian noise preserve privacy?**  
*A*: $L_2$ clipping bounds the magnitude of local model updates to a maximum threshold $C$, preventing any single patient's data from exerting extreme influence on weight updates. Adding Gaussian noise $\mathcal{N}(0, \sigma^2 C^2)$ obfuscates exact gradients, protecting against model update inversion and membership inference attacks.

**Q5: Why is Centralized Learning included in the comparison?**  
*A*: Centralized learning pools all patient records into a single dataset before training. It serves as an empirical performance reference (upper bound) to demonstrate how closely Federated Learning approaches centralized accuracy without compromising patient data privacy.
