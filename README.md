# Project XAU V32: Neuro-Symbolic Adversarial PPO

### An Institutional-Grade Quantitative Research & Production Execution Engine for Gold (XAUUSD)

## Architecture: V8 NS-iT-RACPPO

This repository implements the **Neuro-Symbolic iTransformer-RNN Adversarial Continuous PPO (NS-iT-RACPPO)** architecture[cite: 3, 5, 8]. It bridges high-dimensional deep learning with financial physics to produce a high-conviction, low-drawdown execution strategy[cite: 3, 6].

### Core Mathematical Pillars:

*   **Variational Mode Decomposition (VMD)**: Employs non-recursive physics decomposition to extract 5 mathematically pure Intrinsic Mode Functions (IMFs), isolating structural trends from high-frequency noise without phase-lag[cite: 3, 5].
*   **Liquid CfC (Closed-form Continuous-time) Cells**: Provides continuous-time signal processing to handle irregular temporal intervals and maintain gradient stability during volatile market regimes[cite: 3, 5].
*   **Bayesian Governor (Monte Carlo Self-Correction)**: Conducts 30 mental simulations per inference pass[cite: 3, 5]. If the standard deviation (uncertainty) exceeds 0.80, the trade is rejected as "low-quality noise"[cite: 5].

---

## Intermarket Data Fusion & Feature Universe

To achieve higher predictive accuracy, the model utilizes a multi-asset input matrix, performing real-time data fusion across several key markets[cite: 5]:

*   **Primary Target**: XAUUSD (Gold)[cite: 5].
*   **Correlation Proxy**: XAGUSD (Silver)[cite: 5].
*   **Macro-Sentiment Proxy**: US500.cash (S&P 500) used as a dynamic Volatility proxy[cite: 5].
*   **Custom DXY Calculation**: Real-time synthetic US Dollar Index using EURUSD, USDJPY, and GBPUSD[cite: 5].
*   **Weighting Formula**: ((EURUSD^-0.576) * (USDJPY^0.136) * (GBPUSD^-0.119)) * 40.0[cite: 5].

---

## Market Regime Detection & Temporal Decay

The engine implements a Regime-Aware Temporal Decay mechanism to ensure modern dynamics take precedence over historical data[cite: 7]:

*   **Regime Shift Detection**: Utilizes a proxy for Kullback–Leibler Divergence (DKL) calculated as abs(advantages) / batch_variance[cite: 7].
*   **Surprise-Based Decay**: When the prediction error relative to variance (Surprise) spikes, the model recognize a potential regime change and triggers an exponentially accelerated decay of old memory[cite: 7].
*   **Weighting Formula**: Regime Weights = Base Time Weights * exp(-alpha * KL Proxy)[cite: 7].

---

## The Veteran Marathon: Adversarial Regime Polish

This model undergoes a specialized two-phase training lifecycle[cite: 3, 8]:

1.  **Phase 1: Maturity Champion**: Training on "Pure Physics" to master the baseline geometry of the Gold market[cite: 3].
2.  **Phase 2: Adversarial Polish**: An Adversary is introduced to inject dynamic noise specifically targeting directional biases, forcing the "Veteran" brain to develop immunity against market manipulation[cite: 3, 8].

---

## Validation & Quality Control

The system is built on the principles of Applied Mathematics and Statistics, utilizing institutional validation to prevent overfitting[cite: 3, 5]:

*   **Purged Walk-Forward Optimization (WFO)**: Enforces strict embargo periods between training and testing folds[cite: 3].
*   **Purged Combinatorial CV (CPCV)**: Evaluates the model across multiple non-linear chronological paths[cite: 3].
*   **Bi-Level Optimization**: Decouples hyperparameter tuning (Optuna) from the primary RL loop[cite: 3].

---

## Production Implementation

The /production layer features a mimic-perfect execution engine[cite: 5]:

*   **2000-Bar Dynamic Buffer**: Stabilizes VMD center-frequencies to ensure live IMFs match research precision[cite: 5].
*   **Z-Score Epsilon Sync**: Matches the 1e-9 precision of the Cython training core[cite: 5].
*   **Diagnostic Logging**: Every execution is logged with its corresponding Mu (Conviction) and Std (Uncertainty)[cite: 5].

---

## Audit: Tournament Matrix

The final "Veteran" model is audited against the "Pure Physics" model to verify the impact of adversarial training[cite: 6]:

| Model Brain | Win Rate | Max Drawdown | Avg Uncertainty |
| :--- | :---: | :---: | :---: |
| Maturity (Pure Physics) | 49.25% | 4.49% | 0.2194 |
| **Veteran (Adversarial)** | **54.76%** | **0.91%** | **0.1654** |

**Outcome**: The Adversarial Polish reduced Max Drawdown by approximately 80% while significantly increasing overall signal quality[cite: 6].

---

## Technical Evolution & Research Roadmap

The current V8 architecture is the culmination of an iterative development path focused on solving specific market physics challenges[cite: 3, 5, 8]:

| Date | Engine / Architecture | Key Technical Breakthrough |
| :--- | :--- | :--- |
| 2025-09-02 | **MT5 Scripting** | Initial plotting and automated moving average testing[cite: 8]. |
| 2025-10-04 | **LSTM Model** | First transition into temporal sequence modeling[cite: 8]. |
| 2026-01-08 | **CNN-BiLSTM** | Integration of spatial feature extraction and bidirectional memory[cite: 8]. |
| 2026-02-01 | **TGAF-CNN-BiLSTM** | Introduction of gated activation functions for gradient control[cite: 8]. |
| 2026-02-06 | **Transformer-CNN-BiLSTM** | Implementation of Multi-Head Attention mechanisms[cite: 8]. |
| 2026-03-03 | **CEEMDAN-iT-PPO** | **Major Pivot**: Addition of PPO Reinforcement Learning, PINN Physics, and Purged Validation[cite: 3, 5, 8]. |
| 2026-03-22 | **VMD-TCN-TLSTM-CFC** | Migration to non-recursive VMD physics and Liquid CfC cells[cite: 3, 5, 8]. |
| 2026-04-10 | **NS-iT-RACPPO** | **Current SOTA**: Integration of Adversarial Polish and Bayesian Governor[cite: 3, 5, 8]. |

---

**Developed by Anthony Cooke**
BSc Applied Mathematics and Statistics Student

---

### Instructions for Recruiters:

*   To view the mathematical discovery process, see `/research/v8_model_research.py`.
*   To view the live MT5 implementation, see `/production/mt5_execution_engine.py`.
