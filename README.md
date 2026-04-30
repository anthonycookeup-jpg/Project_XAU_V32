# Project XAU V32: Neuro-Symbolic Adversarial PPO

### An Institutional-Grade Quantitative Research & Production Execution Engine for Gold (XAUUSD)

## Architecture: V8 NS-iT-RACPPO

This repository implements the **Neuro-Symbolic iTransformer-RNN Adversarial Continuous PPO (NS-iT-RACPPO)** architecture. It bridges high-dimensional deep learning with financial physics to produce a high-conviction, low-drawdown execution strategy.

### Core Mathematical Pillars:

*   **Variational Mode Decomposition (VMD)**: Employs non-recursive physics decomposition to extract 5 mathematically pure Intrinsic Mode Functions (IMFs), isolating structural trends from high-frequency noise without phase-lag.
*   **Liquid CfC (Closed-form Continuous-time) Cells**: Provides continuous-time signal processing to handle irregular temporal intervals and maintain gradient stability during volatile market regimes.
*   **Bayesian Governor (Monte Carlo Self-Correction)**: Conducts 30 mental simulations per inference pass. If the standard deviation (uncertainty) exceeds 0.80, the trade is rejected as "low-quality noise".

---

## Intermarket Data Fusion & Feature Universe

To achieve higher predictive accuracy, the model utilizes a multi-asset input matrix, performing real-time data fusion across several key markets:

*   **Primary Target**: XAUUSD (Gold).
*   **Correlation Proxy**: XAGUSD (Silver).
*   **Macro-Sentiment Proxy**: US500.cash (S&P 500) used as a dynamic Volatility proxy.
*   **Custom DXY Calculation**: Real-time synthetic US Dollar Index using EURUSD, USDJPY, and GBPUSD.
*   **Weighting Formula**: ((EURUSD^-0.576) * (USDJPY^0.136) * (GBPUSD^-0.119)) * 40.0.

---

## Market Regime Detection & Temporal Decay

The engine implements a Regime-Aware Temporal Decay mechanism to ensure modern dynamics take precedence over historical data:

*   **Regime Shift Detection**: Utilizes a proxy for Kullback–Leibler Divergence (DKL) calculated as abs(advantages) / batch_variance.
*   **Surprise-Based Decay**: When the prediction error relative to variance (Surprise) spikes, the model recognizes a potential regime change and triggers an exponentially accelerated decay of old memory.
*   **Weighting Formula**: Regime Weights = Base Time Weights * exp(-alpha * KL Proxy).

---

## The Veteran Marathon: Adversarial Regime Polish

This model undergoes a specialized two-phase training lifecycle:

1.  **Phase 1: Maturity Champion**: Training on "Pure Physics" to master the baseline geometry of the Gold market.
2.  **Phase 2: Adversarial Polish**: Once matured, an Adversary is introduced to inject dynamic noise specifically targeting directional biases, forcing the "Veteran" brain to develop immunity against market manipulation and volatility traps.

---

## Validation & Quality Control

The system is built on the principles of Applied Mathematics and Statistics, utilizing institutional validation to prevent overfitting:

*   **Purged Walk-Forward Optimization (WFO)**: Enforces strict embargo periods between training and testing folds to prevent data leakage.
*   **Purged Combinatorial CV (CPCV)**: Evaluates the model across multiple non-linear chronological paths to ensure robustness across diverse economic cycles.
*   **Bi-Level Optimization**: Decouples hyperparameter tuning (Optuna) from the primary reinforcement learning loop to prevent strategy lock-in.

---

## Production Implementation

The /production layer features a mimic-perfect execution engine designed to synchronize the live environment with the research laboratory:

*   **2000-Bar Dynamic Buffer**: Stabilizes VMD center-frequencies to ensure live IMFs match research precision.
*   **Z-Score Epsilon Sync**: Matches the 1e-9 precision of the Cython training core.
*   **Diagnostic Logging**: Every execution is logged with its corresponding Mu (Conviction) and Std (Uncertainty) for post-trade quantitative audits.

---

## Audit: Tournament Matrix

> **Audit Period**: January 2022 – March 2026 (51 Continuous Months)

The final "Veteran" model is audited against the baseline "Pure Physics" model to verify the impact of adversarial training:

| Model Brain | Total Trades | Win Rate | Max Drawdown | Net PnL | Profit Factor | Avg Uncertainty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Maturity (Pure Physics) | 1,594 | 49.25% | 4.49% | -$2,605.91 | 0.98 | 0.2194 |
| **Veteran (Adversarial)** | **1,114** | **54.76%** | **0.91%** | **+$4,870.02** | **1.24** | **0.1654** |

**Outcome**: The Adversarial Polish reduced Max Drawdown by approximately 80%. By filtering for high-conviction entries through the Bayesian Governor, the model transformed a negative expectancy baseline into a stable, positive-carry system.

---

## Technical Evolution & Research Roadmap

The current V8 architecture is the culmination of an iterative development path focused on solving specific market physics challenges:

| Date | Engine / Architecture | Key Technical Breakthrough |
| :--- | :--- | :--- |
| 2025-09-02 | **MT5 Scripting** | Initial plotting and automated moving average testing. |
| 2025-10-04 | **LSTM Model** | First transition into temporal sequence modeling. |
| 2026-01-08 | **CNN-BiLSTM** | Integration of spatial feature extraction and bidirectional memory. |
| 2026-02-01 | **TGAF-CNN-BiLSTM** | Introduction of gated activation functions for gradient control. |
| 2026-02-06 | **Transformer-CNN-BiLSTM** | Implementation of Multi-Head Attention mechanisms. |
| 2026-03-03 | **CEEMDAN-iT-PPO** | **Major Pivot**: Addition of PPO Reinforcement Learning, PINN Physics, and Purged Validation. |
| 2026-03-22 | **VMD-TCN-TLSTM-CFC** | Migration to non-recursive VMD physics and Liquid CfC cells. |
| 2026-04-10 | **NS-iT-RACPPO** | **Current SOTA**: Integration of Adversarial Polish and Bayesian Governor. |

---

**Developed by Anthony Cooke**  
BSc Applied Mathematics and Statistics Student

---

### Instructions for Recruiters:

*   To view the mathematical discovery process, see `/research/v8_model_research.py`.
*   To view the live MT5 implementation, see `/production/mt5_execution_engine.py`.
