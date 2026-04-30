# Project XAU V32: Neuro-Symbolic Adversarial PPO

### An Institutional-Grade Quantitative Research & Production Execution Engine for Gold (XAUUSD)

## Architecture: V8 NS-iT-RACPPO

This repository implements the **Neuro-Symbolic iTransformer-RNN Adversarial Continuous PPO (NS-iT-RACPPO)** architecture[cite: 3]. It bridges high-dimensional deep learning with financial physics to produce a high-conviction, low-drawdown execution strategy[cite: 3, 6].

### Core Mathematical Pillars:

*   **Variational Mode Decomposition (VMD)**: Employs non-recursive physics decomposition to extract 5 mathematically pure Intrinsic Mode Functions (IMFs), isolating structural trends from high-frequency noise without the phase-lag inherent in traditional indicators[cite: 3].
*   **Liquid CfC (Closed-form Continuous-time) Cells**: Provides continuous-time signal processing to handle irregular temporal intervals and maintain gradient stability during volatile market regimes[cite: 3].
*   **Bayesian Governor (Monte Carlo Self-Correction)**: Conducts 30 mental simulations per inference pass[cite: 3]. By calculating the variance across these simulations, the model self-corrects: if the standard deviation (uncertainty) exceeds 0.80, the trade is rejected as "low-quality noise"[cite: 5].

---

## Market Regime Detection & Temporal Decay

To ensure the system remains relevant in modern 2026 conditions, the engine implements a Regime-Aware Temporal Decay mechanism that prioritizes modern market dynamics over stale historical data:

*   **Regime Shift Detection**: The system utilizes a proxy for Kullback–Leibler Divergence ($D_{KL}$) calculated as `abs(advantages) / batch_variance`[cite: 7].
*   **Surprise-Based Decay**: When the "Surprise" (prediction error relative to variance) spikes, the model recognizes a potential regime change[cite: 7]. It then triggers an exponentially accelerated decay of old memory, ensuring current physics outweigh historical assumptions[cite: 7].
*   **Weighting Formula**: $Regime Weights = Base Time Weights \times e^{-\alpha \cdot KL Proxy}$[cite: 7].

---

## The Veteran Marathon: Adversarial Regime Polish

Unlike standard Reinforcement Learning agents, this model undergoes a two-phase "Veteran Marathon" training lifecycle[cite: 3]:

1.  **Phase 1: Maturity Champion**: Training on "Pure Physics" to master the baseline geometry of the Gold market[cite: 3].
2.  **Phase 2: Adversarial Polish**: Once matured, an Adversary is introduced to inject dynamic noise and specifically target the model's directional biases[cite: 3]. This forces the "Veteran" brain to develop an immune system against market manipulation and volatility traps[cite: 3].

---

## Validation & Quality Control

The system is built on the principles of **BSc Applied Mathematics and Statistics**, utilizing strict institutional validation to prevent overfitting[cite: 2, 3]:

*   **Purged Walk-Forward Optimization (WFO)**: Enforces strict embargo periods between training and testing folds to prevent data leakage[cite: 3].
*   **Purged Combinatorial CV (CPCV)**: Evaluates the model across multiple non-linear chronological paths to ensure robustness across diverse economic cycles[cite: 3].
*   **Bi-Level Optimization**: Decouples hyperparameter tuning (Optuna) from the primary reinforcement learning loop to prevent strategy lock-in[cite: 3].

---

## Production Implementation

The `/production` layer features a mimic-perfect execution engine designed to synchronize the live environment with the research laboratory[cite: 5]:

*   **2000-Bar Dynamic Buffer**: Stabilizes VMD center-frequencies, ensuring that live IMFs match the mathematical precision of historical research IMFs[cite: 5].
*   **Z-Score Epsilon Sync**: Matches the $1 \times 10^{-9}$ precision of the Cython training core[cite: 5].
*   **Diagnostic Logging**: Every execution is logged with its corresponding Mu (Conviction) and Std (Uncertainty) for post-trade quantitative audits[cite: 5].

---

## Audit: Tournament Matrix

The final "Veteran" model is audited against the "Pure Physics" model to verify the impact of adversarial training:

| Model Brain | Win Rate | Max Drawdown | Avg Uncertainty |
| :--- | :---: | :---: | :---: |
| Maturity (Pure Physics) | 49.25% | 4.49% | 0.2194 |
| **Veteran (Adversarial)** | **54.76%** | **0.91%** | **0.1654** |

**Outcome**: The Adversarial Polish reduced Max Drawdown by approximately 80% while significantly increasing overall signal quality[cite: 6].

---

**Developed by Anthony Cooke**
*BSc Applied Mathematics and Statistics Student*

---

### Instructions for Recruiters:

*   To view the mathematical discovery process, see `/research/v8_model_research.py`.
*   To view the live MT5 implementation, see `/production/mt5_execution_engine.py`.
