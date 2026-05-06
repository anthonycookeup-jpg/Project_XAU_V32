# Project XAU V32: Neuro-Symbolic Adversarial PPO

### An Institutional-Grade Quantitative Research & Production Execution Engine for Gold (XAUUSD)

## Architecture: V8 NS-iT-RACPPO

This repository implements the **Neuro-Symbolic iTransformer-RNN Adversarial Continuous PPO (NS-iT-RACPPO)** architecture. It bridges high-dimensional deep learning with financial physics to produce a high-conviction, low-drawdown execution strategy.

### Core Mathematical Pillars:

*   **Variational Mode Decomposition (VMD)**: Employs non-recursive physics decomposition to extract 5 mathematically pure Intrinsic Mode Functions (IMFs), isolating structural trends from high-frequency noise without phase-lag.
*   **12-Head RoPE Council (Rotary Position Embedding)**: Upgrades standard attention mechanisms to inject precise absolute and relative temporal positioning, allowing the model to analyze micro-market structures without losing the macro-context.
*   **Liquid CfC (Closed-form Continuous-time) Cells**: Provides continuous-time signal processing to handle irregular temporal intervals and maintain gradient stability during volatile market regimes.
*   **Bayesian Governor (Monte Carlo Self-Correction)**: Conducts 30 mental simulations per inference pass. If the standard deviation (uncertainty) exceeds 0.80, the trade is rejected as "low-quality noise".

---

## The Communication Highway (PhD Bypass)

*   **Direct-to-Actor Routing**: Traditional AI architectures force all data through a deep analytical bottleneck. This model routes high-frequency, non-stationary "Chaos" sensors directly to the Actor head, skipping the main perception layers. This "PhD Bypass" allows the execution engine to react instantly to sudden regime breaks and volatility shocks before the slower analytical "brain" has finished computing the long-term trend.

---

## Asymmetric Reward Shaping (PINN)

Built on Physics-Informed Neural Network (PINN) principles, the architecture uses Asymmetric Reward Shaping to solve the common reinforcement learning failures of "fear" and "suicide spirals":

*   **The Participation Tax**: Cures model "fear" (hesitation to trade to avoid losses) by applying a micro-penalty for remaining flat when high-probability setups exist. This mathematically forces the bot to actively hunt for alpha.
*   **The Sudden-Death Governor**: Cures "suicide spirals" (revenge trading during drawdowns). This strict quality control circuit breaker acts as a hard fail-safe, immediately halting the bot and flattening positions if a defined drawdown threshold is breached.

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

*   **Regime Shift Detection**: Utilizes a proxy for Kullback–Leibler Divergence ($D_{KL}$) calculated as `abs(advantages) / batch_variance`.
*   **Surprise-Based Decay**: When the prediction error relative to variance (Surprise) spikes, the model recognizes a potential regime change and triggers an exponentially accelerated decay of old memory.
*   **Weighting Formula**: $Regime Weights = Base Time Weights \times exp(-\alpha \times KL Proxy)$.

---

## The Veteran Marathon: Adversarial Regime Polish

This model undergoes a specialized two-phase training lifecycle:

1.  **Phase 1: Maturity Champion**: Training on "Pure Physics" to master the baseline geometry of the Gold market.
2.  **Phase 2: Adversarial Polish**: Once matured, an Adversary is introduced to inject dynamic noise and specifically target the model's directional biases. This forces the "Veteran" brain to develop immunity against market manipulation and volatility traps.

---

## Validation & Quality Control

The system is built on the principles of **Applied Mathematics and Statistics**, utilizing strict institutional validation to prevent overfitting:

*   **Purged Walk-Forward Optimization (WFO)**: Enforces strict embargo periods between training and testing folds to prevent data leakage.
*   **Purged Combinatorial CV (CPCV)**: Evaluates the model across multiple non-linear chronological paths to ensure robustness across diverse economic cycles.
*   **Bi-Level Optimization**: Decouples hyperparameter tuning (Optuna) from the primary reinforcement learning loop to prevent strategy lock-in.

---

## Production Implementation

The `/production` layer features a mimic-perfect execution engine designed to synchronize the live environment with the research laboratory:

*   **2000-Bar Dynamic Buffer**: Stabilizes VMD center-frequencies, ensuring that live IMFs match the mathematical precision of historical research IMFs.
*   **Z-Score Epsilon Sync**: Matches the $1 \times 10^{-9}$ precision of the Cython training core.
*   **Diagnostic Logging**: Every execution is logged with its corresponding Mu (Conviction) and Std (Uncertainty) for post-trade quantitative audits.

---

## Audit: Tournament Matrix

> **Audit Period**: January 2022 – March 2026 (51 Continuous Months)

The final "Veteran" model is audited against the baseline "Pure Physics" model to verify the impact of adversarial training:

| Model Brain | Total Trades | Win Rate | Max Drawdown | Net PnL | Profit Factor | Recovery Factor | Sharpe Ratio | Avg Uncertainty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Maturity (Pure Physics) | 1,597 | 48.65% | 4.77% | -$3,345.93 | 0.89 | -0.70 | -1.37 | 0.2192 |
| **Veteran (Adversarial)** | **1,129** | **54.12%** | **0.95%** | **+$4,592.12** | **1.27** | **4.86** | **2.24** | **0.1653** |

**Outcome:** The Adversarial Polish reduced Max Drawdown by over 80%. By filtering for high-conviction entries through the Bayesian Governor, the model transformed a negative expectancy baseline into a highly stable, positive-carry system with a Recovery Factor of 4.86 and a Sharpe Ratio of 2.24.
---
## Bayesian Calibration and Harmonic Risk Scaling

The V8 architecture utilizes a multi-stage risk management framework that operates on the output of the Bayesian Governor. While the neural network provides directional conviction ($\mu$) and uncertainty ($\sigma$), the execution engine (Cells 11 and 12) optimizes capital allocation based on the model’s self-reported reliability.

### Research Stage: Cell 11 - Calibration & Sensitivity Engine

This stage performs a statistical audit of the model’s "honesty." By binning thousands of inference passes into performance tiers based on Bayesian standard deviation, we identified the specific regimes where the model possesses a true mathematical edge.

* **Institutional Binning:** Trades are categorized from Tier 1 (High Conviction) to Tier 5 (Noise/Rejected).
* **The Profit Factor Diagnostic:** Testing revealed that while the "Toxic" bin (Tier 5) maintained a high win rate, its Profit Factor was significantly lower than the High Conviction tiers. This indicated that high-uncertainty trades were "picking up pennies in front of a steamroller."
* **Threshold Detection:** A High-Risk Threshold was detected at std > 0.1892, allowing for the automatic damping of low-quality signals.

### Execution Stage: Cell 12 - Harmonic Position Scaling

To translate these research findings into production, Cell 12 replaces static lot sizing with a Harmonic Scaling algorithm. This ensures that the system rewards high-conviction signals while protecting the equity curve during periods of Bayesian confusion.

**The Harmonic Multiplier Formula:**
$$LotSize = \frac{Risk \cdot |\mu| \cdot (\frac{ideal\_std}{std})^{1.5}}{Price \cdot 0.01}$$

This formula ensures that position sizes are weighted by the inverse of their uncertainty, amplified by a power exponent of 1.5 to maximize the Sharpe Ratio.

### Head-to-Head Performance: Standard vs. Harmonic Execution

This comparison demonstrates the impact of applying the Harmonic Risk Layer to the Veteran Champion on the same 25,000-bar audit period.

| Metric | Veteran (Standard) | Veteran (Harmonic) | Impact |
| :--- | :--- | :--- | :--- |
| **Net PnL** | $4,592.12 | $5,735.38 | +24.9% |
| **Win Rate** | 54.12% | 54.12% | 0.00% |
| **Max Drawdown** | 0.95% | 0.94% | -0.01% |
| **Profit Factor** | 1.27 | 1.33 | +4.7% |
| **Recovery Factor** | 4.86 | 5.75 | +18.3% |
| **Sharpe Ratio** | 2.24 | 2.56 | +14.3% |
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

## ⚙️ Industrial Hardware Requirements

To maintain the **0.94% Max Drawdown** and execute the **30-pass Bayesian uncertainty auditing**, the hardware stack is partitioned by operational mode:

### **1. Research & Audit Mode (High-Throughput Testing)**
*   **System RAM:** 50GB Minimum (Required for high-dimensional tensor staging and parallel regime decomposition during backtesting).
*   **VRAM:** 22GB Minimum (Optimized for RTX 3090/4090, NVIDIA A100, or H100).
*   **Compute:** CUDA 12.4+ for non-stationary signal decomposition.

### **2. Live Production Mode (Market Execution)**
*   **System RAM:** Optimized for lean, low-latency execution (<1GB binary footprint).
*   **VRAM:** <1GB (Retained for real-time Bayesian Governor auditing and 12-Head RoPE Council operations).
*   **Environment:** **Python 3.11.9** (TensorFlow/XLA Optimized).

> **Architect’s Note:** The 50GB System RAM ceiling is specifically utilized during the "Stress-Test" phase to ensure a strict **Factor of Safety** consistent with quality control standards. Live execution is scaled for maximum throughput efficiency.

**Warning:** Attempting to execute the V8 research suite on hardware with <22GB VRAM will lead to a kernel-level memory overflow during the VMD signal isolation phase.

---

**Developed by Anthony Cooke**  
BSc Applied Mathematics and Statistics Student

---

### Instructions for Recruiters:

*   To view the mathematical discovery process, see [`research/v8_model_research.py`](research/v8_model_research.py).
*   To view the live MT5 implementation, see [`production/mt5_execution_engine.py`](production/mt5_execution_engine.py).
*   To view the raw institutional audit logs, see [`audit/veteran_audit.csv`](audit/veteran_audit.csv) and [`audit/maturity_audit.csv`](audit/maturity_audit.csv).
*   To view the high-resolution Bayesian Uncertainty Heatmap, see [`assets/Veteran_vs_Maturity_Heatmap.png`](assets/Veteran_vs_Maturity_Heatmap.png).
* To view the empirical calibration data, see [`audit/v8_qc_calibration_matrix.csv`](audit/v8_qc_calibration_matrix.csv).
* To view the head-to-head harmonic scaling metrics, see [`audit/v8_harmonic_scaling_report.csv`](audit/v8_harmonic_scaling_report.csv).
* To view the professional diagnostic visualization, see [`assets/v8_calibration_performance_distribution.png`](assets/v8_calibration_performance_distribution.png).
