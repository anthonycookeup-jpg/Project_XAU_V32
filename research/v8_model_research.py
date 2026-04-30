# --- CELL 0 & 1: SETUP & QUANTITATIVE DEPENDENCIES ---
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
print("Precision Policy: FLOAT32 Engaged for Maximum Gradient Stability.")

print("--- Installing Institutional Quant Dependencies ---")
!pip install EMD-signal ta backtrader optuna tf2onnx gymnasium scipy statsmodels arch vmdpy > /dev/null

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings, gc, itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as stats
from scipy.stats import entropy

try:
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.optimizer.set_jit(True)
    print("Hardware Turbo: XLA and Multi-threading Engaged.")
except RuntimeError:
    print("TF Memory locked.")

import gymnasium as gym
from vmdpy import VMD 
import ta
from sklearn.preprocessing import RobustScaler
import scipy.signal

warnings.filterwarnings("ignore")
print("--- GPU and Environment Ready ---")


# --- CELL 2: INTERMARKET FUSION & VMD PRE-COMPUTATION ---
from google.colab import drive
import os
import pandas as pd
import numpy as np
from vmdpy import VMD
import ta
import scipy.stats as stats
from scipy.stats import entropy

print("--- Mounting Google Drive ---")
drive.mount('/content/drive')
data_path = "/content/drive/MyDrive/Forex Data"

primary_asset = 'XAUUSD'
intermarket_assets = ['XAGUSD', 'USDJPY', 'EURUSD', 'GBPUSD', 'US500.cash']
all_assets = [primary_asset] + intermarket_assets
timeframes = ['H1', 'H4']

def get_weights_fracdiff(d, size):
    w = [1.]
    for k in range(1, size):
        w.append(-w[-1] / k * (d - k + 1))
    return np.array(w[::-1])

def frac_diff(series, d, window=12):
    weights = get_weights_fracdiff(d, window)
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

def load_tf_data(asset, tf_str):
    filename = f"{asset}_{tf_str}_Recent.csv"
    fpath = os.path.join(data_path, filename)
    if not os.path.exists(fpath):
        print(f"Could not find {fpath}")
        return pd.DataFrame()

    df = pd.read_csv(fpath, sep='\\t', header=0,
                     names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'SpreadRaw'],
                     engine='python')

    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                 format='%Y.%m.%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime')

    if asset == 'XAUUSD':
        df['Spread'] = df['SpreadRaw'] * 0.01

    cols = {
        'Open': f'{asset}_{tf_str}_Open',
        'High': f'{asset}_{tf_str}_High',
        'Low': f'{asset}_{tf_str}_Low',
        'Close': f'{asset}_{tf_str}_Close',
        'TickVol': f'{asset}_{tf_str}_Volume'
    }
    if asset == 'XAUUSD' and tf_str == 'H1':
        cols['Spread'] = 'Spread'

    return df[list(cols.keys())].rename(columns=cols).astype(np.float32)

def apply_feature_bank(df):
    df = df.copy()
    close = df['XAUUSD_H1_Close']
    high = df['XAUUSD_H1_High']
    low = df['XAUUSD_H1_Low']
    vol = df['XAUUSD_H1_Volume']

    df['Log_Return'] = np.log(close / close.shift(1))
    df['FracDiff_045'] = frac_diff(close, 0.45, window=12)
    df['Body_Size'] = abs(df['XAUUSD_H1_Open'] - close)
    df['Upper_Wick'] = high - df[['XAUUSD_H1_Open', 'XAUUSD_H1_Close']].max(axis=1)
    df['Lower_Wick'] = df[['XAUUSD_H1_Open', 'XAUUSD_H1_Close']].min(axis=1) - low

    for w in [2, 3, 5, 8, 13, 21, 27, 34, 44, 55, 89, 144]:
        df[f'ATR_{w}'] = ta.volatility.average_true_range(high, low, close, window=w)

    for w in [2, 5, 8, 13, 21, 34, 55, 89, 144]:
        df[f'RSI_{w}'] = ta.momentum.rsi(close, window=w)

    for w in [8, 13, 21, 34, 55, 89, 144, 200, 233, 300]:
        df[f'EMA_{w}'] = ta.trend.ema_indicator(close, window=w)

    for w in [13, 21, 34, 55, 144]:
        df[f'VWAP_{w}'] = ta.volume.volume_weighted_average_price(high, low, close, vol, window=w)

    macd_pairs = [(5,13), (8,21), (12,26), (21,55), (34,89), (55,144), (144,300)]
    for f, s in macd_pairs:
        df[f'MACD_{f}_{s}'] = ta.trend.macd(close, window_fast=f, window_slow=s)

    for w in [13, 21, 34, 55, 89]:
        df[f'BB_Width_{w}'] = ta.volatility.bollinger_wband(close, window=w)

    df['OBV'] = ta.volume.on_balance_volume(close, vol)

    df['Skew_14'] = close.rolling(14).skew()
    df['Kurt_14'] = close.rolling(14).kurt()

    lag1_var = close.diff(1).rolling(14).var()
    lag5_var = close.diff(5).rolling(14).var()
    df['Hurst_Proxy'] = np.log(lag5_var / (lag1_var + 1e-8)) / np.log(5)

    def rolling_entropy(series, window):
        return series.rolling(window).apply(lambda x: entropy(np.histogram(x, bins=10)[0] + 1e-9), raw=True)
    df['Shannon_Entropy_14'] = rolling_entropy(df['Log_Return'].fillna(0), 14)

    df['OU_Tension'] = (close - ta.trend.sma_indicator(close, 55)) / (df['ATR_55'] + 1e-8)

    df['FVG_Bull'] = (low - high.shift(2)).clip(lower=0)
    df['FVG_Bear'] = (low.shift(2) - high).clip(lower=0)
    df['Liquidity_Gap'] = df['FVG_Bull'] - df['FVG_Bear']

    if 'DXY_H1_Close' in df.columns:
        dxy_ret = np.log(df['DXY_H1_Close'] / df['DXY_H1_Close'].shift(1))
        dxy_z = (dxy_ret - dxy_ret.rolling(24).mean()) / (dxy_ret.rolling(24).std() + 1e-8)
        df['Macro_Shock'] = dxy_z.where(abs(dxy_z) > 2.5, 0.0)
    else:
        df['Macro_Shock'] = 0.0

    return df.bfill().fillna(0) 

print("\\n--- Initiating Intermarket Data Fusion ---")

h1_frames = {a: load_tf_data(a, 'H1') for a in all_assets}
h4_frames = {a: load_tf_data(a, 'H4') for a in all_assets}

df_final = pd.concat(h1_frames.values(), axis=1, join='inner')

df_final['DXY_H1_Close'] = ((df_final['EURUSD_H1_Close']**-0.576) * (df_final['USDJPY_H1_Close']**0.136) * (df_final['GBPUSD_H1_Close']**-0.119)) * 40.0

df_h4_all = pd.concat(h4_frames.values(), axis=1, join='inner')
df_h4_all['DXY_H4_Close'] = ((df_h4_all['EURUSD_H4_Close']**-0.576) * (df_h4_all['USDJPY_H4_Close']**0.136) * (df_h4_all['GBPUSD_H4_Close']**-0.119)) * 40.0

h4_context = df_h4_all[['XAUUSD_H4_Close', 'XAGUSD_H4_Close', 'DXY_H4_Close', 'US500.cash_H4_Close']].reindex(df_final.index, method='ffill')
df_final = pd.concat([df_final, h4_context], axis=1)

df_final = apply_feature_bank(df_final)

for asset in ['XAGUSD', 'USDJPY', 'DXY', 'US500.cash']:
    df_final[f'{asset}_H1_LogRet'] = np.log(df_final[f'{asset}_H1_Close'] / df_final[f'{asset}_H1_Close'].shift(1))

df_final['VIX_Proxy_ATR_14'] = ta.volatility.average_true_range(
    df_final['US500.cash_H1_High'], df_final['US500.cash_H1_Low'], df_final['US500.cash_H1_Close'], window=14
)

# =====================================================================
# THE V8 VMD UPGRADE: Non-Recursive Physics Decomposition
# =====================================================================
if len(df_final) % 2 != 0:
    df_final = df_final.iloc[1:].copy()
    print(f"Quality Control: Adjusted DataFrame length to {len(df_final)} (Even) to prevent VMD truncation.")

print(f"Executing Elite VMD (Variational Mode Decomposition) on {len(df_final)} synced rows...")
alpha = 2000       
tau = 0            
K = 5              
DC = 0             
init = 1           
tol = 1e-7         

u, u_hat, omega = VMD(df_final['XAUUSD_H1_Close'].values, alpha, tau, K, DC, init, tol)

for i in range(K):
    df_final[f'IMF_{i+1}'] = u[i, :]

df_final = df_final.dropna()
print(f"Data Fusion, Omni-Spectrum, & VMD Compilation Complete.")
print(f"Total Columns: {len(df_final.columns)}")


# --- NEW CELL 2.5: QUALITY CONTROL MATH PHYSICS UNIT TEST ---
import numpy as np
import pandas as pd
from vmdpy import VMD

print("--- INITIALIZING QUALITY CONTROL FACTORY TEST ---")

def test_fvg_logic():
    data = {
        'High': [100.0, 120.0, 130.0],
        'Low': [90.0, 105.0, 110.0]
    }
    df_test = pd.DataFrame(data)
    fvg_bull = (df_test['Low'] - df_test['High'].shift(2)).clip(lower=0)

    assert fvg_bull.iloc[2] == 10.0, f"FVG Math Failure! Expected 10.0, got {fvg_bull.iloc[2]}"
    print("Liquidity Gap (FVG) Physics: PASS")

def test_vmd_integrity():
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

    alpha = 2000
    tau = 0
    K = 2 
    DC = 0
    init = 1
    tol = 1e-7

    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)

    assert u.shape[0] == K, f"VMD Failure! Expected {K} modes, got {u.shape[0]}"
    print("VMD Non-Recursive Decomposition: PASS")

test_fvg_logic()
test_vmd_integrity()
print("--- ALL V8 SENSORS CALIBRATED AND VERIFIED ---")


# --- CELL 3: PURGED WFO & CPCV GENERATORS ---

import numpy as np
import itertools

class PurgedWalkForward:
    def __init__(self, n_splits=5, purge_gap=20):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, data_len):
        indices = np.arange(data_len)
        fold_size = data_len // self.n_splits
        splits = []
        for i in range(self.n_splits - 1):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.purge_gap
            test_end = min((i + 2) * fold_size, data_len)
            splits.append((indices[:train_end], indices[test_start:test_end]))
        return splits

class PurgedCombinatorialCV:
    def __init__(self, n_groups=6, test_groups=2, purge_gap=20):
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.purge_gap = purge_gap

    def split(self, data_len):
        group_size = data_len // self.n_groups
        groups = [np.arange(i * group_size, (i + 1) * group_size) for i in range(self.n_groups)]
        test_combinations = list(itertools.combinations(range(self.n_groups), self.test_groups))
        splits = []
        for test_idx in test_combinations:
            train_idx = [i for i in range(self.n_groups) if i not in test_idx]
            test_indices = np.concatenate([groups[i] for i in test_idx])
            train_indices = []
            for t_group in train_idx:
                chunk = groups[t_group].copy()
                if (t_group + 1) in test_idx: chunk = chunk[:-self.purge_gap]
                if (t_group - 1) in test_idx: chunk = chunk[self.purge_gap:]
                if len(chunk) > 0: train_indices.append(chunk)
            splits.append((np.concatenate(train_indices), test_indices))
        return splits


# Cell 4:
%load_ext cython

# Cell 4 A:
%%cython -a
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport log, sqrt

cdef class FastMarketEnvCore:
    cdef:
        double[:, :] df_view
        double[:] spread_array
        int lookback, n_features, max_steps, atr_index, close_index
        public int current_step
        public double current_exposure  
        double action_tax, max_allowable_spread
        float[:, :] obs_buffer

        double initial_balance, current_balance, peak_balance

    def __init__(self, double[:, :] df_array, double[:] spread_array, int lookback, int atr_index, int close_index):
        self.df_view = df_array
        self.spread_array = spread_array
        self.lookback = lookback
        self.n_features = df_array.shape[1]
        self.atr_index = atr_index
        self.close_index = close_index
        self.max_steps = df_array.shape[0] - 1
        self.current_step = lookback

        self.action_tax = 0.00030
        self.max_allowable_spread = 0.60
        self.current_exposure = 0.0 

        self.initial_balance = 10000.0
        self.current_balance = 10000.0
        self.peak_balance = 10000.0

        self.obs_buffer = np.zeros((self.lookback, self.n_features), dtype=np.float32)

    cpdef float[:, :] get_obs(self):
        cdef int i, j
        cdef int start = self.current_step - self.lookback
        cdef double val, mean, std, sum_val, sq_sum, variance

        for j in range(self.n_features):
            sum_val = 0.0
            sq_sum = 0.0

            for i in range(self.lookback):
                val = self.df_view[start + i, j]
                sum_val += val
                sq_sum += val * val

            mean = sum_val / self.lookback

            variance = (sq_sum / self.lookback) - (mean * mean)
            if variance < 0:
                variance = 0.0

            std = sqrt(variance) + 1e-9

            for i in range(self.lookback):
                self.obs_buffer[i, j] = <float>((self.df_view[start + i, j] - mean) / std)

        return self.obs_buffer

    cpdef tuple step(self, double conviction):
        if conviction > 1.0: conviction = 1.0
        if conviction < -1.0: conviction = -1.0

        cdef double current_price = self.df_view[self.current_step, self.close_index]
        cdef double next_price = self.df_view[self.current_step + 1, self.close_index]
        cdef double atr = self.df_view[self.current_step, self.atr_index]
        cdef double current_spread = self.spread_array[self.current_step]

        cdef double physics_penalty = 0.0

        if current_spread > self.max_allowable_spread:
            conviction = 0.0 
            physics_penalty += 0.0001

        cdef double log_ret = log(next_price / current_price)
        cdef double spread_percent = current_spread / current_price

        cdef double delta_exposure = conviction - self.current_exposure
        if delta_exposure < 0: delta_exposure = -delta_exposure 

        cdef double transaction_cost = delta_exposure * (spread_percent + self.action_tax)

        self.current_exposure = conviction

        cdef double financial_return = (conviction * log_ret) - transaction_cost

        if conviction > -0.05 and conviction < 0.05:
            financial_return -= 0.00005

        if financial_return < 0:
            financial_return *= 2.5

        if conviction > 0.05 or conviction < -0.05:
            if atr < (current_spread * 3.0):
                physics_penalty += (current_spread * 3.0) - atr

        self.current_balance *= (1.0 + financial_return)

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        cdef double drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        cdef double risk_penalty = 0.0
        cdef bint done = False

        if drawdown > 0.05:
            risk_penalty = 5.0
            done = True

        cdef double final_reward = financial_return - physics_penalty - risk_penalty

        self.current_step += 1

        if self.current_step >= self.max_steps or self.current_balance < self.initial_balance * 0.5:
            done = True

        return final_reward, done

    cpdef void reset(self):
        self.current_step = self.lookback
        self.current_exposure = 0.0
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance


# --- CELL 4B: GYMNASIUM WRAPPER FOR CYTHON CORE ---
import gymnasium as gym
import numpy as np

class PurgedMarketEnv(gym.Env):
    def __init__(self, df, lookback=64):
        super(PurgedMarketEnv, self).__init__()

        self.df_values = np.ascontiguousarray(df.values, dtype=np.float64)
        self.spread_values = np.ascontiguousarray(df['Spread'].values, dtype=np.float64)
        self.lookback = lookback

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, df.shape[1]), dtype=np.float32)

        self.atr_index = list(df.columns).index('ATR_13')
        self.close_index = list(df.columns).index('XAUUSD_H1_Close')

        self.core = FastMarketEnvCore(
            self.df_values,
            self.spread_values,
            self.lookback,
            self.atr_index,
            self.close_index
        )

    def reset(self, seed=None, options=None):
        self.core.reset()
        return np.asarray(self.core.get_obs()), {}

    def step(self, action):
        conviction = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
        reward, done = self.core.step(conviction)
        return np.asarray(self.core.get_obs()), reward, done, False, {}


# --- TEST CELL: ENV STRESS TEST (Spectrum Upgraded) ---
import numpy as np
import pandas as pd

print("Testing Environment for NaN stability...")

num_features = len(df_final.columns)
dummy_df = pd.DataFrame(np.ones((1000, num_features)) * 2600.0, columns=df_final.columns)

dummy_df['Spread'] = 0.1
dummy_df['XAUUSD_H1_Close'] = 2600.0

dummy_df['ATR_S'] = 1.0

test_env = PurgedMarketEnv(dummy_df, lookback=180)
obs, _ = test_env.reset()

nan_detected = False
for i in range(500):
    obs, reward, done, _, _ = test_env.step(0)
    if np.isnan(obs).any():
        print(f"FAIL: NaN detected in observation at step {i}!")
        nan_detected = True
        break

if not nan_detected:
    print("PASS: Environment is mathematically stable even with flat data.")



# --- CELL 5: V8 FLUID INTELLIGENCE ARCHITECTURE (OMNISCIENT LIQUIDITY ENGINE) ---
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, Dense, LayerNormalization, MultiHeadAttention, Concatenate, GlobalAveragePooling1D, Reshape, Multiply, RNN, Add

@tf.keras.utils.register_keras_serializable()
class TLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(TLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = (units, units)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units * 4), initializer='glorot_uniform', name='W')
        self.U = self.add_weight(shape=(self.units, self.units * 4), initializer='orthogonal', name='U')
        self.b = self.add_weight(shape=(self.units * 4,), initializer='zeros', name='b')
        self.W_d = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='W_d')
        self.built = True

    def call(self, inputs, states):
        h_prev, c_prev = states
        decay = tf.math.exp(-tf.nn.softplus(tf.matmul(inputs, self.W_d)))
        c_prev_decayed = c_prev * decay

        z = tf.matmul(inputs, self.W) + tf.matmul(h_prev, self.U) + self.b
        i, f, o, c_tilde = tf.split(z, 4, axis=-1)

        i = tf.math.sigmoid(i)
        f = tf.math.sigmoid(f)
        o = tf.math.sigmoid(o)
        c_tilde = tf.math.tanh(c_tilde)

        c_t = f * c_prev_decayed + i * c_tilde
        h_t = o * tf.math.tanh(c_t)

        return h_t, [h_t, c_t]

    def get_config(self):
        config = super(TLSTMCell, self).get_config()
        config.update({"units": self.units})
        return config

@tf.keras.utils.register_keras_serializable()
class RoPEMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(RoPEMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def apply_rotary_positional_embedding(self, tensor):
        seq_len = tf.shape(tensor)[1]
        dim = tf.shape(tensor)[-1]
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, dim, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / tf.cast(dim, tf.float32)))
        pos_emb_even = tf.sin(positions * div_term)
        pos_emb_odd = tf.cos(positions * div_term)

        pos_emb = tf.reshape(tf.stack([pos_emb_even, pos_emb_odd], axis=-1), [seq_len, dim])
        return tensor + pos_emb[tf.newaxis, :, :]

    def call(self, query, value, key):
        q_rope = self.apply_rotary_positional_embedding(query)
        k_rope = self.apply_rotary_positional_embedding(key)
        return self.mha(query=q_rope, value=value, key=k_rope)

    def get_config(self):
        config = super(RoPEMultiHeadAttention, self).get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class VariableSelectionNetwork(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_dense = Dense(input_shape[-1], activation='softmax', name='feature_weights')
        super(VariableSelectionNetwork, self).build(input_shape)

    def call(self, inputs):
        attention_weights = self.attention_dense(inputs)
        gated_inputs = Multiply()([inputs, attention_weights])
        return gated_inputs, attention_weights

    def get_config(self):
        return super(VariableSelectionNetwork, self).get_config()

@tf.keras.utils.register_keras_serializable()
class LiquidCfC(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LiquidCfC, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.backbone = Dense(self.units, activation='gelu')
        self.time_gate = Dense(self.units, activation='sigmoid')
        self.A = Dense(self.units, activation='tanh')
        super(LiquidCfC, self).build(input_shape)

    def call(self, inputs):
        features, vsn_weights = inputs
        vsn_pooled = tf.reduce_mean(vsn_weights, axis=1)
        tau_modulator = self.time_gate(vsn_pooled)
        x = self.backbone(features)
        time_decay = tau_modulator
        return x * time_decay + self.A(x) * (1.0 - time_decay)

    def get_config(self):
        config = super(LiquidCfC, self).get_config()
        config.update({"units": self.units})
        return config

@tf.keras.utils.register_keras_serializable()
class BayesianGovernor(tf.keras.layers.Layer):
    def __init__(self, rate, mc_passes=30, **kwargs):
        super(BayesianGovernor, self).__init__(**kwargs)
        self.rate = rate
        self.mc_passes = mc_passes 

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        else:
            tiled = tf.tile(tf.expand_dims(inputs, 0), [self.mc_passes, 1, 1])
            dropped = tf.nn.dropout(tiled, rate=self.rate)
            mean_rep = tf.reduce_mean(dropped, axis=0)
            var_rep = tf.math.reduce_variance(dropped, axis=0)
            mean_var = tf.reduce_mean(var_rep, axis=-1, keepdims=True)
            confidence_gate = tf.exp(-mean_var * 10.0)
            return mean_rep * confidence_gate

    def get_config(self):
        config = super(BayesianGovernor, self).get_config()
        config.update({"rate": self.rate, "mc_passes": self.mc_passes})
        return config

def build_ppo_agent(lookback, n_features, n_phd_sensors=7):
    inputs = Input(shape=(lookback, n_features), name='Intermarket_Input')

    pattern_inputs = tf.keras.layers.Lambda(lambda x: x[:, :, :-n_phd_sensors], name='split_patterns')(inputs)
    phd_inputs = tf.keras.layers.Lambda(lambda x: x[:, :, -n_phd_sensors:], name='split_phd')(inputs)

    vsn_gated_inputs, vsn_weights = VariableSelectionNetwork()(pattern_inputs)

    tcn = vsn_gated_inputs
    for d in [1, 2, 4, 8]:
        tcn = Conv1D(64, kernel_size=3, dilation_rate=d, padding='causal', activation='gelu')(tcn)
        tcn = LayerNormalization()(tcn)

    tcn = Conv1D(128, kernel_size=3, padding='causal', activation='gelu')(tcn)
    tcn = LayerNormalization()(tcn)

    tlstm_layer = RNN(TLSTMCell(128), return_sequences=True)
    lstm = Bidirectional(tlstm_layer)(vsn_gated_inputs)
    lstm = LayerNormalization()(lstm)

    rope_attn = RoPEMultiHeadAttention(num_heads=12, key_dim=64)
    cnn_attended_raw = rope_attn(query=tcn, value=lstm, key=lstm)

    cnn_added = Add()([tcn, cnn_attended_raw])
    cnn_attended = LayerNormalization()(cnn_added)

    cnn_flat = GlobalAveragePooling1D()(cnn_attended)
    tft_flat = GlobalAveragePooling1D()(lstm)
    tft_projected = Dense(128, activation='gelu', name='tft_projection')(tft_flat)

    bilinear = Multiply()([cnn_flat, tft_projected])
    merged_patterns = Concatenate(axis=-1)([cnn_flat, tft_projected, bilinear])

    liquid_cfc = LiquidCfC(128)
    fluid_features = liquid_cfc([merged_patterns, vsn_weights])

    phd_inverted = tf.keras.layers.Permute((2, 1), name='iTransformer_Invert')(phd_inputs)

    phd_projected = Dense(32, activation='swish', name='phd_projection')(phd_inverted)

    phd_gate = Dense(32, activation='sigmoid', name='phd_glu_gate')(phd_projected)
    phd_gated = Multiply()([phd_projected, phd_gate])

    phd_flat = GlobalAveragePooling1D()(phd_gated)

    governor = BayesianGovernor(rate=0.3)(fluid_features)
    dense_core = Dense(128, activation='gelu')(governor)

    actor_input = Concatenate()([dense_core, phd_flat])

    actor_mu = Dense(1, activation='tanh', name='actor_mu')(actor_input)
    actor_std_raw = Dense(1, activation='softplus', name='actor_std_raw')(actor_input)
    actor_std = tf.keras.layers.Lambda(lambda x: x + 1e-4, name='actor_std')(actor_std_raw)

    action_params = Concatenate(name='actor_out')([actor_mu, actor_std])

    state_value = Dense(1, activation='linear', name='critic_out')(dense_core)

    return tf.keras.models.Model(inputs=inputs, outputs=[action_params, state_value])

def freeze_foundation(agent):
    print("Initiating Base Freezing for Transfer Learning...")
    for layer in agent.layers:
        if 'actor_out' not in layer.name and 'critic_out' not in layer.name and 'dense' not in layer.name[-5:] and 'liquid' not in layer.name.lower():
            layer.trainable = False
            print(f"   Frozen: {layer.name}")
    print("Base Physics Locked. Actor-Critic Heads ready for 12-Month Regime Polish.")


# Cell 6:
import tensorflow as tf
import numpy as np
import math

def train_ppo_agent_vectorized(envs, agent, optimizer, epochs=1, trajectory_steps=2048, batch_size=64,
                               clip_ratio=0.2, lambda_pinn=0.1, entropy_coef=0.01, adv_strength=0.0):

    is_vectorized = hasattr(envs, 'num_envs')
    n_envs = envs.num_envs if is_vectorized else len(envs)

    if not hasattr(envs, 'current_states') or envs.current_states is None:
        if is_vectorized:
            envs.current_states, _ = envs.reset()
        else:
            envs.current_states = np.array([env.reset()[0] for env in envs])

    states = envs.current_states

    @tf.function(reduce_retracing=True)
    def ppo_update_step(obs_b, act_b, ret_b, old_logp_b, adv_b, tw_b, c_ratio, ent_coef):
        with tf.GradientTape() as tape:
            action_params, state_values = agent(obs_b, training=True)
            state_values = tf.reshape(state_values, [-1])

            mu = action_params[:, 0:1]
            std = action_params[:, 1:2]

            act_b = tf.reshape(act_b, [-1, 1])

            variance = tf.square(std)
            new_logps = -0.5 * (tf.square(act_b - mu) / variance + tf.math.log(variance) + tf.math.log(2.0 * math.pi))
            new_logps = tf.reshape(new_logps, [-1])

            ratio = tf.exp(new_logps - old_logp_b)
            unclipped = ratio * adv_b
            clipped = tf.clip_by_value(ratio, 1.0 - c_ratio, 1.0 + c_ratio) * adv_b

            actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped) * tw_b)
            critic_loss = tf.reduce_mean(tf.square(ret_b - state_values) * tw_b)

            entropy = tf.reduce_mean(0.5 + 0.5 * tf.math.log(2.0 * math.pi * variance))

            total_loss = actor_loss + 0.5 * critic_loss - (ent_coef * entropy)

        grads = tape.gradient(total_loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.trainable_variables))
        return total_loss

    for epoch in range(epochs):
        batch_obs, batch_acts, batch_advs, batch_rets, batch_logps = [], [], [], [], []

        for _ in range(trajectory_steps // n_envs):
            action_params, values = agent(states, training=False)

            mu = action_params[:, 0:1]
            std = action_params[:, 1:2]

            dist = tf.random.normal(tf.shape(mu))
            actions = mu + dist * std
            actions = tf.clip_by_value(actions, -1.0, 1.0) 

            variance = tf.square(std)
            logps = -0.5 * (tf.square(actions - mu) / variance + tf.math.log(variance) + tf.math.log(2.0 * math.pi))

            actions = tf.reshape(actions, [-1]).numpy()
            logps = tf.reshape(logps, [-1]).numpy()

            if is_vectorized:
                next_states, rewards, dones, _, _ = envs.step(np.expand_dims(actions, axis=-1))

                if adv_strength > 0.0:
                    adversary_noise = np.random.normal(0, adv_strength, size=next_states.shape)
                    for i in range(n_envs):
                        if actions[i] > 0.05: 
                            next_states[i] -= abs(adversary_noise[i])
                        elif actions[i] < -0.05: 
                            next_states[i] += abs(adversary_noise[i])

                for i in range(n_envs):
                    batch_obs.append(states[i])
                    batch_acts.append(actions[i])
                    batch_rets.append(rewards[i])
                    batch_logps.append(logps[i])
                states = next_states
            else:
                next_states, rewards, dones = [], [], []
                for i in range(n_envs):
                    s_next, r, d, _, _ = envs[i].step([actions[i]])

                    if adv_strength > 0.0:
                        adversary_noise = np.abs(np.random.normal(0, adv_strength, size=s_next.shape))
                        if actions[i] > 0.05:
                            s_next -= adversary_noise
                        elif actions[i] < -0.05:
                            s_next += adversary_noise

                    next_states.append(s_next)
                    rewards.append(r)
                    dones.append(d)

                    batch_obs.append(states[i])
                    batch_acts.append(actions[i])
                    batch_rets.append(r)
                    batch_logps.append(logps[i])

                states = np.array(next_states)
                for i in range(n_envs):
                    if dones[i]:
                        states[i] = envs[i].reset()[0]

        obs_t = tf.convert_to_tensor(batch_obs, dtype=tf.float32)
        act_t = tf.convert_to_tensor(batch_acts, dtype=tf.float32)
        ret_t = tf.convert_to_tensor(batch_rets, dtype=tf.float32)
        old_logp_t = tf.convert_to_tensor(batch_logps, dtype=tf.float32)

        val_chunks = []
        obs_length = tf.shape(obs_t)[0]
        for i in range(0, obs_length, batch_size):
            _, v_chunk = agent(obs_t[i:i+batch_size], training=False)
            val_chunks.append(tf.reshape(v_chunk, [-1]))
        values = tf.concat(val_chunks, axis=0)

        advantages = ret_t - values
        batch_variance = tf.math.reduce_std(advantages) + 1e-8
        advantages = (advantages - tf.reduce_mean(advantages)) / batch_variance

        seq_len = tf.cast(tf.shape(act_t)[0], tf.float32)
        base_time_weights = tf.exp(0.5 * (tf.range(seq_len, dtype=tf.float32) / seq_len - 1.0))

        kl_proxy = tf.math.abs(advantages) / batch_variance

        alpha_decay = 0.1
        regime_weights = base_time_weights * tf.exp(-alpha_decay * kl_proxy)

        time_weights = regime_weights / (tf.reduce_mean(regime_weights) + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices((
            obs_t, act_t, ret_t, old_logp_t, advantages, time_weights
        ))
        dataset = dataset.cache().batch(batch_size).prefetch(buffer_size=20)

        for obs_b, act_b, ret_b, old_logp_b, adv_b, tw_b in dataset:
            ppo_update_step(
                obs_b, act_b, ret_b, old_logp_b, adv_b, tw_b,
                tf.convert_to_tensor(clip_ratio, dtype=tf.float32),
                tf.convert_to_tensor(entropy_coef, dtype=tf.float32)
            )

    envs.current_states = states

    return np.sum(batch_rets) / n_envs


# --- CORRECTED TEST CELL: GRADIENT EXPLOSION CHECK ---
import tensorflow as tf
import numpy as np

print("Testing Neural Architecture for Gradient Stability...")
tf.keras.mixed_precision.set_global_policy('float32')

test_agent = build_ppo_agent(lookback=180, n_features=78)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

random_obs = tf.random.normal((32, 180, 78))

with tf.GradientTape() as tape:
    probs, values = test_agent(random_obs, training=True)
    loss = tf.reduce_mean(tf.square(values)) + tf.reduce_mean(tf.square(probs))

grads = tape.gradient(loss, test_agent.trainable_variables)

nan_found = False
for g in grads:
    if g is not None:
        if tf.reduce_any(tf.math.is_nan(g)):
            nan_found = True
            break

if nan_found:
    print("FAIL: Gradients exploded into NaNs!")
else:
    print("PASS: Gradients are clean. The brain's math is stable.")



# --- CELL 7: GLOBAL ARCHITECTURE OPTIMIZATION (2026 CONTINUOUS VERSION) ---
import optuna
import tensorflow as tf
import gc
import os
import shutil
import gymnasium as gym
import numpy as np

class CompatWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, *, seed=None, options=None):
        return self.env.reset()

def objective(trial):
    tf.keras.backend.clear_session()
    gc.collect()

    lookback = trial.suggest_int('lookback', 180, 512)
    entropy_coef = trial.suggest_float('entropy_coef', 0.0001, 0.01, log=True)
    clip_ratio = trial.suggest_float('clip_ratio', 0.1, 0.3)
    lambda_pinn = trial.suggest_float('lambda_pinn', 0.1, 0.4)

    df_trial_full = df_final.copy()
    df_trial_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_trial_full.dropna(inplace=True)

    split_idx = int(len(df_trial_full) * 0.8)
    df_train = df_trial_full.iloc[:split_idx - lookback]
    df_val = df_trial_full.iloc[split_idx:]

    n_features = df_train.shape[1]

    def make_train_env():
        return CompatWrapper(PurgedMarketEnv(df_train, lookback=lookback))

    train_envs = gym.vector.AsyncVectorEnv([make_train_env for _ in range(24)])

    agent = build_ppo_agent(lookback=lookback, n_features=n_features)
    global optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    gc.disable()

    try:
        _ = train_ppo_agent_vectorized(
            train_envs, agent, optimizer=optimizer, epochs=8, trajectory_steps=1024,
            batch_size=128, clip_ratio=clip_ratio,
            lambda_pinn=lambda_pinn, entropy_coef=entropy_coef,
            adv_strength=0.0
        )

        def make_val_env():
            return CompatWrapper(PurgedMarketEnv(df_val, lookback=lookback))

        val_envs = gym.vector.AsyncVectorEnv([make_val_env for _ in range(24)])

        honest_reward = train_ppo_agent_vectorized(
            val_envs, agent, optimizer=optimizer, epochs=1, trajectory_steps=1024,
            batch_size=128, clip_ratio=clip_ratio,
            lambda_pinn=lambda_pinn, entropy_coef=entropy_coef,
            adv_strength=0.0
        )

    except Exception as e:
        print(f"Trial failed: {e}")
        gc.enable()
        return -999.0

    gc.enable()
    return honest_reward

print("\\n--- Initiating Optuna: 2026 Continuous Signal Discovery Engaged ---")

ram_disk_path = "/dev/shm/optuna_study"
os.makedirs(ram_disk_path, exist_ok=True)

db_path = "/content/drive/MyDrive/Forex Data/optuna_study_continuous_v32.db"
local_db_path = os.path.join(ram_disk_path, "gold_sniper_continuous_v32.db")

if os.path.exists(db_path):
    shutil.copy(db_path, local_db_path)
    print("Database loaded into high-speed RAM Disk.")

storage_name = f"sqlite:///{local_db_path}"

study = optuna.create_study(
    study_name="gold_sniper_continuous_v32",
    storage=storage_name,
    load_if_exists=True,
    direction="maximize",
    pruner=optuna.pruners.MedianPruner() 
)

for _ in range(1):
    study.optimize(objective, n_trials=13)
    shutil.copy(local_db_path, db_path)
    print(f"Progress force-saved to Drive. Best Reward so far: {study.best_value:.6f}")



# --- CELL 9.1: THE VETERAN MARATHON (CONTINUOUS v32 HARMONIC) ---
import gc
import os
import json
import tensorflow as tf
import numpy as np
import shutil
import gymnasium as gym

print("--- 0. Quality Control: Initializing High-Speed Production Line (Harmonic Neuro-Symbolic) ---")

class CompatWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, *, seed=None, options=None):
        return self.env.reset()

ram_disk_path = "/dev/shm/Checkpoints_Marathon"
os.makedirs(ram_disk_path, exist_ok=True)

drive_path = "/content/drive/MyDrive/Forex Data/Checkpoints"
os.makedirs(drive_path, exist_ok=True)

best_checkpoint = os.path.join(ram_disk_path, "champion_brain_gold_v32.weights.h5")
current_run_path = os.path.join(ram_disk_path, "pinn_ppo_agent_gold_v32.weights.h5")
state_path = os.path.join(ram_disk_path, "training_state_gold_v32.json")

drive_best = os.path.join(drive_path, "champion_brain_gold_v32.weights.h5")
drive_current = os.path.join(drive_path, "pinn_ppo_agent_gold_v32.weights.h5")
drive_state = os.path.join(drive_path, "training_state_gold_v32.json")

if os.path.exists(drive_current) and os.path.exists(drive_state):
    shutil.copy(drive_current, current_run_path)
    shutil.copy(drive_state, state_path)
    print("Resuming: Copied saved continuous states from Drive to high-speed RAM.")
if os.path.exists(drive_best):
    shutil.copy(drive_best, best_checkpoint)

best_params = {
    'lookback': 332,
    'entropy_coef': 0.00014476,
    'clip_ratio': 0.2241,
    'lambda_pinn': 0.3185,
    'imf_count': 5,
    'num_envs': 24
}

df_prod = df_final.copy()
print("Scrubbing dataset for corrupted rows...")
df_prod.replace([np.inf, -np.inf], np.nan, inplace=True)
df_prod.dropna(inplace=True)

df_recent_12m = df_prod.iloc[-6000:]

def make_base_env():
    return CompatWrapper(PurgedMarketEnv(df_prod, lookback=best_params['lookback']))

def make_polish_env():
    return CompatWrapper(PurgedMarketEnv(df_recent_12m, lookback=best_params['lookback']))

envs_base = gym.vector.AsyncVectorEnv([make_base_env for _ in range(best_params['num_envs'])])
envs_polish = gym.vector.AsyncVectorEnv([make_polish_env for _ in range(best_params['num_envs'])])

tf.keras.backend.clear_session()
gc.collect()

agent_prod = build_ppo_agent(lookback=best_params['lookback'], n_features=df_prod.shape[1])

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=0.5)

start_epoch = 1
best_reward_ever = -float('inf')
total_epochs = 600
base_epochs = 250
maturity_epoch = 350

if os.path.exists(current_run_path) and os.path.exists(state_path):
    print("Found interrupted run in RAM. Resuming production...")
    agent_prod.load_weights(current_run_path)
    with open(state_path, "r") as f:
        state = json.load(f)
        start_epoch = state.get("epoch", 0) + 1
        best_reward_ever = state.get("best_reward", -float('inf'))
else:
    print(f"New Harmonic Production Engaged | Target Maturity: Epoch {maturity_epoch}")

gc.enable()

try:
    for epoch in range(start_epoch, total_epochs + 1):

        if epoch > maturity_epoch:
            adv_strength = min(0.001, (epoch - maturity_epoch) * 0.000004)
        else:
            adv_strength = 0.0

        if epoch <= base_epochs:
            current_envs = envs_base
            h_max = best_params['entropy_coef']
        else:
            if epoch == base_epochs + 1:
                print("\\n" + "="*50 + "\\n PHASE 2: CONTINUOUS ADVERSARIAL REGIME POLISH\\n" + "="*50)
                freeze_foundation(agent_prod)
                best_reward_ever = -float('inf')
            current_envs = envs_polish
            h_max = best_params['entropy_coef'] * 1.5

        h_min, eta = 0.00005, 0.05
        current_entropy = h_min + (h_max - h_min) / (1.0 + eta * ((epoch - 1) % base_epochs))

        avg_reward = train_ppo_agent_vectorized(
            current_envs, agent_prod, optimizer=optimizer, epochs=1,
            trajectory_steps=1024, batch_size=128,
            clip_ratio=best_params['clip_ratio'],
            lambda_pinn=best_params['lambda_pinn'],
            entropy_coef=current_entropy,
            adv_strength=adv_strength
        )

        if epoch < maturity_epoch:
            if epoch > base_epochs and avg_reward > best_reward_ever:
                best_reward_ever = avg_reward
                safe_path = os.path.join(drive_path, "safe_maturity_gold_v32.weights.h5")
                agent_prod.save_weights(safe_path)
                print(f"SAFEGUARD: New Pristine Maturity Champion Saved (Reward: {avg_reward:.6f})")
            else:
                print(f"Young Score {avg_reward:.6f} at Epoch {epoch} (Waiting for maturity)")
        else:
            if epoch == maturity_epoch:
                best_reward_ever = -float('inf')

            weighted_score = avg_reward * (1.0 + adv_strength * 1000)

            if weighted_score > best_reward_ever:
                best_reward_ever = weighted_score
                agent_prod.save_weights(best_checkpoint)
                shutil.copy(best_checkpoint, drive_best)
                print(f"NEW HARMONIC VETERAN CHAMPION: Reward {avg_reward:.6f} at Strength {adv_strength:.6f} (Weighted: {weighted_score:.6f})")
            else:
                print(f"Mature Epoch {epoch}: Reward {avg_reward:.6f} (Did not beat Veteran High Score {best_reward_ever:.6f})")

        if adv_strength > 0:
            print(f"Adversary Strength: {adv_strength:.6f}")

        print(f"Epoch {epoch}/{total_epochs} | Avg Reward: {avg_reward:.6f} | Ent: {current_entropy:.8f}")

        agent_prod.save_weights(current_run_path)
        with open(state_path, "w") as f:
            json.dump({"epoch": epoch, "best_reward": best_reward_ever}, f)

        if epoch >= 400 or epoch % 5 == 0:
            shutil.copy(current_run_path, drive_current)
            shutil.copy(state_path, drive_state)
            print(f"Standard Save Checkpoint Synced to Google Drive.")

        gc.collect()

except KeyboardInterrupt:
    print("\\n Production manually halted by operator.")

finally:
    gc.enable()

print("\\n Production Run Complete. Your Neuro-Symbolic 'Mind' is now a battle-hardened veteran.")


# --- CELL 10: V8 CONTINUOUS MC AUDIT & TOURNAMENT MATRIX ---
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import gc
import matplotlib.pyplot as plt 

print("--- 1. Initializing V8 Normalized Quality Control Audit (Tournament Matrix) ---")

lookback = 332 
audit_bars = 25000
activation_mu = 0.15   
max_uncertainty = 0.80 
risk_per_trade_dollars = 1000.0

if 'df_prod' not in locals():
    df_prod = df_final.copy()
    df_prod.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prod.dropna(inplace=True)

df_audit = df_prod.iloc[-audit_bars:].copy()
features = df_audit[df_final.columns.tolist()].columns.tolist()
n_features = len(features)

raw_data = df_audit[features].values
all_windows = []
for t in range(lookback, len(df_audit) - 1):
    window = raw_data[t-lookback:t]
    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0) + 1e-8
    obs = (window - means) / stds
    all_windows.append(obs)
obs_batch = np.array(all_windows).astype(np.float32)

drive_dir = "/content/drive/MyDrive/Forex Data/Checkpoints"
models_to_test = {
    "Maturity Champion (Pure Physics)": os.path.join(drive_dir, "safe_maturity_gold_v32.weights.h5"),
    "Veteran Champion (Adversarial)": os.path.join(drive_dir, "champion_brain_gold_v32.weights.h5")
}

closes = df_audit['XAUUSD_H1_Close'].values[lookback:]
dates = df_audit.index[lookback:]

tournament_results = []
heatmap_data = {} 

for model_name, model_path in models_to_test.items():
    print(f"\\n" + "="*65)
    print(f" AUDITING: {model_name}")
    print("="*65)

    tf.keras.backend.clear_session()
    gc.collect()

    audit_agent = build_ppo_agent(lookback=lookback, n_features=n_features)
    try:
        audit_agent.load_weights(model_path)
        print(f" {model_name} Weights Loaded Successfully.")
    except Exception as e:
        print(f" Could not load {model_name}. Skipping. Error: {e}")
        continue

    print(" Executing V8 Monte Carlo Inference (30 Passes via Bayesian Governor)...")
    action_params, _ = audit_agent.predict(obs_batch, batch_size=512, verbose=1)

    account_balance = 100000.0
    peak_balance = account_balance
    max_drawdown = 0.0
    current_pos = 0
    entry_price = 0.0
    trades = []
    uncertainties = []

    for i in range(len(action_params)):
        mu = action_params[i][0]
        std = action_params[i][1]
        uncertainties.append(std)

        action = 1 

        if std < max_uncertainty:
            if mu > activation_mu: action = 2 
            elif mu < -activation_mu: action = 0 

        current_price, time_stamp = closes[i], dates[i]

        if current_pos == 0:
            if action == 2:
                current_pos, entry_price, entry_time = 1, current_price, time_stamp
                lot_multiplier = (risk_per_trade_dollars * abs(mu)) / (entry_price * 0.01)
                entry_std = std
            elif action == 0:
                current_pos, entry_price, entry_time = -1, current_price, time_stamp
                lot_multiplier = (risk_per_trade_dollars * abs(mu)) / (entry_price * 0.01)
                entry_std = std

        elif current_pos == 1 and action != 2:
            pnl = (current_price - entry_price) * lot_multiplier
            account_balance += pnl
            trades.append({'Entry': entry_time, 'Exit': time_stamp, 'Type': 'LONG', 'PnL': pnl, 'Balance': account_balance, 'Uncertainty': entry_std})
            current_pos = -1 if action == 0 else 0
            entry_price = current_price if current_pos != 0 else 0
            if current_pos != 0:
                lot_multiplier = (risk_per_trade_dollars * abs(mu)) / (entry_price * 0.01)
                entry_std = std

        elif current_pos == -1 and action != 0:
            pnl = (entry_price - current_price) * lot_multiplier
            account_balance += pnl
            trades.append({'Entry': entry_time, 'Exit': time_stamp, 'Type': 'SHORT', 'PnL': pnl, 'Balance': account_balance, 'Uncertainty': entry_std})
            current_pos = 1 if action == 2 else 0
            entry_price = current_price if current_pos != 0 else 0
            if current_pos != 0:
                lot_multiplier = (risk_per_trade_dollars * abs(mu)) / (entry_price * 0.01)
                entry_std = std

        if account_balance > peak_balance:
            peak_balance = account_balance
        max_drawdown = max(max_drawdown, (peak_balance - account_balance) / peak_balance)

    heatmap_data[model_name] = uncertainties

    if trades:
        win_rate = (len([t for t in trades if t['PnL'] > 0]) / len(trades)) * 100
        avg_uncertainty = np.mean([t['Uncertainty'] for t in trades])
        net_pnl = account_balance - 100000

        print(f"Final Balance: ${account_balance:,.2f} | Net: ${net_pnl:,.2f}")
        print(f"Total Trades: {len(trades)} | Win Rate: {win_rate:.2f}% | Max DD: {max_drawdown*100:.2f}%")
        print(f"Average AI Uncertainty on Entry: {avg_uncertainty:.4f} (Lower = Higher Quality)")

        df_trades = pd.DataFrame(trades)
        df_trades['Month'] = pd.to_datetime(df_trades['Exit']).dt.strftime('%Y-%m')
        monthly_stats = []

        for month, group in df_trades.groupby('Month'):
            m_trades = len(group)
            m_win_rate = (len(group[group['PnL'] > 0]) / m_trades) * 100
            m_balances = group['Balance'].values
            m_peaks = np.maximum.accumulate(m_balances)
            m_dd = (m_peaks - m_balances) / np.where(m_peaks == 0, 1, m_peaks)
            m_uncert = group['Uncertainty'].mean()
            monthly_stats.append({
                'Month': month,
                'Trades': m_trades,
                'Win%': f"{m_win_rate:.1f}%",
                'Net PnL': f"${group['PnL'].sum():,.2f}",
                'Max DD': f"{np.max(m_dd)*100:.2f}%",
                'Avg Uncertainty': f"{m_uncert:.4f}"
            })

        with pd.option_context('display.max_rows', None):
            print("\\nMonthly Breakdown:")
            print(pd.DataFrame(monthly_stats).to_string(index=False))

        tournament_results.append({
            "Brain": model_name,
            "Net PnL": f"${net_pnl:,.2f}",
            "Win Rate": f"{win_rate:.2f}%",
            "Max DD": f"{max_drawdown*100:.2f}%",
            "Avg Uncertainty": f"{avg_uncertainty:.4f}"
        })
    else:
        print(" No trades executed. (If unexpected, check if max_uncertainty is too strict).")
        tournament_results.append({
            "Brain": model_name,
            "Net PnL": "$0.00",
            "Win Rate": "0.00%",
            "Max DD": "0.00%",
            "Avg Uncertainty": "N/A"
        })

print("\\n" + "="*25)
print(" V8 TOURNAMENT MATRIX: MATURITY VS VETERAN")
print("="*25)
df_matrix = pd.DataFrame(tournament_results)
print(df_matrix.to_string(index=False))

print("\\n" + "="*25)
print(" GENERATING CONFIDENCE HEATMAPS...")
print("="*25)

num_models = len(heatmap_data)
if num_models > 0:
    fig, axes = plt.subplots(num_models, 1, figsize=(16, 6 * num_models), sharex=True)
    if num_models == 1: axes = [axes]

    for ax, (model_name, uncerts) in zip(axes, heatmap_data.items()):
        sc = ax.scatter(dates[:len(uncerts)], closes[:len(uncerts)], c=uncerts, cmap='Reds', s=5, alpha=0.8)
        ax.set_title(f"{model_name}: 2026 Gold Confusion Heatmap", fontsize=14, fontweight='bold')
        ax.set_ylabel("XAUUSD Price", fontsize=12)
        ax.grid(True, alpha=0.3)

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Bayesian Uncertainty (Std)", fontsize=10)

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    plt.show()
