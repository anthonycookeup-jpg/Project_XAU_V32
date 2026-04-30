# --- THE MT5 LIVE EXECUTION BRIDGE (Trading_Bot2 V8 CONTINUOUS) ---
# Ensure you run: pip install MetaTrader5 tensorflow pandas numpy ta optuna scipy arch vmdpy
# Python Version: 3.11.9

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import gc 
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalAveragePooling1D, Multiply, LayerNormalization, RNN, Bidirectional, MultiHeadAttention, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
import ta
from vmdpy import VMD
import scipy.stats as stats
from scipy.stats import entropy

mixed_precision.set_global_policy('float32')

WORK_DIR = r"C:\\Users\\Antho\\OneDrive\\Desktop\\XAUmetaC"
WEIGHTS_FILE = os.path.join(WORK_DIR, "champion_brain_gold_v32.weights.h5") 
TRADE_LOG_CSV = os.path.join(WORK_DIR, "live_trades_log.csv") 

MAGIC_NUMBER = 777777
LOT_SIZE = 0.01 
LOOKBACK = 332
ACTIVATION_MU = 0.15
MAX_UNCERTAINTY = 0.80

BROKER_SYMBOLS = {
    'XAUUSD': 'XAUUSD',
    'XAGUSD': 'XAGUSD',
    'USDJPY': 'USDJPY',
    'EURUSD': 'EURUSD',
    'GBPUSD': 'GBPUSD',
    'US500.cash': 'US500.cash' 
}
PRIMARY_SYMBOL = BROKER_SYMBOLS['XAUUSD']

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

def get_weights_fracdiff(d, size):
    w = [1.]
    for k in range(1, size):
        w.append(-w[-1] / k * (d - k + 1))
    return np.array(w[::-1])

def frac_diff(series, d, window=12):
    weights = get_weights_fracdiff(d, window)
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

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

def fetch_mt5_live_data(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('datetime', inplace=True)
    
    colab_name = [k for k, v in BROKER_SYMBOLS.items() if v == symbol][0]
    tf_str = 'H1' if timeframe == mt5.TIMEFRAME_H1 else 'H4'
    
    cols = {
        'open': f'{colab_name}_{tf_str}_Open',
        'high': f'{colab_name}_{tf_str}_High',
        'low': f'{colab_name}_{tf_str}_Low',
        'close': f'{colab_name}_{tf_str}_Close',
        'tick_volume': f'{colab_name}_{tf_str}_Volume'
    }
    
    if colab_name == 'XAUUSD' and tf_str == 'H1':
        df['spread'] = df['spread'] * 0.01
        cols['spread'] = 'Spread'
        
    df = df.rename(columns=cols)
    keep_cols = list(cols.values())
        
    return df[keep_cols].astype(np.float32)

def prepare_live_inference_data():
    fetch_len = 2000 
    h1_frames = {}
    h4_frames = {}
    
    for colab_sym, broker_sym in BROKER_SYMBOLS.items():
        h1_df = fetch_mt5_live_data(broker_sym, mt5.TIMEFRAME_H1, fetch_len)
        h4_df = fetch_mt5_live_data(broker_sym, mt5.TIMEFRAME_H4, fetch_len // 2)
        if h1_df is None or h4_df is None: return None
        h1_frames[colab_sym] = h1_df
        h4_frames[colab_sym] = h4_df

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

    df_final = df_final.bfill().fillna(0)
    
    if len(df_final) % 2 != 0:
        df_final = df_final.iloc[1:].copy()
        
    alpha, tau, K, DC, init, tol = 2000, 0, 5, 0, 1, 1e-7
    u, _, _ = VMD(df_final['XAUUSD_H1_Close'].values, alpha, tau, K, DC, init, tol)
    for i in range(K):
        df_final[f'IMF_{i+1}'] = u[i, :]

    df_final = df_final.dropna()
    
    window = df_final.iloc[-LOOKBACK:].values
    means = np.mean(window, axis=0)
    stds = np.std(window, axis=0) + 1e-9
    obs = (window - means) / stds
    
    return np.expand_dims(obs, axis=0)

def log_completed_trade(pos, exit_price, current_mu, current_std):
    try:
        entry_time = datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S')
        exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        trade_type = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
        
        df_log = pd.DataFrame([{
            'Ticket': pos.ticket,
            'Type': trade_type,
            'Volume': pos.volume,
            'Entry_Time': entry_time,
            'Entry_Price': pos.price_open,
            'Exit_Time': exit_time,
            'Exit_Price': exit_price,
            'Gross_PnL': pos.profit,
            'Swap': pos.swap,
            'Exit_Mu': round(current_mu, 4),
            'Exit_Std': round(current_std, 4)
        }])
        
        if not os.path.isfile(TRADE_LOG_CSV):
            df_log.to_csv(TRADE_LOG_CSV, index=False)
        else:
            df_log.to_csv(TRADE_LOG_CSV, mode='a', header=False, index=False)
            
        print(f"Diagnostics logged to: live_trades_log.csv")
    except Exception as e:
        print(f"Failed to log trade to CSV: {e}")

def execute_trade(action):
    symbol_info = mt5.symbol_info(PRIMARY_SYMBOL)
    if symbol_info is None: return None
    if not symbol_info.visible:
        mt5.symbol_select(PRIMARY_SYMBOL, True)

    order_type = mt5.ORDER_TYPE_BUY if action == 2 else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(PRIMARY_SYMBOL).ask if action == 2 else mt5.symbol_info_tick(PRIMARY_SYMBOL).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PRIMARY_SYMBOL,
        "volume": float(LOT_SIZE), 
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Trading_Bot2_V8",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Failed: {result.comment}")
        return None
    
    print(f"Trade Executed: {'LONG' if action == 2 else 'SHORT'} at {price}")
    return result.order

def close_position(pos, current_mu, current_std):
    tick = mt5.symbol_info_tick(PRIMARY_SYMBOL)
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": PRIMARY_SYMBOL,
        "volume": pos.volume,
        "type": order_type,
        "position": pos.ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Bot2_Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Position Closed Successfully at {price}.")
        log_completed_trade(pos, price, current_mu, current_std)
    else:
        print(f"Failed to close position: {result.comment}")

if __name__ == "__main__":
    print(f"--- INITIALIZING TRADING_BOT2 V8 CONTINUOUS ENGINE ---")
    if not mt5.initialize():
        print("MT5 Initialization Failed.")
        mt5.shutdown()
        exit()

    print("Building Neuro-Symbolic Agent Architecture...")
    agent = build_ppo_agent(lookback=LOOKBACK, n_features=109)
    
    try:
        agent.load_weights(WEIGHTS_FILE)
        print(f"Veteran Weights Loaded from: {WEIGHTS_FILE}")
    except Exception as e:
        print(f"FATAL ERROR loading weights: {e}")
        mt5.shutdown()
        exit()

    print("\\n SYSTEM ONLINE: Hunting for high-probability LP Gaps...\\n")

    try:
        while True:
            obs = prepare_live_inference_data()
            if obs is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Data fetch incomplete. Retrying...")
                time.sleep(10)
                continue

            action_params, _ = agent.predict(obs, verbose=0)
            mu = action_params[0][0]
            std = action_params[0][1]

            action = 1 
            if std < MAX_UNCERTAINTY:
                if mu > ACTIVATION_MU: action = 2 
                elif mu < -ACTIVATION_MU: action = 0 

            my_positions = mt5.positions_get(symbol=PRIMARY_SYMBOL, magic=MAGIC_NUMBER)
            
            if my_positions is None or len(my_positions) == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] FLAT | Mu: {mu:.4f} | Std: {std:.4f} | AI Signal: {action}")
                
                if action != 1:
                    execute_trade(action)
            else:
                pos = my_positions[0]
                current_direction = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] IN POSITION ({current_direction}) | Mu: {mu:.4f} | Std: {std:.4f} | AI Signal: {action}")
                
                if (pos.type == mt5.ORDER_TYPE_BUY and action != 2) or (pos.type == mt5.ORDER_TYPE_SELL and action != 0):
                    close_position(pos, mu, std)

            time.sleep(60)
            gc.collect()

    except KeyboardInterrupt:
        print("\\n Live Execution manually halted by operator.")
    finally:
        mt5.shutdown()
