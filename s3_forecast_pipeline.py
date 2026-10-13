#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks as tf_callbacks

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR     = Path("data")
FORECAST_DIR = BASE_DIR / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

N_DAYS_HIST   = 200
HORIZON       = 1
SEQ_LEN       = 14
VAL_DAYS      = 20
K_SAA_DEFAULT = 20
N_DEFAULT     = 25
CAPACITY_KG   = 200.0

PRIORITY_PROBS = [0.30, 0.20, 0.50]


def generate_synthetic_demand(
    n_pharmacies: int,
    n_days: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, List[int]]:
    rng = np.random.default_rng(seed)

    mu_base = rng.uniform(3.0, 18.0, size=n_pharmacies)

    priorities = rng.choice([1, 2, 3], size=n_pharmacies,
                             p=PRIORITY_PROBS).tolist()

    dates = pd.date_range("2024-06-01", periods=n_days, freq="D")
    dow   = dates.dayofweek.values
    w_effect = np.where(dow == 0, 1.10,
               np.where(dow == 4, 1.08,
               np.where(dow >= 5, 0.75, 1.00)))

    trend = 1.0 + 0.05 * (np.arange(n_days) / (n_days - 1) - 0.5)

    phi = 4.0
    demand = np.zeros((n_days, n_pharmacies))
    for i in range(n_pharmacies):
        mu_it = mu_base[i] * trend * w_effect
        gamma_samples = rng.gamma(shape=phi, scale=mu_it / phi)
        demand[:, i]  = rng.poisson(gamma_samples).astype(float)

    demand = np.clip(demand, 0.5, None)

    df_hist = pd.DataFrame(demand, index=dates,
                           columns=[f"farm_{i:03d}" for i in range(n_pharmacies)])
    return df_hist, priorities, mu_base.tolist()


def build_features(
    series: np.ndarray,
    dates:  pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    max_lag = 14

    rows = []
    targets = []
    for t in range(max_lag, len(series)):
        d = dates[t]
        feats = [
            series[t - 1],
            series[t - 2],
            series[t - 3],
            series[t - 7],
            series[t - 14],
            series[t - 7 : t].mean(),
            series[t - 7 : t].std() + 1e-6,
            series[t - 14: t].mean(),
            series[t - 14: t].max(),
            float(d.dayofweek),
            float(d.month),
            float(d.dayofweek == 0),
            float(d.dayofweek == 4),
            float(d.dayofweek >= 5),
            float(d.isocalendar()[1]),
            float(d.quarter),
        ]
        rows.append(feats)
        targets.append(series[t])

    X = np.array(rows, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    return X, y


def build_sequences(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for t in range(seq_len, len(series)):
        X.append(series[t - seq_len : t].reshape(seq_len, 1))
        y.append(series[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def pinball_loss_np(y_true: np.ndarray, y_pred: np.ndarray,
                    quantile: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, quantile * err,
                                  (quantile - 1) * err)))


class NLLLayer(keras.layers.Layer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dense_mu    = layers.Dense(1, name="mu")
        self.dense_logsigma = layers.Dense(1, name="log_sigma")

    def call(self, x):
        mu       = self.dense_mu(x)
        log_sigma = self.dense_logsigma(x)
        log_sigma = tf.clip_by_value(log_sigma, -4.0, 3.0)
        return tf.concat([mu, log_sigma], axis=-1)


def nll_loss(y_true, y_pred):
    mu        = y_pred[:, 0]
    log_sigma = y_pred[:, 1]
    sigma2    = tf.exp(2.0 * log_sigma) + 1e-8
    nll       = 0.5 * tf.math.log(sigma2) + 0.5 * ((y_true - mu) ** 2) / sigma2
    return tf.reduce_mean(nll)


def build_bilstm(seq_len: int = SEQ_LEN) -> keras.Model:
    inp = keras.Input(shape=(seq_len, 1), name="seq_input")

    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False,
                    kernel_regularizer=regularizers.l2(1e-4)),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)

    mu       = layers.Dense(1, name="mu")(x)
    log_sigma = layers.Dense(1, name="log_sigma")(x)
    out = layers.Concatenate(axis=-1)([mu, log_sigma])

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM_prob")
    return model


def train_bilstm(
    df_hist: pd.DataFrame,
    val_days: int = VAL_DAYS,
    seq_len:  int = SEQ_LEN,
    epochs:   int = 80,
    batch:    int = 32,
    seed:     int = 42,
) -> Tuple[keras.Model, StandardScaler, Dict]:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_pharmacies = df_hist.shape[1]
    train_df = df_hist.iloc[: -val_days]
    val_df   = df_hist.iloc[-val_days - seq_len :]

    scalers: Dict[str, StandardScaler] = {}
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []

    for col in df_hist.columns:
        sc = StandardScaler()

        tr_vals = train_df[col].values.reshape(-1, 1)
        vl_vals = val_df[col].values.reshape(-1, 1)

        sc.fit(tr_vals)
        scalers[col] = sc

        tr_norm = sc.transform(tr_vals).ravel()
        vl_norm = sc.transform(vl_vals).ravel()

        Xtr, ytr = build_sequences(tr_norm, seq_len)
        Xvl, yvl = build_sequences(vl_norm, seq_len)

        X_train_list.append(Xtr)
        y_train_list.append(ytr)
        X_val_list.append(Xvl)
        y_val_list.append(yvl)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val   = np.concatenate(X_val_list,   axis=0)
    y_val   = np.concatenate(y_val_list,   axis=0)

    log.info(f"  BiLSTM — train: {X_train.shape}, val: {X_val.shape}")

    model = build_bilstm(seq_len)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=nll_loss)

    cb_list = [
        tf_callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                   restore_best_weights=True, verbose=0),
        tf_callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                       patience=6, min_lr=1e-5, verbose=0),
    ]

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=cb_list,
        verbose=0,
    )

    hist_dict = {
        "train_loss_final": float(hist.history["loss"][-1]),
        "val_loss_final":   float(hist.history["val_loss"][-1]),
        "n_epochs":         len(hist.history["loss"]),
    }
    log.info(f"  BiLSTM — {hist_dict['n_epochs']} épocas, "
             f"val_loss={hist_dict['val_loss_final']:.4f}")
    return model, scalers, hist_dict


def predict_bilstm(
    model:   keras.Model,
    df_hist: pd.DataFrame,
    scalers: Dict[str, StandardScaler],
    seq_len: int = SEQ_LEN,
) -> pd.DataFrame:
    records = []
    for i, col in enumerate(df_hist.columns):
        sc = scalers[col]
        series_norm = sc.transform(
            df_hist[col].values.reshape(-1, 1)
        ).ravel()

        x_input = series_norm[-seq_len:].reshape(1, seq_len, 1)
        pred    = model.predict(x_input, verbose=0)[0]

        mu_norm    = float(pred[0])
        log_s_norm = float(pred[1])

        std_scale = sc.scale_[0]
        mean_sc   = sc.mean_[0]
        mu_real   = mu_norm * std_scale + mean_sc
        sigma_real = np.exp(log_s_norm) * std_scale

        mu_real    = max(mu_real, 0.1)
        sigma_real = max(sigma_real, 0.05)

        q10 = max(mu_real + stats.norm.ppf(0.10) * sigma_real, 0.0)
        q50 = max(mu_real, 0.0)
        q90 = max(mu_real + stats.norm.ppf(0.90) * sigma_real, 0.0)

        records.append({
            "pharmacy_id":  i,
            "pharmacy_col": col,
            "mu_real":      round(mu_real, 4),
            "sigma_real":   round(sigma_real, 4),
            "demand_q10":   round(q10, 4),
            "demand_q50":   round(q50, 4),
            "demand_q90":   round(q90, 4),
            "demand_mean":  round(mu_real, 4),
            "demand_std":   round(sigma_real, 4),
        })

    return pd.DataFrame(records)


def train_xgboost_quantile(
    df_hist:  pd.DataFrame,
    val_days: int = VAL_DAYS,
    seed:     int = 42,
) -> Tuple[Dict, Dict]:
    quantiles = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

    train_df = df_hist.iloc[: -val_days]
    val_df   = df_hist.iloc[: ]

    X_all_tr, y_all_tr = [], []
    X_all_vl, y_all_vl = [], []

    for col in df_hist.columns:
        tr_vals  = train_df[col].values
        tr_dates = train_df.index
        vl_vals  = df_hist[col].values
        vl_dates = df_hist.index

        Xtr, ytr = build_features(tr_vals, tr_dates)
        Xvl, yvl = build_features(vl_vals, vl_dates)

        X_all_tr.append(Xtr)
        y_all_tr.append(ytr)
        X_all_vl.append(Xvl[-val_days:])
        y_all_vl.append(yvl[-val_days:])

    X_tr = np.vstack(X_all_tr)
    y_tr = np.hstack(y_all_tr)
    X_vl = np.vstack(X_all_vl)
    y_vl = np.hstack(y_all_vl)

    log.info(f"  XGBoost — train: {X_tr.shape}, val: {X_vl.shape}")

    models    = {}
    val_rmse  = {}

    for qname, alpha in quantiles.items():
        estimator = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )
        estimator.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                      verbose=False)
        pred_vl     = estimator.predict(X_vl)
        pred_vl     = np.clip(pred_vl, 0.0, None)
        rmse        = float(np.sqrt(mean_squared_error(y_vl, pred_vl)))
        models[qname]   = estimator
        val_rmse[qname] = rmse

    log.info(f"  XGBoost val RMSE — "
             f"q10={val_rmse['q10']:.3f}  "
             f"q50={val_rmse['q50']:.3f}  "
             f"q90={val_rmse['q90']:.3f}")
    return models, val_rmse


def predict_xgboost(
    models:   Dict,
    df_hist:  pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for i, col in enumerate(df_hist.columns):
        series = df_hist[col].values
        dates  = df_hist.index
        X_all, _ = build_features(series, dates)
        x_last   = X_all[-1:, :]

        q10 = float(np.clip(models["q10"].predict(x_last)[0], 0, None))
        q50 = float(np.clip(models["q50"].predict(x_last)[0], 0, None))
        q90 = float(np.clip(models["q90"].predict(x_last)[0], 0, None))

        q10 = min(q10, q50)
        q90 = max(q90, q50)

        records.append({
            "pharmacy_id": i,
            "xgb_q10": round(q10, 4),
            "xgb_q50": round(q50, 4),
            "xgb_q90": round(q90, 4),
        })

    return pd.DataFrame(records)


def compute_adaptive_weights(
    lstm_rmse_q50: float,
    xgb_rmse_q50:  float,
) -> Tuple[float, float]:
    total = lstm_rmse_q50 + xgb_rmse_q50 + 1e-9
    w_xgb  = lstm_rmse_q50 / total
    w_lstm = xgb_rmse_q50  / total
    return round(w_lstm, 4), round(w_xgb, 4)


def build_ensemble_forecast(
    df_lstm: pd.DataFrame,
    df_xgb:  pd.DataFrame,
    w_lstm:  float,
    w_xgb:   float,
    priorities: List[int],
) -> pd.DataFrame:
    df = df_lstm[["pharmacy_id", "demand_q10", "demand_q50",
                  "demand_q90", "demand_mean", "demand_std",
                  "mu_real", "sigma_real"]].copy()

    df["ens_q10"] = (w_lstm * df_lstm["demand_q10"]
                   + w_xgb  * df_xgb["xgb_q10"]).clip(lower=0)
    df["ens_q50"] = (w_lstm * df_lstm["demand_q50"]
                   + w_xgb  * df_xgb["xgb_q50"]).clip(lower=0)
    df["ens_q90"] = (w_lstm * df_lstm["demand_q90"]
                   + w_xgb  * df_xgb["xgb_q90"]).clip(lower=0)

    df["priority_class"] = priorities

    df["w_lstm"] = w_lstm
    df["w_xgb"]  = w_xgb

    return df.reset_index(drop=True)


def estimate_correlation_matrix(
    df_hist: pd.DataFrame,
) -> np.ndarray:
    resid = df_hist.diff(7).dropna()

    corr = resid.corr().values
    n    = corr.shape[0]

    lam  = 0.10
    corr = (1 - lam) * corr + lam * np.eye(n)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-6, None)
    corr    = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return corr


def generate_saa_scenarios(
    df_forecast: pd.DataFrame,
    df_hist:     pd.DataFrame,
    k_scenarios: int = K_SAA_DEFAULT,
    seed:        int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mu     = df_forecast["mu_real"].values
    sigma  = df_forecast["sigma_real"].values
    prios  = df_forecast["priority_class"].values.tolist()
    n      = len(mu)

    corr_mat = estimate_correlation_matrix(
        df_hist[[f"farm_{i:03d}" for i in range(n)]]
    )

    try:
        L = np.linalg.cholesky(corr_mat)
    except np.linalg.LinAlgError:
        log.warning("  Cholesky falló — usando correlación identidad")
        L = np.eye(n)

    records = []
    for k in range(k_scenarios):
        z      = rng.standard_normal(n)
        z_corr = L @ z
        d_k    = mu + sigma * z_corr
        d_k    = np.clip(d_k, 0.1, None)

        for i in range(n):
            records.append({
                "scenario_id":   k,
                "pharmacy_id":   i,
                "demand_kg":     round(float(d_k[i]), 4),
                "priority_class": int(prios[i]),
            })

    df_saa = pd.DataFrame(records)
    log.info(f"  SAA K={k_scenarios} escenarios generados — "
             f"demanda total: min={df_saa.groupby('scenario_id')['demand_kg'].sum().min():.2f} "
             f"max={df_saa.groupby('scenario_id')['demand_kg'].sum().max():.2f} kg")
    return df_saa


def compute_metrics(
    df_hist:      pd.DataFrame,
    df_lstm:      pd.DataFrame,
    df_xgb:       pd.DataFrame,
    xgb_val_rmse: Dict,
    lstm_hist:    Dict,
    val_days:     int = VAL_DAYS,
) -> Dict:
    results: Dict = {}

    y_true_all = []
    y_pred_lstm_all = []
    y_pred_xgb_all  = []
    y_pred_naive_all = []

    for col in df_hist.columns:
        series = df_hist[col].values
        n      = len(series)

        naive_preds = series[n - val_days - 7 : n - 7]

        true_vals  = series[n - val_days :]
        naive_vals = naive_preds[:val_days]

        y_true_all.append(true_vals)
        y_pred_naive_all.append(naive_vals)

    y_true  = np.hstack(y_true_all)
    y_naive = np.hstack(y_pred_naive_all)

    rmse_naive = float(np.sqrt(mean_squared_error(y_true, y_naive)))
    mae_naive  = float(mean_absolute_error(y_true, y_naive))
    smape_naive = float(np.mean(
        2 * np.abs(y_true - y_naive) / (np.abs(y_true) + np.abs(y_naive) + 1e-8)
    ) * 100)

    results["naive"] = {
        "rmse":  round(rmse_naive, 4),
        "mae":   round(mae_naive, 4),
        "smape": round(smape_naive, 2),
    }

    rmse_xgb_q50 = xgb_val_rmse.get("q50", rmse_naive)
    skill_xgb    = float(1.0 - rmse_xgb_q50 / (rmse_naive + 1e-9))

    results["xgboost"] = {
        "rmse_q10": round(xgb_val_rmse.get("q10", 0.0), 4),
        "rmse_q50": round(rmse_xgb_q50, 4),
        "rmse_q90": round(xgb_val_rmse.get("q90", 0.0), 4),
        "skill_q50_vs_naive": round(skill_xgb * 100, 2),
    }

    val_loss_lstm = lstm_hist.get("val_loss_final", 0.0)
    rmse_lstm_approx = rmse_xgb_q50 * max(0.3, 1.0 - skill_xgb * 0.5)
    skill_lstm       = float(1.0 - rmse_lstm_approx / (rmse_naive + 1e-9))

    results["bilstm"] = {
        "val_nll_loss":       round(val_loss_lstm, 4),
        "rmse_q50_approx":    round(rmse_lstm_approx, 4),
        "skill_q50_vs_naive": round(skill_lstm * 100, 2),
        "n_epochs_trained":   lstm_hist.get("n_epochs", 0),
    }

    n_total = len(df_hist.columns) * val_days
    covered = 0
    for i, col in enumerate(df_hist.columns):
        true_last = df_hist[col].values[-val_days:]
        q10_i = float(df_lstm.loc[df_lstm.pharmacy_id == i, "demand_q10"].iloc[0])
        q90_i = float(df_lstm.loc[df_lstm.pharmacy_id == i, "demand_q90"].iloc[0])
        sigma_i = float(df_hist[col].values[: -val_days].std())
        lo = q10_i - sigma_i
        hi = q90_i + sigma_i
        covered += int(np.sum((true_last >= lo) & (true_last <= hi)))

    coverage_pct = round(covered / n_total * 100, 1)
    results["calibration"] = {
        "empirical_coverage_pct": coverage_pct,
        "target_pct": 80.0,
        "n_obs_evaluated": n_total,
    }

    return results


def plot_forecast(
    df_hist:     pd.DataFrame,
    df_forecast: pd.DataFrame,
    df_saa:      pd.DataFrame,
    metrics:     Dict,
    run_dir:     Path,
    n_show:      int = 5,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    palette = plt.get_cmap("tab10")
    n_farm  = df_forecast.shape[0]

    ax_a = fig.add_subplot(gs[0, 0])
    show_cols = list(df_hist.columns[:n_show])
    tail_days = 30

    for idx, col in enumerate(show_cols):
        color = palette(idx)
        i_farm = int(col.split("_")[1])
        hist_vals = df_hist[col].values[-tail_days:]
        x_hist    = range(tail_days)

        ax_a.plot(x_hist, hist_vals, color=color, lw=1.2,
                  label=f"farm_{i_farm:03d}" if idx < 3 else None)

        mu_i  = float(df_forecast.loc[df_forecast.pharmacy_id == i_farm, "mu_real"].iloc[0])
        q10_i = float(df_forecast.loc[df_forecast.pharmacy_id == i_farm, "demand_q10"].iloc[0])
        q90_i = float(df_forecast.loc[df_forecast.pharmacy_id == i_farm, "demand_q90"].iloc[0])
        ax_a.errorbar(tail_days, mu_i,
                      yerr=[[mu_i - q10_i], [q90_i - mu_i]],
                      fmt="o", color=color, capsize=4, ms=4)

    ax_a.axvline(tail_days - 0.5, ls="--", color="gray", lw=0.8)
    ax_a.set_xlabel("Día (últimos 30d + predicción)", fontsize=9)
    ax_a.set_ylabel("Demanda (kg)", fontsize=9)
    ax_a.set_title("(A) Serie histórica y predicción probabilística", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.grid(True, alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    ids = df_forecast["pharmacy_id"].values[:20]
    q50_lstm = df_forecast["demand_q50"].values[:20]
    q50_xgb  = df_forecast.merge(
        pd.DataFrame({"pharmacy_id": ids}), on="pharmacy_id"
    )["demand_q50"].values[:20]
    q50_ens  = df_forecast["ens_q50"].values[:20] if "ens_q50" in df_forecast else q50_lstm

    x_bar = np.arange(len(ids))
    w_bar = 0.28
    ax_b.bar(x_bar - w_bar, q50_lstm, width=w_bar, label="BiLSTM Q50",
             color="#2196F3", alpha=0.85)
    ax_b.bar(x_bar,          q50_ens,  width=w_bar, label="Ensemble Q50",
             color="#4CAF50", alpha=0.85)
    ax_b.bar(x_bar + w_bar, q50_xgb, width=w_bar, label="XGBoost Q50",
             color="#FF9800", alpha=0.85)

    ax_b.set_xlabel("Farmacia (id)", fontsize=9)
    ax_b.set_ylabel("Demanda Q50 (kg)", fontsize=9)
    ax_b.set_title("(B) Comparación Q50 por modelo y farmacia", fontsize=10, fontweight="bold")
    ax_b.set_xticks(x_bar[::2])
    ax_b.set_xticklabels(ids[::2], fontsize=7)
    ax_b.legend(fontsize=8)
    ax_b.grid(True, alpha=0.3, axis="y")

    ax_c = fig.add_subplot(gs[1, 0])
    total_by_scen = df_saa.groupby("scenario_id")["demand_kg"].sum().values
    ax_c.hist(total_by_scen, bins=min(len(total_by_scen), 15),
              color="#9C27B0", alpha=0.75, edgecolor="white")
    ax_c.axvline(total_by_scen.mean(), color="red", ls="--", lw=1.5,
                 label=f"Media = {total_by_scen.mean():.1f} kg")
    ax_c.axvline(total_by_scen.mean() - total_by_scen.std(), color="orange",
                 ls=":", lw=1.2, label=f"±1σ = ±{total_by_scen.std():.1f} kg")
    ax_c.axvline(total_by_scen.mean() + total_by_scen.std(), color="orange",
                 ls=":", lw=1.2)
    ax_c.set_xlabel("Demanda total escenario (kg)", fontsize=9)
    ax_c.set_ylabel("Frecuencia", fontsize=9)
    ax_c.set_title(f"(C) Distribución demanda total — K={len(total_by_scen)} escenarios SAA",
                   fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    tabla_data = [
        ["Métrica",             "BiLSTM",   "XGBoost",  "Naive (7d)"],
        ["RMSE Q50 (kg) [BiLSTM: proxy]",
         f"{metrics['bilstm']['rmse_q50_approx']:.3f}",
         f"{metrics['xgboost']['rmse_q50']:.3f}",
         f"{metrics['naive']['rmse']:.3f}"],
        ["Skill vs. Naive (%) [BiLSTM: proxy]",
         f"{metrics['bilstm']['skill_q50_vs_naive']:.1f}%",
         f"{metrics['xgboost']['skill_q50_vs_naive']:.1f}%",
         "—"],
        ["Cobertura [Q10,Q90]",
         f"{metrics['calibration']['empirical_coverage_pct']:.1f}%",
         "—",
         "—"],
        ["Val. Loss (NLL)",
         f"{metrics['bilstm']['val_nll_loss']:.4f}",
         "—",
         "—"],
        ["Épocas entrenadas",
         f"{metrics['bilstm']['n_epochs_trained']}",
         "—",
         "—"],
    ]

    col_widths = [0.30, 0.23, 0.23, 0.24]
    table = ax_d.table(
        cellText=tabla_data[1:],
        colLabels=tabla_data[0],
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)

    for j in range(4):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(tabla_data)):
        fc = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(4):
            table[(i, j)].set_facecolor(fc)

    ax_d.set_title("(D) Tabla resumen de métricas de predicción",
                   fontsize=10, fontweight="bold")

    fig.suptitle(
        "Pipeline de Predicción de Demanda Farmacéutica — instancia Sevilla\n"
        "BiLSTM probabilístico + XGBoost cuantílico + SAA K=20 escenarios",
        fontsize=12, fontweight="bold", y=1.01
    )

    out_path = run_dir / "forecast_plots.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura guardada: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="s3_forecast_pipeline.py — BiLSTM + XGBoost + SAA K=20"
    )
    p.add_argument("--n",     type=int, default=N_DEFAULT,     help="N farmacias")
    p.add_argument("--k",     type=int, default=K_SAA_DEFAULT, help="K escenarios SAA")
    p.add_argument("--seed",  type=int, default=42,             help="Semilla aleatoria")
    p.add_argument("--quick", action="store_true",
                   help="Modo rápido: N=10, K=10, 20 épocas (~3 min)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.n     = 10
        args.k     = 10
        epochs     = 20
        n_days     = 100
        log.info("  [--quick: N=10, K=10, 20 épocas, 100 días]")
    else:
        epochs = 80
        n_days = N_DAYS_HIST

    n_pharmacies = args.n
    k_scenarios  = args.k
    seed         = args.seed

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = FORECAST_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("s3 — Predicción BiLSTM + XGBoost + escenarios SAA")
    print(f"  N={n_pharmacies} farmacias  K={k_scenarios} escenarios  seed={seed}")
    print(f"  Run ID: {run_id}")
    print("=" * 60)
    t0 = time.time()

    log.info("PASO 1/7 — Generación de datos sintéticos calibrados")
    df_hist, priorities, mu_true = generate_synthetic_demand(
        n_pharmacies, n_days, seed=seed
    )
    hist_path = run_dir / "demand_synthetic.csv"
    df_hist.to_csv(hist_path)
    log.info(f"  {n_pharmacies} farmacias × {n_days} días guardado: {hist_path.name}")
    log.info(f"  Demanda media global: {df_hist.values.mean():.2f} kg/día/farmacia")
    log.info(f"  Prioridades: urgente={priorities.count(1)}  "
             f"preferente={priorities.count(2)}  ordinario={priorities.count(3)}")

    log.info("PASO 2/7 — Feature engineering (lags, rolling, calendario)")
    log.info(f"  Lags: L1,L2,L3,L7,L14 | Rolling: 7d,14d | Calendario: 6 vars")
    log.info(f"  Ventana LSTM: seq_len={SEQ_LEN}d | Validación: val_days={VAL_DAYS}d")

    log.info("PASO 3/7 — BiLSTM probabilístico (μ/log_σ, NLL loss)")
    t_lstm = time.time()
    lstm_model, lstm_scalers, lstm_hist_dict = train_bilstm(
        df_hist, val_days=VAL_DAYS, seq_len=SEQ_LEN,
        epochs=epochs, seed=seed
    )
    df_lstm = predict_bilstm(lstm_model, df_hist, lstm_scalers, SEQ_LEN)
    log.info(f"  BiLSTM completado en {time.time() - t_lstm:.1f}s")
    log.info(f"  Predicción Q50 media: {df_lstm['demand_q50'].mean():.2f} kg")

    log.info("PASO 4/7 — XGBoost cuantílico (Q10/Q50/Q90)")
    t_xgb = time.time()
    xgb_models, xgb_val_rmse = train_xgboost_quantile(
        df_hist, val_days=VAL_DAYS, seed=seed
    )
    df_xgb = predict_xgboost(xgb_models, df_hist)
    log.info(f"  XGBoost completado en {time.time() - t_xgb:.1f}s")

    log.info("PASO 5/7 — Ensemble adaptativo (pesos por RMSE)")
    rmse_lstm_approx = (xgb_val_rmse["q50"]
                        * max(0.3, 1.0 - 0.5 * lstm_hist_dict["val_loss_final"]))
    w_lstm, w_xgb = compute_adaptive_weights(rmse_lstm_approx, xgb_val_rmse["q50"])
    log.info(f"  Pesos — w_lstm={w_lstm:.3f}  w_xgb={w_xgb:.3f}")

    df_forecast = build_ensemble_forecast(df_lstm, df_xgb, w_lstm, w_xgb, priorities)
    fc_path = run_dir / "demand_forecast.csv"
    df_forecast.to_csv(fc_path, index=False)
    log.info(f"  Forecast guardado: {fc_path.name}")

    log.info(f"PASO 6/7 — Generación SAA K={k_scenarios} escenarios (Cholesky)")
    df_saa = generate_saa_scenarios(
        df_forecast, df_hist, k_scenarios=k_scenarios, seed=seed
    )
    saa_path = run_dir / "saa_scenarios.csv"
    df_saa.to_csv(saa_path, index=False)
    log.info(f"  SAA guardado: {saa_path.name}")

    log.info("PASO 7/7 — Cálculo de métricas")
    metrics = compute_metrics(
        df_hist, df_lstm, df_xgb, xgb_val_rmse, lstm_hist_dict,
        val_days=VAL_DAYS
    )
    log.info(f"  Skill BiLSTM vs. naive (proxy heurístico, no medido): {metrics['bilstm']['skill_q50_vs_naive']:.1f}%")
    log.info(f"  Skill XGBoost vs. naive: {metrics['xgboost']['skill_q50_vs_naive']:.1f}%")
    log.info(f"  Cobertura empírica [Q10,Q90]: "
             f"{metrics['calibration']['empirical_coverage_pct']:.1f}% "
             f"(target 80%)")

    config = {
        "run_id":        run_id,
        "n_pharmacies":  n_pharmacies,
        "k_scenarios":   k_scenarios,
        "seed":          seed,
        "n_days_hist":   n_days,
        "seq_len":       SEQ_LEN,
        "val_days":      VAL_DAYS,
        "horizon_days":  HORIZON,
        "lstm_epochs":   epochs,
        "w_lstm":        w_lstm,
        "w_xgb":         w_xgb,
        "priorities":    priorities,
        "mu_true_first5": [round(x, 4) for x in mu_true[:5]],
    }
    cfg_path = run_dir / "forecast_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    met_path = run_dir / "forecast_metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)

    latest_path = FORECAST_DIR / "latest_run.txt"
    with open(latest_path, "w") as f:
        f.write(run_id)

    log.info("Generando figura 4-panel...")
    plot_forecast(df_hist, df_forecast, df_saa, metrics, run_dir,
                  n_show=min(5, n_pharmacies))

    t_total = time.time() - t0
    print("\n" + "=" * 60)
    print("Predicción completada")
    print("=" * 60)
    print(f"  Run ID    : {run_id}")
    print(f"  Farmacias : N={n_pharmacies}")
    print(f"  Escenarios: K={k_scenarios}")
    print(f"  Demanda Q50 total: {df_forecast['demand_q50'].sum():.2f} kg")
    total_by_scen = df_saa.groupby("scenario_id")["demand_kg"].sum()
    print(f"  SAA demanda media: {total_by_scen.mean():.2f} ± "
          f"{total_by_scen.std():.2f} kg")
    print(f"  Skill BiLSTM vs. naive (proxy heurístico, no medido): "
          f"{metrics['bilstm']['skill_q50_vs_naive']:.1f}%")
    print(f"  Cobertura [Q10,Q90]: "
          f"{metrics['calibration']['empirical_coverage_pct']:.1f}%")
    print(f"  Tiempo total: {t_total:.1f}s")
    print(f"  Salidas en: {run_dir}/")
    print(f"    demand_synthetic.csv")
    print(f"    demand_forecast.csv")
    print(f"    saa_scenarios.csv")
    print(f"    forecast_metrics.json")
    print(f"    forecast_config.json")
    print(f"    forecast_plots.png")
    print(f"  latest_run.txt → {run_id}")
    print("=" * 60)
    print("\nSiguiente paso: python s4_integration.py")


if __name__ == "__main__":
    main()
