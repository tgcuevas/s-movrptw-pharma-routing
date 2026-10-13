import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = {25: "20260615_200957", 50: "20260615_205432"}
PRIORITY_PROBS = [0.30, 0.20, 0.50]
K_REAL = 10_000
SEED_REAL = 42
VAL_DAYS = 20


def historico(n, n_days, seed):
    rng = np.random.default_rng(seed)
    mu_base = rng.uniform(3.0, 18.0, size=n)
    rng.choice([1, 2, 3], size=n, p=PRIORITY_PROBS)
    dates = pd.date_range("2024-06-01", periods=n_days, freq="D")
    dow = dates.dayofweek.values
    w = np.where(dow == 0, 1.10, np.where(dow == 4, 1.08, np.where(dow >= 5, 0.75, 1.00)))
    trend = 1.0 + 0.05 * (np.arange(n_days) / (n_days - 1) - 0.5)
    dem = np.zeros((n_days, n))
    for i in range(n):
        g = rng.gamma(shape=4.0, scale=mu_base[i] * trend * w / 4.0)
        dem[:, i] = rng.poisson(g).astype(float)
    return mu_base, np.clip(dem, 0.5, None), dates


def rmse(p, y):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def evalua(n, run):
    d = Path("data/forecast") / run
    fc = pd.read_csv(d / "demand_forecast.csv").sort_values("pharmacy_id").reset_index(drop=True)
    hist = pd.read_csv(d / "demand_synthetic.csv", index_col=0)
    cfg = json.loads((d / "forecast_config.json").read_text())
    nd, seed = int(cfg["n_days_hist"]), int(cfg["seed"])

    mu_base, dem, dates = historico(n, nd, seed)
    if not np.allclose(dem, hist.values[:, :n]):
        sys.exit(f"ERROR N={n}: el histórico regenerado no coincide con demand_synthetic.csv")

    fecha = dates[-1] + pd.Timedelta(days=1)
    dw = fecha.dayofweek
    w = 1.10 if dw == 0 else 1.08 if dw == 4 else 0.75 if dw >= 5 else 1.00
    mu_true = mu_base * (1.0 + 0.05 * (nd / (nd - 1) - 0.5)) * w
    rng = np.random.default_rng(SEED_REAL)
    Y = np.clip(rng.poisson(rng.gamma(shape=4.0, scale=mu_true / 4.0, size=(K_REAL, n))).astype(float), 0.5, None)

    b = fc["demand_q50"].values[:n]
    e = fc["ens_q50"].values[:n]
    wl, wx = float(fc["w_lstm"].iloc[0]), float(fc["w_xgb"].iloc[0])
    if (e <= 0).any():
        sys.exit("ERROR: hay medianas del ensemble recortadas a cero; no se puede reconstruir XGBoost")
    x = (e - wl * b) / wx
    naive = hist.values[-7, :n]

    cob = lambda lo, hi: round(100 * float(np.mean((Y >= lo) & (Y <= hi))), 1)
    modelos = {"naive_t7": naive, "BiLSTM": b, "XGBoost": x, "ensemble": e}
    r_real = {k: rmse(p, Y) for k, p in modelos.items()}
    r_media = {k: rmse(p, mu_true) for k, p in modelos.items()}

    cub = tot = 0
    for i in range(n):
        y = hist.values[-VAL_DAYS:, i]
        s = hist.values[:-VAL_DAYS, i].std()
        cub += int(np.sum((y >= fc["demand_q10"][i] - s) & (y <= fc["demand_q90"][i] + s))); tot += VAL_DAYS

    return {
        "script": "recompute_forecast_eval.py",
        "run_s3": run, "n_farmacias": n, "dia_previsto": str(fecha.date()),
        "realizaciones": K_REAL, "seed_realizaciones": SEED_REAL,
        "historico_regenerado_identico": True,
        "cobertura_Q10_Q90_frente_a_demanda_verdadera_pct": {
            "BiLSTM": cob(fc["demand_q10"].values[:n], fc["demand_q90"].values[:n]),
            "ensemble": cob(fc["ens_q10"].values[:n], fc["ens_q90"].values[:n]),
            "nominal": 80.0},
        "rmse_frente_a_realizaciones_kg": {k: round(v, 3) for k, v in r_real.items()},
        "skill_frente_a_naive_realizaciones_pct": {k: round(100 * (1 - r_real[k] / r_real["naive_t7"]), 1)
                                                    for k in ("BiLSTM", "XGBoost", "ensemble")},
        "rmse_frente_a_media_verdadera_kg": {k: round(v, 3) for k, v in r_media.items()},
        "metrica_s3_banda_ensanchada_pct": round(100 * cub / tot, 1),
        "nota": ("El plan determinista y los escenarios SAA de s4 usan la media y la desviación del BiLSTM; "
                 "XGBoost y el ensemble se evalúan pero no alimentan la optimización. La métrica de s3 mide "
                 "la cobertura de una banda ensanchada ±σ_hist sobre días del histórico, no la de [Q10, Q90]."),
    }


def main():
    for n, run in RUNS.items():
        out = evalua(n, run)
        p = Path("data/forecast") / run / "forecast_eval_verdad.json"
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        c = out["cobertura_Q10_Q90_frente_a_demanda_verdadera_pct"]; s = out["skill_frente_a_naive_realizaciones_pct"]
        print(f"N={n} ({out['dia_previsto']}): cobertura BiLSTM {c['BiLSTM']} % · ensemble {c['ensemble']} % · "
              f"skill BiLSTM {s['BiLSTM']} % · XGBoost {s['XGBoost']} % · ensemble {s['ensemble']} % · "
              f"métrica de s3 {out['metrica_s3_banda_ensanchada_pct']} % → {p}")


if __name__ == "__main__":
    main()
