#!/usr/bin/env python3
import argparse, glob, json, os, sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats


def permutacion_pareada_media(d: np.ndarray, n_mc: int = 200000, seed: int = 42) -> float:
    K = len(d)
    obs = d.mean()
    if K <= 20:
        from itertools import product
        total = 0; mayores = 0
        for signos in product((1.0, -1.0), repeat=K):
            total += 1
            if (d * np.array(signos)).mean() >= obs - 1e-12:
                mayores += 1
        return mayores / total
    rng = np.random.default_rng(seed)
    signos = rng.choice((1.0, -1.0), size=(n_mc, K))
    return float(((signos * d).mean(axis=1) >= obs - 1e-12).mean())


def localizar_s4(data_root: str) -> str:
    cands = sorted(glob.glob(os.path.join(data_root, "integration", "*_s4")))
    if not cands:
        sys.exit(f"[ERROR] No se encontró ninguna carpeta *_s4 bajo {data_root}/integration")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="raíz de datos (por defecto: data)")
    ap.add_argument("--s4-dir", default=None, help="carpeta *_s4 concreta (si se omite, la última)")
    args = ap.parse_args()

    s4 = args.s4_dir or localizar_s4(args.data_root)
    eev = pd.read_csv(os.path.join(s4, "s4_eev_eval.csv"))
    saa = pd.read_csv(os.path.join(s4, "s4_saa_fronts.csv"))
    df = eev.merge(saa, on="scenario_id", suffixes=("", "_saa"))
    K = len(df)

    pares = {
        "f1": ("f1_ev_eval", "f1_best"),
        "f2": ("f2_ev_eval", "f2_best"),
        "f3": ("f3_ev_eval", "f3_best"),
    }
    resultados = {}
    print(f"\nWilcoxon pareado EEV vs RP por escenario · {os.path.basename(s4)} · K={K}")
    print(f"{'obj':>4} {'mediana EEV':>16} {'mediana RP':>16} {'Δ mediana':>14} {'W':>8} {'p Wilcoxon':>11} {'p permutación':>14}")
    for obj, (c_eev, c_rp) in pares.items():
        d = df[c_eev].to_numpy(dtype=float) - df[c_rp].to_numpy(dtype=float)
        if np.allclose(d, 0):
            w, p = float("nan"), 1.0
        else:
            w, p = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
        p_perm = permutacion_pareada_media(d)
        resultados[obj] = {
            "K": K,
            "media_diferencia_EEV_menos_RP": float(d.mean()),
            "p_permutacion_media_EEV_mayor_que_RP": float(p_perm),
            "significativo_perm_0_05": bool(p_perm < 0.05),
            "mediana_EEV": float(np.median(df[c_eev])),
            "mediana_RP": float(np.median(df[c_rp])),
            "mediana_diferencia": float(np.median(d)),
            "n_escenarios_EEV_mayor": int((d > 0).sum()),
            "W": None if np.isnan(w) else float(w),
            "p_unilateral_EEV_mayor_que_RP": float(p),
            "significativo_0_05": bool(p < 0.05),
        }
        print(f"{obj:>4} {np.median(df[c_eev]):>16.2f} {np.median(df[c_rp]):>16.2f} "
              f"{np.median(d):>14.2f} {w:>8.1f} {p:>11.4g} {p_perm:>14.4g}")

    salida = {
        "script": "recompute_vss_tests.py",
        "proposito": "P8: test pareado por escenario del estadistico del VSS de s4 (EEV vs media de mejores por objetivo, columnas f*_best) para H1/H2, Tabla 6.1",
        "s4_dir": os.path.abspath(s4),
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "tests": "Wilcoxon pareado (mediana) + permutacion pareada de signos sobre la MEDIA de las diferencias (el estadistico que define el VSS); ambos unilaterales EEV > RP",
        "resultados": resultados,
    }
    out = os.path.join(s4, "recompute_vss_tests.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nRegistro JSON: {out}")
    print("Tabla 6.1: usar el p de PERMUTACIÓN (testa la media, que es lo que define el VSS): f1 → H1 · f2 → H2")


if __name__ == "__main__":
    main()
