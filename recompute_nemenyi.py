#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    print("[AVISO] scikit-posthocs no instalado: se usará Wilcoxon por pares.\n"
          "        Instala con:  pip install scikit-posthocs==0.12.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sensitivity/solomon100/solomon100_results.csv")
    ap.add_argument("--metrics", default="hv,igd,spacing")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"[ERROR] No se encuentra el CSV: {csv_path}\n"
                         f"        Ejecuta antes s5b_solomon100.py para generarlo.")

    df = pd.read_csv(csv_path)
    metrics = [m.strip() for m in args.metrics.split(",")]
    algos = list(df["algorithm"].unique())
    instances = list(df["instance"].unique())

    print(f"Cargado: {csv_path}")
    print(f"Algoritmos: {algos}")
    print(f"Instancias: {instances}\n")

    out = {"method": "nemenyi" if SP_AVAILABLE else "wilcoxon_pairwise",
           "per_instance": {}}

    for inst in instances:
        out["per_instance"][inst] = {}
        df_i = df[df["instance"] == inst]
        for metric in metrics:
            groups = [df_i[df_i["algorithm"] == a][metric].dropna().values
                      for a in algos]
            try:
                stat, pval = stats.friedmanchisquare(*groups)
            except Exception:
                stat, pval = np.nan, np.nan

            posthoc = {}
            min_len = min((len(g) for g in groups), default=0)
            if min_len >= 2:
                bal = np.column_stack([g[:min_len] for g in groups])
                if SP_AVAILABLE:
                    ph = sp.posthoc_nemenyi_friedman(bal)
                    ph.columns = algos
                    ph.index = algos
                    posthoc = ph.round(4).to_dict()
                else:
                    for i, j in combinations(range(len(algos)), 2):
                        try:
                            _, wp = stats.wilcoxon(bal[:, i], bal[:, j])
                            posthoc[f"{algos[i]}_vs_{algos[j]}"] = round(float(wp), 6)
                        except Exception:
                            posthoc[f"{algos[i]}_vs_{algos[j]}"] = None

            out["per_instance"][inst][metric] = {
                "friedman_stat": round(float(stat), 4) if not np.isnan(stat) else None,
                "friedman_p": round(float(pval), 6) if not np.isnan(pval) else None,
                "n_blocks": int(min_len),
                "posthoc": posthoc,
            }

            print(f"── {inst} · {metric.upper()} "
                  f"(Friedman p={pval:.2e}, n={min_len}) ──")
            if SP_AVAILABLE and posthoc:
                print(pd.DataFrame(posthoc).round(4).to_string())
            else:
                for k, v in posthoc.items():
                    print(f"   {k}: p={v}")
            print()

    out_path = csv_path.parent / "solomon100_posthoc_nemenyi.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[OK] Guardado: {out_path}")
    print("     Copia estos p-valores a §4.2.1 y a la tabla H3 de la memoria.")


if __name__ == "__main__":
    main()
