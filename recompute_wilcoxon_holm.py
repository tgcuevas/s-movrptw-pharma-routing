import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

BASE = Path("data/sensitivity/solomon100")
ALG = ["NSGA-II", "NSGA-III", "MOEA/D"]
MEJOR_ALTO = {"hv": True, "igd": False, "spacing": False}


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def main():
    d = pd.read_csv(BASE / "solomon100_results.csv")
    filas = []
    for (inst, met) in [(i, m) for i in sorted(d.instance.unique()) for m in ("hv", "igd", "spacing")]:
        sub = d[d.instance == inst].pivot(index="run", columns="algorithm", values=met)
        terna = []
        for a, b in combinations(ALG, 2):
            x, y = sub[a].values, sub[b].values
            dif = (x - y) if MEJOR_ALTO[met] else (y - x)
            nz = dif[dif != 0]
            rk = rankdata(np.abs(nz)); wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
            r = float((wp - wm) / (wp + wm)) if (wp + wm) > 0 else 0.0
            p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue) if len(nz) else 1.0
            terna.append({"instancia": inst, "indicador": met, "a": a, "b": b, "p": p, "r_biserial": round(r, 3),
                          "mediana_a": float(np.median(x)), "mediana_b": float(np.median(y))})
        for f, pa in zip(terna, holm([f["p"] for f in terna])):
            f["p_holm_terna"] = float(pa)
        filas += terna
    for f, pa in zip(filas, holm([f["p"] for f in filas])):
        f["p_holm_global"] = float(pa)
    for f in filas:
        sig = f["p_holm_terna"] < 0.05
        f["resultado"] = ("empate" if not sig else ("gana " + f["a"] if f["r_biserial"] > 0 else "gana " + f["b"]))

    def recuento(rival, clave):
        s = [f for f in filas if f["a"] == "NSGA-II" and f["b"] == rival]
        sup = sum(1 for f in s if f[clave] < 0.05 and f["r_biserial"] > 0)
        inf = sum(1 for f in s if f[clave] < 0.05 and f["r_biserial"] < 0)
        return {"superior": sup, "inferior": inf, "sin_diferencia": len(s) - sup - inf}

    out = {"metodo": "Wilcoxon de rangos con signo emparejado por réplica, bilateral; corrección de Holm",
           "referencia": "Benavoli, Corani y Mangili (2016)",
           "NSGA-II_vs_NSGA-III": {"holm_terna": recuento("NSGA-III", "p_holm_terna"), "holm_global": recuento("NSGA-III", "p_holm_global")},
           "NSGA-II_vs_MOEA/D": {"holm_terna": recuento("MOEA/D", "p_holm_terna"), "holm_global": recuento("MOEA/D", "p_holm_global")},
           "comparaciones": filas}
    json.dump(out, open(BASE / "solomon100_wilcoxon_holm.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    for k in ("NSGA-II_vs_NSGA-III", "NSGA-II_vs_MOEA/D"):
        print(k, "· Holm por terna:", out[k]["holm_terna"], "· Holm global:", out[k]["holm_global"])
    print("Guardado:", BASE / "solomon100_wilcoxon_holm.json")


if __name__ == "__main__":
    main()
