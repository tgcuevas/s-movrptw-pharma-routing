import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_integration as s4
import robustez_vss as rv
import s5b_equiparable as eq
import s5b_solomon100 as s5b
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

DATA = Path("data")
NUEVAS = {"C101-25-s1001": ("C101", 1001), "C101-25-s1002": ("C101", 1002), "R101-25-s1003": ("R101", 1003), "R101-25-s1004": ("R101", 1004)}
PRUEBA = False


def sevilla(n):
    df_fc, _, _, _ = s4.load_forecast_data(rv.REF[n][0])
    return {"label": f"Sevilla-{n}", "n": n, "D": s4.get_distance_matrix(n).tolist(), "demands": df_fc["demand_q50"].values[:n].tolist(),
            "priorities": df_fc["priority_class"].values[:n].tolist(), "service_times": s4.get_service_times(n).tolist(), "capacity": s4.VEHICLE_CAPACITY}


def instancias():
    I = {f"Sevilla-{n}": ("sevilla", sevilla(n)) for n in (25, 50)}
    for k, (tipo, sd) in NUEVAS.items():
        I[k] = ("nueva", eq.gen_25(tipo, sd))
    return I


def ejecuta(tarea):
    key, (clase, inst), r, alg = tarea
    prob = s4.CVRPTW_3obj(inst) if clase == "sevilla" else eq.Problema(inst, eq.demanda_replica(inst, r))
    presupuesto = 600 if PRUEBA else eq.PRESUPUESTO
    ops = dict(sampling=s5b.PermutationSampling(), crossover=s5b.OrderCrossover(), mutation=s5b.SwapMutation())
    ref = get_reference_directions("das-dennis", 3, n_partitions=eq.PARTICIONES)
    a = (NSGA2(pop_size=eq.POP, eliminate_duplicates=True, **ops) if alg == "NSGA-II" else
         NSGA3(pop_size=eq.POP, ref_dirs=ref, eliminate_duplicates=True, **ops) if alg == "NSGA-III" else
         MOEAD(ref_dirs=ref, n_neighbors=eq.VECINOS, **ops))
    res = minimize(prob, a, ("n_eval", presupuesto), seed=40000 + r, verbose=False)
    F = res.pop.get("F"); F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    return key, r, alg, np.unique(F, axis=0), int(res.algorithm.evaluator.n_eval)


def decision_c6(df):
    claves = list(df["instancia"].unique()); p, filas = [], []
    for k in claves:
        H = df[df.instancia == k].pivot(index="replica", columns="algoritmo", values="hv")
        p.append(wilcoxon(H["NSGA-II"], H["MOEA/D"]).pvalue if not np.allclose(H["NSGA-II"], H["MOEA/D"]) else 1.0)
        filas.append({"instancia": k, "hv_mediano": {a: float(H[a].median()) for a in eq.ALGS},
                      "mejor_NSGA-II_vs_MOEA/D": "NSGA-II" if H["NSGA-II"].median() > H["MOEA/D"].median() else "MOEA/D",
                      "rango_medio": {a: float(v) for a, v in H[list(eq.ALGS)].rank(axis=1, ascending=False).mean().items()}})
    pa = eq.holm(np.array(p))
    for f, q in zip(filas, pa):
        f["p_holm"] = float(q); f["significativo"] = bool(q < 0.05)
    gana = lambda f, a: f["significativo"] and f["mejor_NSGA-II_vs_MOEA/D"] == a
    sev = [f for f in filas if f["instancia"].startswith("Sevilla")]; nue = [f for f in filas if not f["instancia"].startswith("Sevilla")]
    if all(gana(f, "NSGA-II") for f in sev) and sum(gana(f, "NSGA-II") for f in nue) >= 3:
        v = "confirmada"
    elif any(gana(f, "MOEA/D") for f in sev) or sum(gana(f, "MOEA/D") for f in nue) >= 3:
        v = "refutada"
    else:
        v = "no concluyente"
    return {"veredicto": v, "instancias": filas}


def main():
    global PRUEBA
    ap = argparse.ArgumentParser(); ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prueba", action="store_true"); a = ap.parse_args(); PRUEBA = a.prueba
    out = DATA / ("consolidacion_prueba" if PRUEBA else "consolidacion") / "complementarias"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    I = instancias(); claves = list(I)[:3] if PRUEBA else list(I); reps = range(2 if PRUEBA else eq.N_REP)
    T = [(k, I[k], r, alg) for k in claves for r in reps for alg in eq.ALGS]
    print(f"{len(T)} ejecuciones con {a.procesos} procesos{' · MODO PRUEBA' if PRUEBA else ''}", flush=True)
    frentes, evals = {}, []
    with ProcessPoolExecutor(max_workers=a.procesos, initializer=_init, initargs=(PRUEBA,)) as ex:
        for i, (k, r, alg, F, ne) in enumerate(ex.map(ejecuta, T, chunksize=1)):
            frentes.setdefault((k, r), {})[alg] = F; evals.append({"instancia": k, "replica": r, "algoritmo": alg, "n_eval": ne})
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(T)} · {time.time()-t0:.0f} s", flush=True)
    presup = 600 if PRUEBA else eq.PRESUPUESTO
    w4 = all(presup <= e["n_eval"] <= presup + eq.POP for e in evals); print("W4:", w4, flush=True)
    if not w4:
        sys.exit("ERROR: W4 fallida; algún algoritmo no ha usado el presupuesto fijado")
    df = pd.DataFrame([{"instancia": k, "replica": r, "algoritmo": al, **m} for (k, r), fr in frentes.items() for al, m in eq.indicadores(fr).items()])
    if PRUEBA:
        print(f"Comprobación técnica correcta: {len(df)} filas de indicadores; no se calcula el veredicto · {time.time()-t0:.0f} s"); return
    df.to_csv(out / "C6_indicadores.csv", index=False); pd.DataFrame(evals).to_csv(out / "C6_evaluaciones.csv", index=False)
    d = decision_c6(df)
    json.dump({"W4": w4, "decision_C6": d, "fecha": datetime.now().isoformat(timespec="seconds"), "protocolo": "protocolo_consolidacion.md (enmienda 1.2)"},
              open(out / "C6_decision.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"DECISIÓN C6: {d['veredicto']} · {time.time()-t0:.0f} s")


def _init(prueba):
    global PRUEBA
    PRUEBA = prueba; eq.PRUEBA = prueba


if __name__ == "__main__":
    main()
