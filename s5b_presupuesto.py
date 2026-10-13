import argparse
import json
import os
import sys
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_integration as s4
import s5b_equiparable as eq
import s5b_sevilla as sv
import s5b_solomon100 as s5b
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

DATA = Path("data")
CONDICIONES = [("Sevilla-50", 40_000, True), ("R1-100", 80_000, True), ("Sevilla-50", 80_000, False), ("R1-100", 40_000, False)]
PRUEBA = False


def instancias():
    return {"Sevilla-50": ("sevilla", sv.sevilla(50)), "R1-100": ("nueva", eq.gen_100("R1", eq.SEED + zlib.crc32(b"R1") % 1000))}


def ejecuta(tarea):
    cond, (clase, inst), presupuesto, r, alg = tarea
    prob = s4.CVRPTW_3obj(inst) if clase == "sevilla" else eq.Problema(inst, eq.demanda_replica(inst, r))
    ops = dict(sampling=s5b.PermutationSampling(), crossover=s5b.OrderCrossover(), mutation=s5b.SwapMutation())
    ref = get_reference_directions("das-dennis", 3, n_partitions=eq.PARTICIONES)
    a = (NSGA2(pop_size=eq.POP, eliminate_duplicates=True, **ops) if alg == "NSGA-II" else
         NSGA3(pop_size=eq.POP, ref_dirs=ref, eliminate_duplicates=True, **ops) if alg == "NSGA-III" else
         MOEAD(ref_dirs=ref, n_neighbors=eq.VECINOS, **ops))
    res = minimize(prob, a, ("n_eval", presupuesto), seed=40000 + r, verbose=False)
    F = res.pop.get("F"); F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    return cond, r, alg, np.unique(F, axis=0), int(res.algorithm.evaluator.n_eval), presupuesto


def resumen(df):
    filas = {}
    for cond, g in df.groupby("condicion"):
        H = g.pivot(index="replica", columns="algoritmo", values="hv")
        p = wilcoxon(H["NSGA-II"], H["MOEA/D"]).pvalue if not np.allclose(H["NSGA-II"], H["MOEA/D"]) else 1.0
        ns = g.groupby("algoritmo")["n_sol"].median()
        filas[cond] = {"p": float(p), "hv_mediano": {a: float(H[a].median()) for a in eq.ALGS},
                       "mejor_NSGA-II_vs_MOEA/D": "NSGA-II" if H["NSGA-II"].median() > H["MOEA/D"].median() else "MOEA/D",
                       "rango_medio": {a: float(v) for a, v in H[list(eq.ALGS)].rank(axis=1, ascending=False).mean().items()},
                       "replicas_frente_NSGA-II_dominado_entero": int((H["NSGA-II"] < 1e-9).sum()),
                       "replicas_frente_MOEA/D_dominado_entero": int((H["MOEA/D"] < 1e-9).sum()),
                       "soluciones_mediana": {a: float(ns[a]) for a in eq.ALGS}}
    return filas


def decision_c8(filas):
    dec = [f"{i}|{b}" for i, b, d in CONDICIONES if d]; des = [f"{i}|{b}" for i, b, d in CONDICIONES if not d]
    for grupo in (dec, des):
        pa = eq.holm(np.array([filas[c]["p"] for c in grupo]))
        for c, q in zip(grupo, pa):
            filas[c]["p_holm"] = float(q); filas[c]["significativo"] = bool(q < 0.05); filas[c]["decide"] = c in dec
    gana = lambda c, a: filas[c]["significativo"] and filas[c]["mejor_NSGA-II_vs_MOEA/D"] == a
    v = "confirmada" if all(gana(c, "NSGA-II") for c in dec) else "refutada" if all(gana(c, "MOEA/D") for c in dec) else "no concluyente"
    return {"veredicto": v, "condiciones": filas}


def main():
    global PRUEBA
    ap = argparse.ArgumentParser(); ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prueba", action="store_true"); a = ap.parse_args(); PRUEBA = a.prueba
    out = DATA / ("consolidacion_prueba" if PRUEBA else "consolidacion") / "complementarias"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    I = instancias(); v5 = eq.w5({"R1-100": I["R1-100"][1]}); print("W5:", v5, flush=True)
    if not all(v5.values()):
        sys.exit("ERROR: W5 fallida; el generador replicado no reproduce la instancia original")
    reps = range(2 if PRUEBA else eq.N_REP)
    T = [(f"{i}|{b}", I[i], 600 if PRUEBA else b, r, alg) for i, b, _ in CONDICIONES for r in reps for alg in eq.ALGS]
    T.sort(key=lambda t: -t[2] * (2 if t[1][0] == "nueva" else 1))
    print(f"{len(T)} ejecuciones con {a.procesos} procesos{' · MODO PRUEBA' if PRUEBA else ''}", flush=True)
    frentes, evals = {}, []
    with ProcessPoolExecutor(max_workers=a.procesos, initializer=_init, initargs=(PRUEBA,)) as ex:
        for i, (c, r, alg, F, ne, pres) in enumerate(ex.map(ejecuta, T, chunksize=1)):
            frentes.setdefault((c, r), {})[alg] = F; evals.append({"condicion": c, "replica": r, "algoritmo": alg, "n_eval": ne, "presupuesto": pres})
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(T)} · {time.time()-t0:.0f} s", flush=True)
    w4 = all(e["presupuesto"] <= e["n_eval"] <= e["presupuesto"] + eq.POP for e in evals); print("W4:", w4, flush=True)
    if not w4:
        sys.exit("ERROR: W4 fallida; algún algoritmo no ha usado el presupuesto fijado")
    df = pd.DataFrame([{"condicion": c, "replica": r, "algoritmo": al, **m} for (c, r), fr in frentes.items() for al, m in eq.indicadores(fr).items()])
    if PRUEBA:
        print(f"Comprobación técnica correcta: {len(df)} filas de indicadores; no se calcula el veredicto · {time.time()-t0:.0f} s"); return
    df.to_csv(out / "C8_indicadores.csv", index=False); pd.DataFrame(evals).to_csv(out / "C8_evaluaciones.csv", index=False)
    d = decision_c8(resumen(df))
    json.dump({"W4": w4, "W5": v5, "decision_C8": d, "fecha": datetime.now().isoformat(timespec="seconds"), "protocolo": "protocolo_consolidacion.md (enmienda 1.3)"},
              open(out / "C8_decision.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"DECISIÓN C8: {d['veredicto']} · {time.time()-t0:.0f} s")


def _init(prueba):
    global PRUEBA
    PRUEBA = prueba; eq.PRUEBA = prueba; sv.PRUEBA = prueba


if __name__ == "__main__":
    main()
