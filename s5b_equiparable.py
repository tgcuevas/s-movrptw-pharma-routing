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
from scipy.spatial.distance import cdist
from scipy.stats import friedmanchisquare, wilcoxon

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s5b_solomon100 as s5b
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

DATA = Path("data")
SEED = 42
FAMILIAS_100 = ("C1", "C2", "R1", "R2", "RC1", "RC2")
ALGS = ("NSGA-II", "NSGA-III", "MOEA/D")
N_REP, PRESUPUESTO, POP, PARTICIONES, VECINOS = 30, 20_000, 92, 12, 20
Q_NEMENYI_3 = 2.343
PRUEBA = False


def gen_100(family, seed):
    rng = np.random.default_rng(seed); n = 100
    if family.startswith("C"):
        centers = rng.uniform(15, 85, (5, 2))
        xy = np.clip(np.vstack([rng.normal(centers[i % 5], 7, (1, 2)) for i in range(n)]), 0, 100)
        tw = 600 if family == "C1" else 3000
    elif family.startswith("R"):
        xy = rng.uniform(0, 100, (n, 2)); tw = 900 if family == "R1" else 4500
    else:
        n_clust = n // 3; centers = rng.uniform(20, 80, (3, 2))
        xy_c = np.vstack([rng.normal(centers[i % 3], 7, (1, 2)) for i in range(n_clust)])
        xy = np.clip(np.vstack([xy_c, rng.uniform(0, 100, (n - n_clust, 2))]), 0, 100)
        tw = 1200 if family == "RC1" else 3600
    D = cdist(np.vstack([[[50.0, 50.0]], xy]), np.vstack([[[50.0, 50.0]], xy])) * 60.0; np.fill_diagonal(D, 0.0)
    mu = rng.uniform(5, 15, n); dem = np.clip(rng.poisson(rng.gamma(shape=4.0, scale=mu / 4.0)).astype(float), 0.5, None)
    pri = rng.choice([1, 2, 3], size=n, p=s5b.PRIO_PROBS).tolist(); svc = np.full(n + 1, 120.0); svc[0] = 0.0
    return {"label": f"{family}-100", "n": n, "D": D, "mu": mu, "dem_original": dem, "pri": pri, "svc": svc,
            "cap": float(s5b.CAPACITY), "tw": tw, "co2": float(s5b.CO2_ALPHA)}


def gen_25(stype, seed):
    import s5_sensitivity as s5
    rng = np.random.default_rng(seed); n = 25
    if stype == "C101":
        centers = rng.uniform(20, 80, (3, 2))
        xy = np.clip(np.vstack([rng.normal(centers[i % 3], 8, (1, 2)) for i in range(n)]), 0, 100); tw = 600
    else:
        xy = rng.uniform(0, 100, (n, 2)); tw = 1800
    D = cdist(np.vstack([[[50.0, 50.0]], xy]), np.vstack([[[50.0, 50.0]], xy])) * 2.0 * 60; np.fill_diagonal(D, 0.0)
    mu = rng.uniform(5, 15, n); dem = np.clip(rng.poisson(rng.gamma(shape=4.0, scale=mu / 4.0)).astype(float), 0.5, None)
    pri = rng.choice([1, 2, 3], size=n, p=s5.PRIO_PROBS).tolist(); svc = np.full(n + 1, 120.0); svc[0] = 0.0
    return {"label": f"{stype}-25", "n": n, "D": D, "mu": mu, "dem_original": dem, "pri": pri, "svc": svc,
            "cap": float(s5.CAPACITY), "tw": tw, "co2": float(s5.CO2_ALPHA)}


def instancias():
    I = {f"{f}-100": gen_100(f, SEED + zlib.crc32(f.encode()) % 1000) for f in FAMILIAS_100}
    I["C101-25"] = gen_25("C101", SEED); I["R101-25"] = gen_25("R101", SEED + 1)
    return I


def w5(I):
    import s5_sensitivity as s5
    res = {}
    for k, inst in I.items():
        if k.endswith("-100"):
            o = s5b.gen_solomon_100(k[:-4], SEED + zlib.crc32(k[:-4].encode()) % 1000)
        else:
            o = s5.gen_solomon_instance(k[:4], 25, SEED if k.startswith("C") else SEED + 1)
        res[k] = bool(np.allclose(np.array(o["D"]), inst["D"]) and np.array_equal(np.array(o["demands"]), inst["dem_original"])
                      and list(o["priorities"]) == list(inst["pri"]) and np.allclose(np.array(o["service_times"]), inst["svc"]))
    return res


class Problema(ElementwiseProblem):
    def __init__(self, inst, dem):
        self.D, self.dem, self.pri, self.svc, self.cap, self.co2 = inst["D"], dem, inst["pri"], inst["svc"], inst["cap"], inst["co2"]
        self.b = inst["D"][0, 1:] + inst["tw"]
        super().__init__(n_var=inst["n"], n_obj=3, xl=0, xu=inst["n"] - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        routes, cur, cd = [], [], 0.0
        for idx in np.asarray(x, dtype=int):
            if cd + self.dem[idx] > self.cap and cur:
                routes.append(cur); cur, cd = [], 0.0
            cur.append(int(idx)); cd += float(self.dem[idx])
        routes.append(cur)
        f1 = f2 = f3 = 0.0
        for rt in routes:
            t, load, prev = 0.0, 0.0, 0
            for idx in rt:
                nd = idx + 1; t += self.D[prev, nd] + self.svc[nd]; load += self.dem[idx]; prev = nd
                f2 += s5b._ALPHA_TW[self.pri[idx]] * max(0.0, t - self.b[idx])
            t += self.D[prev, 0]; f1 += t; f3 += self.co2 * t * (load / self.cap)
        out["F"] = [f1, f2, f3]


def demanda_replica(inst, r):
    rng = np.random.default_rng(30000 + r)
    return np.clip(rng.poisson(rng.gamma(shape=4.0, scale=inst["mu"] / 4.0)).astype(float), 0.5, None)


def ejecuta(tarea):
    key, inst, r, alg = tarea
    prob = Problema(inst, demanda_replica(inst, r)); presupuesto = 600 if PRUEBA else PRESUPUESTO
    ops = dict(sampling=s5b.PermutationSampling(), crossover=s5b.OrderCrossover(), mutation=s5b.SwapMutation())
    ref = get_reference_directions("das-dennis", 3, n_partitions=PARTICIONES)
    if alg == "NSGA-II":
        a = NSGA2(pop_size=POP, eliminate_duplicates=True, **ops)
    elif alg == "NSGA-III":
        a = NSGA3(pop_size=POP, ref_dirs=ref, eliminate_duplicates=True, **ops)
    else:
        a = MOEAD(ref_dirs=ref, n_neighbors=VECINOS, **ops)
    res = minimize(prob, a, ("n_eval", presupuesto), seed=40000 + r, verbose=False)
    F = res.pop.get("F"); F = F[NonDominatedSorting().do(F, only_non_dominated_front=True)]
    return key, r, alg, np.unique(F, axis=0), int(res.algorithm.evaluator.n_eval)


def spacing(F):
    if len(F) < 2:
        return 0.0
    d = cdist(F, F); np.fill_diagonal(d, np.inf); m = d.min(1)
    return float(np.sqrt(((m - m.mean()) ** 2).sum() / (len(F) - 1)))


def indicadores(frentes):
    U = np.vstack(list(frentes.values())); Und = U[NonDominatedSorting().do(U, only_non_dominated_front=True)]
    lo, hi = U.min(0), Und.max(0); Z = lambda F: (F - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    hv, igd = HV(ref_point=np.full(3, 1.1)), IGD(Z(Und))
    return {a: {"hv": float(hv(Z(F))), "igd": float(igd(Z(F))), "spacing": spacing(Z(F)), "n_sol": int(len(F))} for a, F in frentes.items()}


def holm(p):
    orden = np.argsort(p); m = len(p); adj = np.empty(m); acc = 0.0
    for i, j in enumerate(orden):
        acc = max(acc, min(1.0, (m - i) * p[j])); adj[j] = acc
    return adj


def contrastes(df):
    out = {}
    for key, g in df.groupby("instancia"):
        H = g.pivot(index="replica", columns="algoritmo", values="hv")[list(ALGS)]
        rk = H.rank(axis=1, ascending=False).mean(); N = len(H)
        cd = Q_NEMENYI_3 * np.sqrt(3 * 4 / (6 * N))
        pares = [("NSGA-II", "NSGA-III"), ("NSGA-II", "MOEA/D"), ("NSGA-III", "MOEA/D")]
        p = np.array([wilcoxon(H[a], H[b]).pvalue if not np.allclose(H[a], H[b]) else 1.0 for a, b in pares]); pa = holm(p)
        out[key] = {"friedman_p": float(friedmanchisquare(*[H[a] for a in ALGS]).pvalue), "rango_medio": {a: float(rk[a]) for a in ALGS},
                    "nemenyi_cd": float(cd), "hv_medio": {a: float(H[a].mean()) for a in ALGS},
                    "wilcoxon_holm": {f"{a} vs {b}": {"p_holm": float(pa[i]), "mejor": a if H[a].median() > H[b].median() else b,
                                                       "significativo": bool(pa[i] < 0.05)} for i, (a, b) in enumerate(pares)}}
    return out


def decision_c3(con):
    rangos = {a: float(np.mean([c["rango_medio"][a] for c in con.values()])) for a in ALGS}; mejor = min(rangos, key=rangos.get)
    def gana(a, b):
        n = 0
        for c in con.values():
            w = c["wilcoxon_holm"].get(f"{a} vs {b}") or c["wilcoxon_holm"].get(f"{b} vs {a}")
            n += int(w["significativo"] and w["mejor"] == a)
        return n
    sup = {b: gana("NSGA-II", b) for b in ALGS if b != "NSGA-II"}
    if mejor == "NSGA-II" and all(v >= 5 for v in sup.values()):
        v = "confirmada"
    elif mejor != "NSGA-II" and gana(mejor, "NSGA-II") >= 5:
        v = "refutada"
    else:
        v = "no concluyente"
    return {"veredicto": v, "rango_medio_global": rangos, "mejor_rango": mejor, "NSGA-II_superior_significativo_en": sup,
            "mejor_superior_a_NSGA-II_en": (gana(mejor, "NSGA-II") if mejor != "NSGA-II" else None), "instancias": len(con)}


def main():
    global PRUEBA
    ap = argparse.ArgumentParser(); ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prueba", action="store_true"); a = ap.parse_args(); PRUEBA = a.prueba
    out = DATA / ("consolidacion_prueba" if PRUEBA else "consolidacion"); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    I = instancias(); v5 = w5(I); print("W5:", v5, flush=True)
    if not all(v5.values()):
        sys.exit("ERROR: W5 fallida; el generador replicado no reproduce alguna instancia original")
    claves = list(I)[:2] if PRUEBA else list(I); reps = range(2 if PRUEBA else N_REP)
    tareas = [(k, I[k], r, alg) for k in claves for r in reps for alg in ALGS]
    print(f"{len(tareas)} ejecuciones con {a.procesos} procesos{' · MODO PRUEBA' if PRUEBA else ''}", flush=True)
    frentes, evals = {}, []
    with ProcessPoolExecutor(max_workers=a.procesos, initializer=_init, initargs=(PRUEBA,)) as ex:
        for i, (k, r, alg, F, ne) in enumerate(ex.map(ejecuta, tareas, chunksize=1)):
            frentes.setdefault((k, r), {})[alg] = F; evals.append({"instancia": k, "replica": r, "algoritmo": alg, "n_eval": ne})
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(tareas)} · {time.time()-t0:.0f} s", flush=True)
    presup = 600 if PRUEBA else PRESUPUESTO
    w4 = all(presup <= e["n_eval"] <= presup + POP for e in evals); print("W4:", w4, flush=True)
    if not w4:
        sys.exit("ERROR: W4 fallida; algún algoritmo no ha usado el presupuesto fijado")
    filas = [{"instancia": k, "replica": r, "algoritmo": alg, **m} for (k, r), fr in frentes.items() for alg, m in indicadores(fr).items()]
    df = pd.DataFrame(filas)
    if PRUEBA:
        print(f"Comprobación técnica correcta: {len(df)} filas de indicadores; no se calcula el veredicto · {time.time()-t0:.0f} s"); return
    df.to_csv(out / "P4_indicadores.csv", index=False); pd.DataFrame(evals).to_csv(out / "P4_evaluaciones.csv", index=False)
    con = contrastes(df); d = decision_c3(con)
    json.dump({"W4": w4, "W5": v5, "contrastes": con, "decision_C3": d, "fecha": datetime.now().isoformat(timespec="seconds"),
               "protocolo": "protocolo_consolidacion.md v1.0 (enmienda 1.1)"}, open(out / "P4_decision_C3.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"DECISIÓN C3: {d['veredicto']} · rangos {d['rango_medio_global']} · {time.time()-t0:.0f} s")


def _init(prueba):
    global PRUEBA
    PRUEBA = prueba


if __name__ == "__main__":
    main()
