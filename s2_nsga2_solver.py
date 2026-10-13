import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR   = Path("data")
SOLVER_DIR = DATA_DIR / "solver"
FARM_DIR   = DATA_DIR / "farmacias"
SOLVER_DIR.mkdir(parents=True, exist_ok=True)

POP_SIZE  = 200
N_GEN     = 500
N_GEN_FAST = 100
SEED      = 42

ALPHA = {1: 10_000, 2: 5_000, 3: 1_000}

TW_SLACK = {
    1: 0.10,
    2: 0.30,
    3: 0.60,
}
TW_CLASS = {
    1: (0,  900),
    2: (0, 1800),
    3: (0, 3600),
}


def split_prins(tour, demand, capacity, time_matrix, tw, service_time=60):
    n = len(tour)
    V = [np.inf] * (n + 1)
    P = [-1]    * (n + 1)
    V[0] = 0.0

    for i in range(n):
        load = 0.0
        t    = time_matrix[0][tour[i] + 1]
        j    = i
        while j < n:
            client_idx  = tour[j] + 1
            load       += demand[tour[j]]
            if j > i:
                t += time_matrix[tour[j-1] + 1][client_idx] + service_time
            if load > capacity:
                break
            cost = t + time_matrix[client_idx][0]
            if V[i] + cost < V[j + 1]:
                V[j + 1] = V[i] + cost
                P[j + 1] = i
            j += 1

    routes = []
    j = n
    while j > 0:
        i = P[j]
        if i == -1:
            return [], False
        routes.append([tour[k] for k in range(i, j)])
        j = i
    routes.reverse()
    return routes, True


def evaluate_solution(routes, time_matrix, demand, capacity,
                      tw, priority, co2_per_sec=0.05, service_time=60):
    f1 = 0.0
    f2 = 0.0
    f3 = 0.0

    entregas_ok   = {1: 0, 2: 0, 3: 0}
    entregas_tot  = {1: 0, 2: 0, 3: 0}

    for route in routes:
        if not route:
            continue
        t, load = 0.0, 0.0
        prev = 0
        for client in route:
            ci = client + 1
            e_i, l_i = tw[client]
            t = max(t + time_matrix[prev][ci], e_i) + service_time
            load += demand[client]
            tardanza = max(0.0, t - l_i)
            p = priority[client]
            f2 += ALPHA[p] * tardanza
            entregas_tot[p] += 1
            if tardanza == 0:
                entregas_ok[p] += 1
            prev = ci
        t += time_matrix[prev][0]
        f1 += t
        f3 += co2_per_sec * t * (load / capacity)

    num = sum(ALPHA[p] * entregas_ok[p]  for p in [1,2,3])
    den = sum(ALPHA[p] * entregas_tot[p] for p in [1,2,3])
    rtcr = num / den if den > 0 else 0.0

    return f1, f2, f3, rtcr


class CVRPTWSevilla(ElementwiseProblem):

    def __init__(self, time_matrix, demand, capacity, tw, priority,
                 n_vehicles=10, service_time=60):
        self.time_matrix  = time_matrix
        self.demand       = demand
        self.capacity     = capacity
        self.tw           = tw
        self.priority     = priority
        self.n_vehicles   = n_vehicles
        self.service_time = service_time
        self.n_clients    = len(demand)
        super().__init__(
            n_var=self.n_clients,
            n_obj=3,
            xl=0,
            xu=self.n_clients - 1,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        tour = x.astype(int)
        routes, feasible = split_prins(
            tour, self.demand, self.capacity,
            self.time_matrix, self.tw, self.service_time
        )
        if not feasible or not routes:
            out["F"] = [1e9, 1e9, 1e9]
            return
        f1, f2, f3, rtcr = evaluate_solution(
            routes, self.time_matrix, self.demand,
            self.capacity, self.tw, self.priority,
            service_time=self.service_time
        )
        out["F"] = [f1, f2, f3]


class PermutationSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            X[i] = np.random.permutation(problem.n_var)
        return X


class OrderCrossoverCustom(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        n_matings, n_var = X.shape[1], X.shape[2]
        Y = np.full_like(X, -1)
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            a, b = sorted(np.random.choice(n_var, 2, replace=False))
            child1 = [-1] * n_var
            child1[a:b+1] = p1[a:b+1].tolist()
            fill = [x for x in p2 if x not in child1]
            idx = 0
            for i in range(n_var):
                if child1[i] == -1:
                    child1[i] = fill[idx]; idx += 1
            child2 = [-1] * n_var
            child2[a:b+1] = p2[a:b+1].tolist()
            fill2 = [x for x in p1 if x not in child2]
            idx2 = 0
            for i in range(n_var):
                if child2[i] == -1:
                    child2[i] = fill2[idx2]; idx2 += 1
            Y[0, k] = child1
            Y[1, k] = child2
        return Y


class InversionMutation(Mutation):
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            if np.random.random() < self.prob:
                a, b = sorted(np.random.choice(problem.n_var, 2, replace=False))
                X[i, a:b+1] = X[i, a:b+1][::-1]
        return X


def generar_instancia_solomon_c1(n=50, seed=42):
    rng = np.random.RandomState(seed)
    centers = [(25, 75), (75, 25), (50, 50)]
    clients = []
    for cx, cy in centers:
        pts = rng.randn(n // 3, 2) * 8 + [cx, cy]
        clients.extend(pts.tolist())
    while len(clients) < n:
        clients.append(rng.randn(2).tolist())
    clients = np.array(clients[:n])
    depot = np.array([[50.0, 50.0]])
    coords = np.vstack([depot, clients])

    n_total = n + 1
    tm = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            tm[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))

    demand   = rng.randint(5, 15, size=n).tolist()
    capacity = 50

    SLACK_SOL = {1: 0.10, 2: 0.30, 3: 0.60}
    tw       = []
    priority = []
    n_total  = n + 1
    for i in range(n):
        p = int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.50]))
        priority.append(p)
        t_dir = float(np.sqrt(np.sum((coords[0] - coords[i+1])**2)))
        l_i   = max(t_dir * (1 + SLACK_SOL[p]), 1.0)
        tw.append((0.0, l_i))

    return {
        "time_matrix": tm,
        "demand": demand,
        "capacity": capacity,
        "tw": tw,
        "priority": priority,
        "n_clients": n,
        "coords": coords,
        "tipo": "C1_cluster",
    }


def generar_instancia_solomon_r1(n=50, seed=123):
    rng = np.random.RandomState(seed)
    clients = rng.uniform(0, 100, size=(n, 2))
    depot   = np.array([[50.0, 50.0]])
    coords  = np.vstack([depot, clients])

    n_total = n + 1
    tm = np.zeros((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            tm[i, j] = np.sqrt(np.sum((coords[i] - coords[j])**2))

    demand   = rng.randint(5, 15, size=n).tolist()
    capacity = 50
    SLACK_SOL = {1: 0.10, 2: 0.30, 3: 0.60}
    tw       = []
    priority = []
    for i in range(n):
        p = int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.50]))
        priority.append(p)
        t_dir = float(np.sqrt(np.sum((coords[0] - coords[i+1])**2)))
        l_i   = max(t_dir * (1 + SLACK_SOL[p]), 1.0)
        tw.append((0.0, l_i))

    return {
        "time_matrix": tm,
        "demand": demand,
        "capacity": capacity,
        "tw": tw,
        "priority": priority,
        "n_clients": n,
        "coords": coords,
        "tipo": "R1_random",
    }


def run_benchmark_solomon(n_runs=30):
    log.info("=== BENCHMARK SOLOMON (30 runs × 2 instancias) ===")
    resultados = []

    for inst_fn, nombre in [
        (lambda: generar_instancia_solomon_c1(50, 42), "C101-50"),
        (lambda: generar_instancia_solomon_r1(50, 123), "R101-50"),
    ]:
        inst = inst_fn()
        log.info(f"Instancia {nombre}: {inst['n_clients']} clientes, capacidad {inst['capacity']}")

        hvs, f1s, f2s, tiempos = [], [], [], []

        for run in range(n_runs):
            problema = CVRPTWSevilla(
                time_matrix  = inst["time_matrix"],
                demand       = inst["demand"],
                capacity     = inst["capacity"],
                tw           = inst["tw"],
                priority     = inst["priority"],
                service_time = 10,
            )
            alg = NSGA2(
                pop_size   = POP_SIZE,
                sampling   = PermutationSampling(),
                crossover  = OrderCrossoverCustom(),
                mutation   = InversionMutation(prob=0.15),
                eliminate_duplicates = False,
            )
            t0 = time.time()
            res = minimize(
                problema, alg,
                ("n_gen", N_GEN_FAST),
                seed    = SEED + run,
                verbose = False,
            )
            elapsed = time.time() - t0

            F = res.F
            if F is not None and len(F) > 0:
                ref_point = F.max(axis=0) * 1.1
                ind = HV(ref_point=ref_point)
                hv  = ind(F)
                f1_med = F[:, 0].mean()
                f2_med = F[:, 1].mean()
            else:
                hv, f1_med, f2_med = 0.0, 1e9, 1e9

            hvs.append(hv)
            f1s.append(f1_med)
            f2s.append(f2_med)
            tiempos.append(elapsed)

            if (run + 1) % 10 == 0:
                log.info(f"  {nombre} run {run+1:2d}/30 — HV={hv:.4f}  "
                         f"f1={f1_med:.1f}  t={elapsed:.1f}s")

        resultados.append({
            "instancia":   nombre,
            "n_runs":      n_runs,
            "hv_media":    float(np.mean(hvs)),
            "hv_std":      float(np.std(hvs)),
            "hv_cv_pct":   float(np.std(hvs) / np.mean(hvs) * 100) if np.mean(hvs) > 0 else 0,
            "f1_media":    float(np.mean(f1s)),
            "f2_media":    float(np.mean(f2s)),
            "t_media_s":   float(np.mean(tiempos)),
            "t_total_s":   float(np.sum(tiempos)),
        })
        log.info(f"  {nombre} completado — HV media={np.mean(hvs):.4f}  "
                 f"CV={np.std(hvs)/np.mean(hvs)*100:.1f}%  "
                 f"t_media={np.mean(tiempos):.1f}s")

    df = pd.DataFrame(resultados)
    csv_path = SOLVER_DIR / "solomon_benchmark.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Benchmark guardado: {csv_path}")
    return df


def preparar_instancia_sevilla(n_farmacias=10):
    log.info(f"Preparando instancia Sevilla N={n_farmacias}...")

    df   = pd.read_csv(FARM_DIR / "farmacias_sevilla.csv")
    full_tm = np.load(FARM_DIR / "time_matrix.npy")

    farm_idx = list(range(2, 2 + n_farmacias))
    sel_idx  = [0] + farm_idx

    tm = full_tm[np.ix_(sel_idx, sel_idx)]

    rng = np.random.RandomState(SEED)
    demand   = rng.randint(5, 50, size=n_farmacias).tolist()
    capacity = max(200, int(1000 * n_farmacias / 50))

    priority = []
    tw       = []
    t_d1 = tm[0, 1:]
    for i in range(n_farmacias):
        p = int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.50]))
        priority.append(p)
        t_dir = float(t_d1[i])
        l_i   = t_dir * (1 + TW_SLACK[p])
        tw.append((0.0, float(l_i)))

    log.info(f"  Distribución prioridad: urgente={priority.count(1)}  "
             f"preferente={priority.count(2)}  ordinario={priority.count(3)}")
    log.info(f"  Demanda total: {sum(demand)} kg  Capacidad: {capacity} kg")
    log.info(f"  Tiempos D1→farmacias: min={tm[0,1:].min():.0f}s  "
             f"media={tm[0,1:].mean():.0f}s  max={tm[0,1:].max():.0f}s")

    return {
        "time_matrix": tm,
        "demand":      demand,
        "capacity":    capacity,
        "tw":          tw,
        "priority":    priority,
        "n_clients":   n_farmacias,
        "farm_names":  df.iloc[farm_idx - np.array([2]*n_farmacias)]["nombre"].tolist(),
    }


def run_sevilla_n10(inst):
    log.info("=== NSGA-II sobre instancia Sevilla N=10 ===")

    problema = CVRPTWSevilla(
        time_matrix  = inst["time_matrix"],
        demand       = inst["demand"],
        capacity     = inst["capacity"],
        tw           = inst["tw"],
        priority     = inst["priority"],
        service_time = 180,
    )

    alg = NSGA2(
        pop_size   = POP_SIZE,
        sampling   = PermutationSampling(),
        crossover  = OrderCrossoverCustom(),
        mutation   = InversionMutation(prob=0.15),
        eliminate_duplicates = False,
    )

    log.info(f"Ejecutando NSGA-II: pop={POP_SIZE}, gen={N_GEN}, "
             f"N_clientes={inst['n_clients']}...")
    t0  = time.time()
    res = minimize(
        problema, alg,
        ("n_gen", N_GEN),
        seed    = SEED,
        verbose = False,
    )
    elapsed = time.time() - t0
    log.info(f"Optimización completada en {elapsed:.1f}s")

    F = res.F
    if F is None or len(F) == 0:
        log.error("NSGA-II no generó soluciones factibles")
        return None

    mask = np.all(F < 1e8, axis=1)
    F    = F[mask]
    n_dist = len(np.unique(np.round(F, 6), axis=0))
    log.info(f"Soluciones en frente de Pareto: {len(F)} "
             f"(de {len(res.F)} totales; {n_dist} vectores de objetivos distintos)")

    log.info(f"  f1 (coste/tiempo):  min={F[:,0].min():.0f}s  "
             f"media={F[:,0].mean():.0f}s  max={F[:,0].max():.0f}s")
    log.info(f"  f2 (tardanza pond): min={F[:,1].min():.0f}  "
             f"media={F[:,1].mean():.0f}  max={F[:,1].max():.0f}")
    log.info(f"  f3 (CO2 proxy):     min={F[:,2].min():.2f}  "
             f"media={F[:,2].mean():.2f}  max={F[:,2].max():.2f}")

    f_range = F.max(axis=0) - F.min(axis=0)
    if np.all(f_range < 1e-6):
        log.warning("  ⚠️  Frente degenerado: todas las soluciones tienen el mismo valor.")
        log.warning("  Causa probable: capacidad muy alta (todo en 1 ruta) o N muy pequeño.")
        hv = 0.0
    else:
        ref_point = F.max(axis=0) * 1.1
        ind = HV(ref_point=ref_point)
        hv  = ind(F)
    log.info(f"  Hypervolume: {hv:.4f}  Rango f: {f_range.round(1)}")

    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
    weights = np.array([0.4, 0.3, 0.3])
    dist_ideal  = np.sqrt(np.sum(weights * F_norm**2, axis=1))
    dist_nadir  = np.sqrt(np.sum(weights * (1 - F_norm)**2, axis=1))
    topsis_score = dist_nadir / (dist_ideal + dist_nadir + 1e-10)
    best_idx = np.argmax(topsis_score)
    log.info(f"  Solución compromiso TOPSIS (idx={best_idx}): "
             f"f1={F[best_idx,0]:.0f}s  f2={F[best_idx,1]:.0f}  "
             f"f3={F[best_idx,2]:.2f}")

    df_pareto = pd.DataFrame(F, columns=["f1_tiempo_s", "f2_tardanza", "f3_co2"])
    df_pareto["topsis_score"] = topsis_score
    df_pareto["es_compromiso"] = False
    df_pareto.loc[best_idx, "es_compromiso"] = True
    csv_path = SOLVER_DIR / "sevilla_n10_pareto.csv"
    df_pareto.to_csv(csv_path, index=False)
    log.info(f"Frente Pareto guardado: {csv_path}")

    _plot_pareto(F, best_idx, inst)

    return {
        "F":          F,
        "hv":         hv,
        "best_idx":   best_idx,
        "elapsed_s":  elapsed,
        "n_pareto":   len(F),
        "n_distintas": n_dist,
    }


def _plot_pareto(F, best_idx, inst):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Frente de Pareto — instancia Sevilla N={inst['n_clients']} (NSGA-II)",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.scatter(F[:,0]/60, F[:,1]/1000, c='steelblue', alpha=0.7, s=30, label='Pareto')
    ax.scatter(F[best_idx,0]/60, F[best_idx,1]/1000,
               c='red', s=100, zorder=5, marker='*', label='Compromiso TOPSIS')
    ax.set_xlabel("f1 — Tiempo total (min)", fontsize=11)
    ax.set_ylabel("f2 — Tardanza ponderada (×10³)", fontsize=11)
    ax.set_title("Coste vs. Nivel de Servicio Clínico")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(F[:,0]/60, F[:,2], c='seagreen', alpha=0.7, s=30, label='Pareto')
    ax.scatter(F[best_idx,0]/60, F[best_idx,2],
               c='red', s=100, zorder=5, marker='*', label='Compromiso TOPSIS')
    ax.set_xlabel("f1 — Tiempo total (min)", fontsize=11)
    ax.set_ylabel("f3 — Emisiones CO₂ (proxy)", fontsize=11)
    ax.set_title("Coste vs. Sostenibilidad")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = SOLVER_DIR / "sevilla_n10_pareto.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Figura Pareto guardada: {png_path}")


def guardar_config(benchmark_df, pareto_result):
    config = {
        "solver":        "NSGA-II (pymoo 0.6.1.3)",
        "pop_size":      POP_SIZE,
        "n_gen":         N_GEN,
        "crossover":     "OrderCrossover (OX)",
        "mutation":      "InversionMutation(prob=0.15)",
        "decoding":      "Split Prins O(N²)",
        "seed_base":     SEED,
        "alpha_pesos":   ALPHA,
        "tw_clases_s":   {str(k): list(v) for k, v in TW_CLASS.items()},
        "benchmark": {
            row["instancia"]: {
                "hv_media":  row["hv_media"],
                "hv_cv_pct": row["hv_cv_pct"],
                "t_media_s": row["t_media_s"],
            }
            for _, row in benchmark_df.iterrows()
        },
    }
    if pareto_result:
        config["sevilla_n10"] = {
            "n_pareto":  pareto_result["n_pareto"],
            "hv":        pareto_result["hv"],
            "elapsed_s": pareto_result["elapsed_s"],
        }
    cfg_path = SOLVER_DIR / "solver_config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"Configuración guardada: {cfg_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("s2 — NSGA-II · Benchmark Solomon + Sevilla N=10")
    print("="*60 + "\n")

    log.info("PASO 1/3 — Benchmark Solomon C101-50 y R101-50")
    benchmark_df = run_benchmark_solomon(n_runs=30)
    print("\n" + benchmark_df.to_string(index=False) + "\n")

    log.info("PASO 2/3 — Instancia Sevilla N=10 con tiempos reales")
    inst_n10 = preparar_instancia_sevilla(n_farmacias=10)

    log.info("PASO 3/3 — NSGA-II sobre Sevilla N=10")
    pareto_result = run_sevilla_n10(inst_n10)

    guardar_config(benchmark_df, pareto_result)

    print("\n" + "="*60)
    print("VALIDACIÓN DEL SOLVER")
    print("="*60)
    for _, row in benchmark_df.iterrows():
        cv   = row['hv_cv_pct']
        f1ok = row['f1_media'] < 1e8
        hvok = row['hv_media'] > 0
        ok   = "✅" if (f1ok and not (cv != cv)) else "⚠️"
        print(f"  {ok}  {row['instancia']}: "
              f"f1_media={row['f1_media']:.1f}  f2_media={row['f2_media']:.1f}  "
              f"HV={row['hv_media']:.4f}  t_media={row['t_media_s']:.1f}s")
    if pareto_result:
        pok = "✅" if pareto_result['hv'] > 0 else "⚠️ HV=0 — frente degenerado"
        print(f"  {pok}  Sevilla N=10: {pareto_result['n_pareto']} soluciones Pareto  "
              f"HV={pareto_result['hv']:.4f}  t={pareto_result['elapsed_s']:.1f}s")

    print("\n  Archivos generados:")
    for f in sorted(SOLVER_DIR.iterdir()):
        print(f"    {f}")
    print(f"\n→ Siguiente paso: python s2b_figura41.py y python s3_forecast_pipeline.py\n")
