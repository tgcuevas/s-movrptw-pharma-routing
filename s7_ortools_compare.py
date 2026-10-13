#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED            = 42
VEHICLE_CAPACITY = 200.0
CO2_ALPHA       = 0.05
PRIO_PROBS      = [0.30, 0.20, 0.50]
ALPHA_TW        = {1: 10_000, 2: 5_000, 3: 1_000}
TW_CLASS        = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}
SERVICE_T       = 180.0


def build_instance(n: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    size = n + 1

    raw = np.exp(rng.normal(6.3, 0.35, size=(size, size)))
    raw = (raw + raw.T) / 2.0
    np.fill_diagonal(raw, 0.0)
    raw[0, 1:] += 300.0
    raw[1:, 0] += 300.0
    D = raw

    service_t = np.full(size, SERVICE_T)
    service_t[0] = 0.0

    priorities = rng.choice([1, 2, 3], size=n, p=PRIO_PROBS).tolist()

    demands = rng.uniform(3.0, 18.0, size=n)

    log.info(f"  Instancia N={n}: D media={D[D>0].mean():.0f}s · "
             f"demanda total={demands.sum():.1f} kg · "
             f"capacidad={VEHICLE_CAPACITY:.0f} kg")
    return D, service_t, priorities, demands


def evaluate_routes(routes, D, service_t, priorities, demands,
                    capacity=VEHICLE_CAPACITY):
    depot = 0
    f1 = f2 = f3 = 0.0
    for route in routes:
        if not route:
            continue
        t, load, prev = 0.0, 0.0, depot
        for idx in route:
            node = int(idx) + 1
            t += float(D[prev, node]) + float(service_t[node])
            load += float(demands[idx])
            prev = node
            tw_end = TW_CLASS[priorities[idx]][1]
            f2 += ALPHA_TW[priorities[idx]] * max(0.0, t - tw_end)
        t += float(D[prev, depot])
        f1 += t
        f3 += CO2_ALPHA * t * (load / capacity)
    return f1, f2, f3, len([r for r in routes if r])


def routes_from_permutation(order, demands, capacity=VEHICLE_CAPACITY):
    routes, cur, cur_dem = [], [], 0.0
    for idx in order:
        d_i = float(demands[idx])
        if cur_dem + d_i > capacity and cur:
            routes.append(cur); cur, cur_dem = [], 0.0
        cur.append(int(idx)); cur_dem += d_i
    if cur:
        routes.append(cur)
    return routes


def load_real_instance(data_dir, n=25, seed=SEED):
    import csv as _csv
    from pathlib import Path as _P
    data_dir = _P(data_dir)

    fdir = None
    for d in sorted((data_dir / "forecast").glob("2*")):
        cfg = d / "forecast_config.json"
        if cfg.exists():
            import json as _j
            if _j.loads(cfg.read_text()).get("n_pharmacies") == n:
                fdir = d
    if fdir is None:
        raise SystemExit(f"No se encontró un run de forecast con n_pharmacies={n} en {data_dir}/forecast")

    rows = list(_csv.DictReader(open(fdir / "demand_forecast.csv")))
    rows = sorted(rows, key=lambda r: int(r["pharmacy_id"]))[:n]
    demands = np.array([float(r["demand_q50"]) for r in rows])
    priorities = [int(float(r["priority_class"])) for r in rows]

    D = build_instance(n, seed)[0]
    service_t = np.full(n + 1, SERVICE_T); service_t[0] = 0.0

    s4dir = None
    for d in sorted((data_dir / "integration").glob("2*_s4")):
        import json as _j
        summ = d / "s4_summary.json"
        if summ.exists():
            sj = _j.loads(summ.read_text())
            if sj.get("n_pharmacies") == n and sj.get("run_id_s3") == fdir.name:
                s4dir = d
    front, compromise = None, None
    if s4dir and (s4dir / "s4_ev_pareto.csv").exists():
        pr = list(_csv.DictReader(open(s4dir / "s4_ev_pareto.csv")))
        front = np.array([[float(r["f1_s"]), float(r["f2_pond"]), float(r["f3_co2"])] for r in pr])
        comp_rows = [r for r in pr if str(r.get("is_compromise")).lower() == "true"]
        if comp_rows:
            c = comp_rows[0]
            compromise = {"f1": float(c["f1_s"]), "f2": float(c["f2_pond"]), "f3": float(c["f3_co2"])}
    log.info(f"  Instancia REAL N={n} cargada de {fdir.name} · demanda total={demands.sum():.1f} kg · "
             f"frente NSGA-II EV: {0 if front is None else len(front)} puntos")
    return D, service_t, priorities, demands, front, compromise


def solve_nsga2(D, service_t, priorities, demands, n,
                pop_size=100, n_gen=200, seed=SEED):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import PermutationRandomSampling
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation
    from pymoo.optimize import minimize

    class CVRPTW(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=3, xl=0, xu=n - 1, vtype=int)

        def _evaluate(self, x, out, *a, **k):
            order = np.argsort(x)
            routes = routes_from_permutation(order, demands)
            f1, f2, f3, _ = evaluate_routes(routes, D, service_t, priorities, demands)
            out["F"] = [f1, f2, f3]

    algo = NSGA2(pop_size=pop_size,
                 sampling=PermutationRandomSampling(),
                 crossover=OrderCrossover(),
                 mutation=InversionMutation(),
                 eliminate_duplicates=True)
    t0 = time.time()
    res = minimize(CVRPTW(), algo, ("n_gen", n_gen), seed=seed, verbose=False)
    cpu = time.time() - t0

    F = np.atleast_2d(res.F)
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    Fn = (F - ideal) / span
    comp_idx = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    X = np.atleast_2d(res.X)
    order = np.argsort(X[comp_idx])
    routes = routes_from_permutation(order, demands)
    f1, f2, f3, nr = evaluate_routes(routes, D, service_t, priorities, demands)
    return {"f1": f1, "f2": f2, "f3": f3, "n_routes": nr,
            "pareto_size": int(F.shape[0]), "cpu_s": cpu, "routes": routes,
            "front": F.tolist()}


def solve_ortools(D, service_t, priorities, demands, n,
                  capacity=VEHICLE_CAPACITY, time_limit=15):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    size = n + 1
    n_vehicles = n
    depot = 0

    Dint = np.rint(D).astype(int)
    serv = np.rint(service_t).astype(int)
    dem_int = np.rint(demands).astype(int)

    mgr = pywrapcp.RoutingIndexManager(size, n_vehicles, depot)
    routing = pywrapcp.RoutingModel(mgr)

    def time_cb(i, j):
        a = mgr.IndexToNode(i); b = mgr.IndexToNode(j)
        return int(Dint[a, b] + serv[b])
    tcb = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(tcb)

    def dem_cb(i):
        node = mgr.IndexToNode(i)
        return 0 if node == depot else int(dem_int[node - 1])
    dcb = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(
        dcb, 0, [int(capacity)] * n_vehicles, True, "Capacity")

    horizon = int(Dint.max() * size + serv.sum() + 10_000)
    routing.AddDimension(tcb, 0, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for node in range(1, size):
        idx = mgr.NodeToIndex(node)
        prio = priorities[node - 1]
        tw_end = TW_CLASS[prio][1]
        time_dim.SetCumulVarSoftUpperBound(idx, tw_end, ALPHA_TW[prio])

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(int(time_limit))

    t0 = time.time()
    sol = routing.SolveWithParameters(params)
    cpu = time.time() - t0
    if sol is None:
        raise RuntimeError("OR-Tools no encontró solución.")

    routes = []
    for v in range(n_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = mgr.IndexToNode(idx)
            if node != depot:
                route.append(node - 1)
            idx = sol.Value(routing.NextVar(idx))
        if route:
            routes.append(route)

    f1, f2, f3, nr = evaluate_routes(routes, D, service_t, priorities, demands)
    return {"f1": f1, "f2": f2, "f3": f3, "n_routes": nr,
            "cpu_s": cpu, "routes": routes}


def dominance_analysis(ort, front):
    p = np.array([ort["f1"], ort["f2"], ort["f3"]], dtype=float)
    F = np.array(front, dtype=float)
    dom_by = np.sum(np.all(F <= p, axis=1) & np.any(F < p, axis=1))
    dominates = np.sum(np.all(p <= F, axis=1) & np.any(p < F, axis=1))
    if dom_by > 0:
        verdict = (f"La solución OR-Tools está DOMINADA por {int(dom_by)} de "
                   f"{len(F)} soluciones del frente NSGA-II.")
    elif dominates > 0:
        verdict = (f"La solución OR-Tools DOMINA a {int(dominates)} soluciones "
                   f"del frente, y es no-dominada por el resto.")
    else:
        verdict = ("La solución OR-Tools es NO-DOMINADA respecto al frente "
                   "NSGA-II (ocupa un punto de compromiso distinto del espacio "
                   "de objetivos: prioriza f2/f3 sobre f1).")
    return {"dominated_by": int(dom_by), "dominates": int(dominates),
            "verdict": verdict}


def make_figure(ort, nsga, out_png):
    objs = [
        ("f\u2081 \u2014 Tiempo (h)", ort["f1"] / 3600, nsga["f1"] / 3600),
        ("f\u2082 \u2014 Tardanza", ort["f2"], nsga["f2"]),
        ("f\u2083 \u2014 CO\u2082 (kg)", ort["f3"], nsga["f3"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    colors = ["#E07B39", "#2E6DA4"]
    for ax, (title, ov, nv) in zip(axes, objs):
        bars = ax.bar(["OR-Tools", "NSGA-II"], [ov, nv], color=colors, width=0.6)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():,.1f}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.18)
    fig.suptitle("OR-Tools vs. NSGA-II (compromiso) \u2014 instancia Sevilla N=25",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--time-limit", type=int, default=15, help="límite OR-Tools (s)")
    ap.add_argument("--n-gen", type=int, default=200, help="generaciones NSGA-II")
    ap.add_argument("--from-data", default=None,
                    help="ruta a la carpeta data/ del pipeline: usa la instancia real "
                         "y el frente NSGA-II EV real (recomendado para la memoria)")
    args = ap.parse_args()

    out_dir = Path("data/compare_ortools"); out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  COMPARACIÓN OR-Tools vs. NSGA-II — instancia N=%d", args.n)
    log.info("=" * 60)

    if args.from_data:
        D, service_t, priorities, demands, front, comp = load_real_instance(
            args.from_data, n=args.n, seed=args.seed)
        if comp is None:
            raise SystemExit("No se pudo cargar la solución compromiso real (s4_ev_pareto.csv).")
        nsga = {"f1": comp["f1"], "f2": comp["f2"], "f3": comp["f3"],
                "n_routes": int(np.ceil(demands.sum() / VEHICLE_CAPACITY)),
                "pareto_size": int(len(front)), "cpu_s": float("nan"),
                "front": front.tolist(), "source": "s4_ev_pareto.csv (real)"}
        log.info("  [1/2] Compromiso NSGA-II REAL (s4): f1=%.2f h · f2=%.0f · f3=%.1f kg · Pareto=%d",
                 nsga["f1"] / 3600, nsga["f2"], nsga["f3"], nsga["pareto_size"])
    else:
        D, service_t, priorities, demands = build_instance(args.n, args.seed)
        log.info("  [1/2] Resolviendo con NSGA-II (pymoo)...")
        nsga = solve_nsga2(D, service_t, priorities, demands, args.n, n_gen=args.n_gen,
                           seed=args.seed)
        log.info("        compromiso: f1=%.2f h · f2=%.0f · f3=%.1f kg · rutas=%d · "
                 "Pareto=%d · %.1fs",
                 nsga["f1"] / 3600, nsga["f2"], nsga["f3"], nsga["n_routes"],
                 nsga["pareto_size"], nsga["cpu_s"])

    log.info("  [2/2] Resolviendo con OR-Tools (CVRPTW + TW blandas)...")
    ort = solve_ortools(D, service_t, priorities, demands, args.n,
                        time_limit=args.time_limit)
    log.info("        solución:  f1=%.2f h · f2=%.0f · f3=%.1f kg · rutas=%d · %.1fs",
             ort["f1"] / 3600, ort["f2"], ort["f3"], ort["n_routes"], ort["cpu_s"])

    dom = dominance_analysis(ort, nsga["front"])
    log.info("        dominancia: %s", dom["verdict"])

    def fmt(d):
        cpu = d["cpu_s"]
        cpu = None if (cpu is None or (isinstance(cpu, float) and np.isnan(cpu))) else round(cpu, 2)
        return {"f1_h": round(d["f1"] / 3600, 3), "f2": round(d["f2"], 1),
                "f3_kg": round(d["f3"], 2), "n_routes": d["n_routes"],
                "cpu_s": cpu}
    summary = {
        "instance": f"Sevilla_N{args.n}", "seed": args.seed,
        "ortools": fmt(ort), "nsga2_compromise": fmt(nsga),
        "nsga2_pareto_size": nsga["pareto_size"],
        "dominance": dom,
        "note": ("OR-Tools entrega 1 solución mono-objetivo agregada; NSGA-II "
                 "entrega un frente de Pareto y habilita el cálculo del VSS. "
                 "Comparación sobre instancia determinista idéntica (seed=%d)."
                 % args.seed),
    }
    (out_dir / "ortools_vs_nsga2.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    import csv
    with open(out_dir / "ortools_vs_nsga2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metodo", "f1_h", "f2", "f3_kg", "n_rutas", "cpu_s", "pareto"])
        w.writerow(["OR-Tools", *fmt(ort).values(), 1])
        w.writerow(["NSGA-II (compromiso)", *fmt(nsga).values(),
                    nsga["pareto_size"]])

    make_figure(ort, nsga, out_dir / "ortools_vs_nsga2.png")

    print("\n" + "=" * 64)
    print("  RESULTADO — OR-Tools vs. NSGA-II (instancia N=%d)" % args.n)
    print("=" * 64)
    print(f"  {'Métrica':<22}{'OR-Tools':>14}{'NSGA-II compr.':>18}")
    print(f"  {'f1 Tiempo (h)':<22}{ort['f1']/3600:>14.3f}{nsga['f1']/3600:>18.3f}")
    print(f"  {'f2 Tardanza':<22}{ort['f2']:>14.0f}{nsga['f2']:>18.0f}")
    print(f"  {'f3 CO2 (kg)':<22}{ort['f3']:>14.2f}{nsga['f3']:>18.2f}")
    print(f"  {'Nº rutas':<22}{ort['n_routes']:>14d}{nsga['n_routes']:>18d}")
    print(f"  {'Soluciones':<22}{1:>14d}{nsga['pareto_size']:>18d}")
    print("=" * 64)
    print(f"  Dominancia: {dom['verdict']}")
    print(f"  Guardado en: {out_dir}/")
    print("  Lectura: OR-Tools reduce la tardanza y las emisiones a costa de más tiempo de")
    print("  flota y entrega UNA solución; NSGA-II ofrece el frente de Pareto y habilita el VSS.")


if __name__ == "__main__":
    main()
