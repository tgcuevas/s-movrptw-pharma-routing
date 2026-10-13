import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

VEHICLE_CAPACITY = 200.0
CO2_ALPHA = 0.05
SERVICE_T = 180.0
ALPHA_TW = {1: 10_000, 2: 5_000, 3: 1_000}
TW_CLASS = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}

NAVY = "#0B2E5B"
PRIO_COLOR = {1: "#C0392B", 2: "#E67E22", 3: "#5D6D7E"}
PRIO_NAME = {1: "urgente (≤ 15 min)", 2: "preferente (≤ 30 min)", 3: "ordinaria (≤ 60 min)"}
ROUTE_COLORS = ["#2D6CB5", "#C0392B", "#2E8B57", "#8E44AD", "#E67E22", "#16A085",
                "#D35400", "#7F8C8D", "#1ABC9C", "#9B59B6", "#34495E", "#F39C12"]


def split_routes(order: np.ndarray, demands: np.ndarray, capacity: float) -> List[List[int]]:
    routes, cur, cur_dem = [], [], 0.0
    for idx in order:
        d = float(demands[idx])
        if cur_dem + d > capacity and cur:
            routes.append(cur)
            cur, cur_dem = [], 0.0
        cur.append(int(idx))
        cur_dem += d
    if cur:
        routes.append(cur)
    return routes


def eval_route(order, demands, D, service_t, priorities, capacity) -> Tuple[float, float, float, int]:
    routes = split_routes(np.asarray(order, dtype=int), demands, capacity)
    f1 = f2 = f3 = 0.0
    for route in routes:
        t, load, prev = 0.0, 0.0, 0
        for idx in route:
            node = idx + 1
            t += float(D[prev, node]) + float(service_t[node])
            load += float(demands[idx])
            prev = node
            f2 += ALPHA_TW[priorities[idx]] * max(0.0, t - TW_CLASS[priorities[idx]][1])
        t += float(D[prev, 0])
        f1 += t
        f3 += CO2_ALPHA * t * (load / capacity)
    return f1, f2, f3, max(0, len(routes) - 1)


class CVRPTW3(ElementwiseProblem):
    def __init__(self, D, demands, priorities, service_t, capacity):
        self.D, self.demands, self.priorities = D, demands, priorities
        self.service_t, self.capacity = service_t, capacity
        super().__init__(n_var=len(demands), n_obj=3, xl=0, xu=len(demands) - 1)

    def _evaluate(self, x, out, *a, **k):
        f1, f2, f3, _ = eval_route(x.astype(int), self.demands, self.D, self.service_t,
                                   self.priorities, self.capacity)
        out["F"] = [f1, f2, f3]


class PermutationSampling(Sampling):
    def _do(self, problem, n_samples, **kw):
        return np.array([np.random.permutation(problem.n_var) for _ in range(n_samples)], dtype=int)


class OrderCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kw):
        _, n_mat, n_var = X.shape
        Y = np.zeros_like(X)
        for i in range(n_mat):
            p1, p2 = X[0, i], X[1, i]
            for k, (a, b) in enumerate([(p1, p2), (p2, p1)]):
                c1, c2 = sorted(np.random.choice(n_var, 2, replace=False))
                child = np.full(n_var, -1, dtype=int)
                child[c1:c2] = a[c1:c2]
                pos = c2 % n_var
                for gene in np.roll(b, -c2):
                    if gene not in child:
                        child[pos] = gene
                        pos = (pos + 1) % n_var
                Y[k, i] = child
        return Y


class SwapMutation(Mutation):
    def _do(self, problem, X, **kw):
        for i in range(len(X)):
            a, b = np.random.choice(problem.n_var, 2, replace=False)
            X[i, a], X[i, b] = X[i, b], X[i, a]
        return X


def select_compromise(F: np.ndarray) -> int:
    Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    return int(np.argmin(np.linalg.norm(Fn, axis=1)))


def parse_args():
    p = argparse.ArgumentParser(description="Mapa ilustrativo de rutas sobre la red real de Sevilla")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run", default=None, help="run_id de s3 (por defecto: data/forecast/latest_run.txt)")
    p.add_argument("--n", type=int, default=25, help="farmacias (las N primeras del CSV de s1)")
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_inputs(data_dir: Path, run: str, n: int):
    graphml = data_dir / "network" / "sevilla_drive.graphml"
    csv = data_dir / "farmacias" / "farmacias_sevilla.csv"
    tm_path = data_dir / "farmacias" / "time_matrix.npy"
    for pth in (graphml, csv, tm_path):
        if not pth.exists():
            print(f"ERROR: falta {pth} — ejecuta desde la carpeta del pipeline o usa --data-dir")
            sys.exit(1)

    df = pd.read_csv(csv)
    farm = df[df["tipo"] == "farmacia"].reset_index(drop=True)
    dep = df[df["tipo"] == "deposito"].reset_index(drop=True)
    if len(farm) < n:
        print(f"ERROR: el CSV solo tiene {len(farm)} farmacias, no {n}")
        sys.exit(1)
    d1 = dep[dep["id"] == "D1"].iloc[0]

    full_tm = np.load(tm_path)
    n_dep = len(dep)
    if full_tm.shape[0] != n_dep + len(farm):
        print(f"AVISO: time_matrix.npy es {full_tm.shape} y el CSV tiene {n_dep}+{len(farm)} filas; "
              "asumo índices 0..{n_dep-1} depósitos y el resto farmacias en el orden del CSV")
    sel = [0] + list(range(n_dep, n_dep + n))
    D = full_tm[np.ix_(sel, sel)].astype(float)
    farm = farm.iloc[:n].reset_index(drop=True)

    fc_dir = data_dir / "forecast"
    if run is None:
        latest = fc_dir / "latest_run.txt"
        if not latest.exists():
            print("ERROR: no hay data/forecast/latest_run.txt; pasa --run <run_id de s3>")
            sys.exit(1)
        run = latest.read_text().strip()
    fc_csv = fc_dir / run / "demand_forecast.csv"
    if not fc_csv.exists():
        print(f"ERROR: falta {fc_csv}")
        sys.exit(1)
    fc = pd.read_csv(fc_csv).sort_values("pharmacy_id").reset_index(drop=True)
    if len(fc) < n:
        print(f"ERROR: el run {run} de s3 tiene {len(fc)} farmacias (< {n}); usa un run con N ≥ {n}")
        sys.exit(1)
    demands = fc["demand_q50"].values[:n].astype(float)
    priorities = fc["priority_class"].values[:n].astype(int).tolist()
    service_t = np.full(n + 1, SERVICE_T)
    service_t[0] = 0.0

    print(f"  Run s3 para demandas/prioridades: {run}")
    print(f"  N={n} · demanda total q50 = {demands.sum():.1f} kg · capacidad {VEHICLE_CAPACITY:.0f} kg")
    print(f"  Prioridades: urgente={priorities.count(1)} preferente={priorities.count(2)} ordinaria={priorities.count(3)}")
    print(f"  Tiempos reales D1→farmacias: {D[0,1:].min():.0f}–{D[0,1:].max():.0f} s (media {D[0,1:].mean():.0f} s)")
    return graphml, farm, d1, D, demands, priorities, service_t, run


def node_of(G, row, col_x="x_utm", col_y="y_utm"):
    if "node_osm" in row.index and not pd.isna(row["node_osm"]):
        nd = int(row["node_osm"])
        if nd in G.nodes:
            return nd
    return ox.distance.nearest_nodes(G, X=float(row[col_x]), Y=float(row[col_y]))


def draw_plan(ax, G, farm, d1, nodes, routes, priorities, title, subtitle, path_cache):
    ox.plot_graph(G, ax=ax, bgcolor="white", node_size=0, edge_color="#D3DAE3",
                  edge_linewidth=0.3, show=False, close=False)
    dep_node = nodes[0]
    for r_i, route in enumerate(routes):
        color = ROUTE_COLORS[r_i % len(ROUTE_COLORS)]
        seq = [dep_node] + [nodes[idx + 1] for idx in route] + [dep_node]
        for a, b in zip(seq[:-1], seq[1:]):
            key = (a, b)
            if key not in path_cache:
                try:
                    path_cache[key] = nx.shortest_path(G, a, b, weight="travel_time")
                except nx.NetworkXNoPath:
                    path_cache[key] = None
            path = path_cache[key]
            if path and len(path) > 1:
                xs = [G.nodes[u]["x"] for u in path]
                ys = [G.nodes[u]["y"] for u in path]
            else:
                xs = [G.nodes[a]["x"], G.nodes[b]["x"]]
                ys = [G.nodes[a]["y"], G.nodes[b]["y"]]
            ax.plot(xs, ys, color=color, lw=1.8, alpha=0.9, zorder=3, solid_capstyle="round")
        for k, idx in enumerate(route, start=1):
            ax.annotate(str(k), (farm.loc[idx, "x_utm"], farm.loc[idx, "y_utm"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=6, color=color, zorder=6)
    for p in (3, 2, 1):
        idxs = [i for i, pr in enumerate(priorities) if pr == p]
        ax.scatter(farm.loc[idxs, "x_utm"], farm.loc[idxs, "y_utm"], s=34, c=PRIO_COLOR[p],
                   edgecolors="white", linewidths=0.7, zorder=5)
    ax.scatter([d1["x_utm"]], [d1["y_utm"]], s=220, marker="*", c=NAVY, edgecolors="white",
               linewidths=0.8, zorder=7)
    pad = 900
    xs = np.r_[farm["x_utm"].values, d1["x_utm"]]
    ys = np.r_[farm["y_utm"].values, d1["y_utm"]]
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11.5, color=NAVY, loc="left", pad=18)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=8.5, color="#6B7280", va="bottom")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = data_dir / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Datos: {data_dir.resolve()}")

    graphml, farm, d1, D, demands, priorities, service_t, run = load_inputs(data_dir, args.run, args.n)

    np.random.seed(args.seed)
    problem = CVRPTW3(D, demands, priorities, service_t, VEHICLE_CAPACITY)
    algo = NSGA2(pop_size=args.pop, sampling=PermutationSampling(), crossover=OrderCrossover(),
                 mutation=SwapMutation(), eliminate_duplicates=True)
    t0 = time.time()
    res = minimize(problem, algo, termination=("n_gen", args.n_gen), seed=int(args.seed), verbose=False)
    F, X = np.asarray(res.F), np.asarray(res.X).astype(int)
    print(f"  NSGA-II: {len(F)} soluciones no dominadas en {time.time() - t0:.1f} s")

    i_t = int(np.argmin(F[:, 0]))
    i_c = int(np.argmin(F[:, 1]))
    i_k = select_compromise(F)
    picks = [("(a) Plan de menor tiempo de ruta", i_t),
             ("(b) Plan de menor tardanza clínica", i_c),
             ("(c) Plan de compromiso (distancia al punto utópico)", i_k)]

    print("Cargando red y calculando caminos mínimos sobre la red real...")
    G = ox.load_graphml(graphml)
    nodes = [node_of(G, d1)] + [node_of(G, farm.loc[i]) for i in range(len(farm))]
    path_cache: Dict = {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    record = {"instancia": {"n": args.n, "run_s3": run, "seed": args.seed, "pop": args.pop, "n_gen": args.n_gen,
                            "capacidad_kg": VEHICLE_CAPACITY, "servicio_s": SERVICE_T,
                            "matriz": "data/farmacias/time_matrix.npy (OSMnx, real)",
                            "nota": "Instancia ilustrativa; no comparable con la ejecución de referencia de s4"},
              "planes": []}
    for ax, (label, i) in zip(axes, picks):
        routes = split_routes(X[i], demands, VEHICLE_CAPACITY)
        f1, f2, f3, _ = eval_route(X[i], demands, D, service_t, priorities, VEHICLE_CAPACITY)
        sub = f"f₁ = {f1/3600:.2f} h  ·  f₂ = {f2/1000:,.0f} ×10³  ·  f₃ = {f3:.1f}  ·  {len(routes)} ruta(s)"
        draw_plan(ax, G, farm, d1, nodes, routes, priorities, label, sub, path_cache)
        record["planes"].append({"plan": label, "f1_s": f1, "f1_h": f1 / 3600, "f2": f2, "f3": f3,
                                 "n_rutas": len(routes),
                                 "rutas": [[farm.loc[idx, "id"] for idx in r] for r in routes],
                                 "permutacion": X[i].tolist()})
        print(f"  {label}: {sub}")

    handles = [Line2D([0], [0], marker="*", ls="", color=NAVY, markersize=13, label="Depósito (Polígono El Pino)")]
    handles += [Line2D([0], [0], marker="o", ls="", color=PRIO_COLOR[p], markersize=7, label=PRIO_NAME[p]) for p in (1, 2, 3)]
    handles += [Line2D([0], [0], color="#555555", lw=2, label="rutas (caminos mínimos por tiempo, OSMnx)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Tres planes del frente de Pareto sobre la red real de Sevilla", fontsize=13, color=NAVY,
                 x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.945, f"Instancia ilustrativa N = {args.n}: matriz de tiempos OSMnx, demanda q50 y prioridades "
             "de la ejecución de referencia de s3 · mismo evaluador que s4 · "
             "no es la ejecución de referencia de la memoria", fontsize=9, color="#6B7280", ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))

    png = out_dir / "s8_mapa_rutas_sevilla.png"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "s8_mapa_rutas_sevilla.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(F, columns=["f1_s", "f2_pond", "f3_co2"]).assign(
        menor_tiempo=lambda d: d.index == i_t, menor_tardanza=lambda d: d.index == i_c,
        compromiso=lambda d: d.index == i_k).to_csv(out_dir / "s8_frente_ilustrativo.csv", index=False)
    with open(out_dir / "s8_rutas_ilustrativas.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Figura guardada: {png}")
    print(f"Trazabilidad:    {out_dir / 's8_rutas_ilustrativas.json'} · {out_dir / 's8_frente_ilustrativo.csv'}")


if __name__ == "__main__":
    main()
