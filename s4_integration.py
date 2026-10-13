#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = Path("data")
INTEG_DIR   = BASE_DIR / "integration"
FORECAST_DIR = BASE_DIR / "forecast"
INTEG_DIR.mkdir(parents=True, exist_ok=True)

VEHICLE_CAPACITY = 200.0
PENALTY_EXTRA_TRIP = 1800.0
CO2_ALPHA = 0.05
N_GEN_DEFAULT  = 200
N_GEN_QUICK    = 80
POP_SIZE       = 100
SEED           = 42


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

_S2_COORDS: Optional[Dict] = None
try:
    sys.path.insert(0, str(Path(__file__).parent))
    import s2_nsga2_solver as _s2_mod
    _S2_COORDS = {"loaded": True}
    log.info("  s2_nsga2_solver detectado (sólo para referencia geográfica)")
except ImportError:
    log.info("  s2_nsga2_solver no encontrado — usando geometría calibrada Sevilla")


def load_forecast_data(run_id: Optional[str] = None) -> Tuple[pd.DataFrame,
                                                               pd.DataFrame,
                                                               Dict, str]:
    if run_id is None:
        latest = FORECAST_DIR / "latest_run.txt"
        if not latest.exists():
            raise FileNotFoundError(
                "No se encontró latest_run.txt. "
                "Ejecuta primero: python s3_forecast_pipeline.py"
            )
        run_id = latest.read_text().strip()

    run_dir = FORECAST_DIR / run_id

    for fname in ["demand_forecast.csv", "saa_scenarios.csv", "forecast_config.json"]:
        fpath = run_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {fpath}")

    df_fc  = pd.read_csv(run_dir / "demand_forecast.csv")
    df_saa = pd.read_csv(run_dir / "saa_scenarios.csv")
    with open(run_dir / "forecast_config.json") as f:
        config = json.load(f)

    n_farm = len(df_fc)
    k_scen = df_saa["scenario_id"].nunique()
    log.info(f"  Run s3: {run_id}")
    log.info(f"  N={n_farm} farmacias, K={k_scen} escenarios SAA")
    log.info(f"  Demanda Q50 total: {df_fc['demand_q50'].sum():.2f} kg")
    log.info(f"  Demanda SAA media: "
             f"{df_saa.groupby('scenario_id')['demand_kg'].sum().mean():.2f} kg")

    return df_fc, df_saa, config, run_id


class PermutationSampling(Sampling):
    def _do(self, problem, n_samples, **kw):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            X[i] = np.random.permutation(problem.n_var)
        return X


class OrderCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kw):
        _, n_matings, n_var = X.shape
        Y = np.zeros_like(X)
        for i in range(n_matings):
            p1, p2 = X[0, i], X[1, i]
            for k, (a, b) in enumerate([(p1, p2), (p2, p1)]):
                c1, c2 = sorted(np.random.choice(n_var, 2, replace=False))
                child   = np.full(n_var, -1, dtype=int)
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


_ALPHA_TW = {1: 10_000, 2: 5_000, 3: 1_000}
_TW_CLASS  = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}


class CVRPTW_3obj(ElementwiseProblem):
    def __init__(self, inst: Dict):
        self.n_farm    = inst["n"]
        self.D         = np.array(inst["D"], dtype=float)
        self.demands   = np.array(inst["demands"], dtype=float)
        self.priorities = inst["priorities"]
        self.service_t = np.array(inst["service_times"], dtype=float)
        self.capacity  = float(inst.get("capacity", VEHICLE_CAPACITY))
        super().__init__(n_var=self.n_farm, n_obj=3,
                         xl=0, xu=self.n_farm - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3, _ = _eval_route(
            x.astype(int), self.demands, self.D,
            self.service_t, self.priorities, self.capacity
        )
        out["F"] = [f1, f2, f3]


def _eval_route(
    order:      np.ndarray,
    demands:    np.ndarray,
    D:          np.ndarray,
    service_t:  np.ndarray,
    priorities: List[int],
    capacity:   float,
) -> Tuple[float, float, float, int]:
    depot   = 0
    routes  = []
    cur_rt  = []
    cur_dem = 0.0

    for idx in order:
        node = int(idx) + 1
        d_i  = float(demands[idx])
        if cur_dem + d_i > capacity and cur_rt:
            routes.append(cur_rt)
            cur_rt, cur_dem = [], 0.0
        cur_rt.append((int(idx), node, d_i))
        cur_dem += d_i
    if cur_rt:
        routes.append(cur_rt)

    f1 = f2 = f3 = 0.0
    n_extra = max(0, len(routes) - 1)

    for route in routes:
        t, load, prev = 0.0, 0.0, depot
        for (idx, node, d_i) in route:
            t    += float(D[prev, node]) + float(service_t[node])
            load += d_i
            prev  = node
            tw_end = _TW_CLASS[priorities[idx]][1]
            f2    += _ALPHA_TW[priorities[idx]] * max(0.0, t - tw_end)
        t   += float(D[prev, depot])
        f1  += t
        f3  += CO2_ALPHA * t * (load / capacity)

    return f1, f2, f3, n_extra


def get_distance_matrix(n: int, seed: int = SEED) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    size = n + 1

    raw = np.exp(rng.normal(6.3, 0.35, size=(size, size)))
    raw = (raw + raw.T) / 2.0
    np.fill_diagonal(raw, 0.0)

    raw[0, 1:] += 300.0
    raw[1:, 0] += 300.0

    log.info(f"  Matriz D calibrada Sevilla: {raw.shape}  "
             f"media={raw[raw>0].mean():.0f}s  "
             f"min={raw[raw>0].min():.0f}s  max={raw.max():.0f}s")
    return raw


def get_service_times(n: int) -> np.ndarray:
    st    = np.full(n + 1, 180.0)
    st[0] = 0.0
    return st


def evaluate_solution(
    route_order: List[int],
    demands:     np.ndarray,
    D:           np.ndarray,
    service_t:   np.ndarray,
    priorities:  List[int],
    capacity:    float = VEHICLE_CAPACITY,
    alpha:       Dict  = None,
    tw_class:    Dict  = None,
) -> Tuple[float, float, float, int]:
    if alpha is None:
        alpha = {1: 10_000, 2: 5_000, 3: 1_000}
    if tw_class is None:
        tw_class = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}

    n = len(route_order)
    depot = 0

    routes      = []
    current_rt  = []
    current_dem = 0.0

    for idx in route_order:
        farm_node = idx + 1
        dem_i     = float(demands[idx])

        if current_dem + dem_i > capacity and current_rt:
            routes.append(current_rt)
            current_rt  = []
            current_dem = 0.0

        current_rt.append((idx, farm_node, dem_i))
        current_dem += dem_i

    if current_rt:
        routes.append(current_rt)

    f1_total = 0.0
    f2_total = 0.0
    f3_total = 0.0
    n_extra  = max(0, len(routes) - 1)

    for route in routes:
        t_curr  = 0.0
        load    = 0.0
        prev    = depot

        for (idx, farm_node, dem_i) in route:
            travel = float(D[prev, farm_node])
            t_curr += travel + float(service_t[farm_node])
            load   += dem_i
            prev    = farm_node

            tw_end = tw_class[priorities[idx]][1]
            tardanza = max(0.0, t_curr - tw_end)
            f2_total += alpha[priorities[idx]] * tardanza

        t_curr  += float(D[prev, depot])
        f1_total += t_curr
        f3_total += CO2_ALPHA * t_curr * (load / capacity)

    return f1_total, f2_total, f3_total, n_extra


def run_nsga2_for_instance(
    demands:    np.ndarray,
    priorities: List[int],
    D:          np.ndarray,
    service_t:  np.ndarray,
    n_gen:      int  = N_GEN_DEFAULT,
    pop_size:   int  = POP_SIZE,
    seed:       int  = SEED,
    label:      str  = "instancia",
    return_X:   bool = False,
):
    n = len(demands)
    inst = {
        "label":         label,
        "n":             n,
        "D":             D.tolist(),
        "demands":       demands.tolist(),
        "priorities":    priorities,
        "service_times": service_t.tolist(),
        "capacity":      VEHICLE_CAPACITY,
    }

    problem   = CVRPTW_3obj(inst)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationSampling(),
        crossover=OrderCrossover(),
        mutation=SwapMutation(),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem, algorithm,
        termination=("n_gen", n_gen),
        seed=int(seed),
        verbose=False,
    )

    if result.F is not None and len(result.F) > 0:
        return (result.F, result.X) if return_X else result.F

    log.warning(f"  NSGA-II sin frente para {label}; usando solución greedy")
    f1, f2, f3, _ = _eval_route(
        np.arange(n), demands, D, service_t, priorities, VEHICLE_CAPACITY
    )
    F_greedy = np.array([[f1, f2, f3]])
    return (F_greedy, np.arange(n)[None, :]) if return_X else F_greedy


def select_compromise(F: np.ndarray) -> Tuple[np.ndarray, int]:
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    utopia = np.zeros(F.shape[1])
    dists  = np.linalg.norm(F_norm - utopia, axis=1)
    idx    = int(np.argmin(dists))
    return F[idx], idx


def evaluate_ev_on_scenarios(
    f_ev_compromise: np.ndarray,
    df_saa:          pd.DataFrame,
    demands_ev:      np.ndarray,
    capacity:        float = VEHICLE_CAPACITY,
    ev_perm:         Optional[np.ndarray] = None,
) -> pd.DataFrame:
    k_list = sorted(df_saa["scenario_id"].unique())
    n      = demands_ev.shape[0]

    load_factor_ev = demands_ev.sum() / max(capacity, 1.0)
    f1_ev, f2_ev, f3_ev = float(f_ev_compromise[0]), \
                           float(f_ev_compromise[1]), \
                           float(f_ev_compromise[2])

    ev_routes = None
    if ev_perm is not None:
        ev_routes, cur, cd = [], [], 0.0
        for i in np.asarray(ev_perm, dtype=int):
            if cd + demands_ev[i] > capacity and cur:
                ev_routes.append(cur); cur, cd = [], 0.0
            cur.append(int(i)); cd += float(demands_ev[i])
        ev_routes.append(cur)
    else:
        log.warning("  evaluate_ev_on_scenarios sin ev_perm: se usa la aproximación agregada (no recomendada)")

    records = []
    for k in k_list:
        df_k    = df_saa[df_saa["scenario_id"] == k].sort_values("pharmacy_id")
        dem_k   = df_k["demand_kg"].values[:n]

        load_factor_k = dem_k.sum() / capacity

        if ev_routes is not None:
            n_extra = int(sum(max(0, int(np.ceil(dem_k[r].sum() / capacity)) - 1)
                              for r in ev_routes))
        else:
            n_extra = max(0, int(np.ceil(dem_k.sum() / capacity)) - 1)

        f1_k = f1_ev + PENALTY_EXTRA_TRIP * n_extra
        f2_k = f2_ev
        f3_k = f3_ev * (load_factor_k / (load_factor_ev + 1e-9))

        records.append({
            "scenario_id":      k,
            "f1_ev_eval":       round(f1_k, 2),
            "f2_ev_eval":       round(f2_k, 2),
            "f3_ev_eval":       round(f3_k, 4),
            "demand_total_kg":  round(float(dem_k.sum()), 2),
            "load_factor":      round(float(load_factor_k), 4),
            "n_extra_trips":    n_extra,
        })

    return pd.DataFrame(records)


def compute_vss(
    df_eev:     pd.DataFrame,
    saa_fronts: Dict[int, np.ndarray],
) -> Dict:
    eev_f1 = float(df_eev["f1_ev_eval"].mean())
    eev_f2 = float(df_eev["f2_ev_eval"].mean())
    eev_f3 = float(df_eev["f3_ev_eval"].mean())

    rp_f1_list, rp_f2_list, rp_f3_list = [], [], []
    for k, F_k in saa_fronts.items():
        if F_k is not None and len(F_k) > 0:
            rp_f1_list.append(float(F_k[:, 0].min()))
            rp_f2_list.append(float(F_k[:, 1].min()))
            rp_f3_list.append(float(F_k[:, 2].min()))

    rp_f1 = float(np.mean(rp_f1_list)) if rp_f1_list else eev_f1
    rp_f2 = float(np.mean(rp_f2_list)) if rp_f2_list else eev_f2
    rp_f3 = float(np.mean(rp_f3_list)) if rp_f3_list else eev_f3

    vss_f1 = max(0.0, eev_f1 - rp_f1)
    vss_f2 = max(0.0, eev_f2 - rp_f2)
    vss_f3 = max(0.0, eev_f3 - rp_f3)

    vss_pct_f1 = vss_f1 / (eev_f1 + 1e-9) * 100
    vss_pct_f2 = vss_f2 / (eev_f2 + 1e-9) * 100
    vss_pct_f3 = vss_f3 / (eev_f3 + 1e-9) * 100

    return {
        "EEV": {"f1": round(eev_f1, 2), "f2": round(eev_f2, 2),
                "f3": round(eev_f3, 4)},
        "RP":  {"f1": round(rp_f1, 2),  "f2": round(rp_f2, 2),
                "f3": round(rp_f3, 4)},
        "VSS": {"f1": round(vss_f1, 2), "f2": round(vss_f2, 2),
                "f3": round(vss_f3, 4)},
        "VSS_pct": {"f1": round(vss_pct_f1, 2), "f2": round(vss_pct_f2, 2),
                    "f3": round(vss_pct_f3, 2)},
        "n_scenarios": len(rp_f1_list),
    }


def select_robust_saa(
    saa_fronts: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, Dict]:
    compromises = []
    for k, F_k in sorted(saa_fronts.items()):
        if F_k is not None and len(F_k) > 0:
            f_c, _ = select_compromise(F_k)
            compromises.append(f_c)

    if not compromises:
        return np.zeros(3), {"error": "no fronts available"}

    C = np.array(compromises)

    col_norms = np.linalg.norm(C, axis=0) + 1e-12
    C_norm    = C / col_norms

    ideal      = C_norm.min(axis=0)
    anti_ideal = C_norm.max(axis=0)

    d_ideal      = np.linalg.norm(C_norm - ideal,      axis=1)
    d_anti_ideal = np.linalg.norm(C_norm - anti_ideal, axis=1)

    scores = d_anti_ideal / (d_ideal + d_anti_ideal + 1e-12)
    best   = int(np.argmax(scores))

    info = {
        "topsis_score":      round(float(scores[best]), 4),
        "selected_scenario": best,
        "n_candidates":      len(compromises),
        "f_robust":          [round(float(v), 4) for v in C[best]],
    }
    return C[best], info


def plot_integration(
    f_ev_front:     np.ndarray,
    f_ev_comp:      np.ndarray,
    saa_fronts:     Dict[int, np.ndarray],
    f_robust:       np.ndarray,
    df_eev:         pd.DataFrame,
    vss:            Dict,
    run_dir:        Path,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
    pal = plt.get_cmap("tab20")

    ax_a = fig.add_subplot(gs[0, 0])

    for k, F_k in saa_fronts.items():
        if F_k is not None:
            idx_sorted = np.argsort(F_k[:, 0])
            ax_a.scatter(F_k[:, 0] / 3600, F_k[:, 1] / 1000,
                         s=8, alpha=0.25, color=pal(k % 20))

    idx_ev = np.argsort(f_ev_front[:, 0])
    ax_a.scatter(f_ev_front[:, 0] / 3600, f_ev_front[:, 1] / 1000,
                 s=30, color="#E53935", zorder=5, label="Pareto EV (det.)")

    ax_a.scatter(f_ev_comp[0] / 3600, f_ev_comp[1] / 1000,
                 s=120, marker="*", color="#B71C1C", zorder=10,
                 label="Compromiso EV (x̄)")

    ax_a.scatter(f_robust[0] / 3600, f_robust[1] / 1000,
                 s=120, marker="D", color="#1565C0", zorder=10,
                 label="Robusta SAA (TOPSIS)")

    ax_a.set_xlabel("f₁ — Tiempo total ruta (h)", fontsize=9)
    ax_a.set_ylabel("f₂ — Tardanza ponderada (×10³)", fontsize=9)
    ax_a.set_title("(A) Frentes de Pareto: EV vs. SAA (f₁ × f₂)",
                   fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.grid(True, alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])

    objs       = ["f₁ (h)", "f₂ (×10³)", "f₃ (kg CO₂)"]
    scales     = [3600, 1000, 1]
    eev_vals   = [vss["EEV"]["f1"] / scales[0],
                  vss["EEV"]["f2"] / scales[1],
                  vss["EEV"]["f3"] / scales[2]]
    rp_vals    = [vss["RP"]["f1"]  / scales[0],
                  vss["RP"]["f2"]  / scales[1],
                  vss["RP"]["f3"]  / scales[2]]

    x_bar  = np.arange(3)
    w_bar  = 0.35
    b_eev  = ax_b.bar(x_bar - w_bar / 2, eev_vals, width=w_bar,
                      color="#EF5350", alpha=0.85, label="EEV (det.)")
    b_rp   = ax_b.bar(x_bar + w_bar / 2, rp_vals,  width=w_bar,
                      color="#42A5F5", alpha=0.85, label="RP  (SAA)")

    for i, (ev_v, rp_v) in enumerate(zip(eev_vals, rp_vals)):
        vss_p = vss["VSS_pct"][["f1","f2","f3"][i]]
        ax_b.annotate(f"VSS={vss_p:.1f}%",
                      xy=(i, max(ev_v, rp_v) * 1.02),
                      ha="center", va="bottom", fontsize=8,
                      color="#1B5E20", fontweight="bold")

    ax_b.set_xticks(x_bar)
    ax_b.set_xticklabels(objs, fontsize=9)
    ax_b.set_ylabel("Valor objetivo (escala normalizada)", fontsize=9)
    ax_b.set_title("(B) EEV vs. RP por objetivo — VSS",
                   fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3, axis="y")

    ax_c = fig.add_subplot(gs[1, 0])
    sc_ids = df_eev["scenario_id"].values
    f1_eev = df_eev["f1_ev_eval"].values / 3600
    f3_eev = df_eev["f3_ev_eval"].values

    ax_c2 = ax_c.twinx()
    ax_c.bar(sc_ids, f1_eev, color="#EF9A9A", alpha=0.7, label="f₁ EEV (h)")
    ax_c2.plot(sc_ids, f3_eev, "o-", color="#7B1FA2", ms=5,
               lw=1.5, label="f₃ EEV (kg CO₂)")

    ax_c.axhline(vss["EEV"]["f1"] / 3600, ls="--", color="#C62828",
                 lw=1.2, label=f"EEV f₁ media={vss['EEV']['f1']/3600:.1f}h")
    ax_c.set_xlabel("Escenario SAA (k)", fontsize=9)
    ax_c.set_ylabel("f₁ — Tiempo ruta (h)", fontsize=9, color="#C62828")
    ax_c2.set_ylabel("f₃ — CO₂ proxy (kg)", fontsize=9, color="#7B1FA2")
    ax_c.set_title("(C) Evaluación EEV de x̄ en cada escenario SAA",
                   fontsize=10, fontweight="bold")

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax_c.grid(True, alpha=0.3, axis="y")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    tabla = [
        ["Métrica",      "f₁ (tiempo, h)",   "f₂ (tardanza, ×10³)", "f₃ (CO₂, kg)"],
        ["EEV (det.)",
         f"{vss['EEV']['f1']/3600:.2f}",
         f"{vss['EEV']['f2']/1000:.2f}",
         f"{vss['EEV']['f3']:.3f}"],
        ["RP (SAA)",
         f"{vss['RP']['f1']/3600:.2f}",
         f"{vss['RP']['f2']/1000:.2f}",
         f"{vss['RP']['f3']:.3f}"],
        ["VSS (abs.)",
         f"{vss['VSS']['f1']/3600:.2f}",
         f"{vss['VSS']['f2']/1000:.2f}",
         f"{vss['VSS']['f3']:.3f}"],
        ["VSS (%)",
         f"{vss['VSS_pct']['f1']:.1f}%",
         f"{vss['VSS_pct']['f2']:.1f}%",
         f"{vss['VSS_pct']['f3']:.1f}%"],
    ]

    tbl = ax_d.table(cellText=tabla[1:], colLabels=tabla[0],
                     colWidths=[0.25, 0.25, 0.27, 0.23],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    for j in range(4):
        tbl[(0, j)].set_facecolor("#37474F")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tabla)):
        fc = "#F5F5F5" if i % 2 == 0 else "white"
        if i == 4:
            fc = "#E8F5E9"
        for j in range(4):
            tbl[(i, j)].set_facecolor(fc)
            if i == 4:
                tbl[(i, j)].set_text_props(color="#1B5E20", fontweight="bold")

    ax_d.set_title("(D) Tabla VSS — Value of Stochastic Solution",
                   fontsize=10, fontweight="bold")

    fig.suptitle(
        "Integración SAA: NSGA-II Estocástico — instancia Sevilla\n"
        f"N=25 farmacias · K={vss['n_scenarios']} escenarios SAA · "
        f"VSS f₁={vss['VSS_pct']['f1']:.1f}% | "
        f"f₂={vss['VSS_pct']['f2']:.1f}% | "
        f"f₃={vss['VSS_pct']['f3']:.1f}%",
        fontsize=11, fontweight="bold", y=1.01,
    )

    out_path = run_dir / "s4_plots.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura guardada: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="s4_integration.py — SAA VSS: NSGA-II estocástico (instancia Sevilla)"
    )
    p.add_argument("--run",    type=str, default=None,
                   help="run_id de s3 (por defecto: latest_run.txt)")
    p.add_argument("--n_gen",  type=int, default=N_GEN_DEFAULT,
                   help="Generaciones NSGA-II por instancia")
    p.add_argument("--n",      type=int, default=25,
                   help="N farmacias (debe coincidir con el run de s3)")
    p.add_argument("--quick",  action="store_true",
                   help="Modo rápido: n_gen=80, sólo primeras 5 farmacias × 5 escenarios")
    return p.parse_args()


def main():
    args   = parse_args()
    n_gen  = N_GEN_QUICK if args.quick else args.n_gen
    n_farm = args.n

    if args.quick:
        log.info("  [--quick: n_gen=80, K≤5]")

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_s4"
    run_dir = INTEG_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("s4 — Integración SAA: NSGA-II → VSS")
    print(f"  N={n_farm} farmacias  n_gen={n_gen}  Run: {run_id}")
    print("=" * 60)
    t0 = time.time()

    log.info("PASO 1/6 — Cargando datos de forecast (s3)")
    df_fc, df_saa, fc_config, s3_run_id = load_forecast_data(args.run)

    n_farm_s3   = len(df_fc)
    demands_ev  = df_fc["demand_q50"].values[:n_farm]
    priorities  = df_fc["priority_class"].values[:n_farm].tolist()
    k_scenarios = sorted(df_saa["scenario_id"].unique())

    if args.quick:
        k_scenarios = k_scenarios[:5]
        log.info(f"  [--quick: usando K={len(k_scenarios)} escenarios]")

    D         = get_distance_matrix(n_farm)
    service_t = get_service_times(n_farm)

    log.info("PASO 2/6 — Problema EV (NSGA-II con demanda determinista q50)")
    t2 = time.time()
    F_ev, X_ev = run_nsga2_for_instance(
        demands_ev, priorities, D, service_t,
        n_gen=n_gen, pop_size=POP_SIZE, seed=SEED, label="EV_q50", return_X=True
    )
    f_ev_comp, ev_idx = select_compromise(F_ev)
    ev_perm = np.asarray(X_ev[ev_idx], dtype=int)
    log.info(f"  EV: {len(F_ev)} soluciones Pareto — "
             f"compromiso: f1={f_ev_comp[0]/3600:.2f}h  "
             f"f2={f_ev_comp[1]/1000:.1f}k  f3={f_ev_comp[2]:.2f}kg  "
             f"({time.time()-t2:.1f}s)")

    df_ev_front = pd.DataFrame(F_ev, columns=["f1_s", "f2_pond", "f3_co2"])
    df_ev_front["is_compromise"] = (df_ev_front.index == ev_idx)
    df_ev_front.to_csv(run_dir / "s4_ev_pareto.csv", index=False)

    log.info(f"PASO 3/6 — Problema SAA/RP: {len(k_scenarios)} instancias NSGA-II")
    saa_fronts:  Dict[int, np.ndarray] = {}
    saa_records: List[Dict] = []
    t3 = time.time()

    for k in k_scenarios:
        df_k   = df_saa[df_saa["scenario_id"] == k].sort_values("pharmacy_id")
        dem_k  = df_k["demand_kg"].values[:n_farm]
        prio_k = df_k["priority_class"].values[:n_farm].tolist()

        F_k = run_nsga2_for_instance(
            dem_k, prio_k, D, service_t,
            n_gen=n_gen, pop_size=POP_SIZE,
            seed=int(SEED + k), label=f"SAA_k{int(k):02d}"
        )
        saa_fronts[int(k)] = F_k
        f_k_comp, _   = select_compromise(F_k)

        saa_records.append({
            "scenario_id":     k,
            "n_pareto_sols":   len(F_k),
            "f1_best":         round(float(F_k[:, 0].min()), 2),
            "f2_best":         round(float(F_k[:, 1].min()), 2),
            "f3_best":         round(float(F_k[:, 2].min()), 4),
            "f1_compromise":   round(float(f_k_comp[0]), 2),
            "f2_compromise":   round(float(f_k_comp[1]), 2),
            "f3_compromise":   round(float(f_k_comp[2]), 4),
            "demand_total_kg": round(float(dem_k.sum()), 2),
        })
        if (k + 1) % 5 == 0 or k == k_scenarios[-1]:
            log.info(f"  Escenario k={k:02d}: {len(F_k)} sols Pareto, "
                     f"f1_best={F_k[:,0].min()/3600:.2f}h — "
                     f"{time.time()-t3:.1f}s acumulados")

    df_saa_fronts = pd.DataFrame(saa_records)
    df_saa_fronts.to_csv(run_dir / "s4_saa_fronts.csv", index=False)
    log.info(f"  RP completado: {len(k_scenarios)} escenarios en "
             f"{time.time()-t3:.1f}s")

    log.info("PASO 4/6 — Evaluación EEV: x̄ en cada escenario SAA")
    df_eev = evaluate_ev_on_scenarios(f_ev_comp, df_saa, demands_ev, ev_perm=ev_perm)
    df_eev_k = df_eev[df_eev["scenario_id"].isin(k_scenarios)].copy()
    df_eev_k.to_csv(run_dir / "s4_eev_eval.csv", index=False)
    log.info(f"  EEV f1 media: {df_eev_k['f1_ev_eval'].mean()/3600:.2f}h  "
             f"infactibles: {df_eev_k['n_extra_trips'].sum()} viajes extra")

    log.info("PASO 5/6 — Cálculo VSS = EEV − RP")
    vss = compute_vss(df_eev_k, saa_fronts)
    log.info(f"  VSS f1 = {vss['VSS']['f1']/3600:.2f}h "
             f"({vss['VSS_pct']['f1']:.1f}%)")
    log.info(f"  VSS f2 = {vss['VSS']['f2']:.0f} "
             f"({vss['VSS_pct']['f2']:.1f}%)")
    log.info(f"  VSS f3 = {vss['VSS']['f3']:.3f} kg CO₂ "
             f"({vss['VSS_pct']['f3']:.1f}%)")

    log.info("PASO 6/6 — Solución robusta SAA (TOPSIS) + outputs")
    f_robust, topsis_info = select_robust_saa(saa_fronts)

    summary = {
        "run_id_s4":   run_id,
        "run_id_s3":   s3_run_id,
        "n_pharmacies": n_farm,
        "k_scenarios_used": len(k_scenarios),
        "n_gen":       n_gen,
        "seed":        SEED,
        "s2_available": _S2_COORDS is not None,
        "EV_compromise": {
            "f1_h":     round(float(f_ev_comp[0]) / 3600, 3),
            "f2_pond":  round(float(f_ev_comp[1]) / 1000, 3),
            "f3_co2":   round(float(f_ev_comp[2]), 4),
        },
        "robust_SAA": {
            "f1_h":  round(float(f_robust[0]) / 3600, 3),
            "f2_k":  round(float(f_robust[1]) / 1000, 3),
            "f3_co2": round(float(f_robust[2]), 4),
            **topsis_info,
        },
        "VSS": vss,
    }

    with open(run_dir / "s4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    (INTEG_DIR / "latest_integration.txt").write_text(run_id)

    log.info("Generando figura 4-panel...")
    plot_integration(F_ev, f_ev_comp, saa_fronts, f_robust,
                     df_eev_k, vss, run_dir)

    t_total = time.time() - t0
    print("\n" + "=" * 60)
    print("Integración SAA completada")
    print("=" * 60)
    print(f"  Run ID    : {run_id}")
    print(f"  s3 Run    : {s3_run_id}")
    print(f"  N farmacias: {n_farm}  |  K escenarios: {len(k_scenarios)}")
    print(f"\n  ── Value of Stochastic Solution (VSS) ──")
    print(f"  {'Objetivo':<25} {'EEV':>10} {'RP':>10} {'VSS':>10} {'VSS%':>7}")
    print(f"  {'-'*64}")
    print(f"  {'f1 — Tiempo ruta (h)':<25} "
          f"{vss['EEV']['f1']/3600:>10.2f} "
          f"{vss['RP']['f1']/3600:>10.2f} "
          f"{vss['VSS']['f1']/3600:>10.2f} "
          f"{vss['VSS_pct']['f1']:>6.1f}%")
    print(f"  {'f2 — Tardanza (×10³)':<25} "
          f"{vss['EEV']['f2']/1000:>10.2f} "
          f"{vss['RP']['f2']/1000:>10.2f} "
          f"{vss['VSS']['f2']/1000:>10.2f} "
          f"{vss['VSS_pct']['f2']:>6.1f}%")
    print(f"  {'f3 — CO₂ proxy (kg)':<25} "
          f"{vss['EEV']['f3']:>10.3f} "
          f"{vss['RP']['f3']:>10.3f} "
          f"{vss['VSS']['f3']:>10.3f} "
          f"{vss['VSS_pct']['f3']:>6.1f}%")
    print(f"\n  Solución robusta SAA (TOPSIS):")
    print(f"    f1={f_robust[0]/3600:.2f}h  "
          f"f2={f_robust[1]/1000:.2f}k  f3={f_robust[2]:.3f}kg CO₂")
    print(f"    TOPSIS score={topsis_info['topsis_score']:.4f}")
    print(f"\n  Tiempo total: {t_total:.1f}s")
    print(f"  Salidas en: {run_dir}/")
    print(f"    s4_ev_pareto.csv")
    print(f"    s4_saa_fronts.csv")
    print(f"    s4_eev_eval.csv")
    print(f"    s4_summary.json")
    print(f"    s4_plots.png")
    print("=" * 60)

    target_ok = (vss["VSS_pct"]["f1"] >= 3.0 or
                 vss["VSS_pct"]["f2"] >= 3.0 or
                 vss["VSS_pct"]["f3"] >= 3.0)
    if target_ok:
        print("\n  VSS ≥ 3 % en al menos un objetivo")
    else:
        print("\n  ⚠️  VSS < 3% — considerar aumentar K o revisar "
              "dispersión de escenarios SAA")

    print("\nSiguiente paso: python s5_sensitivity.py")


if __name__ == "__main__":
    main()
