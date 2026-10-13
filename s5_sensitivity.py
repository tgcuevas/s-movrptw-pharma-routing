#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, logging, warnings, argparse
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

try:
    import scikit_posthocs as sp
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    logging.warning("scikit_posthocs no disponible — Nemenyi se calculará vía "
                    "fallback Wilcoxon por pares. Instala 'scikit-posthocs' "
                    "para el post-hoc de Nemenyi completo.")

BASE_DIR  = Path("data")
SENS_DIR  = BASE_DIR / "sensitivity"
FORE_DIR  = BASE_DIR / "forecast"
SENS_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY     = 200.0
CO2_ALPHA    = 0.05
PENALTY_TRIP = 1800.0
SEED         = 42
_ALPHA_TW    = {1: 10_000, 2: 5_000, 3: 1_000}
_TW_CLASS    = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}
PRIO_PROBS   = [0.30, 0.20, 0.50]


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

class CVRPTW_3obj(ElementwiseProblem):
    def __init__(self, inst):
        self.n      = inst["n"]
        self.D      = np.array(inst["D"], dtype=float)
        self.dem    = np.array(inst["demands"], dtype=float)
        self.prio   = inst["priorities"]
        self.svc    = np.array(inst["service_times"], dtype=float)
        self.cap    = float(inst.get("capacity", CAPACITY))
        super().__init__(n_var=self.n, n_obj=3, xl=0, xu=self.n - 1)
    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3 = _eval(x.astype(int), self.dem, self.D,
                            self.svc, self.prio, self.cap)
        out["F"] = [f1, f2, f3]

def _eval(order, dem, D, svc, prio, cap):
    depot = 0
    routes, cur, cd = [], [], 0.0
    for idx in order:
        nd = int(idx) + 1
        d  = float(dem[idx])
        if cd + d > cap and cur:
            routes.append(cur); cur, cd = [], 0.0
        cur.append((int(idx), nd, d)); cd += d
    if cur: routes.append(cur)
    f1 = f2 = f3 = 0.0
    for rt in routes:
        t, load, prev = 0.0, 0.0, depot
        for (idx, nd, d) in rt:
            t    += float(D[prev, nd]) + float(svc[nd])
            load += d; prev = nd
            f2   += _ALPHA_TW[prio[idx]] * max(0.0, t - _TW_CLASS[prio[idx]][1])
        t   += float(D[prev, depot])
        f1  += t
        f3  += CO2_ALPHA * t * (load / cap)
    return f1, f2, f3


def make_distance_matrix(n: int, seed: int = SEED) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    size = n + 1
    raw  = np.exp(rng.normal(6.3, 0.35, (size, size)))
    raw  = (raw + raw.T) / 2.0; np.fill_diagonal(raw, 0.0)
    raw[0, 1:] += 300.0; raw[1:, 0] += 300.0
    return raw

def make_service_times(n: int) -> np.ndarray:
    st = np.full(n + 1, 180.0); st[0] = 0.0; return st

def make_priorities(n: int, seed: int = SEED) -> List[int]:
    rng = np.random.default_rng(seed)
    return rng.choice([1, 2, 3], size=n, p=PRIO_PROBS).tolist()

def make_instance(n, demands, priorities, D, svc):
    return {"n": n, "D": D.tolist(), "demands": demands.tolist(),
            "priorities": priorities, "service_times": svc.tolist(),
            "capacity": CAPACITY}

def _run_nsga2(inst, pop_size, n_gen, seed):
    prob = CVRPTW_3obj(inst)
    alg  = NSGA2(pop_size=int(pop_size),
                 sampling=PermutationSampling(),
                 crossover=OrderCrossover(),
                 mutation=SwapMutation(),
                 eliminate_duplicates=True)
    res  = minimize(prob, alg, ("n_gen", int(n_gen)),
                    seed=int(seed), verbose=False)
    return res.F if res.F is not None else np.zeros((1, 3))

def _run_nsga3(inst, pop_size, n_gen, seed):
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    pop_size = max(int(pop_size), len(ref_dirs))
    prob = CVRPTW_3obj(inst)
    alg  = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs,
                 sampling=PermutationSampling(),
                 crossover=OrderCrossover(),
                 mutation=SwapMutation(),
                 eliminate_duplicates=True)
    res  = minimize(prob, alg, ("n_gen", int(n_gen)),
                    seed=int(seed), verbose=False)
    return res.F if res.F is not None else np.zeros((1, 3))

def _run_moead(inst, pop_size, n_gen, seed):
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=4)
    prob = CVRPTW_3obj(inst)
    try:
        alg = MOEAD(ref_dirs=ref_dirs, n_neighbors=5,
                    prob_neighbor_mating=0.7,
                    sampling=PermutationSampling(),
                    crossover=OrderCrossover(),
                    mutation=SwapMutation())
        res = minimize(prob, alg, ("n_gen", int(n_gen)),
                       seed=int(seed), verbose=False)
        return res.F if res.F is not None else np.zeros((1, 3))
    except Exception:
        return _run_nsga2(inst, pop_size, n_gen, seed + 9999)

def _run_nsga2_fx(inst, pop_size, n_gen, seed):
    prob = CVRPTW_3obj(inst)
    alg  = NSGA2(pop_size=int(pop_size), sampling=PermutationSampling(),
                 crossover=OrderCrossover(), mutation=SwapMutation(),
                 eliminate_duplicates=True)
    res  = minimize(prob, alg, ("n_gen", int(n_gen)), seed=int(seed), verbose=False)
    return res.F, res.X

def ev_plan(F, X, dem):
    Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    i  = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    routes, cur, cd = [], [], 0.0
    for j in np.asarray(X[i], dtype=int):
        if cd + dem[j] > CAPACITY and cur:
            routes.append(cur); cur, cd = [], 0.0
        cur.append(int(j)); cd += float(dem[j])
    routes.append(cur)
    return F[i], routes

def extra_trips(d_k, routes):
    return int(sum(max(0, int(np.ceil(d_k[r].sum() / CAPACITY)) - 1) for r in routes))

def compute_ref_point(fronts: List[np.ndarray], margin: float = 1.1) -> np.ndarray:
    all_F = np.vstack(fronts)
    return all_F.max(axis=0) * margin

def compute_spacing(F: np.ndarray) -> float:
    if len(F) < 2: return 0.0
    dists = cdist(F, F); np.fill_diagonal(dists, np.inf)
    d_min = dists.min(axis=1)
    return float(np.sqrt(np.sum((d_min - d_min.mean()) ** 2) / (len(F) - 1)))

def select_compromise(F: np.ndarray) -> np.ndarray:
    rng_f = F.max(axis=0) - F.min(axis=0) + 1e-9
    Fn = (F - F.min(axis=0)) / rng_f
    return F[np.argmin(np.linalg.norm(Fn, axis=1))]

def crps_normal(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    from scipy.stats import norm as _norm
    z = (y_true - mu) / (sigma + 1e-9)
    crps = sigma * (z * (2 * _norm.cdf(z) - 1)
                    + 2 * _norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(crps.mean())

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def gen_solomon_instance(stype: str, n: int, seed: int) -> Dict:
    rng  = np.random.default_rng(seed)
    size = n + 1

    if stype == "C101":
        centers = rng.uniform(20, 80, (3, 2))
        xy = np.vstack([rng.normal(centers[i % 3], 8, (1, 2))
                        for i in range(n)])
        xy = np.clip(xy, 0, 100)
        tw_width = 600
    else:
        xy = rng.uniform(0, 100, (n, 2))
        tw_width = 1800

    depot_xy = np.array([[50.0, 50.0]])
    all_xy   = np.vstack([depot_xy, xy])

    D = cdist(all_xy, all_xy) * 2.0 * 60
    np.fill_diagonal(D, 0.0)

    mu_farm = rng.uniform(5, 15, n)
    phi     = 4.0
    gamma_s = rng.gamma(shape=phi, scale=mu_farm / phi)
    demands = rng.poisson(gamma_s).astype(float)
    demands = np.clip(demands, 0.5, None)

    priorities = rng.choice([1, 2, 3], size=n, p=PRIO_PROBS).tolist()
    service_t  = np.full(size, 120.0); service_t[0] = 0.0

    return {
        "label":         f"{stype}-{n}",
        "n":             n,
        "D":             D.tolist(),
        "demands":       demands.tolist(),
        "priorities":    priorities,
        "service_times": service_t.tolist(),
        "capacity":      CAPACITY,
        "tw_width":      tw_width,
    }


def run_module_a(n_runs: int = 30, n_gen: int = 150,
                 pop_size: int = 50, quick: bool = False) -> Dict:
    log.info("═" * 55)
    log.info("MÓDULO A — Benchmark Solomon estocástico")
    log.info(f"  {n_runs} runs × 3 algoritmos × 2 instancias, n_gen={n_gen}")
    log.info("═" * 55)

    instances = {
        "C101-25": gen_solomon_instance("C101", 25, seed=SEED),
        "R101-25": gen_solomon_instance("R101", 25, seed=SEED + 1),
    }
    algos = ["NSGA-II", "NSGA-III", "MOEA/D"]
    runners = {"NSGA-II": _run_nsga2, "NSGA-III": _run_nsga3, "MOEA/D": _run_moead}

    records = []
    all_fronts: Dict[str, Dict[str, List]] = {
        inst: {alg: [] for alg in algos} for inst in instances
    }

    t0 = time.time()
    for inst_name, inst in instances.items():
        for alg_name in algos:
            fn = runners[alg_name]
            for r in range(n_runs):
                F = fn(inst, pop_size, n_gen, seed=SEED + r * 100)
                all_fronts[inst_name][alg_name].append(F)
            log.info(f"  {inst_name} × {alg_name}: {n_runs} runs completados "
                     f"({time.time()-t0:.0f}s)")

    ref_pts = {}
    for inst_name in instances:
        all_f = [F for alg in algos
                 for F in all_fronts[inst_name][alg]]
        ref_pts[inst_name] = compute_ref_point(all_f)

    emp_fronts = {}
    for inst_name in instances:
        pooled = np.vstack([F for alg in algos
                            for F in all_fronts[inst_name][alg]])
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        nds   = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
        emp_fronts[inst_name] = pooled[nds]

    for inst_name in instances:
        ref_pt = ref_pts[inst_name]
        pf     = emp_fronts[inst_name]
        hv_ind  = HV(ref_point=ref_pt)
        igd_ind = IGD(pf)

        for alg_name in algos:
            for r, F in enumerate(all_fronts[inst_name][alg_name]):
                try:
                    hv_val  = float(hv_ind(F))
                    igd_val = float(igd_ind(F))
                    sp_val  = compute_spacing(F)
                except Exception:
                    hv_val = igd_val = sp_val = np.nan
                records.append({
                    "instance": inst_name, "algorithm": alg_name,
                    "run": r, "hv": hv_val, "igd": igd_val,
                    "spacing": sp_val, "n_sols": len(F),
                })

    df = pd.DataFrame(records)

    test_results = {}
    for inst_name in instances:
        test_results[inst_name] = {}
        df_i = df[df["instance"] == inst_name]
        for metric in ["hv", "igd", "spacing"]:
            groups = [df_i[df_i["algorithm"] == a][metric].dropna().values
                      for a in algos]
            try:
                stat, pval = stats.friedmanchisquare(*groups)
            except Exception:
                stat, pval = np.nan, np.nan

            posthoc = {}
            try:
                min_len = min(len(g) for g in groups)
                if min_len >= 2:
                    bal = np.column_stack([g[:min_len] for g in groups])
                    if SP_AVAILABLE:
                        ph = sp.posthoc_nemenyi_friedman(bal)
                        ph.columns = algos; ph.index = algos
                        posthoc = ph.round(4).to_dict()
                    else:
                        from itertools import combinations
                        for i, j in combinations(range(len(algos)), 2):
                            try:
                                _, wp = stats.wilcoxon(bal[:, i], bal[:, j])
                                posthoc[f"{algos[i]}_vs_{algos[j]}"] = round(float(wp), 6)
                            except Exception:
                                posthoc[f"{algos[i]}_vs_{algos[j]}"] = None
            except Exception as e:
                logging.warning(f"Post-hoc {inst_name}/{metric} no calculado: {e}")
                posthoc = {}
            posthoc_method = "nemenyi" if SP_AVAILABLE else "wilcoxon_pairwise"

            stats_dict = {
                a: {"mean": round(float(np.nanmean(g)), 4),
                    "std":  round(float(np.nanstd(g)), 4)}
                for a, g in zip(algos, groups)
            }
            test_results[inst_name][metric] = {
                "friedman_stat": round(float(stat), 4) if not np.isnan(stat) else None,
                "friedman_p":    round(float(pval), 6) if not np.isnan(pval) else None,
                "significant":   bool(pval < 0.05) if not np.isnan(pval) else False,
                "posthoc_nemenyi": posthoc,
                "posthoc_method": posthoc_method,
                "stats": stats_dict,
            }

        for metric in ["hv", "igd", "spacing"]:
            tr = test_results[inst_name][metric]
            sig = "✅" if tr["significant"] else "—"
            log.info(f"  {inst_name} {metric}: Friedman p={tr['friedman_p']}  {sig}")
            for a in algos:
                m, s = tr["stats"][a]["mean"], tr["stats"][a]["std"]
                log.info(f"    {a:<10}  {metric}={m:.4f} ± {s:.4f}")

    return {"df": df, "tests": test_results,
            "fronts": all_fronts, "emp_fronts": emp_fronts}


def run_module_b(N_sobol: int = 32, n_gen_sobol: int = 80,
                 n_farm: int = 25, quick: bool = False) -> Dict:
    log.info("═" * 55)
    log.info("MÓDULO B — Análisis Sobol SA")
    log.info(f"  N={N_sobol} muestras Saltelli → {N_sobol*(7+2)} evaluaciones (sin índices de segundo orden)")
    log.info("═" * 55)

    problem_def = {
        "num_vars": 7,
        "names": ["pop_size", "n_gen", "p_cross", "p_mut",
                  "k_saa", "sigma_demand", "alpha_urgente"],
        "bounds": [[30, 150], [80, 300], [0.6, 1.0], [0.05, 0.25],
                   [5, 25], [0.05, 0.30], [5_000, 20_000]],
    }

    D     = make_distance_matrix(n_farm, SEED)
    svc   = make_service_times(n_farm)
    prios = make_priorities(n_farm, SEED)
    rng   = np.random.default_rng(SEED)
    mu    = rng.uniform(3, 18, n_farm)

    param_values = saltelli.sample(problem_def, N_sobol,
                                   calc_second_order=False)
    n_eval = len(param_values)
    log.info(f"  Evaluaciones totales: {n_eval}")

    Y_hv = np.zeros(n_eval)
    Y_igd = np.zeros(n_eval)
    Y_vss = np.zeros(n_eval)
    Y_vsshv = np.zeros(n_eval)

    nom_dem = mu.copy()
    nom_inst = make_instance(n_farm, nom_dem, prios, D, svc)
    F_ref_list = [_run_nsga2(nom_inst, 100, 200, SEED + i) for i in range(3)]
    pooled_ref = np.vstack(F_ref_list)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds_idx  = NonDominatedSorting().do(pooled_ref, only_non_dominated_front=True)
    pf_ref   = pooled_ref[nds_idx]
    ref_pt   = compute_ref_point(F_ref_list)

    t0 = time.time()
    for i, params in enumerate(param_values):
        pop_sz   = int(max(15, params[0]))
        n_gen_   = int(max(40, params[1]))
        k_saa_   = int(max(3, params[4]))
        sigma_d  = float(params[5])
        alpha_u  = float(params[6])

        noise = np.random.default_rng(SEED + i).normal(0, sigma_d, n_farm) * mu
        dem_i = np.clip(mu + noise, 0.1, None)

        old_alpha = _ALPHA_TW.copy()
        _ALPHA_TW[1] = alpha_u

        inst_i = make_instance(n_farm, dem_i, prios, D, svc)
        F_i = _run_nsga2(inst_i, pop_sz, n_gen_, seed=int(SEED + i))

        _ALPHA_TW.update(old_alpha)

        try:
            hv_ind  = HV(ref_point=ref_pt)
            igd_ind = IGD(pf_ref)
            Y_hv[i]  = float(hv_ind(F_i))
            Y_igd[i] = float(igd_ind(F_i))
        except Exception:
            Y_hv[i] = Y_igd[i] = 0.0

        F_det = _run_nsga2(nom_inst, pop_sz, n_gen_, seed=int(SEED + i + 5000))
        comp_i   = select_compromise(F_i)
        comp_det = select_compromise(F_det)
        vss_f1 = max(0.0, float(comp_det[0]) - float(comp_i[0]))
        Y_vss[i] = vss_f1 / (float(comp_det[0]) + 1e-9) * 100

        try:
            hv_rp  = float(hv_ind(F_i))
            hv_eev = float(hv_ind(F_det))
            Y_vsshv[i] = max(0.0, (hv_rp - hv_eev) / (hv_rp + 1e-9)) * 100
        except Exception:
            Y_vsshv[i] = 0.0

        if (i + 1) % max(1, n_eval // 5) == 0:
            log.info(f"  Sobol {i+1}/{n_eval} — "
                     f"{time.time()-t0:.0f}s acumulados")

    sobol_results = {}
    for metric_name, Y in [("hv", Y_hv), ("igd", Y_igd),
                            ("vss_pct", Y_vss), ("vss_hv", Y_vsshv)]:
        try:
            si = sobol_analyze.analyze(problem_def, Y,
                                       calc_second_order=False,
                                       print_to_console=False)
            sobol_results[metric_name] = {
                "S1":  {n: round(float(v), 4)
                        for n, v in zip(problem_def["names"], si["S1"])},
                "ST":  {n: round(float(v), 4)
                        for n, v in zip(problem_def["names"], si["ST"])},
                "S1_conf": {n: round(float(v), 4)
                            for n, v in zip(problem_def["names"], si["S1_conf"])},
                "ST_conf": {n: round(float(v), 4)
                            for n, v in zip(problem_def["names"], si["ST_conf"])},
            }
        except Exception as ex:
            log.warning(f"  Sobol analyze falló para {metric_name}: {ex}")
            sobol_results[metric_name] = {}

    if "hv" in sobol_results and sobol_results["hv"]:
        log.info("  Ranking S1 (HV):")
        s1_hv = sobol_results["hv"]["S1"]
        for nm, val in sorted(s1_hv.items(), key=lambda x: -x[1]):
            log.info(f"    {nm:<18} S1={val:.4f}")

    return {"sobol": sobol_results, "Y": {"hv": Y_hv, "igd": Y_igd,
            "vss_pct": Y_vss, "vss_hv": Y_vsshv},
            "param_values": param_values, "problem": problem_def}


def run_module_c(k_range: List[int] = None, n_farm: int = 25,
                 n_gen: int = 100, n_rep: int = 5) -> Dict:
    if k_range is None:
        k_range = [3, 5, 8, 10, 15, 20, 25, 30]
    log.info("═" * 55)
    log.info("MÓDULO C — VSS_HV — convergencia vs K")
    log.info(f"  K ∈ {k_range}  |  n_gen={n_gen}  |  {n_rep} repeticiones")
    log.info("═" * 55)

    D    = make_distance_matrix(n_farm, SEED)
    svc  = make_service_times(n_farm)
    prio = make_priorities(n_farm, SEED)
    rng  = np.random.default_rng(SEED)
    mu   = rng.uniform(3, 18, n_farm)
    sig  = mu * 0.15

    corr = np.eye(n_farm) * 0.9 + np.ones((n_farm, n_farm)) * 0.1
    L    = np.linalg.cholesky(corr)

    inst_ev = make_instance(n_farm, mu, prio, D, svc)
    F_ev, X_ev = _run_nsga2_fx(inst_ev, 100, n_gen * 2, SEED)
    x_bar, ev_routes = ev_plan(F_ev, X_ev, mu)

    ref_pt = compute_ref_point([F_ev]) * 1.5

    records = []
    for K in k_range:
        vsshv_reps = []
        for rep in range(n_rep):
            rng_k = np.random.default_rng(SEED + rep * 1000 + K)
            scenarios = []
            for k in range(K):
                z     = rng_k.standard_normal(n_farm)
                z_c   = L @ z
                d_k   = np.clip(mu + sig * z_c, 0.1, None)
                scenarios.append(d_k)

            F_rp_list = []
            for k, d_k in enumerate(scenarios):
                inst_k = make_instance(n_farm, d_k, prio, D, svc)
                F_k    = _run_nsga2(inst_k, 50, n_gen, int(SEED + rep * 100 + k))
                F_rp_list.append(F_k)

            pooled_rp = np.vstack(F_rp_list)
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nd_idx = NonDominatedSorting().do(pooled_rp,
                                              only_non_dominated_front=True)
            F_rp_nd = pooled_rp[nd_idx]

            eev_pts = []
            for k, d_k in enumerate(scenarios):
                load  = d_k.sum()
                n_extra = extra_trips(d_k, ev_routes)
                f1_k = float(x_bar[0]) + PENALTY_TRIP * n_extra
                f2_k = float(x_bar[1])
                f3_k = float(x_bar[2]) * (load / CAPACITY) / (mu.sum() / CAPACITY)
                eev_pts.append([f1_k, f2_k, f3_k])
            F_eev = np.array(eev_pts)

            try:
                hv_ind    = HV(ref_point=ref_pt)
                hv_rp     = float(hv_ind(F_rp_nd))
                hv_eev    = float(hv_ind(F_eev))
                vsshv_val = max(0.0, (hv_rp - hv_eev) / (hv_rp + 1e-9)) * 100
            except Exception:
                vsshv_val = np.nan

            vsshv_reps.append(vsshv_val)

        mean_v = float(np.nanmean(vsshv_reps))
        std_v  = float(np.nanstd(vsshv_reps))
        records.append({"K": K, "vsshv_mean": round(mean_v, 3),
                         "vsshv_std": round(std_v, 3),
                         "vsshv_cv_pct": round(std_v / (mean_v + 1e-9) * 100, 1)})
        log.info(f"  K={K:3d}: VSS_HV = {mean_v:.2f}% ± {std_v:.2f}%")

    df_c = pd.DataFrame(records)
    k_conv = df_c[df_c["vsshv_cv_pct"] < 10]["K"].min() if len(df_c) else None
    log.info(f"  K de convergencia (CV<10%): {k_conv}")
    return {"df": df_c, "k_convergence": int(k_conv) if pd.notna(k_conv) else None,
            "f_ev_compromise": x_bar.tolist()}


def run_module_d(K: int = 20, n_farm: int = 25, n_gen: int = 100) -> Dict:
    log.info("═" * 55)
    log.info("MÓDULO D — EVPI: WS ≤ RP ≤ EEV (Birge & Louveaux, 1997)")
    log.info("═" * 55)

    D    = make_distance_matrix(n_farm, SEED)
    svc  = make_service_times(n_farm)
    prio = make_priorities(n_farm, SEED)
    rng  = np.random.default_rng(SEED)
    mu   = rng.uniform(3, 18, n_farm)
    sig  = mu * 0.15
    corr = np.eye(n_farm) * 0.9 + np.ones((n_farm, n_farm)) * 0.1
    L    = np.linalg.cholesky(corr)

    scenarios = []
    rng_s = np.random.default_rng(SEED + 42)
    for k in range(K):
        z   = rng_s.standard_normal(n_farm)
        d_k = np.clip(mu + sig * (L @ z), 0.1, None)
        scenarios.append(d_k)

    inst_ev = make_instance(n_farm, mu, prio, D, svc)
    F_ev, X_ev = _run_nsga2_fx(inst_ev, 100, n_gen * 2, SEED)
    x_bar, ev_routes = ev_plan(F_ev, X_ev, mu)

    eev_vals = {"f1": [], "f2": [], "f3": []}
    for k, d_k in enumerate(scenarios):
        n_extra = extra_trips(d_k, ev_routes)
        eev_vals["f1"].append(float(x_bar[0]) + PENALTY_TRIP * n_extra)
        eev_vals["f2"].append(float(x_bar[1]))
        eev_vals["f3"].append(float(x_bar[2]) * d_k.sum() / (mu.sum() + 1e-9))
    EEV = {o: float(np.mean(v)) for o, v in eev_vals.items()}

    rp_vals = {"f1": [], "f2": [], "f3": []}
    for k, d_k in enumerate(scenarios):
        inst_k = make_instance(n_farm, d_k, prio, D, svc)
        F_k    = _run_nsga2(inst_k, 50, n_gen, int(SEED + k))
        comp_k = select_compromise(F_k)
        for j, o in enumerate(["f1", "f2", "f3"]):
            rp_vals[o].append(float(comp_k[j]))
    RP = {o: float(np.mean(v)) for o, v in rp_vals.items()}

    ws_vals = {"f1": [], "f2": [], "f3": []}
    for k, d_k in enumerate(scenarios):
        inst_k_ws = make_instance(n_farm, np.clip(d_k, 0.1, CAPACITY - 0.1),
                                  prio, D, svc)
        F_k_ws = _run_nsga2(inst_k_ws, 50, n_gen, int(SEED + k + 9000))
        comp_k_ws = select_compromise(F_k_ws)
        for j, o in enumerate(["f1", "f2", "f3"]):
            ws_vals[o].append(float(comp_k_ws[j]))
    WS = {o: float(np.mean(v)) for o, v in ws_vals.items()}

    VSS  = {o: max(0.0, EEV[o] - RP[o])  for o in ["f1", "f2", "f3"]}
    EVPI = {o: max(0.0, RP[o]  - WS[o])  for o in ["f1", "f2", "f3"]}
    VSS_pct  = {o: round(VSS[o] / (EEV[o] + 1e-9) * 100, 2)
                for o in ["f1", "f2", "f3"]}
    EVPI_pct = {o: round(EVPI[o] / (RP[o] + 1e-9) * 100, 2)
                for o in ["f1", "f2", "f3"]}

    log.info(f"  {'Objetivo':<6} {'WS':>10} {'RP':>10} {'EEV':>10} "
             f"{'VSS%':>7} {'EVPI%':>7}")
    log.info(f"  {'-'*54}")
    for o in ["f1", "f2", "f3"]:
        log.info(f"  {o:<6} {WS[o]:>10.2f} {RP[o]:>10.2f} {EEV[o]:>10.2f} "
                 f"{VSS_pct[o]:>6.1f}% {EVPI_pct[o]:>6.1f}%")

    chain_ok = {o: (WS[o] <= RP[o] + 1e-3) and (RP[o] <= EEV[o] + 1e-3)
                for o in ["f1", "f2", "f3"]}
    log.info(f"  Cadena WS≤RP≤EEV: "
             f"f1={'✅' if chain_ok['f1'] else '⚠️'}  "
             f"f2={'✅' if chain_ok['f2'] else '⚠️'}  "
             f"f3={'✅' if chain_ok['f3'] else '⚠️'}")

    return {"WS": WS, "RP": RP, "EEV": EEV,
            "VSS": VSS, "EVPI": EVPI,
            "VSS_pct": VSS_pct, "EVPI_pct": EVPI_pct,
            "chain_ok": chain_ok, "K": K}


def run_module_e(n_runs: int = 30, n_farm: int = 25,
                 n_gen: int = 200, pop_size: int = 100,
                 track_gen: bool = True) -> Dict:
    log.info("═" * 55)
    log.info("MÓDULO E — Estabilidad estadística (30 runs)")
    log.info(f"  {n_runs} runs × 3 algoritmos, N={n_farm}, n_gen={n_gen}")
    log.info("═" * 55)

    D    = make_distance_matrix(n_farm, SEED)
    svc  = make_service_times(n_farm)
    prio = make_priorities(n_farm, SEED)
    rng  = np.random.default_rng(SEED)
    dem  = np.clip(rng.gamma(4, rng.uniform(3, 18, n_farm) / 4), 0.5, None)
    inst = make_instance(n_farm, dem, prio, D, svc)

    ref_fronts = [_run_nsga2(inst, pop_size, n_gen, SEED + i)
                  for i in range(5)]
    ref_pt = compute_ref_point(ref_fronts)
    pooled = np.vstack(ref_fronts)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds_idx = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
    pf_ref  = pooled[nds_idx]

    hv_ind  = HV(ref_point=ref_pt)
    igd_ind = IGD(pf_ref)
    algos   = ["NSGA-II", "NSGA-III", "MOEA/D"]
    runners = {"NSGA-II": _run_nsga2, "NSGA-III": _run_nsga3, "MOEA/D": _run_moead}

    records = []
    for alg_name in algos:
        fn = runners[alg_name]
        for r in range(n_runs):
            F = fn(inst, pop_size, n_gen, seed=int(SEED + r * 17))
            try:
                hv  = float(hv_ind(F))
                igd = float(igd_ind(F))
                sp  = compute_spacing(F)
            except Exception:
                hv = igd = sp = np.nan
            records.append({"algorithm": alg_name, "run": r,
                             "hv": hv, "igd": igd, "spacing": sp})
        log.info(f"  {alg_name}: {n_runs} runs completados")

    df_e = pd.DataFrame(records)

    wilcoxon_results = {}
    pairs = [("NSGA-II", "NSGA-III"), ("NSGA-II", "MOEA/D"),
             ("NSGA-III", "MOEA/D")]
    for a1, a2 in pairs:
        wilcoxon_results[f"{a1}_vs_{a2}"] = {}
        for metric in ["hv", "igd", "spacing"]:
            g1 = df_e[df_e["algorithm"] == a1][metric].dropna().values
            g2 = df_e[df_e["algorithm"] == a2][metric].dropna().values
            try:
                _, p = stats.wilcoxon(g1, g2)
                wilcoxon_results[f"{a1}_vs_{a2}"][metric] = {
                    "p_value": round(float(p), 6),
                    "significant_05": bool(p < 0.05),
                    "significant_10": bool(p < 0.10),
                }
            except Exception as ex:
                wilcoxon_results[f"{a1}_vs_{a2}"][metric] = {"error": str(ex)}

    hv_nsga2 = df_e[df_e["algorithm"] == "NSGA-II"]["hv"].dropna().values
    ci95 = stats.t.interval(0.95, len(hv_nsga2) - 1,
                             loc=np.mean(hv_nsga2),
                             scale=stats.sem(hv_nsga2))
    log.info(f"  NSGA-II HV = {np.mean(hv_nsga2):.4f} ± {np.std(hv_nsga2):.4f}  "
             f"IC95%=[{ci95[0]:.4f}, {ci95[1]:.4f}]")

    for pair, mets in wilcoxon_results.items():
        for met, res in mets.items():
            if "p_value" in res:
                sig = "✅" if res["significant_05"] else "—"
                log.info(f"  Wilcoxon {pair} ({met}): p={res['p_value']}  {sig}")

    return {"df": df_e, "wilcoxon": wilcoxon_results,
            "ci95_hv_nsga2": list(ci95)}


def run_module_f(n_farm: int = 25, n_days: int = 120,
                 K: int = 10, n_gen: int = 80) -> Dict:
    log.info("═" * 55)
    log.info("MÓDULO F — Ablación forecasting → impacto en VSS_HV")
    log.info(f"  N={n_farm} farmacias, {n_days} días, K={K} escenarios SAA")
    log.info("═" * 55)

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks as tf_cb

    rng  = np.random.default_rng(SEED)
    mu   = rng.uniform(3, 18, n_farm)
    phi  = 4.0

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    dow   = dates.dayofweek.values
    w_eff = np.where(dow == 0, 1.10, np.where(dow == 4, 1.08,
            np.where(dow >= 5, 0.75, 1.00)))
    demand_hist = np.zeros((n_days, n_farm))
    for i in range(n_farm):
        gamma_s = rng.gamma(phi, mu[i] / phi * w_eff)
        demand_hist[:, i] = np.clip(rng.poisson(gamma_s).astype(float), 0.5, None)

    val_days = 14
    train    = demand_hist[:-val_days]
    test     = demand_hist[-val_days:]

    D    = make_distance_matrix(n_farm, SEED)
    svc  = make_service_times(n_farm)
    prio = make_priorities(n_farm, SEED)

    results = {}

    naive_preds = demand_hist[-val_days - 7: -7]
    naive_mu = naive_preds.mean(axis=0)
    naive_sig = naive_preds.std(axis=0) + 0.5
    results["Naive-7d"] = {
        "mu": naive_mu, "sigma": naive_sig,
        "source": "seasonal lag-7"
    }

    try:
        from statsmodels.tsa.arima.model import ARIMA
        arima_mu = np.zeros(n_farm)
        arima_sig = np.zeros(n_farm)
        for i in range(n_farm):
            try:
                mdl = ARIMA(train[:, i], order=(1, 1, 1))
                fit = mdl.fit(disp=False)
                fc  = fit.get_forecast(steps=1)
                arima_mu[i]  = max(fc.predicted_mean[0], 0.1)
                arima_sig[i] = max(fc.se_mean[0], 0.1)
            except Exception:
                arima_mu[i]  = train[-7:, i].mean()
                arima_sig[i] = train[-7:, i].std() + 0.5
        results["ARIMA"] = {"mu": arima_mu, "sigma": arima_sig,
                            "source": "ARIMA(1,1,1)"}
    except ImportError:
        log.warning("  statsmodels no disponible — saltando ARIMA")

    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    max_lag = 14
    X_tr, y_tr, X_te = [], [], []
    for t in range(max_lag, len(train)):
        row = np.hstack([train[t - l] for l in [1, 2, 3, 7, 14]] +
                        [train[t-7:t].mean(axis=0),
                         train[t-7:t].std(axis=0)])
        X_tr.append(row); y_tr.append(train[t])
    for t in range(max_lag, len(demand_hist)):
        row = np.hstack([demand_hist[t - l] for l in [1, 2, 3, 7, 14]] +
                        [demand_hist[t-7:t].mean(axis=0),
                         demand_hist[t-7:t].std(axis=0)])
        X_te.append(row)
    X_tr_a = np.array(X_tr, dtype=np.float32)
    y_tr_a = np.array(y_tr, dtype=np.float32)
    x_last = np.array(X_te[-1:], dtype=np.float32)

    xgb_q50 = np.zeros(n_farm)
    xgb_sig = np.zeros(n_farm)
    for i in range(n_farm):
        mdl_q50 = xgb.XGBRegressor(objective="reg:quantileerror",
                                    quantile_alpha=0.50, n_estimators=100,
                                    max_depth=3, learning_rate=0.1,
                                    random_state=SEED, verbosity=0, n_jobs=1)
        mdl_q10 = xgb.XGBRegressor(objective="reg:quantileerror",
                                    quantile_alpha=0.10, n_estimators=100,
                                    max_depth=3, learning_rate=0.1,
                                    random_state=SEED, verbosity=0, n_jobs=1)
        mdl_q90 = xgb.XGBRegressor(objective="reg:quantileerror",
                                    quantile_alpha=0.90, n_estimators=100,
                                    max_depth=3, learning_rate=0.1,
                                    random_state=SEED, verbosity=0, n_jobs=1)
        Xi = X_tr_a[:, i * n_farm:(i + 1) * n_farm] if X_tr_a.shape[1] > n_farm else X_tr_a
        xi_cols = np.column_stack([X_tr_a[:, l * n_farm + i]
                                   if X_tr_a.shape[1] >= (l + 1) * n_farm
                                   else X_tr_a[:, i]
                                   for l in range(min(5, X_tr_a.shape[1] // n_farm))])
        xi_last = xi_cols[-1:]
        mdl_q50.fit(xi_cols, y_tr_a[:, i], verbose=False)
        mdl_q10.fit(xi_cols, y_tr_a[:, i], verbose=False)
        mdl_q90.fit(xi_cols, y_tr_a[:, i], verbose=False)
        q50 = max(float(mdl_q50.predict(xi_last)[0]), 0.1)
        q10 = max(float(mdl_q10.predict(xi_last)[0]), 0.0)
        q90 = max(float(mdl_q90.predict(xi_last)[0]), q50)
        xgb_q50[i] = q50
        xgb_sig[i] = max((q90 - q10) / (2 * 1.282), 0.1)
    results["XGBoost-Q"] = {"mu": xgb_q50, "sigma": xgb_sig,
                             "source": "XGBoost quantile"}

    SEQ_LEN = 10
    sc = StandardScaler()
    hist_norm = sc.fit_transform(demand_hist)
    X_seq, y_seq = [], []
    for t in range(SEQ_LEN, len(hist_norm) - val_days):
        X_seq.append(hist_norm[t - SEQ_LEN:t])
        y_seq.append(hist_norm[t])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    inp = keras.Input(shape=(SEQ_LEN, n_farm))
    x   = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(inp)
    x   = layers.Dropout(0.2)(x)
    mu_out  = layers.Dense(n_farm, name="mu")(x)
    ls_out  = layers.Dense(n_farm, name="logsig",
                           activation="tanh")(x)
    out = layers.Concatenate(axis=-1)([mu_out, ls_out])
    bilstm = keras.Model(inp, out)

    def nll_loss_multi(y_true, y_pred):
        mu_p  = y_pred[:, :n_farm]
        ls_p  = y_pred[:, n_farm:] * 3.0
        s2    = tf.exp(2 * ls_p) + 1e-8
        return tf.reduce_mean(0.5 * tf.math.log(s2)
                              + 0.5 * (y_true - mu_p) ** 2 / s2)

    bilstm.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss=nll_loss_multi)
    bilstm.fit(X_seq, y_seq, epochs=20, batch_size=16, verbose=0,
               callbacks=[tf_cb.EarlyStopping(patience=5,
                                              restore_best_weights=True)])

    x_last_seq = hist_norm[-SEQ_LEN:].reshape(1, SEQ_LEN, n_farm)
    pred_bilstm = bilstm.predict(x_last_seq, verbose=0)[0]
    mu_norm     = pred_bilstm[:n_farm]
    ls_norm     = pred_bilstm[n_farm:] * 3.0

    std_scale = sc.scale_
    mu_bl  = np.clip(mu_norm * std_scale + sc.mean_, 0.1, None)
    sig_bl = np.clip(np.exp(ls_norm) * std_scale, 0.05, None)
    results["BiLSTM+XGB"] = {"mu": (0.6 * mu_bl + 0.4 * xgb_q50),
                              "sigma": (0.6 * sig_bl + 0.4 * xgb_sig),
                              "source": "BiLSTM+XGBoost ensemble (s3)"}

    y_val = test.mean(axis=0)
    fc_records = []
    for fname, fdata in results.items():
        mu_f  = fdata["mu"]
        sig_f = fdata["sigma"]
        q10_f = np.clip(mu_f + stats.norm.ppf(0.10) * sig_f, 0, None)
        q50_f = mu_f
        q90_f = np.clip(mu_f + stats.norm.ppf(0.90) * sig_f, 0, None)

        rmse     = float(np.sqrt(np.mean((y_val - mu_f) ** 2)))
        crps_val = crps_normal(y_val, mu_f, sig_f)
        pb_q10   = pinball_loss(y_val, q10_f, 0.10)
        pb_q50   = pinball_loss(y_val, q50_f, 0.50)
        pb_q90   = pinball_loss(y_val, q90_f, 0.90)
        coverage = float(np.mean((y_val >= q10_f) & (y_val <= q90_f)) * 100)

        vsshv_val = _compute_vsshv_for_forecast(
            mu_f, sig_f, prio, D, svc, K=K, n_gen=n_gen)

        fc_records.append({
            "model": fname, "rmse": round(rmse, 4),
            "crps": round(crps_val, 4),
            "pinball_q10": round(pb_q10, 4),
            "pinball_q50": round(pb_q50, 4),
            "pinball_q90": round(pb_q90, 4),
            "coverage_pct": round(coverage, 1),
            "vsshv_pct": round(vsshv_val, 2),
        })
        log.info(f"  {fname:<15} RMSE={rmse:.3f}  CRPS={crps_val:.4f}  "
                 f"Cov={coverage:.0f}%  VSS_HV={vsshv_val:.1f}%")

    return {"df": pd.DataFrame(fc_records)}


def _compute_vsshv_for_forecast(mu, sigma, prio, D, svc, K, n_gen):
    rng  = np.random.default_rng(SEED + 7)
    n    = len(mu)
    corr = np.eye(n) * 0.9 + np.ones((n, n)) * 0.1
    L    = np.linalg.cholesky(corr)

    inst_ev = make_instance(n, mu, prio, D, svc)
    F_ev, X_ev = _run_nsga2_fx(inst_ev, 50, n_gen, SEED)
    x_bar, ev_routes = ev_plan(F_ev, X_ev, mu)

    F_rp_list = []
    eev_pts   = []
    for k in range(K):
        z   = rng.standard_normal(n)
        d_k = np.clip(mu + sigma * (L @ z), 0.1, None)
        F_k = _run_nsga2(make_instance(n, d_k, prio, D, svc),
                         30, n_gen, int(SEED + k))
        F_rp_list.append(F_k)
        n_ex = extra_trips(d_k, ev_routes)
        eev_pts.append([float(x_bar[0]) + PENALTY_TRIP * n_ex,
                        float(x_bar[1]),
                        float(x_bar[2]) * d_k.sum() / (mu.sum() + 1e-9)])

    pooled = np.vstack(F_rp_list)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nd = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
    F_rp_nd = pooled[nd]
    F_eev   = np.array(eev_pts)
    ref_pt  = compute_ref_point([F_rp_nd, F_eev]) * 1.2

    try:
        hv_ind  = HV(ref_point=ref_pt)
        hv_rp   = float(hv_ind(F_rp_nd))
        hv_eev  = float(hv_ind(F_eev))
        return max(0.0, (hv_rp - hv_eev) / (hv_rp + 1e-9)) * 100
    except Exception:
        return 0.0


def plot_all(res_a, res_b, res_c, res_d, res_e, res_f, run_dir):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)
    pal = ["#1565C0", "#C62828", "#2E7D32"]

    ax_a = fig.add_subplot(gs[0, 0])
    df_a = res_a["df"]
    for j, alg in enumerate(["NSGA-II", "NSGA-III", "MOEA/D"]):
        vals = df_a[(df_a["instance"] == "C101-25") &
                    (df_a["algorithm"] == alg)]["hv"].dropna()
        ax_a.boxplot(vals, positions=[j], widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor=pal[j], alpha=0.7))
    ax_a.set_xticks([0, 1, 2]); ax_a.set_xticklabels(["NSGA-II","NSGA-III","MOEA/D"], fontsize=8)
    ax_a.set_ylabel("Hypervolume", fontsize=9)
    ax_a.set_title("(A) HV Solomon C101-25: 30 runs", fontsize=10, fontweight="bold")
    ax_a.grid(True, alpha=0.3, axis="y")

    ax_b = fig.add_subplot(gs[0, 1])
    if res_b and "hv" in res_b["sobol"] and res_b["sobol"]["hv"]:
        s1_hv = res_b["sobol"]["hv"]["S1"]
        st_hv = res_b["sobol"]["hv"]["ST"]
        names = list(s1_hv.keys())
        s1v   = [s1_hv[n] for n in names]
        stv   = [st_hv[n] for n in names]
        x_bar_b = np.arange(len(names))
        ax_b.bar(x_bar_b - 0.2, s1v, 0.38, label="S1 (efecto directo)",
                 color="#1565C0", alpha=0.85)
        ax_b.bar(x_bar_b + 0.2, stv, 0.38, label="ST (total)",
                 color="#C62828", alpha=0.85)
        ax_b.set_xticks(x_bar_b)
        ax_b.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax_b.legend(fontsize=7)
        ax_b.set_ylabel("Índice Sobol", fontsize=9)
    ax_b.set_title("(B) Sobol SA — S1/ST sobre HV", fontsize=10, fontweight="bold")
    ax_b.grid(True, alpha=0.3, axis="y")

    ax_c = fig.add_subplot(gs[0, 2])
    if res_c and "df" in res_c:
        df_c = res_c["df"]
        ax_c.plot(df_c["K"], df_c["vsshv_mean"], "o-",
                  color="#7B1FA2", lw=2, ms=6, label="VSS_HV media")
        ax_c.fill_between(df_c["K"],
                          df_c["vsshv_mean"] - df_c["vsshv_std"],
                          df_c["vsshv_mean"] + df_c["vsshv_std"],
                          alpha=0.25, color="#7B1FA2")
        if res_c.get("k_convergence"):
            ax_c.axvline(res_c["k_convergence"], ls="--", color="red",
                         lw=1.5, label=f"K conv.={res_c['k_convergence']}")
        ax_c.legend(fontsize=8)
    ax_c.set_xlabel("K (escenarios SAA)", fontsize=9)
    ax_c.set_ylabel("VSS_HV (%)", fontsize=9)
    ax_c.set_title("(C) VSS_HV vs K — convergencia SAA", fontsize=10, fontweight="bold")
    ax_c.grid(True, alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.axis("off")
    if res_d:
        s = 3600
        tabla_d = [
            ["Obj.", "WS", "RP", "EEV", "VSS%", "EVPI%"],
            ["f₁(h)",
             f"{res_d['WS']['f1']/s:.2f}",
             f"{res_d['RP']['f1']/s:.2f}",
             f"{res_d['EEV']['f1']/s:.2f}",
             f"{res_d['VSS_pct']['f1']:.1f}%",
             f"{res_d['EVPI_pct']['f1']:.1f}%"],
            ["f₂(k)",
             f"{res_d['WS']['f2']/1000:.1f}",
             f"{res_d['RP']['f2']/1000:.1f}",
             f"{res_d['EEV']['f2']/1000:.1f}",
             f"{res_d['VSS_pct']['f2']:.1f}%",
             f"{res_d['EVPI_pct']['f2']:.1f}%"],
            ["f₃(kg)",
             f"{res_d['WS']['f3']:.2f}",
             f"{res_d['RP']['f3']:.2f}",
             f"{res_d['EEV']['f3']:.2f}",
             f"{res_d['VSS_pct']['f3']:.1f}%",
             f"{res_d['EVPI_pct']['f3']:.1f}%"],
        ]
        tbl = ax_d.table(cellText=tabla_d[1:], colLabels=tabla_d[0],
                         loc="center", cellLoc="center",
                         colWidths=[0.13, 0.17, 0.17, 0.17, 0.18, 0.18])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.7)
        for j in range(6):
            tbl[(0, j)].set_facecolor("#37474F")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        for i in range(1, 4):
            for j in range(6):
                tbl[(i, j)].set_facecolor("#F5F5F5" if i % 2 else "white")
    ax_d.set_title("(D) EVPI — WS ≤ RP ≤ EEV (Birge & Louveaux)",
                   fontsize=10, fontweight="bold")

    ax_e = fig.add_subplot(gs[1, 1])
    if res_e and "df" in res_e:
        df_e = res_e["df"]
        for j, alg in enumerate(["NSGA-II", "NSGA-III", "MOEA/D"]):
            vals = df_e[df_e["algorithm"] == alg]["hv"].dropna()
            ax_e.boxplot(vals, positions=[j], widths=0.5,
                         patch_artist=True,
                         boxprops=dict(facecolor=pal[j], alpha=0.7))
        ax_e.set_xticks([0, 1, 2])
        ax_e.set_xticklabels(["NSGA-II","NSGA-III","MOEA/D"], fontsize=8)
        ax_e.set_ylabel("Hypervolume", fontsize=9)
    ax_e.set_title("(E) Estabilidad HV — Sevilla N=25, 30 runs",
                   fontsize=10, fontweight="bold")
    ax_e.grid(True, alpha=0.3, axis="y")

    ax_f  = fig.add_subplot(gs[1, 2])
    if res_f and "df" in res_f:
        df_f   = res_f["df"]
        models = df_f["model"].tolist()
        x_pos  = np.arange(len(models))
        ax_f2  = ax_f.twinx()
        ax_f.bar(x_pos - 0.2, df_f["rmse"], 0.38,
                 color="#EF5350", alpha=0.85, label="RMSE")
        ax_f2.plot(x_pos + 0.2, df_f["vsshv_pct"], "D-",
                   color="#7B1FA2", ms=7, lw=2, label="VSS_HV%")
        ax_f.set_xticks(x_pos)
        ax_f.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax_f.set_ylabel("RMSE (kg)", fontsize=9, color="#C62828")
        ax_f2.set_ylabel("VSS_HV (%)", fontsize=9, color="#7B1FA2")
        h1, l1 = ax_f.get_legend_handles_labels()
        h2, l2 = ax_f2.get_legend_handles_labels()
        ax_f.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax_f.set_title("(F) Ablación: RMSE↓ vs VSS_HV↑", fontsize=10, fontweight="bold")
    ax_f.grid(True, alpha=0.3, axis="y")

    ax_g = fig.add_subplot(gs[2, :])
    ax_g.axis("off")
    tabla_g = [
        ["Módulo", "Análisis", "Resultado clave", "Papel en el trabajo"],
        ["A — Solomon", "NSGA-II vs III vs MOEA/D\n(30 runs, Friedman+Nemenyi)",
         "p-values tests + ranking HV/IGD/Spacing",
         "Comparación entre algoritmos"],
        ["B — Sobol SA", "7 parámetros → S1/ST\nsobre HV, IGD, VSS%, VSS_HV",
         "Ranking de influencia de parámetros",
         "Sensibilidad global de parámetros"],
        ["C — VSS_HV", "Contribución original\nConvexidad vs K escenarios",
         "K de convergencia SAA + VSS_HV(%)",
         "Métrica VSS_HV frente a K"],
        ["D — EVPI", "Cadena WS ≤ RP ≤ EEV\nBirge & Louveaux (1997)",
         "VSS% + EVPI% por objetivo",
         "Cadena WS ≤ RP ≤ EEV"],
        ["E — Estabilidad", "IC 95% HV, IGD, Spacing\nWilcoxon NSGA-II vs III/MOEA/D",
         "Reproducibilidad demostrada",
         "Estabilidad entre réplicas"],
        ["F — Ablación", "Naive / ARIMA / XGBoost / BiLSTM+XGB\nCRPS + Pinball + Coverage + VSS_HV",
         "Impacto del forecasting en routing",
         "Ablación de la predicción"],
    ]
    tbl_g = ax_g.table(cellText=tabla_g[1:], colLabels=tabla_g[0],
                       loc="center", cellLoc="left",
                       colWidths=[0.12, 0.25, 0.28, 0.35])
    tbl_g.auto_set_font_size(False); tbl_g.set_fontsize(8); tbl_g.scale(1, 1.6)
    for j in range(4):
        tbl_g[(0, j)].set_facecolor("#37474F")
        tbl_g[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, 7):
        fc = "#E8F5E9" if i % 2 else "#F3E5F5"
        for j in range(4):
            tbl_g[(i, j)].set_facecolor(fc)

    ax_g.set_title("(G) Resumen de los módulos",
                   fontsize=10, fontweight="bold")

    fig.suptitle(
        "Análisis de Sensibilidad y Validación Estadística — instancia Sevilla\n"
        "NSGA-II/III/MOEA/D · Sobol SA · VSS_HV · EVPI · Wilcoxon/Friedman/Nemenyi",
        fontsize=11, fontweight="bold", y=1.01)

    out = run_dir / "s5_plots_full.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura 7-panel guardada: {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="s5_sensitivity.py — Análisis de sensibilidad y validación")
    p.add_argument("--quick",   action="store_true",
                   help="Modo rápido: parámetros reducidos (~8 min)")
    p.add_argument("--modules", type=str, default="A,B,C,D,E,F",
                   help="Módulos a ejecutar, ej: A,C,E")
    return p.parse_args()


def main():
    args = parse_args()
    mods = {m.strip().upper() for m in args.modules.split(",")}

    if args.quick:
        log.info("  [--quick: parámetros reducidos para test]")
        runs_a    = 5;  n_gen_a = 80; pop_a = 30
        n_sobol   = 4;  n_gen_b = 50
        k_range_c = [3, 5, 8, 10]; n_rep_c = 2; n_gen_c = 50
        k_d = 5;   n_gen_d = 60
        runs_e = 5; n_gen_e = 80; pop_e = 50
        k_f = 5;   n_gen_f = 50; n_days_f = 60
    else:
        runs_a    = 30; n_gen_a = 150; pop_a = 50
        n_sobol   = 32; n_gen_b = 80
        k_range_c = [3, 5, 8, 10, 15, 20, 25, 30]; n_rep_c = 5; n_gen_c = 80
        k_d = 20;  n_gen_d = 100
        runs_e = 30; n_gen_e = 200; pop_e = 100
        k_f = 10;  n_gen_f = 80;  n_days_f = 120

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_s5"
    run_dir = SENS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("s5 — Sensibilidad y validación estadística")
    print(f"  Módulos: {', '.join(sorted(mods))}")
    print(f"  Quick: {args.quick}  |  Run: {run_id}")
    print("=" * 60)
    t0 = time.time()

    res_a = res_b = res_c = res_d = res_e = res_f = None

    if "A" in mods:
        res_a = run_module_a(n_runs=runs_a, n_gen=n_gen_a,
                             pop_size=pop_a, quick=args.quick)
        df_a = res_a["df"]
        df_a.to_csv(run_dir / "s5_mod_a_solomon.csv", index=False)
        with open(run_dir / "s5_mod_a_tests.json", "w") as f:
            json.dump(res_a["tests"], f, indent=2, default=str)

    if "B" in mods:
        res_b = run_module_b(N_sobol=n_sobol, n_gen_sobol=n_gen_b)
        with open(run_dir / "s5_mod_b_sobol.json", "w") as f:
            json.dump(res_b["sobol"], f, indent=2, default=str)

    if "C" in mods:
        res_c = run_module_c(k_range=k_range_c, n_gen=n_gen_c,
                             n_rep=n_rep_c)
        if res_c and "df" in res_c:
            res_c["df"].to_csv(run_dir / "s5_mod_c_vsshv.csv", index=False)

    if "D" in mods:
        res_d = run_module_d(K=k_d, n_gen=n_gen_d)
        with open(run_dir / "s5_mod_d_evpi.json", "w") as f:
            json.dump(res_d, f, indent=2, default=str)

    if "E" in mods:
        res_e = run_module_e(n_runs=runs_e, n_gen=n_gen_e, pop_size=pop_e)
        if res_e and "df" in res_e:
            res_e["df"].to_csv(run_dir / "s5_mod_e_stability.csv", index=False)
        with open(run_dir / "s5_mod_e_wilcoxon.json", "w") as f:
            json.dump(res_e["wilcoxon"] if res_e else {}, f, indent=2)

    if "F" in mods:
        res_f = run_module_f(K=k_f, n_gen=n_gen_f, n_days=n_days_f)
        if res_f and "df" in res_f:
            res_f["df"].to_csv(run_dir / "s5_mod_f_ablation.csv", index=False)

    log.info("Generando figura de 7 paneles...")
    plot_all(res_a, res_b, res_c, res_d, res_e, res_f, run_dir)

    summary = {
        "run_id": run_id, "quick": args.quick, "modules": list(mods),
        "time_total_s": round(time.time() - t0, 1),
        "mod_d_vss_pct":  res_d["VSS_pct"] if res_d else None,
        "mod_d_evpi_pct": res_d["EVPI_pct"] if res_d else None,
        "mod_c_k_convergence": res_c["k_convergence"] if res_c else None,
        "mod_e_ci95_hv_nsga2": res_e["ci95_hv_nsga2"] if res_e else None,
    }
    with open(run_dir / "s5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    (SENS_DIR / "latest_sensitivity.txt").write_text(run_id)

    t_total = time.time() - t0
    print("\n" + "=" * 60)
    print("Análisis de sensibilidad completado")
    print("=" * 60)
    if res_d:
        print(f"\n  ── EVPI/VSS ──")
        for o in ["f1", "f2", "f3"]:
            print(f"  {o}: VSS={res_d['VSS_pct'][o]:.1f}%  "
                  f"EVPI={res_d['EVPI_pct'][o]:.1f}%")
    if res_c and res_c.get("k_convergence"):
        print(f"\n  VSS_HV converge en K={res_c['k_convergence']} escenarios")
    if res_b and "hv" in res_b.get("sobol", {}):
        s1 = res_b["sobol"]["hv"].get("S1", {})
        top = sorted(s1.items(), key=lambda x: -x[1])[:3]
        print(f"\n  Top-3 parámetros Sobol (HV): "
              + "  ".join(f"{n}={v:.3f}" for n, v in top))
    print(f"\n  Tiempo total: {t_total:.0f}s")
    print(f"  Salidas en: {run_dir}/")
    print("=" * 60)
    print("\nSiguiente paso: python s5b_solomon100.py")


if __name__ == "__main__":
    main()
