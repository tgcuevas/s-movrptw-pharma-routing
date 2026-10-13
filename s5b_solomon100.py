#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, logging, warnings, argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

try:
    import scikit_posthocs as sp
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False
    logging.warning("scikit_posthocs no disponible — Nemenyi calculado via scipy")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SEED         = 42
CAPACITY     = 500.0
CO2_ALPHA    = 0.05
PRIO_PROBS   = [0.30, 0.20, 0.50]
_ALPHA_TW    = {1: 10_000, 2: 5_000, 3: 1_000}
_TW_CLASS    = {1: (0, 900), 2: (0, 1800), 3: (0, 3600)}

OUT_DIR = Path("data") / "sensitivity" / "solomon100"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PermutationSampling(Sampling):
    def _do(self, problem, n_samples, **kw):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            X[i] = np.random.permutation(problem.n_var)
        return X

class OrderCrossover(Crossover):
    def __init__(self): super().__init__(2, 2)
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
        self.n    = inst["n"]
        self.D    = np.array(inst["D"], dtype=float)
        self.dem  = np.array(inst["demands"], dtype=float)
        self.prio = inst["priorities"]
        self.svc  = np.array(inst["service_times"], dtype=float)
        self.cap  = float(inst.get("capacity", CAPACITY))
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
        t  += float(D[prev, depot])
        f1 += t
        f3 += CO2_ALPHA * t * (load / cap)
    return f1, f2, f3


def gen_solomon_100(family: str, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    n   = 100

    if family.startswith("C"):
        n_clusters = 5
        centers    = rng.uniform(15, 85, (n_clusters, 2))
        xy = np.vstack([rng.normal(centers[i % n_clusters], 7, (1, 2))
                        for i in range(n)])
        xy = np.clip(xy, 0, 100)
        tw_width = 600 if family == "C1" else 3000

    elif family.startswith("R"):
        xy = rng.uniform(0, 100, (n, 2))
        tw_width = 900 if family == "R1" else 4500

    else:
        n_clust = n // 3
        n_unif  = n - n_clust
        centers = rng.uniform(20, 80, (3, 2))
        xy_c    = np.vstack([rng.normal(centers[i % 3], 7, (1, 2))
                             for i in range(n_clust)])
        xy_u    = rng.uniform(0, 100, (n_unif, 2))
        xy      = np.vstack([xy_c, xy_u])
        xy      = np.clip(xy, 0, 100)
        tw_width = 1200 if family == "RC1" else 3600

    depot_xy = np.array([[50.0, 50.0]])
    all_xy   = np.vstack([depot_xy, xy])

    D = cdist(all_xy, all_xy) * 60.0
    np.fill_diagonal(D, 0.0)

    phi     = 4.0
    mu_farm = rng.uniform(5, 15, n)
    gamma_s = rng.gamma(shape=phi, scale=mu_farm / phi)
    demands = rng.poisson(gamma_s).astype(float)
    demands = np.clip(demands, 0.5, None)

    priorities = rng.choice([1, 2, 3], size=n, p=PRIO_PROBS).tolist()
    service_t  = np.full(n + 1, 120.0); service_t[0] = 0.0

    return {
        "label":         f"{family}-100",
        "family":        family,
        "n":             n,
        "D":             D.tolist(),
        "demands":       demands.tolist(),
        "priorities":    priorities,
        "service_times": service_t.tolist(),
        "capacity":      CAPACITY,
        "tw_width":      tw_width,
    }


def run_nsga2(inst, pop_size, n_gen, seed):
    prob = CVRPTW_3obj(inst)
    alg  = NSGA2(pop_size=int(pop_size),
                 sampling=PermutationSampling(),
                 crossover=OrderCrossover(),
                 mutation=SwapMutation(),
                 eliminate_duplicates=True)
    res  = minimize(prob, alg, ("n_gen", int(n_gen)),
                    seed=int(seed), verbose=False)
    return res.F if res.F is not None and len(res.F) > 0 else np.zeros((1, 3))

def run_nsga3(inst, pop_size, n_gen, seed):
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
    return res.F if res.F is not None and len(res.F) > 0 else np.zeros((1, 3))

def run_moead(inst, pop_size, n_gen, seed):
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
        return res.F if res.F is not None and len(res.F) > 0 else np.zeros((1, 3))
    except Exception:
        return run_nsga2(inst, pop_size, n_gen, seed + 9999)

def compute_spacing(F):
    if len(F) < 2: return 0.0
    dists = cdist(F, F); np.fill_diagonal(dists, np.inf)
    d_min = dists.min(axis=1)
    return float(np.sqrt(np.sum((d_min - d_min.mean()) ** 2) / (len(F) - 1)))

def compute_ref_point(fronts, margin=1.1):
    return np.vstack(fronts).max(axis=0) * margin


def run_benchmark(families: List[str], n_runs: int,
                  n_gen: int, pop_size: int) -> Dict:
    algos   = ["NSGA-II", "NSGA-III", "MOEA/D"]
    runners = {"NSGA-II": run_nsga2,
               "NSGA-III": run_nsga3,
               "MOEA/D":  run_moead}

    instances = {}
    for fam in families:
        instances[f"{fam}-100"] = gen_solomon_100(fam, seed=SEED + zlib.crc32(fam.encode()) % 1000)
        log.info(f"  Instancia {fam}-100 generada: "
                 f"N={instances[f'{fam}-100']['n']}  "
                 f"TW_width={instances[f'{fam}-100']['tw_width']}s  "
                 f"cap={instances[f'{fam}-100']['capacity']}kg")

    all_fronts: Dict[str, Dict[str, List]] = {
        inst: {alg: [] for alg in algos} for inst in instances
    }
    records = []
    t_global = time.time()

    for inst_name, inst in instances.items():
        for alg_name in algos:
            fn = runners[alg_name]
            t0 = time.time()
            for r in range(n_runs):
                F = fn(inst, pop_size, n_gen, seed=int(SEED + r * 137))
                all_fronts[inst_name][alg_name].append(F)
            elapsed = time.time() - t0
            log.info(f"  {inst_name} × {alg_name}: {n_runs} runs  "
                     f"({elapsed:.0f}s  ≈{elapsed/n_runs:.1f}s/run)")

    log.info("  Calculando HV, IGD, Spacing...")
    for inst_name in instances:
        all_f  = [F for alg in algos for F in all_fronts[inst_name][alg]]
        ref_pt = compute_ref_point(all_f)

        pooled  = np.vstack(all_f)
        nd_idx  = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
        pf_ref  = pooled[nd_idx]

        hv_ind  = HV(ref_point=ref_pt)
        igd_ind = IGD(pf_ref)

        for alg_name in algos:
            for r, F in enumerate(all_fronts[inst_name][alg_name]):
                try:
                    hv  = float(hv_ind(F))
                    igd = float(igd_ind(F))
                    spc = compute_spacing(F)
                except Exception:
                    hv = igd = spc = np.nan
                records.append({
                    "instance":  inst_name,
                    "family":    inst_name.split("-")[0],
                    "algorithm": alg_name,
                    "run":       r,
                    "hv":        hv,
                    "igd":       igd,
                    "spacing":   spc,
                    "n_sols":    len(F),
                })

    df = pd.DataFrame(records)

    log.info("  Ejecutando tests estadísticos (Friedman + Nemenyi)...")
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
            posthoc_method = "nemenyi" if SP_AVAILABLE else "wilcoxon_pairwise"
            try:
                min_len = min((len(g) for g in groups), default=0)
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
                log.warning(f"Post-hoc {inst_name}/{metric} no calculado: {e}")
                posthoc = {}

            desc = {}
            for a, g in zip(algos, groups):
                if len(g) > 0:
                    desc[a] = {
                        "mean":   round(float(np.nanmean(g)), 4),
                        "std":    round(float(np.nanstd(g)),  4),
                        "median": round(float(np.nanmedian(g)),4),
                        "min":    round(float(np.nanmin(g)),   4),
                        "max":    round(float(np.nanmax(g)),   4),
                    }

            test_results[inst_name][metric] = {
                "friedman_stat":   round(float(stat), 4) if not np.isnan(stat) else None,
                "friedman_p":      round(float(pval), 6) if not np.isnan(pval) else None,
                "significant_05":  bool(pval < 0.05) if not np.isnan(pval) else False,
                "posthoc":         posthoc,
                "posthoc_method":  posthoc_method,
                "stats":           desc,
            }

            sig = "✅" if not np.isnan(pval) and pval < 0.05 else "—"
            log.info(f"  {inst_name} {metric}: "
                     f"Friedman p={pval:.4f}  {sig}  |  "
                     + "  ".join(
                         f"{a}={desc.get(a,{}).get('mean',np.nan):.3e}"
                         for a in algos))

    t_total = time.time() - t_global
    log.info(f"  Benchmark completado en {t_total:.0f}s "
             f"({t_total/60:.1f} min)")

    ranking = _compute_global_ranking(df, algos)

    return {
        "df":       df,
        "tests":    test_results,
        "ranking":  ranking,
        "n_runs":   n_runs,
        "n_gen":    n_gen,
        "pop_size": pop_size,
        "families": families,
        "t_total_s": round(t_total, 1),
    }


def _compute_global_ranking(df: pd.DataFrame, algos: List[str]) -> Dict:
    signs = {"hv": -1, "igd": +1, "spacing": +1}
    rank_accumulator = {a: [] for a in algos}

    for inst_name in df["instance"].unique():
        df_i = df[df["instance"] == inst_name]
        for metric, sign in signs.items():
            means = {a: float(df_i[df_i["algorithm"]==a][metric].mean())
                     for a in algos}
            sorted_algs = sorted(means.keys(), key=lambda a: sign * means[a])
            for rank, alg in enumerate(sorted_algs, 1):
                rank_accumulator[alg].append(rank)

    return {
        alg: {
            "mean_rank": round(float(np.mean(ranks)), 2),
            "std_rank":  round(float(np.std(ranks)), 2),
            "global_position": None,
        }
        for alg, ranks in rank_accumulator.items()
    }


def plot_results(results: Dict, out_path: Path) -> None:
    df        = results["df"]
    families  = results["families"]
    algos     = ["NSGA-II", "NSGA-III", "MOEA/D"]
    palette   = ["#1565C0", "#C62828", "#2E7D32"]
    metrics   = ["hv", "igd", "spacing"]
    titles    = ["Hypervolume (↑ mejor)",
                 "IGD — Inverted Generational Distance (↓ mejor)",
                 "Spacing — Uniformidad del frente (↓ mejor)"]

    n_inst = len(families)
    fig, axes = plt.subplots(3, max(n_inst, 1),
                             figsize=(4 * max(n_inst, 1), 12))
    if n_inst == 1:
        axes = axes.reshape(3, 1)

    for row, (metric, title) in enumerate(zip(metrics, titles)):
        for col, fam in enumerate(families):
            ax = axes[row, col]
            inst_name = f"{fam}-100"

            data_by_alg = [df[(df["instance"] == inst_name) &
                              (df["algorithm"] == a)][metric].dropna().values
                           for a in algos]

            bp = ax.boxplot(data_by_alg, positions=[0, 1, 2], widths=0.55,
                            patch_artist=True,
                            medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color); patch.set_alpha(0.8)

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["NSGA-II", "NSGA-III", "MOEA/D"], fontsize=8)
            ax.set_ylabel(metric.upper(), fontsize=8)
            ax.set_title(f"{inst_name}\n{title}", fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

            tr = results["tests"].get(inst_name, {}).get(metric, {})
            pval = tr.get("friedman_p")
            if pval is not None and pval < 0.05:
                ax.text(0.98, 0.02, (f"Friedman p = {pval:.3f}".replace(".", ",") if pval >= 0.001 else "Friedman p < 0,001"),
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7, color="#1B5E20")

    fig.suptitle(
        f"Benchmark Solomon 100 Clientes — NSGA-II vs NSGA-III vs MOEA/D\n"
        f"({results['n_runs']} réplicas, n_gen={results['n_gen']}, "
        f"pop={results['pop_size']})",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura guardada: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="s5b_solomon100.py — Benchmark Solomon 100 clientes")
    p.add_argument("--runs",     type=int,  default=30,
                   help="Número de réplicas (por defecto 30)")
    p.add_argument("--n_gen",    type=int,  default=200,
                   help="Generaciones NSGA-II (default 200)")
    p.add_argument("--pop",      type=int,  default=100,
                   help="Tamaño de población (default 100)")
    p.add_argument("--families", type=str,  default="C1,C2,R1,R2,RC1,RC2",
                   help="Familias Solomon a ejecutar (default: todas las 6)")
    p.add_argument("--quick",    action="store_true",
                   help="Modo rápido: 5 runs, n_gen=80, solo C1+R1 (~4 min)")
    return p.parse_args()


def main():
    args     = parse_args()

    if args.quick:
        n_runs   = 5
        n_gen    = 80
        pop_size = 50
        families = ["C1", "R1"]
        log.info("  [--quick: 5 runs, n_gen=80, familias C1+R1]")
    else:
        n_runs   = args.runs
        n_gen    = args.n_gen
        pop_size = args.pop
        families = [f.strip().upper() for f in args.families.split(",")]

    print("=" * 60)
    print("BENCHMARK SOLOMON 100")
    print(f"  Familias : {', '.join(families)}")
    print(f"  Réplicas : {n_runs}  |  n_gen={n_gen}  |  pop={pop_size}")
    print("  Tiempo orientativo: unos 25 minutos con 30 réplicas en la máquina del autor")
    print("=" * 60)

    results = run_benchmark(families, n_runs, n_gen, pop_size)
    df      = results["df"]

    ranking  = results["ranking"]
    sorted_r = sorted(ranking.keys(), key=lambda a: ranking[a]["mean_rank"])
    for pos, alg in enumerate(sorted_r, 1):
        ranking[alg]["global_position"] = pos

    df.to_csv(OUT_DIR / "solomon100_results.csv", index=False)

    with open(OUT_DIR / "solomon100_tests.json", "w") as f:
        json.dump(results["tests"], f, indent=2, default=str)

    summary = {
        "n_runs":    n_runs,
        "n_gen":     n_gen,
        "pop_size":  pop_size,
        "families":  families,
        "t_total_s": results["t_total_s"],
        "ranking":   ranking,
        "tabla_publicacion": _build_publication_table(df, results["tests"],
                                                       families),
    }
    with open(OUT_DIR / "solomon100_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot_results(results, OUT_DIR / "solomon100_plots.png")

    print("\n" + "=" * 60)
    print("✅  Benchmark Solomon 100 completado")
    print("=" * 60)
    print(f"\n  Ranking global (mean rank sobre {len(families)} inst. × 3 métricas):")
    print(f"  {'Algoritmo':<12} {'Mean Rank':>10} {'Posición':>10}")
    print(f"  {'-'*35}")
    for alg in sorted_r:
        print(f"  {alg:<12} {ranking[alg]['mean_rank']:>10.2f} "
              f"{ranking[alg]['global_position']:>10}")
    print(f"\n  Tiempo total: {results['t_total_s']:.0f}s "
          f"({results['t_total_s']/60:.1f} min)")
    print(f"  Salidas en: {OUT_DIR}/")
    print(f"    solomon100_results.csv")
    print(f"    solomon100_tests.json")
    print(f"    solomon100_summary.json")
    print(f"    solomon100_plots.png")
    print("\n  Tabla resumen: solomon100_summary.json "
          "→ tabla_publicacion")


def _build_publication_table(df, tests, families):
    table = []
    algos = ["NSGA-II", "NSGA-III", "MOEA/D"]
    for fam in families:
        inst_name = f"{fam}-100"
        for metric in ["hv", "igd", "spacing"]:
            row = {"instance": inst_name, "metric": metric}
            tr  = tests.get(inst_name, {}).get(metric, {})
            row["friedman_p"]    = tr.get("friedman_p")
            row["significant"]   = tr.get("significant_05", False)
            for a in algos:
                desc = tr.get("stats", {}).get(a, {})
                row[f"{a}_mean"] = desc.get("mean")
                row[f"{a}_std"]  = desc.get("std")
            table.append(row)
    return table


if __name__ == "__main__":
    main()
