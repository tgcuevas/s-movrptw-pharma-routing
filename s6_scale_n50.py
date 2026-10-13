#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, json, logging, warnings, argparse, subprocess
from datetime import datetime
from pathlib import Path
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
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

BASE_DIR     = Path("data")
SCALE_DIR    = BASE_DIR / "scale_n50"
FORECAST_DIR = BASE_DIR / "forecast"
INTEG_DIR    = BASE_DIR / "integration"
SENS_DIR     = BASE_DIR / "sensitivity"
SCALE_DIR.mkdir(parents=True, exist_ok=True)

SEED         = 42
CAPACITY     = 200.0
CO2_ALPHA    = 0.05
PENALTY_TRIP = 1800.0
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
        self.n   = inst["n"]
        self.D   = np.array(inst["D"], dtype=float)
        self.dem = np.array(inst["demands"], dtype=float)
        self.prio = inst["priorities"]
        self.svc = np.array(inst["service_times"], dtype=float)
        self.cap = float(inst.get("capacity", CAPACITY))
        super().__init__(n_var=self.n, n_obj=3, xl=0, xu=self.n - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, f3 = _eval_route(x.astype(int), self.dem, self.D,
                                  self.svc, self.prio, self.cap)
        out["F"] = [f1, f2, f3]

def _eval_route(order, dem, D, svc, prio, cap):
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
        f1 += t; f3 += CO2_ALPHA * t * (load / cap)
    return f1, f2, f3


def run_s3_n50(quick: bool = False) -> Tuple[str, Dict]:
    log.info("PASO 1/5 — Ejecutando s3_forecast_pipeline.py --n 50")
    t0 = time.time()

    cmd = [sys.executable, "s3_forecast_pipeline.py",
           "--n", "50", "--k", "20"]
    if quick:
        cmd = [sys.executable, "s3_forecast_pipeline.py",
               "--quick"]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=900)

    if result.returncode != 0:
        log.error(f"  s3 falló:\n{result.stderr[-500:]}")
        raise RuntimeError("s3_forecast_pipeline.py falló")

    latest = FORECAST_DIR / "latest_run.txt"
    run_id_s3 = latest.read_text().strip()

    run_dir = FORECAST_DIR / run_id_s3
    with open(run_dir / "forecast_metrics.json") as f:
        metrics = json.load(f)
    with open(run_dir / "forecast_config.json") as f:
        config  = json.load(f)

    n_farm  = config.get("n_pharmacies", "?")
    k_scen  = config.get("k_scenarios", "?")
    elapsed = time.time() - t0

    log.info(f"  s3 completado: N={n_farm}, K={k_scen}  ({elapsed:.0f}s)")
    log.info(f"  Run s3: {run_id_s3}")
    log.info(f"  Skill BiLSTM (proxy heurístico): {metrics.get('bilstm',{}).get('skill_q50_vs_naive',0):.1f}%")
    log.info(f"  Cobertura [Q10,Q90]: {metrics.get('calibration',{}).get('empirical_coverage_pct',0):.1f}%")

    return run_id_s3, metrics, config


def run_s4_n50(quick: bool = False) -> Tuple[str, Dict]:
    log.info("PASO 2/5 — Ejecutando s4_integration.py (N=50)")
    t0 = time.time()

    cmd = [sys.executable, "s4_integration.py", "--n", "50"]
    if quick:
        cmd = [sys.executable, "s4_integration.py", "--quick", "--n", "25"]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=900)

    if result.returncode != 0:
        log.error(f"  s4 falló:\n{result.stderr[-500:]}")
        raise RuntimeError("s4_integration.py falló")

    latest = INTEG_DIR / "latest_integration.txt"
    run_id_s4 = latest.read_text().strip()
    run_dir   = INTEG_DIR / run_id_s4

    with open(run_dir / "s4_summary.json") as f:
        summary = json.load(f)

    elapsed = time.time() - t0
    vss = summary.get("VSS", {})
    log.info(f"  s4 completado ({elapsed:.0f}s)")
    log.info(f"  Run s4: {run_id_s4}")
    log.info(f"  VSS f1={vss.get('VSS_pct',{}).get('f1',0):.1f}%  "
             f"f2={vss.get('VSS_pct',{}).get('f2',0):.1f}%  "
             f"f3={vss.get('VSS_pct',{}).get('f3',0):.1f}%")

    return run_id_s4, summary


def run_stability_n50(n_runs: int = 30, n_gen: int = 200,
                      pop_size: int = 100, quick: bool = False) -> pd.DataFrame:
    if quick:
        n_runs = 5; n_gen = 80; pop_size = 50

    log.info(f"PASO 3/5 — Estabilidad NSGA-II N=50 ({n_runs} réplicas, "
             f"n_gen={n_gen})")
    t0 = time.time()

    rng = np.random.default_rng(SEED)
    n   = 50
    raw = np.exp(rng.normal(6.3, 0.35, (n + 1, n + 1)))
    D   = (raw + raw.T) / 2.0; np.fill_diagonal(D, 0.0)
    D[0, 1:] += 300.0; D[1:, 0] += 300.0

    svc   = np.full(n + 1, 180.0); svc[0] = 0.0
    prios = rng.choice([1, 2, 3], size=n, p=PRIO_PROBS).tolist()
    mu    = rng.uniform(3, 18, n)
    phi   = 4.0
    dem   = np.clip(rng.gamma(phi, mu / phi), 0.5, None)

    inst = {"n": n, "D": D.tolist(), "demands": dem.tolist(),
            "priorities": prios, "service_times": svc.tolist(),
            "capacity": CAPACITY}

    F_pre = []
    for r in range(min(5, n_runs)):
        prob = CVRPTW_3obj(inst)
        alg  = NSGA2(pop_size=int(pop_size),
                     sampling=PermutationSampling(),
                     crossover=OrderCrossover(),
                     mutation=SwapMutation(),
                     eliminate_duplicates=True)
        res  = minimize(prob, alg, ("n_gen", int(n_gen)),
                        seed=int(SEED + r * 7), verbose=False)
        if res.F is not None: F_pre.append(res.F)

    pooled  = np.vstack(F_pre)
    nd_idx  = NonDominatedSorting().do(pooled, only_non_dominated_front=True)
    pf_ref  = pooled[nd_idx]
    ref_pt  = pooled.max(axis=0) * 1.1
    hv_ind  = HV(ref_point=ref_pt)
    igd_ind = IGD(pf_ref)

    records = []
    for r in range(n_runs):
        prob = CVRPTW_3obj(inst)
        alg  = NSGA2(pop_size=int(pop_size),
                     sampling=PermutationSampling(),
                     crossover=OrderCrossover(),
                     mutation=SwapMutation(),
                     eliminate_duplicates=True)
        res  = minimize(prob, alg, ("n_gen", int(n_gen)),
                        seed=int(SEED + r * 17), verbose=False)
        F = res.F if res.F is not None else np.zeros((1, 3))
        try:
            hv  = float(hv_ind(F))
            igd = float(igd_ind(F))
            if len(F) >= 2:
                dists = cdist(F, F); np.fill_diagonal(dists, np.inf)
                d_min = dists.min(axis=1)
                sp = float(np.sqrt(np.sum((d_min - d_min.mean())**2)
                                   / (len(F) - 1)))
            else:
                sp = 0.0
        except Exception:
            hv = igd = sp = np.nan
        records.append({"run": r, "n_farm": 50,
                         "hv": hv, "igd": igd, "spacing": sp,
                         "n_sols": len(F)})

    df = pd.DataFrame(records)
    elapsed = time.time() - t0

    hv_vals = df["hv"].dropna().values
    ci95 = stats.t.interval(0.95, len(hv_vals)-1,
                             loc=np.mean(hv_vals),
                             scale=stats.sem(hv_vals))
    cv = np.std(hv_vals, ddof=1) / np.mean(hv_vals) * 100

    log.info(f"  N=50 completado ({elapsed:.0f}s)")
    log.info(f"  HV = {np.mean(hv_vals):.4e} ± {np.std(hv_vals):.4e}  "
             f"CV={cv:.1f}%  IC95%=[{ci95[0]:.4e}, {ci95[1]:.4e}]")
    log.info(f"  IGD = {df['igd'].mean():.4e}  "
             f"Spacing = {df['spacing'].mean():.4e}")

    return df, inst


def build_comparison_table(
    summary_s4_n25: Optional[Dict],
    summary_s4_n50: Dict,
    metrics_s3_n25: Optional[Dict],
    metrics_s3_n50: Dict,
    df_stab_n25:    Optional[pd.DataFrame],
    df_stab_n50:    pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for label, summ in [("N=25", summary_s4_n25), ("N=50", summary_s4_n50)]:
        if summ is None:
            continue
        vss = summ.get("VSS", {})
        vss_pct = vss.get("VSS_pct", {})
        rows.append({
            "Dimensión":    "VSS f₁ — Tiempo ruta (%)",
            label:          f"{vss_pct.get('f1', 0):.1f}%",
        })
        rows.append({
            "Dimensión":    "VSS f₂ — Tardanza ponderada (%)",
            label:          f"{vss_pct.get('f2', 0):.1f}%",
        })
        rows.append({
            "Dimensión":    "VSS f₃ — CO₂ (%)",
            label:          f"{vss_pct.get('f3', 0):.1f}%",
        })
        ev_comp = summ.get("EV_compromise", {})
        rows.append({
            "Dimensión":    "f₁ compromiso EV (h)",
            label:          f"{ev_comp.get('f1_h', 0):.2f}",
        })
        rows.append({
            "Dimensión":    "K escenarios SAA",
            label:          str(summ.get("k_scenarios_used", "?")),
        })

    for label, met in [("N=25", metrics_s3_n25), ("N=50", metrics_s3_n50)]:
        if met is None:
            continue
        rows.append({
            "Dimensión":    "Skill BiLSTM vs Naive (%, proxy heurístico)",
            label:          f"{met.get('bilstm',{}).get('skill_q50_vs_naive',0):.1f}%",
        })
        rows.append({
            "Dimensión":    "Cobertura [Q10,Q90] (%)",
            label:          f"{met.get('calibration',{}).get('empirical_coverage_pct',0):.1f}%",
        })
        rows.append({
            "Dimensión":    "BiLSTM NLL val. loss",
            label:          f"{met.get('bilstm',{}).get('val_nll_loss',0):.4f}",
        })

    for label, df_s in [("N=25", df_stab_n25), ("N=50", df_stab_n50)]:
        if df_s is None:
            continue
        hv_v  = df_s["hv"].dropna().values
        igd_v = df_s["igd"].dropna().values
        sp_v  = df_s["spacing"].dropna().values
        cv    = np.std(hv_v, ddof=1) / (np.mean(hv_v) + 1e-12) * 100
        rows.append({
            "Dimensión":    "HV NSGA-II (media ± σ)",
            label:          f"{np.mean(hv_v):.3e} ± {np.std(hv_v):.3e}",
        })
        rows.append({
            "Dimensión":    "IGD NSGA-II (media)",
            label:          f"{np.mean(igd_v):.3e}",
        })
        rows.append({
            "Dimensión":    "HV CV (%)",
            label:          f"{cv:.1f}%",
        })

    df_comp = pd.DataFrame(rows)
    if "N=25" in df_comp.columns and "N=50" in df_comp.columns:
        df_comp = df_comp.groupby("Dimensión").first().reset_index()
    return df_comp


def plot_comparison(df_stab_n25: Optional[pd.DataFrame],
                    df_stab_n50: pd.DataFrame,
                    summary_s4_n25: Optional[Dict],
                    summary_s4_n50: Dict,
                    out_path: Path) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    pal = ["#1565C0", "#C62828"]

    ax_a = fig.add_subplot(gs[0, 0])
    data_hv = []
    labels_hv = []
    if df_stab_n25 is not None:
        data_hv.append(df_stab_n25["hv"].dropna().values)
        labels_hv.append("N=25")
    data_hv.append(df_stab_n50["hv"].dropna().values)
    labels_hv.append("N=50")

    bp_a = ax_a.boxplot(data_hv, positions=range(len(data_hv)),
                        widths=0.5, patch_artist=True)
    for patch, color in zip(bp_a["boxes"], pal[:len(data_hv)]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax_a.set_xticks(range(len(data_hv)))
    ax_a.set_xticklabels(labels_hv, fontsize=10)
    ax_a.set_ylabel("Hypervolume", fontsize=9)
    ax_a.set_title("(A) HV NSGA-II: N=25 vs N=50 (30 réplicas)",
                   fontsize=10, fontweight="bold")
    ax_a.grid(True, alpha=0.3, axis="y")

    ax_b = fig.add_subplot(gs[0, 1])
    data_igd = []
    if df_stab_n25 is not None:
        data_igd.append(df_stab_n25["igd"].dropna().values)
    data_igd.append(df_stab_n50["igd"].dropna().values)

    bp_b = ax_b.boxplot(data_igd, positions=range(len(data_igd)),
                        widths=0.5, patch_artist=True)
    for patch, color in zip(bp_b["boxes"], pal[:len(data_igd)]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax_b.set_xticks(range(len(data_igd)))
    ax_b.set_xticklabels(labels_hv, fontsize=10)
    ax_b.set_ylabel("IGD (↓ mejor)", fontsize=9)
    ax_b.set_title("(B) IGD NSGA-II: N=25 vs N=50",
                   fontsize=10, fontweight="bold")
    ax_b.grid(True, alpha=0.3, axis="y")

    ax_c = fig.add_subplot(gs[1, 0])
    objs = ["f₁ (Tiempo)", "f₂ (Tardanza)", "f₃ (CO₂)"]
    keys = ["f1", "f2", "f3"]

    vss_vals = {"N=25": [], "N=50": []}
    for label, summ in [("N=25", summary_s4_n25), ("N=50", summary_s4_n50)]:
        if summ is None:
            vss_vals[label] = [0, 0, 0]
            continue
        vss_pct = summ.get("VSS", {}).get("VSS_pct", {})
        vss_vals[label] = [vss_pct.get(k, 0) for k in keys]

    x_pos = np.arange(3)
    w = 0.35
    colors_use = pal[:2] if summary_s4_n25 else [pal[1]]
    labels_use = ["N=25", "N=50"] if summary_s4_n25 else ["N=50"]

    if summary_s4_n25:
        ax_c.bar(x_pos - w/2, vss_vals["N=25"], w,
                 color=pal[0], alpha=0.85, label="N=25")
    ax_c.bar(x_pos + (w/2 if summary_s4_n25 else 0),
             vss_vals["N=50"], w, color=pal[1], alpha=0.85, label="N=50")

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(objs, fontsize=9)
    ax_c.set_ylabel("VSS (%)", fontsize=9)
    ax_c.set_title("(C) VSS% por objetivo: N=25 vs N=50",
                   fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3, axis="y")
    ax_c.axhline(5, ls="--", color="gray", lw=1, label="Target 5%")

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")

    vss_n50 = summary_s4_n50.get("VSS", {}).get("VSS_pct", {})
    ev_n50  = summary_s4_n50.get("EV_compromise", {})

    tabla = [
        ["Métrica", "N=25", "N=50"],
        ["VSS f₁ (%)",
         f"{(summary_s4_n25 or {}).get('VSS',{}).get('VSS_pct',{}).get('f1',float('nan')):.1f}",
         f"{vss_n50.get('f1',0):.1f}"],
        ["VSS f₂ (%)",
         f"{(summary_s4_n25 or {}).get('VSS',{}).get('VSS_pct',{}).get('f2',float('nan')):.1f}",
         f"{vss_n50.get('f2',0):.1f}"],
        ["VSS f₃ (%)",
         f"{(summary_s4_n25 or {}).get('VSS',{}).get('VSS_pct',{}).get('f3',float('nan')):.1f}",
         f"{vss_n50.get('f3',0):.1f}"],
        ["f₁ compromiso (h)",
         "4.78",
         f"{ev_n50.get('f1_h',0):.2f}"],
        ["Soluciones Pareto",
         "62",
         str(summary_s4_n50.get("n_pareto_ev", "—"))],
        ["Tiempo solver (s)",
         "1.4",
         "—"],
    ]

    tbl = ax_d.table(cellText=tabla[1:], colLabels=tabla[0],
                     colWidths=[0.45, 0.27, 0.28],
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for j in range(3):
        tbl[(0, j)].set_facecolor("#37474F")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(tabla)):
        fc = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(3):
            tbl[(i, j)].set_facecolor(fc)

    ax_d.set_title("(D) Tabla comparativa N=25 vs N=50",
                   fontsize=10, fontweight="bold")

    fig.suptitle(
        "Escalabilidad del Framework: N=25 → N=50 Farmacias\n"
        "Instancia Sevilla",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figura guardada: {out_path}")


def load_n25_results() -> Tuple[Optional[Dict], Optional[Dict],
                                Optional[pd.DataFrame]]:
    summary_n25 = None
    latest_integ = INTEG_DIR / "latest_integration.txt"
    if latest_integ.exists():
        run_id = latest_integ.read_text().strip()
        fp = INTEG_DIR / run_id / "s4_summary.json"
        if fp.exists():
            with open(fp) as f:
                summary_n25 = json.load(f)
            log.info(f"  Resultados s4 N=25 cargados: {run_id}")
            rc = INTEG_DIR / run_id / "recompute_eev_recurso.json"
            if rc.exists():
                with open(rc) as f:
                    summary_n25.setdefault("VSS", {})["VSS_pct"] = json.load(f)["VSS_pct_corregido"]
                log.info("  VSS N=25 tomado de recompute_eev_recurso.json (recurso por ruta)")

    metrics_n25 = None
    latest_fc = FORECAST_DIR / "latest_run.txt"
    if latest_fc.exists():
        run_id_fc = latest_fc.read_text().strip()
        fp_met = FORECAST_DIR / run_id_fc / "forecast_metrics.json"
        fp_cfg = FORECAST_DIR / run_id_fc / "forecast_config.json"
        if fp_met.exists() and fp_cfg.exists():
            with open(fp_met) as f: metrics_n25 = json.load(f)
            with open(fp_cfg) as f: cfg = json.load(f)
            if cfg.get("n_pharmacies", 0) == 25:
                log.info(f"  Resultados s3 N=25 cargados: {run_id_fc}")
            else:
                metrics_n25 = None

    df_n25 = None
    s5_stab = SENS_DIR / "latest_sensitivity.txt"
    if s5_stab.exists():
        run_id_s5 = s5_stab.read_text().strip()
        fp_stab = SENS_DIR / run_id_s5 / "s5_mod_e_stability.csv"
        if fp_stab.exists():
            df_n25 = pd.read_csv(fp_stab)
            df_n25 = df_n25[df_n25["algorithm"] == "NSGA-II"].copy()
            log.info(f"  Estabilidad s5 N=25 cargada: {run_id_s5}")

    return summary_n25, metrics_n25, df_n25


def parse_args():
    p = argparse.ArgumentParser(
        description="s6_scale_n50.py — Escalado completo a N=50")
    p.add_argument("--quick",   action="store_true",
                   help="Modo rápido: parámetros reducidos (~5 min)")
    p.add_argument("--skip_s3", action="store_true",
                   help="Saltar s3 (usar latest_run.txt existente)")
    p.add_argument("--skip_s4", action="store_true",
                   help="Saltar s4 (usar latest_integration.txt existente)")
    return p.parse_args()


def main():
    args = parse_args()

    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S") + "_s6"
    run_dir = SCALE_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ESCALADO N=50 — Pipeline Completo")
    print(f"  Quick: {args.quick}  |  Run: {run_id}")
    if args.quick:
        print("  [--quick: parámetros reducidos para verificación]")
    print("=" * 60)
    t0 = time.time()

    log.info("Cargando resultados N=25 existentes...")
    summary_n25, metrics_n25, df_stab_n25 = load_n25_results()

    if summary_n25 is None:
        log.info("  Sin resultados s4 N=25: la comparativa solo incluye N=50")

    if not args.skip_s3:
        run_id_s3, metrics_n50, config_n50 = run_s3_n50(args.quick)
    else:
        log.info("PASO 1/5 — Saltando s3 (--skip_s3)")
        latest = FORECAST_DIR / "latest_run.txt"
        run_id_s3 = latest.read_text().strip()
        run_dir_s3 = FORECAST_DIR / run_id_s3
        with open(run_dir_s3 / "forecast_metrics.json") as f:
            metrics_n50 = json.load(f)
        with open(run_dir_s3 / "forecast_config.json") as f:
            config_n50 = json.load(f)
        log.info(f"  Usando run s3: {run_id_s3}")

    with open(run_dir / "s6_n50_forecast.json", "w") as f:
        json.dump({
            "run_id_s3": run_id_s3,
            "n_pharmacies": config_n50.get("n_pharmacies"),
            "metrics": metrics_n50,
        }, f, indent=2, default=str)

    if not args.skip_s4:
        run_id_s4, summary_n50 = run_s4_n50(args.quick)
    else:
        log.info("PASO 2/5 — Saltando s4 (--skip_s4)")
        latest = INTEG_DIR / "latest_integration.txt"
        run_id_s4 = latest.read_text().strip()
        with open(INTEG_DIR / run_id_s4 / "s4_summary.json") as f:
            summary_n50 = json.load(f)
        log.info(f"  Usando run s4: {run_id_s4}")

    with open(run_dir / "s6_n50_vss.json", "w") as f:
        json.dump(summary_n50, f, indent=2, default=str)

    n_runs_stab = 5 if args.quick else 30
    n_gen_stab  = 80 if args.quick else 200
    pop_stab    = 50 if args.quick else 100
    df_stab_n50, inst_n50 = run_stability_n50(
        n_runs=n_runs_stab, n_gen=n_gen_stab,
        pop_size=pop_stab, quick=args.quick)
    df_stab_n50.to_csv(run_dir / "s6_n50_stability.csv", index=False)

    log.info("PASO 4/5 — Tabla comparativa N=25 vs N=50")
    df_comp = build_comparison_table(
        summary_n25, summary_n50,
        metrics_n25, metrics_n50,
        df_stab_n25, df_stab_n50)
    df_comp.to_csv(run_dir / "s6_comparison_n25_n50.csv", index=False)
    log.info(f"  Tabla guardada: {run_dir}/s6_comparison_n25_n50.csv")

    wilcoxon_result = None
    if df_stab_n25 is not None:
        hv_25 = df_stab_n25["hv"].dropna().values
        hv_50 = df_stab_n50["hv"].dropna().values
        n_min = min(len(hv_25), len(hv_50))
        if n_min >= 5:
            try:
                _, pval = stats.mannwhitneyu(hv_25[:n_min], hv_50[:n_min],
                                              alternative="two-sided")
                wilcoxon_result = round(float(pval), 6)
                sig = "✅" if pval < 0.05 else "—"
                log.info(f"  Mann-Whitney HV N=25 vs N=50: p={pval:.4f}  {sig}")
            except Exception:
                pass

    log.info("PASO 5/5 — Figura comparativa")
    plot_comparison(df_stab_n25, df_stab_n50,
                    summary_n25, summary_n50,
                    run_dir / "s6_comparison_plot.png")

    summary_s6 = {
        "run_id":        run_id,
        "run_id_s3_n50": run_id_s3,
        "run_id_s4_n50": run_id_s4,
        "quick":         args.quick,
        "t_total_s":     round(time.time() - t0, 1),
        "n50": {
            "skill_bilstm":  metrics_n50.get("bilstm",{}).get(
                             "skill_q50_vs_naive", 0),
            "coverage":      metrics_n50.get("calibration",{}).get(
                             "empirical_coverage_pct", 0),
            "vss_f1_pct":    summary_n50.get("VSS",{}).get(
                             "VSS_pct",{}).get("f1", 0),
            "vss_f2_pct":    summary_n50.get("VSS",{}).get(
                             "VSS_pct",{}).get("f2", 0),
            "vss_f3_pct":    summary_n50.get("VSS",{}).get(
                             "VSS_pct",{}).get("f3", 0),
            "hv_mean":       float(df_stab_n50["hv"].mean()),
            "hv_cv_pct":     float(df_stab_n50["hv"].std() /
                                   (df_stab_n50["hv"].mean() + 1e-12) * 100),
        },
        "n25_cargado": {
            "run_id_s4": (summary_n25 or {}).get("run_id_s4"),
            "vss_pct": (summary_n25 or {}).get("VSS", {}).get("VSS_pct"),
        },
        "wilcoxon_hv_n25_vs_n50": wilcoxon_result,
    }
    with open(run_dir / "s6_summary.json", "w") as f:
        json.dump(summary_s6, f, indent=2, default=str)

    (SCALE_DIR / "latest_scale.txt").write_text(run_id)

    t_total = time.time() - t0
    vss_n50 = summary_s6["n50"]
    print("\n" + "=" * 60)
    print("✅  Escalado N=50 completado")
    print("=" * 60)
    print(f"\n  ── Resultados N=50 ──")
    print(f"  Skill BiLSTM vs Naive (proxy heurístico): {vss_n50['skill_bilstm']:.1f}%")
    print(f"  Cobertura [Q10,Q90]:   {vss_n50['coverage']:.1f}%")
    print(f"  VSS f₁:  {vss_n50['vss_f1_pct']:.1f}%")
    print(f"  VSS f₂:  {vss_n50['vss_f2_pct']:.1f}%")
    print(f"  VSS f₃:  {vss_n50['vss_f3_pct']:.1f}%")
    print(f"  HV NSGA-II: {vss_n50['hv_mean']:.4e}  "
          f"CV={vss_n50['hv_cv_pct']:.1f}%")
    if wilcoxon_result is not None:
        sig = "✅ diferencias significativas" if wilcoxon_result < 0.05 \
              else "— diferencias no significativas"
        print(f"  Mann-Whitney N=25 vs N=50 (HV): p={wilcoxon_result}  {sig}")
    print(f"\n  ── Comparativa N=25 vs N=50 ──")
    print(f"  {'Métrica':<28} {'N=25':>8} {'N=50':>8}")
    print(f"  {'-'*46}")
    v25 = (summary_n25 or {}).get("VSS", {}).get("VSS_pct", {})
    print(f"  {'VSS f₁ (%)':<28} {v25.get('f1', float('nan')):>8.1f} "
          f"{vss_n50['vss_f1_pct']:>8.1f}")
    print(f"  {'VSS f₂ (%)':<28} {v25.get('f2', float('nan')):>8.1f} "
          f"{vss_n50['vss_f2_pct']:>8.1f}")
    print(f"  {'VSS f₃ (%)':<28} {v25.get('f3', float('nan')):>8.1f} "
          f"{vss_n50['vss_f3_pct']:>8.1f}")
    print(f"\n  Tiempo total: {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"  Salidas en: {run_dir}/")
    print(f"    s6_n50_forecast.json")
    print(f"    s6_n50_vss.json")
    print(f"    s6_n50_stability.csv")
    print(f"    s6_comparison_n25_n50.csv")
    print(f"    s6_comparison_plot.png")
    print(f"    s6_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
