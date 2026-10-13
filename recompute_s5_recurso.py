import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("run_s5")
ap.add_argument("--modulos", default="C,D,F")
ap.add_argument("--sobrescribir", action="store_true",
                help="reemplaza los ficheros recompute_* existentes (por defecto se escriben con sufijo de fecha)")
args = ap.parse_args()
sys.argv = sys.argv[:1]
import s5_sensitivity as s5
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

CAP, PEN, SEED = s5.CAPACITY, s5.PENALTY_TRIP, s5.SEED


def run_fx(inst, pop, gen, seed):
    res = minimize(s5.CVRPTW_3obj(inst),
                   NSGA2(pop_size=int(pop), sampling=s5.PermutationSampling(), crossover=s5.OrderCrossover(),
                         mutation=s5.SwapMutation(), eliminate_duplicates=True),
                   ("n_gen", int(gen)), seed=int(seed), verbose=False)
    return res.F, res.X


def compromise(F, X):
    Fn = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-9)
    i = int(np.argmin(np.linalg.norm(Fn, axis=1)))
    return F[i], X[i].astype(int)


def routes_of(x, dem):
    routes, cur, cd = [], [], 0.0
    for i in x:
        if cd + dem[i] > CAP and cur:
            routes.append(cur); cur, cd = [], 0.0
        cur.append(int(i)); cd += dem[i]
    routes.append(cur)
    return routes


def extra_original(d_k):
    return max(0, int(np.ceil(d_k.sum() / CAP)) - 1)


def extra_ruta(d_k, routes):
    return int(sum(max(0, int(np.ceil(d_k[r].sum() / CAP)) - 1) for r in routes))


def hv_pct(F_rp_nd, F_eev, ref):
    ind = HV(ref_point=ref)
    hv_rp, hv_eev = float(ind(F_rp_nd)), float(ind(F_eev))
    return max(0.0, (hv_rp - hv_eev) / (hv_rp + 1e-9)) * 100


def modulo_c(k_range=(3, 5, 8, 10, 15, 20, 25, 30), n_farm=25, n_gen=80, n_rep=5):
    D = s5.make_distance_matrix(n_farm, SEED); svc = s5.make_service_times(n_farm)
    prio = s5.make_priorities(n_farm, SEED); rng = np.random.default_rng(SEED)
    mu = rng.uniform(3, 18, n_farm); sig = mu * 0.15
    L = np.linalg.cholesky(np.eye(n_farm) * 0.9 + np.ones((n_farm, n_farm)) * 0.1)
    F_ev, X_ev = run_fx(s5.make_instance(n_farm, mu, prio, D, svc), 100, n_gen * 2, SEED)
    x_bar, perm = compromise(F_ev, X_ev); rts = routes_of(perm, mu)
    ref = s5.compute_ref_point([F_ev]) * 1.5
    rows, trips = [], {"original": 0, "ruta": 0, "evaluaciones": 0}
    for K in k_range:
        v = {"original": [], "ruta": []}
        for rep in range(n_rep):
            rng_k = np.random.default_rng(SEED + rep * 1000 + K)
            scen = [np.clip(mu + sig * (L @ rng_k.standard_normal(n_farm)), 0.1, None) for _ in range(K)]
            Frp = [s5._run_nsga2(s5.make_instance(n_farm, d, prio, D, svc), 50, n_gen, int(SEED + rep * 100 + k))
                   for k, d in enumerate(scen)]
            pooled = np.vstack(Frp)
            F_rp_nd = pooled[NonDominatedSorting().do(pooled, only_non_dominated_front=True)]
            for modo, fx in (("original", lambda d: extra_original(d)), ("ruta", lambda d: extra_ruta(d, rts))):
                pts = []
                for d in scen:
                    e = fx(d); trips[modo] += e
                    pts.append([float(x_bar[0]) + PEN * e, float(x_bar[1]),
                                float(x_bar[2]) * (d.sum() / CAP) / (mu.sum() / CAP)])
                v[modo].append(hv_pct(F_rp_nd, np.array(pts), ref))
            trips["evaluaciones"] += K
        r = {"K": K}
        for modo in v:
            m, s = float(np.nanmean(v[modo])), float(np.nanstd(v[modo]))
            r.update({f"vsshv_mean_{modo}": round(m, 3), f"vsshv_std_{modo}": round(s, 3),
                      f"vsshv_cv_pct_{modo}": round(s / (m + 1e-9) * 100, 1)})
        rows.append(r)
        print(f"  C · K={K:2d}: VSS_HV original {r['vsshv_mean_original']:.2f} % → corregido {r['vsshv_mean_ruta']:.2f} % "
              f"(CV {r['vsshv_cv_pct_ruta']} %)")
    return pd.DataFrame(rows), {"rutas_EV": len(rts), "viajes_extra": trips}


def modulo_d(K=20, n_farm=25, n_gen=100):
    D = s5.make_distance_matrix(n_farm, SEED); svc = s5.make_service_times(n_farm)
    prio = s5.make_priorities(n_farm, SEED); rng = np.random.default_rng(SEED)
    mu = rng.uniform(3, 18, n_farm); sig = mu * 0.15
    L = np.linalg.cholesky(np.eye(n_farm) * 0.9 + np.ones((n_farm, n_farm)) * 0.1)
    rng_s = np.random.default_rng(SEED + 42)
    scen = [np.clip(mu + sig * (L @ rng_s.standard_normal(n_farm)), 0.1, None) for _ in range(K)]
    F_ev, X_ev = run_fx(s5.make_instance(n_farm, mu, prio, D, svc), 100, n_gen * 2, SEED)
    x_bar, perm = compromise(F_ev, X_ev); rts = routes_of(perm, mu)
    out = {}
    for modo, fx in (("original", extra_original), ("ruta", lambda d: extra_ruta(d, rts))):
        ex = [fx(d) for d in scen]
        out[modo] = {"EEV_f1": float(np.mean([float(x_bar[0]) + PEN * e for e in ex])), "viajes_extra": int(sum(ex)),
                     "escenarios_con_viaje": int(sum(e > 0 for e in ex))}
    rp = [s5.select_compromise(s5._run_nsga2(s5.make_instance(n_farm, d, prio, D, svc), 50, n_gen, int(SEED + k)))
          for k, d in enumerate(scen)]
    ws = [s5.select_compromise(s5._run_nsga2(s5.make_instance(n_farm, np.clip(d, 0.1, CAP - 0.1), prio, D, svc),
                                             50, n_gen, int(SEED + k + 9000))) for k, d in enumerate(scen)]
    RP_f1, WS_f1 = float(np.mean([c[0] for c in rp])), float(np.mean([c[0] for c in ws]))
    for modo in out:
        E = out[modo]["EEV_f1"]
        out[modo].update({"VSS_f1_pct": round(max(0.0, E - RP_f1) / E * 100, 2),
                          "cadena_f1_ok": bool(WS_f1 <= RP_f1 + 1e-3 and RP_f1 <= E + 1e-3)})
    out.update({"RP_f1": RP_f1, "WS_f1": WS_f1, "rutas_EV": len(rts)})
    print(f"  D · EEV f1 {out['original']['EEV_f1']/3600:.2f} h → {out['ruta']['EEV_f1']/3600:.2f} h · "
          f"VSS f1 {out['original']['VSS_f1_pct']} % → {out['ruta']['VSS_f1_pct']} % · RP {RP_f1/3600:.2f} h · WS {WS_f1/3600:.2f} h")
    return out


F_DUAL = []


def vsshv_dual(mu, sigma, prio, D, svc, K, n_gen):
    rng = np.random.default_rng(SEED + 7); n = len(mu)
    L = np.linalg.cholesky(np.eye(n) * 0.9 + np.ones((n, n)) * 0.1)
    F_ev, X_ev = run_fx(s5.make_instance(n, mu, prio, D, svc), 50, n_gen, SEED)
    x_bar, perm = compromise(F_ev, X_ev); rts = routes_of(perm, mu)
    Frp, scen = [], []
    for k in range(K):
        d = np.clip(mu + sigma * (L @ rng.standard_normal(n)), 0.1, None); scen.append(d)
        Frp.append(s5._run_nsga2(s5.make_instance(n, d, prio, D, svc), 30, n_gen, int(SEED + k)))
    pooled = np.vstack(Frp)
    F_rp_nd = pooled[NonDominatedSorting().do(pooled, only_non_dominated_front=True)]
    res = {}
    for modo, fx in (("original", extra_original), ("ruta", lambda d: extra_ruta(d, rts))):
        F_eev = np.array([[float(x_bar[0]) + PEN * fx(d), float(x_bar[1]),
                           float(x_bar[2]) * d.sum() / (mu.sum() + 1e-9)] for d in scen])
        ref = s5.compute_ref_point([F_rp_nd, F_eev]) * 1.2
        try:
            res[modo] = hv_pct(F_rp_nd, F_eev, ref)
        except Exception:
            res[modo] = 0.0
    F_DUAL.append(res)
    return res["original"]


def main():
    s5dir = s5.SENS_DIR / args.run_s5
    if not s5dir.is_dir():
        sys.exit(f"ERROR: no existe {s5dir}")
    mods = {m.strip().upper() for m in args.modulos.split(",")}
    existe = (s5dir / "recompute_s5_recurso.json").exists()
    suf = "" if (args.sobrescribir or not existe) else "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if suf:
        print(f"  Ya existen resultados recompute_*: los nuevos se guardan con el sufijo {suf} (usa --sobrescribir para reemplazarlos)")
    prev = {}
    if not suf and existe:
        prev = json.load(open(s5dir / "recompute_s5_recurso.json", encoding="utf-8"))
    out = {**prev, "script": "recompute_s5_recurso.py", "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
           "run_s5": args.run_s5,
           "recurso": "un viaje de vuelta al depósito (PENALTY_TRIP s) por cada ruta EV cuya carga en el escenario supera Q"}
    t0 = time.time()

    if "C" in mods:
        dfc, info = modulo_c()
        can = pd.read_csv(s5dir / "s5_mod_c_vsshv.csv")
        ok = np.allclose(dfc["vsshv_mean_original"].values, can["vsshv_mean"].values, atol=0.002) and \
             np.allclose(dfc["vsshv_std_original"].values, can["vsshv_std"].values, atol=0.002)
        if not ok:
            sys.exit("ERROR módulo C: el modo original no reproduce s5_mod_c_vsshv.csv. No se escribe nada.")
        k_conv = dfc.loc[dfc["vsshv_cv_pct_ruta"] < 10, "K"].min()
        cv_exact = 100 * dfc["vsshv_std_ruta"] / dfc["vsshv_mean_ruta"]
        k_conv_exact = dfc.loc[cv_exact < 10, "K"].min()
        dfc["vsshv_cv_exacto_ruta"] = cv_exact.round(2)
        dfc.to_csv(s5dir / f"recompute_s5_mod_c_vsshv{suf}.csv", index=False)
        out["modulo_C"] = {"verificacion": "modo original idéntico a s5_mod_c_vsshv.csv", **info,
                           "K_convergencia_original": 8, "K_convergencia_corregido": None if pd.isna(k_conv) else int(k_conv),
                           "K_convergencia_corregido_CV_exacto": None if pd.isna(k_conv_exact) else int(k_conv_exact),
                           "K_con_CV_menor_10_corregido": dfc.loc[cv_exact < 10, "K"].astype(int).tolist()}
        print(f"  C verificado · K de convergencia corregido: {out['modulo_C']['K_convergencia_corregido']}")

    if "D" in mods:
        d = modulo_d()
        can = json.load(open(s5dir / "s5_mod_d_evpi.json", encoding="utf-8"))
        ok = abs(d["original"]["EEV_f1"] - can["EEV"]["f1"]) < 0.5 and abs(d["RP_f1"] - can["RP"]["f1"]) < 0.5 \
            and abs(d["WS_f1"] - can["WS"]["f1"]) < 0.5
        if not ok:
            sys.exit("ERROR módulo D: el modo original no reproduce s5_mod_d_evpi.json. No se escribe nada.")
        out["modulo_D"] = {"verificacion": "modo original idéntico a s5_mod_d_evpi.json (f1)", **d}

    if "F" in mods:
        s5._compute_vsshv_for_forecast = vsshv_dual
        resf = s5.run_module_f(n_farm=25, n_days=120, K=10, n_gen=80)
        dff = resf["df"].copy()
        dff["vsshv_pct_corregido"] = [round(r["ruta"], 2) for r in F_DUAL]
        can = pd.read_csv(s5dir / "s5_mod_f_ablation.csv").set_index("model")
        dff["vsshv_pct_canonico"] = [can.loc[m, "vsshv_pct"] if m in can.index else np.nan for m in dff["model"]]
        dff["reproduce_canonico"] = np.isclose(dff["vsshv_pct"], dff["vsshv_pct_canonico"], atol=0.01)
        dff.to_csv(s5dir / f"recompute_s5_mod_f_ablation{suf}.csv", index=False)
        out["modulo_F"] = dff[["model", "rmse", "vsshv_pct_canonico", "vsshv_pct", "vsshv_pct_corregido",
                               "reproduce_canonico"]].to_dict("records")
        for r in out["modulo_F"]:
            print(f"  F · {r['model']:<11} VSS_HV canónico {r['vsshv_pct_canonico']} · reproducido {r['vsshv_pct']} "
                  f"· corregido {r['vsshv_pct_corregido']} {'(verificado)' if r['reproduce_canonico'] else '(no reproducible: ver nota)'}")
        out["nota_F"] = ("La fila BiLSTM+XGB entrena un BiLSTM en TensorFlow (no determinista): su valor reproducido puede "
                         "diferir del canónico. Las filas deterministas verifican el cálculo.")

    out["tiempo_s"] = round(time.time() - t0, 1)
    with open(s5dir / f"recompute_s5_recurso{suf}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"Guardado: {s5dir / f'recompute_s5_recurso{suf}.json'} ({out['tiempo_s']} s)")


if __name__ == "__main__":
    if not Path("s5_sensitivity.py").exists():
        sys.exit("ERROR: ejecuta desde la carpeta del pipeline (donde está s5_sensitivity.py)")
    main()
