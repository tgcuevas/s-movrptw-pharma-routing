import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
RUNS = sys.argv[1:] or ["20260615_201043_s4", "20260615_205516_s4"]
sys.argv = sys.argv[:1]
import s4_integration as s4
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def perm_test(d):
    d = np.asarray(d, float); obs = d.mean(); a = np.abs(d); cnt = 0
    for s in product((1.0, -1.0), repeat=len(d)):
        cnt += (np.array(s) * a).mean() >= obs - 1e-9
    return cnt / 2 ** len(d)


def recompute(run_s4):
    d4 = s4.INTEG_DIR / run_s4
    summ = json.load(open(d4 / "s4_summary.json", encoding="utf-8"))
    n, n_gen, run_s3 = summ["n_pharmacies"], summ["n_gen"], summ["run_id_s3"]
    Q = s4.VEHICLE_CAPACITY
    df_fc, df_saa, _, _ = s4.load_forecast_data(run_s3)
    dem = df_fc["demand_q50"].values[:n]
    pri = df_fc["priority_class"].values[:n].tolist()
    D, st = s4.get_distance_matrix(n), s4.get_service_times(n)
    inst = {"label": "EV", "n": n, "D": D.tolist(), "demands": dem.tolist(), "priorities": pri,
            "service_times": st.tolist(), "capacity": Q}
    res = minimize(s4.CVRPTW_3obj(inst),
                   NSGA2(pop_size=s4.POP_SIZE, sampling=s4.PermutationSampling(),
                         crossover=s4.OrderCrossover(), mutation=s4.SwapMutation(),
                         eliminate_duplicates=True),
                   termination=("n_gen", n_gen), seed=int(s4.SEED), verbose=False)

    can = pd.read_csv(d4 / "s4_ev_pareto.csv")[["f1_s", "f2_pond", "f3_co2"]].values
    A = np.array(sorted(map(tuple, np.round(res.F, 2))))
    B = np.array(sorted(map(tuple, np.round(can, 2))))
    if A.shape != B.shape or not np.allclose(A, B, atol=0.02):
        sys.exit(f"ERROR {run_s4}: el frente EV reproducido NO coincide con s4_ev_pareto.csv. No se escribe nada.")
    comp, idx = s4.select_compromise(res.F)
    x = res.X[idx].astype(int)

    routes, cur, cd = [], [], 0.0
    for i in x:
        if cd + dem[i] > Q and cur:
            routes.append(cur); cur, cd = [], 0.0
        cur.append(int(i)); cd += dem[i]
    routes.append(cur)

    fronts = pd.read_csv(d4 / "s4_saa_fronts.csv").sort_values("scenario_id")
    eev_old = pd.read_csv(d4 / "s4_eev_eval.csv").sort_values("scenario_id")
    per_k = []
    for k in sorted(df_saa["scenario_id"].unique()):
        dk = df_saa[df_saa["scenario_id"] == k].sort_values("pharmacy_id")["demand_kg"].values[:n]
        loads = [float(dk[r].sum()) for r in routes]
        extra = int(sum(max(0, int(np.ceil(L / Q)) - 1) for L in loads))
        per_k.append({"scenario_id": int(k), "demanda_total_kg": round(float(dk.sum()), 2),
                      "carga_por_ruta_kg": [round(L, 1) for L in loads], "viajes_extra": extra,
                      "f1_eev_corregido": round(float(comp[0]) + s4.PENALTY_EXTRA_TRIP * extra, 2)})
    old = summ["VSS"]
    eev_f1 = float(np.mean([p["f1_eev_corregido"] for p in per_k]))
    EEV = {"f1": eev_f1, "f2": old["EEV"]["f2"], "f3": old["EEV"]["f3"]}
    RP = old["RP"]
    VSS = {o: EEV[o] - RP[o] for o in ("f1", "f2", "f3")}
    VSS_pct = {o: round(100 * VSS[o] / EEV[o], 2) for o in ("f1", "f2", "f3")}

    dif = np.array([p["f1_eev_corregido"] for p in per_k]) - fronts["f1_best"].values
    tests = {"n_escenarios_EEV_mayor": int((dif > 0).sum()), "media_diferencia_s": round(float(dif.mean()), 2),
             "p_wilcoxon_unilateral": float(wilcoxon(dif, alternative="greater").pvalue),
             "p_permutacion_media": perm_test(dif)}

    out = {
        "script": "recompute_eev_recurso.py", "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "run_s4": run_s4, "run_s3": run_s3, "n_farmacias": n, "K": len(per_k),
        "verificacion": "frente EV reproducido idéntico a s4_ev_pareto.csv",
        "recurso": "un viaje de vuelta al depósito (PENALTY_EXTRA_TRIP s) por cada ruta EV cuya carga en el escenario supera Q",
        "plan_EV": {"n_rutas": len(routes), "carga_media_por_ruta_kg": [round(float(dem[r].sum()), 1) for r in routes],
                    "f1_s": round(float(comp[0]), 2)},
        "viajes_extra": {"codigo_original": int(eev_old["n_extra_trips"].sum()),
                         "escenarios_codigo_original": int((eev_old["n_extra_trips"] > 0).sum()),
                         "recurso_por_ruta": int(sum(p["viajes_extra"] for p in per_k)),
                         "escenarios_recurso_por_ruta": int(sum(p["viajes_extra"] > 0 for p in per_k))},
        "EEV_original": old["EEV"], "EEV_corregido": {o: round(v, 4) for o, v in EEV.items()},
        "RP": RP, "VSS_pct_original": old["VSS_pct"], "VSS_pct_corregido": VSS_pct,
        "test_H1_f1_corregido": tests, "escenarios": per_k,
    }
    with open(d4 / "recompute_eev_recurso.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"{run_s4} (N={n}): frente EV verificado · {len(routes)} rutas EV · viajes extra "
          f"{out['viajes_extra']['codigo_original']} → {out['viajes_extra']['recurso_por_ruta']} "
          f"({out['viajes_extra']['escenarios_recurso_por_ruta']}/{len(per_k)} escenarios)")
    print(f"   VSS% original {old['VSS_pct']} → corregido {VSS_pct}")
    print(f"   H1: EEV>RP en {tests['n_escenarios_EEV_mayor']}/{len(per_k)} · p Wilcoxon {tests['p_wilcoxon_unilateral']:.2e} · "
          f"p permutación {tests['p_permutacion_media']:.2e}")
    print(f"   Guardado: {d4 / 'recompute_eev_recurso.json'}")


if __name__ == "__main__":
    if not Path("s4_integration.py").exists():
        sys.exit("ERROR: ejecuta desde la carpeta del pipeline (donde está s4_integration.py)")
    for r in RUNS:
        recompute(r)
