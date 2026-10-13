import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_integration as s4
import robustez_vss as rv
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

DATA = Path("data")
RUN_S5 = "20260615_201114_s5"
SEEDS_EV_EQ = list(range(1042, 1062))
SEED_RP = 2042
P1B_K, P1B_REP = (3, 5, 8, 10, 15, 20, 25, 30), 5
P1E_GENS, P1E_SEEDS = (200, 400, 800, 1600), (42, 43, 44, 45, 46)
K_ORACULO, SEED_ORACULO = 10_000, 20261003
PRUEBA = False
OUT = DATA / "consolidacion"


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def guarda(nombre, obj):
    json.dump(obj, open(OUT / nombre, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)


def ngen(n_gen):
    return 10 if PRUEBA else int(n_gen)


def ctx_de(n, pri):
    return {"D": s4.get_distance_matrix(n), "st": s4.get_service_times(n), "pri": list(pri), "Q": s4.VEHICLE_CAPACITY,
            "alpha": np.array([s4._ALPHA_TW[p] for p in pri], float), "b": np.array([s4._TW_CLASS[p][1] for p in pri], float)}


def estimandos(planes_det, planes_cmp, S_in, oos):
    Ed = np.array([p.S(S_in)[0].mean(0) for p in planes_det])
    Ec = np.array([p.S(S_in)[0].mean(0) for p in planes_cmp])
    nd_d, nd_c = rv.no_dominados(Ed), rv.no_dominados(Ec)
    iA = (nd_d[rv.topsis(Ed[nd_d])], nd_c[rv.topsis(Ec[nd_c])])
    sel = {"E1": (1, (int(np.argmin(Ed[:, 1])), int(np.argmin(Ec[:, 1])))), "E2": (1, iA),
           "E3": (0, (int(np.argmin(Ed[:, 0])), int(np.argmin(Ec[:, 0])))), "E4": (0, iA)}
    out = {"candidatos": {"deterministas": len(planes_det), "comparados": len(planes_cmp)}}
    for tipo, X in oos.items():
        out[tipo] = {e: rv.ic_vss(planes_det[i].S(X)[0][:, o], planes_cmp[j].S(X)[0][:, o])[0] for e, (o, (i, j)) in sel.items()}
        out[f"{tipo}_modeloM"] = {e: rv.ic_vss(planes_det[i].M(X)[0][:, o], planes_cmp[j].M(X)[0][:, o])[0]
                                  for e, (o, (i, j)) in sel.items()}
    return out


def ev_equilibrado(P):
    c, q50 = P["ctx"], P["q50"]; planes = []
    for sd in SEEDS_EV_EQ[: (2 if PRUEBA else 20)]:
        F, X = s4.run_nsga2_for_instance(q50, c["pri"], c["D"], c["st"], n_gen=ngen(P["summ"]["n_gen"]),
                                         pop_size=s4.POP_SIZE, seed=sd, label=f"EVq{sd}", return_X=True)
        planes += [rv.Plan(x, q50, c, f"EV_s{sd}") for x in X]
    return planes


def w1(P):
    c = P["ctx"]
    a = s4.run_nsga2_for_instance(P["q50"], c["pri"], c["D"], c["st"], n_gen=ngen(P["summ"]["n_gen"]), pop_size=s4.POP_SIZE, seed=SEEDS_EV_EQ[0], label="W1a")
    b = s4.run_nsga2_for_instance(P["q50"], c["pri"], c["D"], c["st"], n_gen=ngen(P["summ"]["n_gen"]), pop_size=s4.POP_SIZE, seed=SEEDS_EV_EQ[0], label="W1b")
    return bool(a.shape == b.shape and np.array_equal(a, b))


class ProblemaRecurso(ElementwiseProblem):
    def __init__(self, ctx, q50, S_in):
        self.ctx, self.q50, self.S_in = ctx, q50, S_in
        super().__init__(n_var=len(q50), n_obj=3, xl=0, xu=len(q50) - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = rv.Plan(np.asarray(x, dtype=int), self.q50, self.ctx, "RP").M(self.S_in)[0].mean(0)


def recurso(P):
    res = minimize(ProblemaRecurso(P["ctx"], P["q50"], P["S_in"]),
                   NSGA2(pop_size=s4.POP_SIZE, sampling=s4.PermutationSampling(), crossover=s4.OrderCrossover(),
                         mutation=s4.SwapMutation(), eliminate_duplicates=True),
                   termination=("n_gen", ngen(P["summ"]["n_gen"])), seed=SEED_RP, verbose=False)
    return [rv.Plan(np.asarray(x, dtype=int), P["q50"], P["ctx"], "RP") for x in res.X]


def w2(P, s4_dir):
    val = P["planes_ev"][P["comp_ev"]].M(P["S_in"])[0].mean(0)
    pub = json.load(open(s4_dir / "recompute_eev_recurso.json", encoding="utf-8"))["EEV_corregido"]
    ok = abs(val[0] - pub["f1"]) <= 0.01 and abs(val[1] - pub["f2"]) <= 0.01
    return bool(ok), {"calculado": [round(float(v), 4) for v in val], "publicado": [pub["f1"], pub["f2"], pub["f3"]],
                      "diferencia_f3_informativa": round(float(val[2] - pub["f3"]), 4)}


def oos_de(P):
    if PRUEBA:
        T = np.tile(P["q50"], (5, 1)); return {"oos_modelo": T, "oos_real": T}
    Xr, _ = rv.demanda_verdadera(P["n"], len(P["df_hist"]), int(P["cfg"]["seed"]), rv.K_OOS, rv.SEED_OOS_REAL)
    return {"oos_modelo": rv.escenarios_modelo(P["df_fc"], P["df_hist"], rv.K_OOS, rv.SEED_OOS_MODELO), "oos_real": Xr}


def p2_p3(P, s4_dir):
    oos = oos_de(P)
    r2 = estimandos(ev_equilibrado(P), P["planes_saa"], P["S_in"], oos)
    r3 = estimandos(P["planes_ev"], recurso(P), P["S_in"], oos)
    return {"P2": r2, "P3": r3} if not PRUEBA else {"P2": "calculado (no se guarda en --prueba)", "P3": "calculado (no se guarda en --prueba)"}


def decision_c(ref, rep, prueba_id):
    tipos, conf, refu, info = ("oos_modelo", "oos_real"), True, False, {}
    for n in (25, 50):
        for t in tipos:
            e1, e2 = ref[n][prueba_id][t]["E1"], ref[n][prueba_id][t]["E2"]
            cnt = sum(r[prueba_id][t]["E1"]["vss_pct"] > 0 for r in rep[n]); tot = len(rep[n])
            info[f"N{n}_{t}"] = {"E1_ref": e1, "E2_ref": e2, "replicas_E1_pos": f"{cnt}/{tot}",
                                 "E1_ic_inferior_mayor_10": e1["ic95_pct"][0] > 10,
                                 "E3_ref": ref[n][prueba_id][t]["E3"]}
            conf &= e1["ic95_pct"][0] > 0 and e2["ic95_pct"][0] > 0 and cnt >= 8
            refu |= e1["vss_pct"] <= 0 or cnt < 6
    return {"veredicto": "refutada" if refu else "confirmada" if conf else "no concluyente", "detalle": info}


def p1a(n):
    run_s3, run_s4 = rv.REF[n]; d4 = DATA / "integration" / run_s4
    df_fc, df_saa, _, _ = s4.load_forecast_data(run_s3)
    dem = df_fc["demand_q50"].values[:n]; pri = df_fc["priority_class"].values[:n].tolist(); c = ctx_de(n, pri)
    n_gen = ngen(json.load(open(d4 / "s4_summary.json"))["n_gen"]); K = sorted(df_saa["scenario_id"].unique())[: (3 if PRUEBA else None)]
    F, X = s4.run_nsga2_for_instance(dem, pri, c["D"], c["st"], n_gen=n_gen, pop_size=s4.POP_SIZE, seed=s4.SEED, label="EV", return_X=True)
    comp, idx = s4.select_compromise(F)
    fr = [s4.run_nsga2_for_instance(dem, pri, c["D"], c["st"], n_gen=n_gen, pop_size=s4.POP_SIZE, seed=int(s4.SEED + k), label=f"P{k}") for k in K]
    rp = np.mean([f.min(axis=0) for f in fr], axis=0)
    plan = rv.Plan(X[idx], dem, c, "EV"); Mx, extra = plan.M(np.tile(dem, (len(K), 1)))
    w3 = bool(extra.sum() == 0 if hasattr(extra, "sum") else extra == 0) and bool(np.allclose(Mx.mean(0), plan.base))
    vss = (comp - rp) / comp * 100
    pub = json.load(open(d4 / "recompute_eev_recurso.json", encoding="utf-8"))["VSS_pct_corregido"]
    return {"W3": w3, "VSS_sin_incertidumbre_pct": [round(float(v), 2) for v in vss],
            "VSS_publicado_pct": [pub["f1"], pub["f2"], pub["f3"]],
            "parte_atribuible_a_la_incertidumbre_puntos": [round(pub[o] - float(v), 2) for o, v in zip(("f1", "f2", "f3"), vss)]}


def importa_rc():
    argv = sys.argv; sys.argv = ["consolidacion", RUN_S5]
    import recompute_s5_recurso as rc
    sys.argv = argv
    return rc


def p1b_p1c():
    rc = importa_rc(); s5 = rc.s5; n_farm = 25; n_gen = 10 if PRUEBA else 80
    D = s5.make_distance_matrix(n_farm, rc.SEED); svc = s5.make_service_times(n_farm); prio = s5.make_priorities(n_farm, rc.SEED)
    mu = np.random.default_rng(rc.SEED).uniform(3, 18, n_farm)
    F_ev, X_ev = rc.run_fx(s5.make_instance(n_farm, mu, prio, D, svc), 100, n_gen * 2, rc.SEED)
    x_bar, perm = rc.compromise(F_ev, X_ev); rts = rc.routes_of(perm, mu); ref = s5.compute_ref_point([F_ev]) * 1.5
    pub = pd.read_csv(DATA / "sensitivity" / RUN_S5 / "recompute_s5_mod_c_vsshv.csv").set_index("K")["vsshv_mean_ruta"]
    filas = []
    for K in (P1B_K[:2] if PRUEBA else P1B_K):
        v = []
        for rep in range(1 if PRUEBA else P1B_REP):
            scen = [mu.copy() for _ in range(K)]
            Frp = [s5._run_nsga2(s5.make_instance(n_farm, d, prio, D, svc), 50, n_gen, int(rc.SEED + rep * 100 + k)) for k, d in enumerate(scen)]
            pooled = np.vstack(Frp); F_rp_nd = pooled[rc.NonDominatedSorting().do(pooled, only_non_dominated_front=True)]
            pts = [[float(x_bar[0]) + rc.PEN * rc.extra_ruta(d, rts), float(x_bar[1]), float(x_bar[2]) * (d.sum() / rc.CAP) / (mu.sum() / rc.CAP)] for d in scen]
            v.append(rc.hv_pct(F_rp_nd, np.array(pts), ref))
        filas.append({"K": K, "vsshv_sin_incertidumbre_media": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3),
                      "max": round(float(np.max(v)), 3), "vsshv_publicado_con_incertidumbre": round(float(pub[K]), 3)})
    Kd = 3 if PRUEBA else 20; scen = [mu.copy() for _ in range(Kd)]
    eev = float(x_bar[0]) + rc.PEN * float(np.mean([rc.extra_ruta(d, rts) for d in scen]))
    rp = float(np.mean([s5.select_compromise(s5._run_nsga2(s5.make_instance(n_farm, d, prio, D, svc), 50, n_gen, int(rc.SEED + k)))[0] for k, d in enumerate(scen)]))
    ws = float(np.mean([s5.select_compromise(s5._run_nsga2(s5.make_instance(n_farm, np.clip(d, 0.1, rc.CAP - 0.1), prio, D, svc), 50, n_gen, int(rc.SEED + k + 9000)))[0]
                        for k, d in enumerate(scen)]))
    return {"P1b": filas, "P1c": {"EEV_f1": round(eev, 2), "RP_f1": round(rp, 2), "WS_f1": round(ws, 2),
                                  "VSS_f1_pct_sin_incertidumbre": round((eev - rp) / eev * 100, 2), "EVPI_f1_pct_sin_incertidumbre": round((rp - ws) / rp * 100, 2)}}


def p1d():
    b = json.load(open(DATA / "sensitivity" / RUN_S5 / "s5_mod_b_sobol.json", encoding="utf-8"))
    return {"metricas": {m: b[m] for m in b}, "parametros_sin_efecto_en_el_modelo": ["p_cross", "p_mut", "k_saa"],
            "nota": "Sus índices deberían ser ≈0; un valor distinto de cero mide ruido de la estimación."}


def p1e():
    out = {}
    for n in (25, 50):
        run_s3, _ = rv.REF[n]; df_fc, _, _, _ = s4.load_forecast_data(run_s3)
        dem = df_fc["demand_q50"].values[:n]; pri = df_fc["priority_class"].values[:n].tolist(); c = ctx_de(n, pri)
        gens = (10, 20) if PRUEBA else P1E_GENS; seeds = P1E_SEEDS[:1] if PRUEBA else P1E_SEEDS
        fr = {(g, sd): s4.run_nsga2_for_instance(dem, pri, c["D"], c["st"], n_gen=g, pop_size=s4.POP_SIZE, seed=sd, label=f"C{g}_{sd}") for g in gens for sd in seeds}
        ref = np.vstack(list(fr.values())).max(axis=0) * 1.1; hv = HV(ref_point=ref)
        tabla = {str(g): [round(float(hv(fr[(g, sd)])), 2) for sd in seeds] for g in gens}
        med = {g: float(np.mean(tabla[str(g)])) for g in gens}
        out[f"N{n}"] = {"hv_por_generaciones_y_semilla": tabla, "aumento_pct_respecto_a_200": {str(g): round((med[g] / med[gens[0]] - 1) * 100, 2) for g in gens},
                        "convergencia_practica_800_a_1600": (None if PRUEBA else bool((med[1600] / med[800] - 1) * 100 < 1.0))}
    return out


def p5():
    out = {}
    for n in (25, 50):
        fc = DATA / "forecast" / rv.REF[n][0]; cfg = json.load(open(fc / "forecast_config.json", encoding="utf-8"))
        df_fc = pd.read_csv(fc / "demand_forecast.csv").iloc[:n]; df_hist = pd.read_csv(fc / "demand_synthetic.csv", index_col=0)
        nd = len(df_hist); seed_gen = int(cfg["seed"])
        X, fecha = rv.demanda_verdadera(n, nd, seed_gen, 200 if PRUEBA else K_ORACULO, SEED_ORACULO)
        mu_base, _, dates = rv.proceso_verdadero(n, nd, seed_gen); d = (dates[-1] + pd.Timedelta(days=1)).dayofweek
        w = 1.10 if d == 0 else 1.08 if d == 4 else 0.75 if d >= 5 else 1.00
        mu = mu_base * (1.0 + 0.05 * (nd / (nd - 1) - 0.5)) * w
        naive = df_hist[[f"farm_{i:03d}" for i in range(n)]].iloc[-7].values
        rmse = lambda p: float(np.sqrt(((X - p) ** 2).mean()))
        sk = lambda p: round(100 * (1 - rmse(p) / rmse(naive)), 2)
        S_in = np.array([g.sort_values("pharmacy_id")["demand_kg"].values[:n] for _, g in pd.read_csv(fc / "saa_scenarios.csv").groupby("scenario_id")])
        tot_mod = rv.escenarios_modelo(df_fc, df_hist, 500 if PRUEBA else 20_000, SEED_ORACULO).sum(1); tot_in = S_in.sum(1)
        pub = {}
        fe = fc / "forecast_eval_verdad.json"
        if fe.exists():
            j = json.load(open(fe, encoding="utf-8"))
            pub = {k: v for k, v in pd.json_normalize(j, sep=".").iloc[0].items() if "skill" in k.lower()}
        out[f"N{n}"] = {"fecha_dia_previsto": fecha, "skill_BiLSTM_mismo_procedimiento_pct": sk(df_fc["demand_q50"].values),
                        "skill_maximo_posible_pct": sk(mu), "skill_publicados_forecast_eval_verdad": pub,
                        "CV_demanda_total_modelo_pct": round(100 * float(tot_mod.std() / tot_mod.mean()), 2),
                        "CV_demanda_total_muestra_20_escenarios_pct": round(100 * float(tot_in.std(ddof=1) / tot_in.mean()), 2)}
    return out


def main():
    global PRUEBA, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--solo", default="P1,P2,P3,P5"); ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args(); PRUEBA = a.prueba; OUT = DATA / ("consolidacion_prueba" if PRUEBA else "consolidacion")
    OUT.mkdir(parents=True, exist_ok=True); pruebas = set(a.solo.upper().split(",")); t0 = time.time()
    log(f"Pruebas {sorted(pruebas)}{' · MODO PRUEBA (sin estadísticos de decisión)' if PRUEBA else ''}")
    if "P1" in pruebas:
        r = {"P1a": {f"N{n}": p1a(n) for n in (25, 50)}}; log("P1a hecho")
        r.update(p1b_p1c()); log("P1b y P1c hechos")
        r["P1d"] = p1d(); r["P1e"] = p1e(); log("P1d y P1e hechos")
        guarda("P1_placebos_convergencia.json", r)
        if not all(r["P1a"][k]["W3"] for k in r["P1a"]):
            sys.exit("ERROR: validación W3 fallida")
    if "P5" in pruebas:
        guarda("P5_descriptivos.json", p5()); log("P5 hecho")
    if pruebas & {"P2", "P3"}:
        ref, val = {}, {}
        for n in (25, 50):
            run_s3, run_s4 = rv.REF[n]; s4_dir = DATA / "integration" / run_s4
            P = rv.prepara(DATA / "forecast" / run_s3, s4_dir)
            if P["fallos"]:
                sys.exit(f"ERROR N={n}: {P['fallos']}")
            v = rv.validaciones(P, n)
            ok_w0 = all(x for k, x in v.items() if k.startswith("V") and not k.endswith("detalle") and x is not None)
            ok_w1 = w1(P); ok_w2, det_w2 = w2(P, s4_dir)
            val[n] = {"W0_V0_V5": v, "W1": ok_w1, "W2": ok_w2, "W2_detalle": det_w2}
            guarda("validacion.json", val)
            if not (ok_w0 and ok_w1 and ok_w2):
                sys.exit(f"ERROR N={n}: validación fallida (W0 {ok_w0}, W1 {ok_w1}, W2 {ok_w2}); no se calcula ningún estadístico de decisión")
            ref[n] = p2_p3(P, s4_dir); log(f"Referencia N={n}: validaciones correctas · P2 y P3 calculados · {time.time()-t0:.0f} s")
            if not PRUEBA:
                guarda(f"referencia_N{n}.json", ref[n])
        rep, excl = {25: [], 50: []}, []; parc = OUT / "replicas_parcial"; parc.mkdir(exist_ok=True)
        dirs = sorted(DATA.glob("replicas_vss/*/integration/*_s4"))[: (1 if PRUEBA else None)]
        for s4_dir in dirs:
            cache = parc / f"{s4_dir.name}.json"
            if cache.exists() and not PRUEBA:
                r = json.load(open(cache, encoding="utf-8"))
                (excl.append(r["excluida"]) if "excluida" in r else rep[r["n"]].append(r)); continue
            summ = json.load(open(s4_dir / "s4_summary.json")); fc = s4_dir.parent.parent / "forecast" / summ["run_id_s3"]
            P = rv.prepara(fc, s4_dir)
            if P["fallos"]:
                e = {"replica": s4_dir.name, "motivo": P["fallos"]}; excl.append(e)
                json.dump({"excluida": e}, open(cache, "w", encoding="utf-8"), ensure_ascii=False); continue
            r = p2_p3(P, s4_dir); r.update({"replica": s4_dir.name, "n": P["n"]})
            if not PRUEBA:
                rep[P["n"]].append(r); json.dump(r, open(cache, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
            log(f"Réplica {s4_dir.name} (N={P['n']}) procesada · {time.time()-t0:.0f} s")
        if not PRUEBA:
            guarda("replicas.json", {"replicas": rep, "excluidas": excl})
            d = {"C1_comparacion_equilibrada": decision_c(ref, rep, "P2"), "C2_problema_de_recurso": decision_c(ref, rep, "P3"),
                 "excluidas": excl, "fecha": datetime.now().isoformat(timespec="seconds"), "protocolo": "protocolo_consolidacion.md v1.0 (enmienda 1.1)"}
            guarda("decision_C1_C2.json", d)
            log(f"DECISIÓN C1: {d['C1_comparacion_equilibrada']['veredicto']} · C2: {d['C2_problema_de_recurso']['veredicto']}")
    log(f"Fin · {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
