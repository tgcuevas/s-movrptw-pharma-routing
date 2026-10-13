import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_integration as s4
import robustez_vss as rv
import consolidacion_pruebas as cp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

DATA = Path("data")
SEED_RP_S = 3042
K_ORACULO, SEED_ORACULO = 10_000, 20261003
PRUEBA = False


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


class ProblemaRecursoS(ElementwiseProblem):
    def __init__(self, ctx, q50, S_in):
        self.ctx, self.q50, self.S_in = ctx, q50, S_in
        super().__init__(n_var=len(q50), n_obj=3, xl=0, xu=len(q50) - 1)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = rv.Plan(np.asarray(x, dtype=int), self.q50, self.ctx, "RPS").S(self.S_in)[0].mean(0)


def recurso_S(P):
    res = minimize(ProblemaRecursoS(P["ctx"], P["q50"], P["S_in"]),
                   NSGA2(pop_size=s4.POP_SIZE, sampling=s4.PermutationSampling(), crossover=s4.OrderCrossover(),
                         mutation=s4.SwapMutation(), eliminate_duplicates=True),
                   termination=("n_gen", cp.ngen(P["summ"]["n_gen"])), seed=SEED_RP_S, verbose=False)
    planes = [rv.Plan(np.asarray(x, dtype=int), P["q50"], P["ctx"], "RPS") for x in res.X]
    w6 = bool(np.allclose(np.array([p.S(P["S_in"])[0].mean(0) for p in planes]), res.F, rtol=1e-9))
    return planes, w6


def seleccion(planes_det, planes_cmp, S_in):
    Ed = np.array([p.S(S_in)[0].mean(0) for p in planes_det]); Ec = np.array([p.S(S_in)[0].mean(0) for p in planes_cmp])
    nd_d, nd_c = rv.no_dominados(Ed), rv.no_dominados(Ec)
    return {"E1": (int(np.argmin(Ed[:, 1])), int(np.argmin(Ec[:, 1]))), "E2": (nd_d[rv.topsis(Ed[nd_d])], nd_c[rv.topsis(Ec[nd_c])])}


def desvios(planes_det, planes_cmp, S_in, oos):
    sel = seleccion(planes_det, planes_cmp, S_in); out = {}
    for tipo, X in oos.items():
        out[tipo] = {}
        for e, (i, j) in sel.items():
            dd, dc = planes_det[i].S(X)[1], planes_cmp[j].S(X)[1]; d = dd - dc
            h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            out[tipo][e] = {"delta": round(float(d.mean()), 4), "ic95": [round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)],
                            "desvios_det": round(float(dd.mean()), 4), "desvios_saa": round(float(dc.mean()), 4)}
    return out


def procesa(tarea):
    etiqueta, fc_dir, s4_dir, es_ref = tarea
    cp.PRUEBA = PRUEBA
    P = rv.prepara(Path(fc_dir), Path(s4_dir))
    if P["fallos"]:
        return {"etiqueta": etiqueta, "excluida": P["fallos"]}
    r = {"etiqueta": etiqueta, "n": P["n"], "es_ref": es_ref}
    if es_ref:
        v = rv.validaciones(P, P["n"])
        r["W0"] = bool(all(x for k, x in v.items() if k.startswith("V") and not k.endswith("detalle") and x is not None))
    oos = cp.oos_de(P)
    planes_rps, w6 = recurso_S(P); r["W6"] = w6
    r6 = cp.estimandos(P["planes_ev"], planes_rps, P["S_in"], oos)
    r7 = desvios(cp.ev_equilibrado(P), P["planes_saa"], P["S_in"], oos)
    if not PRUEBA:
        r["P6"], r["P7"] = r6, r7
    return r


def tareas():
    T = [(f"referencia_N{n}", str(DATA / "forecast" / rv.REF[n][0]), str(DATA / "integration" / rv.REF[n][1]), True) for n in (25, 50)]
    for s4_dir in sorted(DATA.glob("replicas_vss/*/integration/*_s4")):
        summ = json.load(open(s4_dir / "s4_summary.json")); fc = s4_dir.parent.parent / "forecast" / summ["run_id_s3"]
        T.append((s4_dir.name, str(fc), str(s4_dir), False))
    return T[:3] if PRUEBA else T


def decision_c5(ref, rep):
    conf, refu, info = True, False, {}
    for n in (25, 50):
        for t in ("oos_modelo", "oos_real"):
            a, b = ref[n]["P7"][t]["E1"], ref[n]["P7"][t]["E2"]
            cnt = sum(r["P7"][t]["E1"]["delta"] > 0 for r in rep[n]); tot = len(rep[n])
            info[f"N{n}_{t}"] = {"E1_ref": a, "E2_ref": b, "replicas_delta_E1_pos": f"{cnt}/{tot}"}
            conf &= a["ic95"][0] > 0 and b["ic95"][0] > 0 and cnt >= 8
            refu |= a["delta"] <= 0 or cnt < 6
    return {"veredicto": "refutada" if refu else "confirmada" if conf else "no concluyente", "detalle": info}


def exceso_rmse(fc_dir, n):
    cfg = json.load(open(fc_dir / "forecast_config.json", encoding="utf-8"))
    df_fc = pd.read_csv(fc_dir / "demand_forecast.csv").iloc[:n]; df_hist = pd.read_csv(fc_dir / "demand_synthetic.csv", index_col=0)
    nd, seed_gen = len(df_hist), int(cfg["seed"])
    X, _ = rv.demanda_verdadera(n, nd, seed_gen, 200 if PRUEBA else K_ORACULO, SEED_ORACULO)
    mu_base, _, dates = rv.proceso_verdadero(n, nd, seed_gen); d = (dates[-1] + pd.Timedelta(days=1)).dayofweek
    w = 1.10 if d == 0 else 1.08 if d == 4 else 0.75 if d >= 5 else 1.00
    mu = mu_base * (1.0 + 0.05 * (nd / (nd - 1) - 0.5)) * w
    rmse = lambda p: float(np.sqrt(((X - p) ** 2).mean()))
    r0 = rmse(mu)
    return {"exceso_BiLSTM_pct": round(100 * (rmse(df_fc["demand_q50"].values) / r0 - 1), 3),
            "exceso_ensemble_pct": round(100 * (rmse(df_fc["ens_q50"].values) / r0 - 1), 3)}


def c7():
    ref = {f"N{n}": exceso_rmse(DATA / "forecast" / rv.REF[n][0], n) for n in (25, 50)}
    rep = {25: [], 50: []}
    for s4_dir in sorted(DATA.glob("replicas_vss/*/integration/*_s4"))[: (2 if PRUEBA else None)]:
        summ = json.load(open(s4_dir / "s4_summary.json")); n = int(summ["n_pharmacies"])
        rep[n].append({"replica": s4_dir.name, **exceso_rmse(s4_dir.parent.parent / "forecast" / summ["run_id_s3"], n)})
    cnt = {n: sum(r["exceso_BiLSTM_pct"] < 2.0 for r in rep[n]) for n in (25, 50)}
    ver = "refutada" if any(cnt[n] < 6 for n in (25, 50)) else "confirmada" if all(cnt[n] >= 8 for n in (25, 50)) else "no concluyente"
    return {"referencia_descriptiva": ref, "replicas": rep, "replicas_exceso_menor_2pct": {f"N{n}": f"{cnt[n]}/{len(rep[n])}" for n in (25, 50)},
            "veredicto": None if PRUEBA else ver}


def main():
    global PRUEBA
    ap = argparse.ArgumentParser(); ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prueba", action="store_true"); a = ap.parse_args(); PRUEBA = a.prueba; cp.PRUEBA = PRUEBA
    out = DATA / ("consolidacion_prueba" if PRUEBA else "consolidacion") / "complementarias"; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log(f"C4, C5 y C7{' · MODO PRUEBA (sin veredictos)' if PRUEBA else ''}")
    r7c = c7(); json.dump(r7c, open(out / "C7_prediccion_replicas.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    log(f"C7 hecho · {time.time()-t0:.0f} s")
    T = tareas(); res = []
    with ProcessPoolExecutor(max_workers=min(a.procesos, len(T)), initializer=_init, initargs=(PRUEBA,)) as ex:
        for r in ex.map(procesa, T, chunksize=1):
            res.append(r); log(f"{r['etiqueta']} procesada{' (EXCLUIDA)' if 'excluida' in r else ''} · {time.time()-t0:.0f} s")
    refs = {r["n"]: r for r in res if r.get("es_ref")}
    malos = [r["etiqueta"] for r in res if r.get("es_ref") and not (r.get("W0") and r.get("W6"))] + [r["etiqueta"] for r in res if "excluida" not in r and not r.get("W6")]
    json.dump({"validaciones": {r["etiqueta"]: {"W0": r.get("W0"), "W6": r.get("W6")} for r in res if "excluida" not in r},
               "excluidas": [r for r in res if "excluida" in r]}, open(out / "validacion_complementarias.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if malos:
        sys.exit(f"ERROR: validación fallida en {sorted(set(malos))}; no se calcula ningún veredicto")
    if PRUEBA:
        log(f"Comprobación técnica correcta · {len(res)} instancias · {time.time()-t0:.0f} s"); return
    rep = {25: [r for r in res if not r.get("es_ref") and "excluida" not in r and r["n"] == 25],
           50: [r for r in res if not r.get("es_ref") and "excluida" not in r and r["n"] == 50]}
    json.dump({"referencias": refs, "replicas": rep}, open(out / "C4_C5_resultados.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    d = {"C4_recurso_simulacion": cp.decision_c(refs, rep, "P6"), "C5_mecanismo_desvios": decision_c5(refs, rep),
         "C7_prediccion_replicas": r7c["veredicto"], "excluidas": [r for r in res if "excluida" in r],
         "fecha": datetime.now().isoformat(timespec="seconds"), "protocolo": "protocolo_consolidacion.md (enmienda 1.2)"}
    json.dump(d, open(out / "decision_C4_C5_C7.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    log(f"DECISIÓN C4: {d['C4_recurso_simulacion']['veredicto']} · C5: {d['C5_mecanismo_desvios']['veredicto']} · C7: {d['C7_prediccion_replicas']}")
    log(f"Fin · {time.time()-t0:.0f} s")


def _init(prueba):
    global PRUEBA
    PRUEBA = prueba; cp.PRUEBA = prueba


if __name__ == "__main__":
    main()
