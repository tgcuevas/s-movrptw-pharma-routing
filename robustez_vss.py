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

DATA = Path("data")
OUT = DATA / "robustez"
REF = {25: ("20260615_200957", "20260615_201043_s4"), 50: ("20260615_205432", "20260615_205516_s4")}
K_OOS, SEED_OOS_MODELO, SEED_OOS_REAL = 1000, 20260925, 20260926
PRIORITY_PROBS = [0.30, 0.20, 0.50]
VSS_DESVIO_REAL_N25 = 6.12


def correlacion(df_hist):
    resid = df_hist.diff(7).dropna(); corr = resid.corr().values; n = corr.shape[0]
    corr = 0.9 * corr + 0.1 * np.eye(n)
    w, V = np.linalg.eigh(corr); w = np.clip(w, 1e-6, None)
    return V @ np.diag(w) @ V.T


def escenarios_modelo(df_fc, df_hist, k, seed):
    rng = np.random.default_rng(seed)
    mu, sigma = df_fc["mu_real"].values, df_fc["sigma_real"].values; n = len(mu)
    L = np.linalg.cholesky(correlacion(df_hist[[f"farm_{i:03d}" for i in range(n)]]))
    return np.array([np.clip(mu + sigma * (L @ rng.standard_normal(n)), 0.1, None) for _ in range(k)])


def proceso_verdadero(n, n_days, seed):
    rng = np.random.default_rng(seed)
    mu_base = rng.uniform(3.0, 18.0, size=n)
    rng.choice([1, 2, 3], size=n, p=PRIORITY_PROBS)
    dates = pd.date_range("2024-06-01", periods=n_days, freq="D"); dow = dates.dayofweek.values
    w = np.where(dow == 0, 1.10, np.where(dow == 4, 1.08, np.where(dow >= 5, 0.75, 1.00)))
    trend = 1.0 + 0.05 * (np.arange(n_days) / (n_days - 1) - 0.5)
    dem = np.zeros((n_days, n))
    for i in range(n):
        g = rng.gamma(shape=4.0, scale=mu_base[i] * trend * w / 4.0)
        dem[:, i] = rng.poisson(g).astype(float)
    return mu_base, np.clip(dem, 0.5, None), dates


def demanda_verdadera(n, n_days, seed_gen, k, seed):
    mu_base, _, dates = proceso_verdadero(n, n_days, seed_gen)
    fecha = dates[-1] + pd.Timedelta(days=1); d = fecha.dayofweek
    w = 1.10 if d == 0 else 1.08 if d == 4 else 0.75 if d >= 5 else 1.00
    mu = mu_base * (1.0 + 0.05 * (n_days / (n_days - 1) - 0.5)) * w
    rng = np.random.default_rng(seed)
    return np.clip(rng.poisson(rng.gamma(shape=4.0, scale=mu / 4.0, size=(k, n))).astype(float), 0.5, None), str(fecha.date())


def simula(R, S, D, st, alpha, b, Q, beta):
    K = S.shape[0]; f = np.zeros((K, 3)); ndes = np.zeros(K)
    for r in R:
        t = np.zeros(K); carga = np.full(K, float(Q)); prev = 0
        for i in r:
            node = i + 1
            t += D[prev, node]
            falta = S[:, i] - carga
            nref = np.where(falta > 0, np.ceil(falta / Q - 1e-12), 0.0)
            t += nref * (D[node, 0] + D[0, node]); carga = carga + nref * Q - S[:, i]
            t += st[node]
            f[:, 1] += alpha[i] * np.maximum(0.0, t - b[i]); ndes += nref; prev = node
        t += D[prev, 0]
        f[:, 0] += t; f[:, 2] += beta * t * (S[:, r].sum(1) / Q)
    return f, ndes


def rutas(perm, dem, Q):
    R, cur, cd = [], [], 0.0
    for i in np.asarray(perm, dtype=int):
        if cd + dem[i] > Q and cur:
            R.append(cur); cur, cd = [], 0.0
        cur.append(int(i)); cd += float(dem[i])
    R.append(cur)
    return R


class Plan:
    def __init__(self, perm, dem, ctx, origen):
        self.R = rutas(perm, dem, ctx["Q"]); self.ctx = ctx; self.origen = origen; self.tot = float(dem.sum())
        f1, f2, f3, _ = s4._eval_route(np.asarray(perm, dtype=int), dem, ctx["D"], ctx["st"], ctx["pri"], ctx["Q"])
        self.base = np.array([f1, f2, f3])

    def S(self, X):
        c = self.ctx
        return simula(self.R, X, c["D"], c["st"], c["alpha"], c["b"], c["Q"], s4.CO2_ALPHA)

    def M(self, X, pen=None):
        pen = s4.PENALTY_EXTRA_TRIP if pen is None else pen; Q = self.ctx["Q"]
        extra = sum(np.maximum(0, np.ceil(X[:, r].sum(1) / Q) - 1) for r in self.R)
        out = np.empty((X.shape[0], 3)); out[:, 0] = self.base[0] + pen * extra
        out[:, 1] = self.base[1]; out[:, 2] = self.base[2] * (X.sum(1) / self.tot)
        return out, extra


def topsis(M):
    Mn = M / (np.linalg.norm(M, axis=0) + 1e-12)
    di = np.linalg.norm(Mn - Mn.min(0), axis=1); da = np.linalg.norm(Mn - Mn.max(0), axis=1)
    return int(np.argmax(da / (di + da + 1e-12)))


def no_dominados(M):
    return [i for i in range(len(M)) if not any(np.all(M[j] <= M[i]) and np.any(M[j] < M[i]) for j in range(len(M)) if j != i)]


def ic_vss(fe, fs):
    d = fe - fs; m = d.mean(); h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)); base = fe.mean()
    return {"vss_pct": round(100 * m / base, 3), "ic95_pct": [round(100 * (m - h) / base, 3), round(100 * (m + h) / base, 3)],
            "favorables_pct": round(100 * float((d > 0).mean()), 1)}, d


def prepara(fc_dir, s4_dir):
    summ = json.load(open(s4_dir / "s4_summary.json", encoding="utf-8"))
    n, n_gen = summ["n_pharmacies"], summ["n_gen"]; cfg = json.load(open(fc_dir / "forecast_config.json", encoding="utf-8"))
    df_fc = pd.read_csv(fc_dir / "demand_forecast.csv").iloc[:n]; df_saa = pd.read_csv(fc_dir / "saa_scenarios.csv")
    df_hist = pd.read_csv(fc_dir / "demand_synthetic.csv", index_col=0)
    D, st = s4.get_distance_matrix(n), s4.get_service_times(n); Q = s4.VEHICLE_CAPACITY
    pri = df_fc["priority_class"].values.tolist(); q50 = df_fc["demand_q50"].values
    ctx = {"D": D, "st": st, "pri": pri, "Q": Q, "alpha": np.array([s4._ALPHA_TW[p] for p in pri], float),
           "b": np.array([s4._TW_CLASS[p][1] for p in pri], float)}
    ks = sorted(df_saa["scenario_id"].unique())
    S_in = np.array([df_saa[df_saa.scenario_id == k].sort_values("pharmacy_id")["demand_kg"].values[:n] for k in ks])
    fallos = []
    if not np.allclose(np.round(escenarios_modelo(df_fc, df_hist, len(ks), int(cfg["seed"])), 4), S_in, atol=1e-4):
        fallos.append("escenarios SAA no reproducidos")
    F_ev, X_ev = s4.run_nsga2_for_instance(q50, pri, D, st, n_gen=n_gen, pop_size=s4.POP_SIZE, seed=s4.SEED, label="EV", return_X=True)
    can = pd.read_csv(s4_dir / "s4_ev_pareto.csv")[["f1_s", "f2_pond", "f3_co2"]].values
    if not np.allclose(np.array(sorted(map(tuple, np.round(F_ev, 2)))), np.array(sorted(map(tuple, np.round(can, 2)))), atol=0.02):
        fallos.append("frente EV no reproducido")
    fr = pd.read_csv(s4_dir / "s4_saa_fronts.csv").sort_values("scenario_id"); planes_saa, frentes = [], []
    for j, k in enumerate(ks):
        prio_k = df_saa[df_saa.scenario_id == k].sort_values("pharmacy_id")["priority_class"].values[:n].tolist()
        F_k, X_k = s4.run_nsga2_for_instance(S_in[j], prio_k, D, st, n_gen=n_gen, pop_size=s4.POP_SIZE,
                                             seed=int(s4.SEED + k), label=f"SAA_k{int(k):02d}", return_X=True)
        row = fr[fr.scenario_id == k].iloc[0]
        if not (len(F_k) == int(row.n_pareto_sols) and abs(F_k[:, 0].min() - row.f1_best) < 0.02 and abs(F_k[:, 1].min() - row.f2_best) < 0.02):
            fallos.append(f"frente SAA k={k} no reproducido")
        frentes.append((j, F_k, X_k)); planes_saa += [Plan(x, S_in[j], ctx, f"SAA_k{int(k):02d}") for x in X_k]
    planes_ev = [Plan(x, q50, ctx, "EV") for x in X_ev]
    return {"n": n, "ctx": ctx, "S_in": S_in, "planes_ev": planes_ev, "planes_saa": planes_saa, "F_ev": F_ev,
            "frentes": frentes, "comp_ev": s4.select_compromise(F_ev)[1], "summ": summ, "cfg": cfg, "fallos": fallos,
            "df_fc": df_fc, "df_hist": df_hist, "q50": q50}


def estimandos(P, oos_sets, detalle=False):
    E_ev = np.array([p.S(P["S_in"])[0].mean(0) for p in P["planes_ev"]])
    E_sa = np.array([p.S(P["S_in"])[0].mean(0) for p in P["planes_saa"]])
    iB = {o: (int(np.argmin(E_ev[:, o])), int(np.argmin(E_sa[:, o]))) for o in (0, 1)}
    nd = no_dominados(E_sa); iA = (topsis(E_ev), nd[topsis(E_sa[nd])])
    sel = {"E1": ("f2", iB[1]), "E2": ("f2", iA), "E3": ("f1", iB[0]), "E4": ("f1", iA)}
    out, filas = {}, {}
    for tipo, X in oos_sets.items():
        out[tipo] = {}
        for e, (lab, (ie, isa)) in sel.items():
            o = 0 if lab == "f1" else 1
            r, d = ic_vss(P["planes_ev"][ie].S(X)[0][:, o], P["planes_saa"][isa].S(X)[0][:, o])
            out[tipo][e] = r; filas[f"{tipo}_{e}"] = d
    if detalle:
        E_evM = np.array([p.M(P["S_in"])[0].mean(0) for p in P["planes_ev"]]); E_saM = np.array([p.M(P["S_in"])[0].mean(0) for p in P["planes_saa"]])
        ndM = no_dominados(E_saM)
        selM = {"E1": ("f2", (int(np.argmin(E_evM[:, 1])), int(np.argmin(E_saM[:, 1])))),
                "E2": ("f2", (topsis(E_evM), ndM[topsis(E_saM[ndM])])),
                "E3": ("f1", (int(np.argmin(E_evM[:, 0])), int(np.argmin(E_saM[:, 0])))),
                "E4": ("f1", (topsis(E_evM), ndM[topsis(E_saM[ndM])]))}
        out["modelo_M"] = {tipo: {e: ic_vss(P["planes_ev"][ie].M(X)[0][:, 0 if lab == "f1" else 1],
                                            P["planes_saa"][isa].M(X)[0][:, 0 if lab == "f1" else 1])[0]
                                  for e, (lab, (ie, isa)) in selM.items()} for tipo, X in oos_sets.items()}
        ref = np.vstack([E_ev, E_sa[nd]]); lo, hi = ref.min(0), ref.max(0)
        Z = lambda M: (M - lo) / (hi - lo + 1e-12)
        reglas = {"x̄ de la memoria": (P["comp_ev"], nd[topsis(E_sa[nd])]), "TOPSIS": iA,
                  "suma ponderada": (int(np.argmin(Z(E_ev).sum(1))), nd[int(np.argmin(Z(E_sa[nd]).sum(1)))]),
                  "punto de codo": (int(np.argmin(np.linalg.norm(Z(E_ev), axis=1))), nd[int(np.argmin(np.linalg.norm(Z(E_sa[nd]), axis=1)))])}
        out["reglas_compromiso"] = {tipo: {nom: {lab: ic_vss(P["planes_ev"][ie].S(X)[0][:, o], P["planes_saa"][isa].S(X)[0][:, o])[0]["vss_pct"]
                                                for o, lab in ((0, "f1"), (1, "f2"), (2, "f3"))} for nom, (ie, isa) in reglas.items()}
                                    for tipo, X in oos_sets.items()}
    return out, pd.DataFrame({k: pd.Series(v) for k, v in filas.items()})


def validaciones(P, n):
    v = {}; ctx = P["ctx"]
    _, hist, _ = proceso_verdadero(n, len(P["df_hist"]), int(P["cfg"]["seed"]))
    v["V0"] = bool(np.array_equal(hist, P["df_hist"].values))
    ok = True
    for p in P["planes_ev"]:
        ok &= np.allclose(p.S(P["q50"][None, :])[0][0], p.base, rtol=1e-6)
    for j, F_k, X_k in P["frentes"]:
        for f, x in zip(F_k, X_k):
            pl = Plan(x, P["S_in"][j], ctx, "v"); ok &= np.allclose(pl.S(P["S_in"][j][None, :])[0][0], f, rtol=1e-6)
    v["V1"] = bool(ok)
    X = np.vstack([P["S_in"], escenarios_modelo(P["df_fc"], P["df_hist"], 300, 7)])
    ok2 = ok4a = True
    for p in P["planes_ev"] + P["planes_saa"][::10]:
        fs, nd_ = p.S(X); fm, ex = p.M(X); sin = nd_ == 0
        ok2 &= np.allclose(fs[sin][:, :2], fm[sin][:, :2], rtol=1e-9); ok4a &= np.array_equal(nd_, ex)
    v["V2"] = bool(ok2)
    Dh = np.zeros((4, 4)); Dh[0, 1] = 5; Dh[1, 2] = 7; Dh[2, 0] = Dh[0, 2] = 9; Dh[2, 3] = 4; Dh[3, 0] = 6
    fh, nh = simula([[0, 1, 2]], np.array([[60.0, 80.0, 30.0]]), Dh, np.array([0, 10, 10, 10.0]), np.ones(3),
                    np.array([100.0, 45.0, 100.0]), 100.0, 0.05)
    v["V3"] = bool(np.allclose(fh[0], [70.0, 5.0, 5.95]) and nh[0] == 1)
    rng = np.random.default_rng(11); todos = P["planes_ev"] + P["planes_saa"]; nodec = True
    for _ in range(200):
        p = todos[rng.integers(len(todos))]; s = X[rng.integers(len(X))][None, :]
        nodec &= p.S(s * 1.1)[1][0] >= p.S(s)[1][0]
    ident = True; Q = ctx["Q"]; D = ctx["D"]
    for p in P["planes_ev"] + P["planes_saa"][::10]:
        f1s = p.S(X)[0][:, 0]; t0 = p.S(np.zeros((1, X.shape[1])))[0][0, 0]; extra = np.zeros(len(X))
        for r in p.R:
            usadas = np.maximum(1, np.ceil(np.cumsum(X[:, r], axis=1) / Q - 1e-12))
            recargas = np.diff(np.hstack([np.ones((len(X), 1)), usadas]), axis=1)
            extra += recargas @ np.array([D[i + 1, 0] + D[0, i + 1] for i in r])
        ident &= np.allclose(f1s, t0 + extra, rtol=0, atol=1e-6)
    v["V4"] = bool(ok4a and nodec and ident)
    v["V4_detalle"] = {"desvios_formula": bool(ok4a), "desvios_no_decrecen": bool(nodec), "identidad_f1": bool(ident)}
    xb = P["planes_ev"][P["comp_ev"]]; rp = P["summ"]["VSS"]["RP"]["f1"]; e = xb.S(P["S_in"])[0][:, 0].mean()
    vss = 100 * (e - rp) / e; D = ctx["D"]
    v["V5"] = bool(abs(vss - VSS_DESVIO_REAL_N25) <= 0.05) if n == 25 else None
    v["V5_detalle"] = {"vss_f1_simulado_pct": round(vss, 3), "asimetria_max_matriz_s": float(np.abs(D - D.T).max())}
    return v


def decision(ref, rep):
    T = ("oos_modelo", "oos_real")
    def ok_ref(e, cmp):
        return all(cmp(ref[n][t][e]) for n in (25, 50) for t in T)
    A1c = ok_ref("E1", lambda r: r["ic95_pct"][0] > 10) and ok_ref("E2", lambda r: r["ic95_pct"][0] > 10)
    A1r = all(sum(r[t]["E1"]["vss_pct"] > 10 for r in rep[n]) >= 8 for n in (25, 50) for t in T)
    A1f = any(ref[n][t]["E1"]["vss_pct"] < 10 for n in (25, 50) for t in T) or \
        any(sum(r[t]["E1"]["vss_pct"] > 10 for r in rep[n]) < 6 for n in (25, 50) for t in T)
    A1 = "confirmada" if (A1c and A1r) else "refutada" if A1f else "no concluyente"
    med = {n: {t: float(np.median([r[t]["E3"]["vss_pct"] for r in rep[n]])) for t in T} for n in (25, 50)}
    A2c = all(med[n][t] < 5 and ref[n][t]["E3"]["ic95_pct"][1] < 5 for n in (25, 50) for t in T)
    A2f = all(med[n][t] >= 5 for n in (25, 50) for t in T)
    A2 = "confirmada" if A2c else "refutada" if A2f else "no concluyente"
    return {"A1_tardanza_solida": A1, "A2_tiempo_modesto": A2,
            "A3_independencia_entrenamiento": "confirmada" if (A1 == "confirmada" and A2 == "confirmada") else "ver recuentos",
            "mediana_E3_replicas": med,
            "recuento_E1_mayor_10": {n: {t: f"{sum(r[t]['E1']['vss_pct'] > 10 for r in rep[n])}/{len(rep[n])}" for t in T} for n in (25, 50)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sin-replicas", action="store_true"); ap.add_argument("--reusar-referencia", action="store_true"); ap.add_argument("--n", default="25,50")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ref, val = {}, {}
    if a.reusar_referencia and all((OUT / f"referencia_N{n}.json").exists() for n in (25, 50)):
        ref = {n: json.load(open(OUT / f"referencia_N{n}.json", encoding="utf-8")) for n in (25, 50)}
        print("Referencia reutilizada de", OUT, flush=True)
    for n in ([] if ref else [int(x) for x in a.n.split(",")]):
        run_s3, run_s4 = REF[n]
        P = prepara(DATA / "forecast" / run_s3, DATA / "integration" / run_s4)
        if P["fallos"]:
            sys.exit(f"ERROR N={n}: {P['fallos']}")
        v = validaciones(P, n); val[n] = v
        if not all(x for k, x in v.items() if k.startswith("V") and not k.endswith("detalle") and x is not None):
            json.dump(val, open(OUT / "validacion.json", "w"), indent=2, default=str)
            sys.exit(f"ERROR N={n}: validación fallida {v}")
        Xr, fecha = demanda_verdadera(n, len(P["df_hist"]), int(P["cfg"]["seed"]), K_OOS, SEED_OOS_REAL)
        oos = {"dentro": P["S_in"], "oos_modelo": escenarios_modelo(P["df_fc"], P["df_hist"], K_OOS, SEED_OOS_MODELO), "oos_real": Xr}
        r, filas = estimandos(P, oos, detalle=True)
        r.update({"n": n, "run_s4": run_s4, "fecha_dia_previsto": fecha, "planes_ev": len(P["planes_ev"]), "planes_saa": len(P["planes_saa"]),
                  "demanda_media_total_kg": {k: round(float(X.sum(1).mean()), 2) for k, X in oos.items()}})
        ref[n] = r
        json.dump(r, open(OUT / f"referencia_N{n}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        filas.to_csv(OUT / f"referencia_N{n}_oos.csv", index=False)
        print(f"N={n}: validación {v} · {len(P['planes_saa'])} planes SAA · {time.time()-t0:.0f} s", flush=True)
        for t in ("oos_modelo", "oos_real"):
            print("   ", t, {e: (x["vss_pct"], x["ic95_pct"]) for e, x in r[t].items()}, flush=True)
    if val:
        json.dump(val, open(OUT / "validacion.json", "w"), indent=2, default=str)
    if a.sin_replicas:
        return
    rep = {25: [], 50: []}; excl = []; parc = OUT / "replicas_parcial"; parc.mkdir(exist_ok=True)
    for s4_dir in sorted(DATA.glob("replicas_vss/*/integration/*_s4")):
        cache = parc / f"{s4_dir.name}.json"
        if cache.exists():
            r = json.load(open(cache, encoding="utf-8"))
            if "excluida" in r:
                excl.append(r["excluida"])
            else:
                rep[r["n"]].append(r)
            continue
        summ = json.load(open(s4_dir / "s4_summary.json")); fc = s4_dir.parent.parent / "forecast" / summ["run_id_s3"]
        P = prepara(fc, s4_dir)
        if P["fallos"]:
            e = {"replica": s4_dir.name, "motivo": P["fallos"]}; excl.append(e)
            json.dump({"excluida": e}, open(cache, "w", encoding="utf-8"), ensure_ascii=False); continue
        Xr, _ = demanda_verdadera(P["n"], len(P["df_hist"]), int(P["cfg"]["seed"]), K_OOS, SEED_OOS_REAL)
        oos = {"oos_modelo": escenarios_modelo(P["df_fc"], P["df_hist"], K_OOS, SEED_OOS_MODELO), "oos_real": Xr}
        r, _ = estimandos(P, oos); r["replica"] = s4_dir.name; r["n"] = P["n"]; rep[P["n"]].append(r)
        json.dump(r, open(cache, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  réplica {s4_dir.name} (N={P['n']}): E1 real {r['oos_real']['E1']['vss_pct']} · E3 real {r['oos_real']['E3']['vss_pct']} · {time.time()-t0:.0f} s", flush=True)
    json.dump({"replicas": rep, "excluidas": excl}, open(OUT / "replicas.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    d = decision(ref, rep); d["excluidas"] = excl; d["fecha"] = datetime.now().isoformat(timespec="seconds")
    json.dump(d, open(OUT / "decision.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("DECISIÓN:", json.dumps(d, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
