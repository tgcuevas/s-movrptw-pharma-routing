import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

D = Path("data")
S3 = {25: D / "forecast/20260615_200957", 50: D / "forecast/20260615_205432"}
S4 = {25: D / "integration/20260615_201043_s4", 50: D / "integration/20260615_205516_s4"}
S5 = D / "sensitivity/20260615_201114_s5"
S6 = D / "scale_n50/20260615_205431_s6"
C = D / "consolidacion"
CC = C / "complementarias"
C_EEV, C_RP, C_HDR, C_ALT, C_VSS = "#EF6F6C", "#5AAEEB", "#2F4858", "#F2F2F2", "#E6F4EA"
ALG = {"NSGA-II": "#4A86C8", "NSGA-III": "#D65F5F", "MOEA/D": "#5B9E5B"}
DPI = 200


def J(p):
    return json.load(open(p, encoding="utf-8"))


def coma(x, nd=1):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)).replace(".", ",")


def tabla(ax, filas, cab, resaltar_ultima=False, fs=10, escala=1.7, anchos=None):
    t = ax.table(cellText=filas, colLabels=cab, cellLoc="center", loc="center", colWidths=anchos)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, escala)
    for (r, c), cel in t.get_celld().items():
        if r == 0:
            cel.set_facecolor(C_HDR); cel.set_text_props(color="white", fontweight="bold")
        elif resaltar_ultima and r == len(filas):
            cel.set_facecolor(C_VSS); cel.set_text_props(fontweight="bold", color="#1E6B34")
        elif r % 2 == 0:
            cel.set_facecolor(C_ALT)
    return t


def cajas(ax, datos, etiquetas, colores):
    bp = ax.boxplot(datos, tick_labels=etiquetas, patch_artist=True, widths=0.55,
                    medianprops=dict(color="#E0A030", lw=1.6))
    for caja, col in zip(bp["boxes"], colores):
        caja.set_facecolor(col); caja.set_alpha(0.9)
    ax.grid(axis="y", alpha=0.3)


def figura_44(out):
    d = S4[25]
    sm, rc = J(d / "s4_summary.json"), J(d / "recompute_eev_recurso.json")
    ev = pd.read_csv(d / "s4_ev_pareto.csv")
    fr = pd.read_csv(d / "s4_saa_fronts.csv").sort_values("scenario_id")
    ee = pd.read_csv(d / "s4_eev_eval.csv").sort_values("scenario_id")
    esc = pd.DataFrame(rc["escenarios"]).sort_values("scenario_id")
    EEV, RP, VSSp = rc["EEV_corregido"], rc["RP"], rc["VSS_pct_corregido"]

    fig, axs = plt.subplots(2, 2, figsize=(13.4, 10.1))
    ax = axs[0, 0]
    ax.scatter(fr["f1_compromise"] / 3600, fr["f2_compromise"] / 1e3, s=28, c="#9DB9D8", alpha=0.85,
               label="Compromiso SAA por escenario")
    evp = ev[ev["is_compromise"].astype(str) != "True"]
    ax.scatter(evp["f1_s"] / 3600, evp["f2_pond"] / 1e3, s=40, c="#D93A35", label="Pareto EV (det.)")
    evc = ev[ev["is_compromise"].astype(str) == "True"].iloc[0]
    ax.scatter(evc["f1_s"] / 3600, evc["f2_pond"] / 1e3, marker="*", s=260, c="#B01E1E", label="Compromiso EV (x̄)")
    rb = sm["robust_SAA"]["f_robust"]
    ax.scatter(rb[0] / 3600, rb[1] / 1e3, marker="D", s=110, c="#1565C0", label="Robusta SAA (TOPSIS)")
    ax.set_xlabel("f₁ — Tiempo total ruta (h)"); ax.set_ylabel("f₂ — Tardanza ponderada (×10³)")
    ax.set_title("(A) Frente EV y compromisos SAA (f₁ × f₂)", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
    ax = axs[0, 1]
    x, w = np.arange(3), 0.36
    rp_rel = [100 * RP[o] / EEV[o] for o in ("f1", "f2", "f3")]
    ax.bar(x - w / 2, [100] * 3, w, color=C_EEV, label="EEV (det.)")
    b = ax.bar(x + w / 2, rp_rel, w, color=C_RP, label="RP  (SAA)")
    for rect, v in zip(b, rp_rel):
        ax.text(rect.get_x() + rect.get_width() / 2, v - 4, coma(v), ha="center", va="top", color="white",
                fontweight="bold", fontsize=9)
    for i, o in enumerate(("f1", "f2", "f3")):
        ax.text(i, 103, f"VSS={coma(VSSp[o])}%", ha="center", color="#1E6B34", fontweight="bold", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["f₁ (tiempo)", "f₂ (tardanza)", "f₃ (CO₂ proxy)"])
    ax.set_ylim(0, 115); ax.set_ylabel("Valor relativo (EEV = 100)")
    ax.set_title("(B) EEV vs. RP por objetivo (EEV = 100)", fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    ax = axs[1, 0]
    col = ["#E57373" if v > 0 else "#F4B6B6" for v in esc["viajes_extra"]]
    ax.bar(esc["scenario_id"], esc["f1_eev_corregido"] / 3600, color=col, width=0.8)
    ax.axhline(EEV["f1"] / 3600, ls="--", c="#B01E1E", lw=1.2, label=f"EEV f₁ media = {coma(EEV['f1'] / 3600, 2)} h")
    ax.set_ylim(0, 7.2); ax.set_xlabel("Escenario SAA (k)"); ax.set_ylabel("f₁ — Tiempo ruta (h)", color="#B01E1E")
    ax2 = ax.twinx()
    ax2.plot(ee["scenario_id"], ee["f3_ev_eval"], "-o", c="#7B1FA2", ms=4, label="f₃ EEV (kg CO₂)")
    ax2.set_ylabel("f₃ — CO₂ proxy (kg)", color="#7B1FA2")
    lo, hi = ee["f3_ev_eval"].min(), ee["f3_ev_eval"].max()
    ax2.set_ylim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper center", ncol=2)
    ax.set_xticks(range(0, len(esc), 2))
    ax.set_title(f"(C) EEV de x̄ por escenario (barras oscuras: {rc['viajes_extra']['recurso_por_ruta']} viajes de recurso)",
                 fontweight="bold", fontsize=10.5)
    ax = axs[1, 1]; ax.axis("off")
    ax.set_title("(D) Tabla VSS — Value of Stochastic Solution", fontweight="bold")
    vabs = {o: EEV[o] - RP[o] for o in EEV}
    filas = [["EEV (det.)", coma(EEV["f1"] / 3600, 2), coma(EEV["f2"] / 1e3, 2), coma(EEV["f3"], 3)],
             ["RP (SAA)", coma(RP["f1"] / 3600, 2), coma(RP["f2"] / 1e3, 2), coma(RP["f3"], 3)],
             ["VSS (abs.)", coma(vabs["f1"] / 3600, 2), coma(vabs["f2"] / 1e3, 2), coma(vabs["f3"], 3)],
             ["VSS (%)", coma(VSSp["f1"]) + "%", coma(VSSp["f2"]) + "%", coma(VSSp["f3"]) + "%"]]
    tabla(ax, filas, ["Métrica", "f₁ (tiempo, h)", "f₂ (tardanza, ×10³)", "f₃ (CO₂, kg)"], resaltar_ultima=True)
    fig.tight_layout(); fig.savefig(out / "Figura_4_4.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def figura_42(out):
    p4 = pd.read_csv(C / "P4_indicadores.csv")
    fams = ["C1-100", "C2-100", "R1-100", "R2-100", "RC1-100", "RC2-100"]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for ax, k in zip(axs.flat, fams):
        g = p4[p4["instancia"] == k]
        cajas(ax, [g[g["algoritmo"] == a]["hv"] for a in ALG], list(ALG), list(ALG.values()))
        ns = g.groupby("algoritmo")["n_sol"].median()
        ax.set_title(f"{k[:-4]} · soluciones: " + " / ".join(f"{int(ns[a])}" for a in ALG), fontweight="bold", fontsize=11)
        ax.set_ylabel("Hipervolumen normalizado")
    fig.suptitle("Comparación equiparable con 100 clientes (20.000 evaluaciones, 30 réplicas)", fontweight="bold", y=1.01)
    fig.tight_layout(); fig.savefig(out / "Figura_4_2.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def figura_45(out):
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    c6 = pd.read_csv(CC / "C6_indicadores.csv"); g = c6[c6["instancia"] == "Sevilla-25"]
    ax = axs[0, 0]
    cajas(ax, [g[g["algoritmo"] == a]["hv"] for a in ALG], list(ALG), list(ALG.values()))
    ax.set_title("(A) HV equiparable, Sevilla N=25 (30 réplicas)", fontweight="bold"); ax.set_ylabel("Hipervolumen normalizado")
    b = J(S5 / "s5_mod_b_sobol.json")["hv"]; ps = list(b["S1"]); ctrl = {"p_cross", "p_mut", "k_saa"}
    ax = axs[0, 1]; x = np.arange(len(ps))
    ax.bar(x, [b["S1"][p] for p in ps], 0.6, yerr=[b["S1_conf"][p] for p in ps], capsize=3,
           color=["#B0B0B0" if p in ctrl else "#3B7DD8" for p in ps], edgecolor="black", lw=0.5)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ps, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("S1 sobre HV (± IC 95 %)"); ax.set_title("(B) Sobol: S1 con control negativo (gris)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    cK = pd.read_csv(S5 / "recompute_s5_mod_c_vsshv.csv"); pb = pd.DataFrame(J(C / "P1_placebos_convergencia.json")["P1b"])
    ax = axs[0, 2]
    ax.fill_between(pb["K"], pb["min"], pb["max"], color="#F4B183", alpha=0.45)
    ax.plot(pb["K"], pb["vsshv_sin_incertidumbre_media"], "-o", c="#C55A11", ms=5, label="Sin incertidumbre (placebo)")
    ax.plot(cK["K"], cK["vsshv_mean_ruta"], "-D", c="#6A1B9A", ms=5, label="Con incertidumbre")
    ax.set_xlabel("K (escenarios SAA)"); ax.set_ylabel("VSS_HV (%)")
    ax.set_title("(C) VSS_HV con y sin incertidumbre", fontweight="bold"); ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    ce = J(C / "P1_placebos_convergencia.json")["P1e"]
    ax = axs[1, 0]
    for n, col in (("N25", "#3B6FB5"), ("N50", "#C0504D")):
        t = ce[n]["hv_por_generaciones_y_semilla"]; gens = sorted(int(k) for k in t); base = np.array(t[str(gens[0])], float)
        for sd in range(len(base)):
            ax.plot(gens, [100 * (t[str(gg)][sd] / base[sd] - 1) for gg in gens], c=col, alpha=0.25, lw=0.8)
        ax.plot(gens, [ce[n]["aumento_pct_respecto_a_200"][str(gg)] for gg in gens], "-o", c=col, lw=2, label=f"N={n[1:]} (media de 5 semillas)")
    ax.set_xscale("log", base=2); ax.set_xticks([200, 400, 800, 1600]); ax.set_xticklabels(["200", "400", "800", "1.600"])
    ax.set_xlabel("Generaciones"); ax.set_ylabel("Aumento del HV (%)")
    ax.set_title("(D) Convergencia de NSGA-II (problema EV)", fontweight="bold"); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    e = pd.read_csv(S5 / "s5_mod_e_stability.csv"); e = e[e["algorithm"] == "NSGA-II"]["hv"]
    ax = axs[1, 1]
    cajas(ax, [e / 1e14], ["NSGA-II"], [ALG["NSGA-II"]])
    ax.text(0.97, 0.95, f"CV = {coma(100 * e.std() / e.mean())} %", transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold")
    ax.set_title("(E) Estabilidad de NSGA-II, Sevilla N=25 (30 semillas)", fontweight="bold"); ax.set_ylabel("Hipervolumen (×10¹⁴)")
    f = pd.read_csv(S5 / "recompute_s5_mod_f_ablation.csv")
    ax = axs[1, 2]
    ax.bar(f["model"], f["rmse"], color="#EF6F6C", label="RMSE")
    ax.set_ylabel("RMSE (kg)", color="#C62828"); ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    ax2.plot(f["model"], f["vsshv_pct_corregido"], "-D", c="#6A1B9A", ms=7, label="VSS_HV% (descriptivo)")
    ax2.set_ylabel("VSS_HV (%)", color="#6A1B9A"); ax2.set_ylim(0, max(f["vsshv_pct_corregido"]) * 1.35)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right")
    ax.set_title("(F) Ablación: RMSE y VSS_HV (descriptivo)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "Figura_4_5.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


def figura_46(out):
    n25 = pd.read_csv(S5 / "s5_mod_e_stability.csv"); n25 = n25[n25["algorithm"] == "NSGA-II"]
    n50 = pd.read_csv(S6 / "s6_n50_stability.csv")
    fm = {n: J(S3[n] / "forecast_metrics.json") for n in (25, 50)}
    fv = {n: J(S3[n] / "forecast_eval_verdad.json") for n in (25, 50)}
    c1 = J(C / "decision_C1_C2.json")["C1_comparacion_equilibrada"]["detalle"]
    c5 = J(CC / "decision_C4_C5_C7.json")["C5_mecanismo_desvios"]["detalle"]
    c7 = J(CC / "C7_prediccion_replicas.json")["referencia_descriptiva"]
    cv = {25: 100 * n25["hv"].std() / n25["hv"].mean(), 50: 100 * n50["hv"].std() / n50["hv"].mean()}
    fig, axs = plt.subplots(2, 2, figsize=(13.4, 10.4))
    rel = lambda x: x / x.mean()
    ax = axs[0, 0]; cajas(ax, [rel(n25["hv"]), rel(n50["hv"])], ["N=25", "N=50"], ["#FFFFFF", "#FFFFFF"])
    ax.set_title("(A) Estabilidad del HV de NSGA-II (30 réplicas)", fontweight="bold"); ax.set_ylabel("HV relativo a su media")
    ax = axs[0, 1]; cajas(ax, [rel(n25["igd"]), rel(n50["igd"])], ["N=25", "N=50"], ["#FFFFFF", "#FFFFFF"])
    ax.set_title("(B) Estabilidad del IGD de NSGA-II (30 réplicas)", fontweight="bold"); ax.set_ylabel("IGD relativo a su media")
    ax = axs[1, 0]; x, w = np.arange(2), 0.36
    for k, (e, col, lab) in enumerate((("E1_ref", "#3B6FB5", "Mejor plan en tardanza"), ("E2_ref", "#8FB0DD", "Plan de compromiso"))):
        val = [c1[f"N{n}_oos_real"][e]["vss_pct"] for n in (25, 50)]
        err = [[val[i] - c1[f"N{n}_oos_real"][e]["ic95_pct"][0] for i, n in enumerate((25, 50))],
               [c1[f"N{n}_oos_real"][e]["ic95_pct"][1] - val[i] for i, n in enumerate((25, 50))]]
        ax.bar(x + (k - 0.5) * w, val, w, yerr=err, capsize=3, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(["N=25", "N=50"]); ax.set_ylabel("Reducción de la tardanza (%)")
    ax.set_title("(C) Planes con escenarios frente a deterministas (C1)", fontweight="bold"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax = axs[1, 1]; ax.axis("off"); ax.set_title("(D) Resumen de la comparativa N=25 vs N=50", fontweight="bold")
    cob = lambda n: fv[n]["cobertura_Q10_Q90_frente_a_demanda_verdadera_pct"]["BiLSTM"]
    des = lambda n: f'{coma(c5[f"N{n}_oos_real"]["E1_ref"]["desvios_det"], 2)} → {coma(c5[f"N{n}_oos_real"]["E1_ref"]["desvios_saa"], 2)}'
    filas = [["Reducción de la tardanza, C1 (%)", coma(c1["N25_oos_real"]["E1_ref"]["vss_pct"]), coma(c1["N50_oos_real"]["E1_ref"]["vss_pct"])],
             ["Desvíos por ronda, C5", des(25), des(50)],
             ["Cobertura BiLSTM, demanda verdadera (%)", coma(cob(25)), coma(cob(50))],
             ["Exceso sobre el error mínimo, C7 (%)", coma(c7["N25"]["exceso_BiLSTM_pct"]), coma(c7["N50"]["exceso_BiLSTM_pct"])],
             ["Skill XGBoost, validación (%)", coma(fm[25]["xgboost"]["skill_q50_vs_naive"]), coma(fm[50]["xgboost"]["skill_q50_vs_naive"])],
             ["CV del HV, 30 réplicas (%)", coma(cv[25]), coma(cv[50])]]
    tabla(ax, filas, ["Métrica", "N=25", "N=50"], fs=9.5, escala=1.7, anchos=[0.56, 0.22, 0.22])
    fig.tight_layout(); fig.savefig(out / "Figura_4_6.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="figuras_v628"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (figura_42, figura_44, figura_45, figura_46):
        f(out); print("generada:", out / f"Figura_{f.__name__[-2]}_{f.__name__[-1]}.png")
