import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import s2_nsga2_solver as s2
except ImportError as e:
    print("ERROR: no encuentro s2_nsga2_solver.py junto a este script:", e)
    sys.exit(1)

NAVY = "#0B2E5B"


def plot_figura_41(F, best_idx, n_clients, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Frente de Pareto NSGA-II — instancia de validación Sevilla N={n_clients}",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax = axes[0]
    ax.scatter(F[:, 0] / 60, F[:, 1] / 1000, c="steelblue", alpha=0.75, s=32, label="Frente de Pareto")
    ax.scatter(F[best_idx, 0] / 60, F[best_idx, 1] / 1000, c="red", s=120, zorder=5, marker="*",
               label="Compromiso TOPSIS")
    ax.set_xlabel("f₁ — Tiempo total (min)", fontsize=11)
    ax.set_ylabel("f₂ — Tardanza ponderada (×10³)", fontsize=11)
    ax.set_title("Coste vs. nivel de servicio clínico")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.scatter(F[:, 0] / 60, F[:, 2], c="seagreen", alpha=0.75, s=32, label="Frente de Pareto")
    ax.scatter(F[best_idx, 0] / 60, F[best_idx, 2], c="red", s=120, zorder=5, marker="*",
               label="Compromiso TOPSIS")
    ax.set_xlabel("f₁ — Tiempo total (min)", fontsize=11)
    ax.set_ylabel("f₃ — Emisiones CO₂ (proxy)", fontsize=11)
    ax.set_title("Coste vs. sostenibilidad")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    farm_csv = s2.FARM_DIR / "farmacias_sevilla.csv"
    tm_path = s2.FARM_DIR / "time_matrix.npy"
    for pth in (farm_csv, tm_path):
        if not pth.exists():
            print(f"ERROR: falta {pth} — ejecuta desde la carpeta del pipeline")
            sys.exit(1)
    print(f"Datos: {s2.DATA_DIR.resolve()}")

    inst = s2.preparar_instancia_sevilla(n_farmacias=10)
    res = s2.run_sevilla_n10(inst)
    if res is None:
        print("ERROR: NSGA-II no devolvió frente para N=10")
        sys.exit(1)

    F, best = res["F"], int(res["best_idx"])
    out_png = s2.SOLVER_DIR / "figura_4_1_n10_pareto.png"
    plot_figura_41(F, best, inst["n_clients"], out_png)

    cfg_path = s2.SOLVER_DIR / "solver_config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    cfg["sevilla_n10"] = {"n_pareto": int(res["n_pareto"]), "hv": float(res["hv"]),
                            "elapsed_s": float(res["elapsed_s"]),
                            "regenerado": datetime.now().isoformat(timespec="seconds"),
                            "motivo": "reubicación de D1 en el Polígono El Pino (s1b)"}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    informe = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_clients": inst["n_clients"], "capacidad_kg": inst["capacity"],
        "demanda_total_kg": float(sum(inst["demand"])),
        "prioridades": {"urgente": inst["priority"].count(1), "preferente": inst["priority"].count(2),
                        "ordinaria": inst["priority"].count(3)},
        "n_pareto": int(res["n_pareto"]), "n_soluciones_distintas": int(len(np.unique(np.round(F, 6), axis=0))),
        "hypervolume": float(res["hv"]), "elapsed_s": float(res["elapsed_s"]),
        "compromiso_topsis": {"f1_min": float(F[best, 0] / 60), "f2_x1000": float(F[best, 1] / 1000),
                              "f3": float(F[best, 2])},
        "rango_f1_min": [float(F[:, 0].min() / 60), float(F[:, 0].max() / 60)],
    }
    with open(s2.SOLVER_DIR / "s2b_figura41_informe.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    print(f"\nFigura 4.1 regenerada: {out_png}")
    print(f"  {res['n_pareto']} soluciones no dominadas ({informe['n_soluciones_distintas']} distintas) · HV {res['hv']:.4g} · "
          f"compromiso TOPSIS: f1 {F[best,0]/60:.1f} min · f2 {F[best,1]/1000:.1f}×10³ · f3 {F[best,2]:.2f}")
    print(f"Informe: {s2.SOLVER_DIR / 's2b_figura41_informe.json'}")


if __name__ == "__main__":
    main()
