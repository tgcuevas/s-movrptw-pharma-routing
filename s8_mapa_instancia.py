import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import osmnx as ox

NAVY = "#0B2E5B"
ZONE_COLORS = {
    "Centro":   "#C0392B",
    "Nervión":  "#E67E22",
    "Este":     "#8E44AD",
    "Norte":    "#2D6CB5",
    "Triana":   "#16A085",
    "Sur":      "#2E8B57",
    "Aljarafe": "#7F8C8D",
}


def parse_args():
    p = argparse.ArgumentParser(description="Mapa de la instancia Sevilla (salidas de s1)")
    p.add_argument("--data-dir", default="data", help="carpeta data/ del pipeline (por defecto: ./data)")
    p.add_argument("--highlight", type=int, default=0,
                   help="resalta las N primeras farmacias del CSV (0 = todas iguales)")
    p.add_argument("--labels", action="store_true", help="escribe el id junto a cada farmacia")
    p.add_argument("--radius-km", type=float, default=15.0, help="radio de servicio a dibujar (0 = no dibujar)")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def check_inputs(data_dir: Path):
    graphml = data_dir / "network" / "sevilla_drive.graphml"
    csv = data_dir / "farmacias" / "farmacias_sevilla.csv"
    missing = [str(p) for p in (graphml, csv) if not p.exists()]
    if missing:
        print("ERROR: no encuentro las salidas de s1 en", data_dir.resolve())
        for m in missing:
            print("   falta:", m)
        print("Ejecuta este script desde la carpeta del pipeline o pasa --data-dir.")
        sys.exit(1)
    return graphml, csv


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    graphml, csv = check_inputs(data_dir)
    out_dir = data_dir / "figuras"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Datos: {data_dir.resolve()}")
    print("Cargando red (puede tardar unos segundos)...")
    G = ox.load_graphml(graphml)
    df = pd.read_csv(csv)

    if "x_utm" not in df.columns or "y_utm" not in df.columns:
        print("ERROR: el CSV no tiene columnas x_utm / y_utm (¿es la versión generada por s1?)")
        sys.exit(1)

    farm = df[df["tipo"] == "farmacia"].reset_index(drop=True)
    dep = df[df["tipo"] == "deposito"].reset_index(drop=True)
    d1 = dep[dep["id"] == "D1"].iloc[0] if (dep["id"] == "D1").any() else dep.iloc[0]
    print(f"  {len(G.nodes):,} nodos · {len(G.edges):,} aristas · {len(farm)} farmacias · depósito {d1['id']}")

    fig, ax = ox.plot_graph(
        G, ax=None, figsize=(12, 6.75), bgcolor="white",
        node_size=0, edge_color="#C9D1DB", edge_linewidth=0.35,
        show=False, close=False,
    )

    if args.radius_km > 0:
        circ = plt.Circle((d1["x_utm"], d1["y_utm"]), args.radius_km * 1000,
                          fill=False, ls="--", lw=1.0, ec=NAVY, alpha=0.6)
        ax.add_patch(circ)

    n_hi = args.highlight if args.highlight > 0 else len(farm)
    for zona, color in ZONE_COLORS.items():
        sub = farm[farm["zona"] == zona]
        if sub.empty:
            continue
        hi = sub[sub.index < n_hi]
        lo = sub[sub.index >= n_hi]
        if not hi.empty:
            ax.scatter(hi["x_utm"], hi["y_utm"], s=46, c=color, edgecolors="white",
                       linewidths=0.8, zorder=5, label=f"{zona} ({len(sub)})")
        if not lo.empty:
            ax.scatter(lo["x_utm"], lo["y_utm"], s=34, facecolors="white", edgecolors=color,
                       linewidths=1.2, zorder=4, label=None if not hi.empty else f"{zona} ({len(sub)})")
    if args.labels:
        for _, r in farm.iterrows():
            ax.annotate(r["id"], (r["x_utm"], r["y_utm"]), xytext=(3, 3), textcoords="offset points",
                        fontsize=5.5, color="#444444", zorder=6)

    ax.scatter([d1["x_utm"]], [d1["y_utm"]], s=260, marker="*", c=NAVY, edgecolors="white",
               linewidths=1.0, zorder=7, label="Depósito (Polígono El Pino)")

    pad = 1500
    xs = np.r_[farm["x_utm"].values, d1["x_utm"]]
    ys = np.r_[farm["y_utm"].values, d1["y_utm"]]
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")

    handles, labels = ax.get_legend_handles_labels()
    if args.highlight > 0:
        handles.append(Line2D([0], [0], marker="o", ls="", markerfacecolor="white",
                              markeredgecolor="#555555", markersize=6))
        labels.append(f"fuera de la instancia N = {args.highlight}")
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=True, framealpha=0.92,
                    title="Farmacias por zona", title_fontsize=8.5)
    leg.get_frame().set_edgecolor("#D9DEE5")
    titulo = "Sevilla: red vial real (OSMnx), depósito y farmacias representativas"
    if args.highlight > 0:
        titulo += f"  ·  instancia N = {args.highlight} resaltada"
    ax.set_title(titulo, fontsize=12, color=NAVY, loc="left", pad=8)
    ax.text(0.995, 0.01, f"Salida de s1_red_sevilla.py · UTM 30N (EPSG:25830) · {len(G.edges):,} tramos",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#6B7280")

    png = out_dir / "s8_mapa_instancia_sevilla.png"
    svg = out_dir / "s8_mapa_instancia_sevilla.svg"
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figura guardada: {png}\n                 {svg}")


if __name__ == "__main__":
    main()
