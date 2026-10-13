import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

CRS_GEO = "EPSG:4326"
CRS_UTM = "EPSG:25830"
EL_PINO_LAT = 37.3803
EL_PINO_LON = -5.9248


def parse_args():
    p = argparse.ArgumentParser(description="Reubicar D1 en el Polígono El Pino y regenerar salidas de s1")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--lat", type=float, default=EL_PINO_LAT)
    p.add_argument("--lon", type=float, default=EL_PINO_LON)
    p.add_argument("--no-backup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    net_dir, farm_dir = data_dir / "network", data_dir / "farmacias"
    graphml = net_dir / "sevilla_drive.graphml"
    csv_path = farm_dir / "farmacias_sevilla.csv"
    gpkg_path = farm_dir / "farmacias_sevilla.gpkg"
    tm_path = farm_dir / "time_matrix.npy"
    for pth in (graphml, csv_path, tm_path):
        if not pth.exists():
            print(f"ERROR: falta {pth} — ejecuta desde la carpeta del pipeline o usa --data-dir")
            sys.exit(1)
    print(f"Datos: {data_dir.resolve()}")

    if not args.no_backup:
        bdir = farm_dir / f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        bdir.mkdir(parents=True, exist_ok=True)
        for pth in (csv_path, gpkg_path, tm_path):
            if pth.exists():
                shutil.copy2(pth, bdir / pth.name)
        print(f"Copia de seguridad de las salidas de s1 en {bdir}")

    df = pd.read_csv(csv_path)
    if not (df["id"] == "D1").any():
        print("ERROR: el CSV no contiene la fila D1")
        sys.exit(1)
    old = df.loc[df["id"] == "D1", ["lat", "lon"]].iloc[0]
    df.loc[df["id"] == "D1", ["lat", "lon"]] = [args.lat, args.lon]
    df.loc[df["id"] == "D1", "nombre"] = "Almacén principal (modelo) — Polígono El Pino"
    if (df["id"] == "D2").any():
        df.loc[df["id"] == "D2", "nombre"] = "Almacén secundario (modelo) — Camas"
    tr = Transformer.from_crs(CRS_GEO, CRS_UTM, always_xy=True)
    df["x_utm"], df["y_utm"] = tr.transform(df["lon"].values, df["lat"].values)
    print(f"D1: ({old['lat']:.4f}, {old['lon']:.4f}) -> ({args.lat:.4f}, {args.lon:.4f})")

    print("Cargando red (puede tardar unos segundos)...")
    G = ox.load_graphml(graphml)
    scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(scc).copy()
    print(f"  Grafo: {len(G.nodes):,} nodos · {len(G.edges):,} aristas · "
          f"componente fuertemente conexa mayor: {len(scc):,} nodos ({100*len(scc)/len(G.nodes):.1f} %)")
    old_nodes = df["node_osm"].astype("Int64").tolist() if "node_osm" in df.columns else [None] * len(df)
    new_nodes = ox.distance.nearest_nodes(G_scc, X=df["x_utm"].values, Y=df["y_utm"].values)
    df["node_osm"] = [int(n) for n in new_nodes]
    df["snap_dist_m"] = [round(float(np.hypot(G.nodes[n]["x"] - x, G.nodes[n]["y"] - y)), 1)
                         for n, x, y in zip(df["node_osm"], df["x_utm"], df["y_utm"])]
    changed = [(r["id"], o, n) for r, o, n in zip(df.to_dict("records"), old_nodes, df["node_osm"])
               if o is not None and not pd.isna(o) and int(o) != int(n)]
    print(f"  Localizaciones re-proyectadas a otro nodo: {len(changed)} "
          f"({', '.join(c[0] for c in changed[:10])}{'…' if len(changed) > 10 else ''})")
    print(f"  Distancia de snap: máx {df['snap_dist_m'].max():.0f} m · media {df['snap_dist_m'].mean():.0f} m")

    nodes = df["node_osm"].tolist()
    n = len(nodes)
    print(f"Calculando matriz de tiempos {n}×{n} (Dijkstra dirigido, travel_time)...")
    t0 = time.time()
    M = np.full((n, n), np.inf)
    for i, o in enumerate(nodes):
        L = nx.single_source_dijkstra_path_length(G, o, weight="travel_time")
        for j, d in enumerate(nodes):
            M[i, j] = 0.0 if i == j else L.get(d, np.inf)
    if not np.isfinite(M).all():
        print("ERROR: quedan pares no alcanzables pese al snapping a la componente fuertemente conexa")
        sys.exit(1)
    print(f"  Matriz calculada en {time.time() - t0:.1f} s")

    M_old = np.load(tm_path)
    if M_old.shape == M.shape:
        off = ~np.eye(n, dtype=bool)
        infl = int((M_old > 10 * np.median(M[off])).sum())
        print(f"  Matriz anterior: media {M_old[off].mean():.0f} s · máx {M_old.max():.0f} s · "
              f"entradas infladas (> 10× mediana nueva): {infl}")
    off = ~np.eye(n, dtype=bool)
    i_d1 = int(df.index[df["id"] == "D1"][0])
    farm_idx = df.index[df["tipo"] == "farmacia"].tolist()
    d1_out = M[i_d1, farm_idx]
    d1_in = M[farm_idx, i_d1]
    stats = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "D1": {"lat": args.lat, "lon": args.lon, "node_osm": int(df.loc[i_d1, "node_osm"]),
               "snap_dist_m": float(df.loc[i_d1, "snap_dist_m"])},
        "componente_fuertemente_conexa_nodos": len(scc),
        "matriz": {"n": n, "media_s": float(M[off].mean()), "min_s": float(M[off].min()),
                   "max_s": float(M[off].max()), "media_min": float(M[off].mean() / 60)},
        "D1_a_farmacias_s": {"min": float(d1_out.min()), "media": float(d1_out.mean()), "max": float(d1_out.max())},
        "farmacias_a_D1_s": {"min": float(d1_in.min()), "media": float(d1_in.mean()), "max": float(d1_in.max())},
        "farmacias_a_mas_de_15km_del_D1_por_tiempo_gt_1800s": int((d1_out > 1800).sum()),
        "localizaciones_reproyectadas": [c[0] for c in changed],
    }
    print(f"  Nueva matriz: media {stats['matriz']['media_s']:.0f} s ({stats['matriz']['media_min']:.1f} min) · "
          f"rango {stats['matriz']['min_s']:.0f}–{stats['matriz']['max_s']:.0f} s")
    print(f"  D1 -> farmacias: {stats['D1_a_farmacias_s']['min']:.0f}–{stats['D1_a_farmacias_s']['max']:.0f} s "
          f"(media {stats['D1_a_farmacias_s']['media']:.0f} s) · farmacias -> D1: media {stats['farmacias_a_D1_s']['media']:.0f} s")

    np.save(str(tm_path), M)
    df.to_csv(csv_path, index=False)
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["x_utm"], df["y_utm"])], crs=CRS_UTM)
    gdf.to_file(str(gpkg_path), driver="GPKG")
    with open(farm_dir / "s1b_reubicacion_informe.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {tm_path} · {csv_path} · {gpkg_path}")
    print(f"Informe:  {farm_dir / 's1b_reubicacion_informe.json'}")
    print("\nPara la memoria (§4.1 y Tabla 4.1):")
    print(f"  Depot: Polígono Industrial El Pino (Sevilla Este) · {args.lat:.4f}°N, {abs(args.lon):.4f}°O")
    print("Siguiente paso: python s2b_figura41.py && python s8_mapa_instancia.py --highlight 25 && "
          "python s8_mapa_rutas.py --run 20260615_200957")


if __name__ == "__main__":
    main()
