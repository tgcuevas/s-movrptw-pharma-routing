import os
import json
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
NETWORK_DIR = DATA_DIR / "network"
FARM_DIR    = DATA_DIR / "farmacias"
for d in [NETWORK_DIR, FARM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CRS_GEO = "EPSG:4326"
CRS_UTM = "EPSG:25830"

def descargar_red_sevilla():
    log.info("Descargando red viaria Sevilla + entorno inmediato...")

    DEPOT_LAT = 37.3826
    DEPOT_LON = -5.9970
    RADIO_M   = 18_000

    G = ox.graph_from_point(
        (DEPOT_LAT, DEPOT_LON),
        dist=RADIO_M,
        network_type="drive",
        simplify=True,
        retain_all=False
    )
    log.info(f"Red descargada: {len(G.nodes):,} nodos · {len(G.edges):,} aristas")

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    G = ox.bearing.add_edge_bearings(G)

    G_utm = ox.projection.project_graph(G, to_crs=CRS_UTM)

    graphml_path = NETWORK_DIR / "sevilla_drive.graphml"
    ox.save_graphml(G_utm, filepath=str(graphml_path))
    log.info(f"GraphML guardado: {graphml_path}")

    nodes, edges = ox.graph_to_gdfs(G_utm)

    nodes = nodes.to_crs(CRS_UTM)
    edges = edges.to_crs(CRS_UTM)

    transformer = Transformer.from_crs(CRS_UTM, CRS_GEO, always_xy=True)
    nodes_geo = nodes.copy()
    nodes_geo["lon"], nodes_geo["lat"] = transformer.transform(
        nodes.geometry.x.values,
        nodes.geometry.y.values
    )

    nodes_export = nodes_geo[["geometry", "x", "y", "lon", "lat"]].copy()
    nodes_export.columns = ["geometry", "x_utm", "y_utm", "lon", "lat"]
    nodes_export["node_id"] = nodes_export.index.astype(str)

    edges_base = edges.reset_index()
    edges_export = edges_base[["geometry"]].copy()
    edges_export["from_nd"]   = edges_base["u"].astype(str)
    edges_export["to_nd"]     = edges_base["v"].astype(str)
    edges_export["length_m"]  = edges_base["length"].round(2)
    edges_export["trav_time"] = edges_base["travel_time"].round(2)
    edges_export["spd_kph"]   = edges_base["speed_kph"].round(1)
    edges_export["highway"]   = edges_base["highway"].astype(str).str[:10]
    edges_gpkg = NETWORK_DIR / "sevilla_edges.gpkg"
    edges_base[["geometry","u","v","length","travel_time","speed_kph",
                "maxspeed","name","highway","bearing"]].to_file(
        str(edges_gpkg), driver="GPKG")

    nodes_shp = NETWORK_DIR / "sevilla_nodes.shp"
    edges_shp = NETWORK_DIR / "sevilla_edges.shp"
    nodes_export.to_file(str(nodes_shp), driver="ESRI Shapefile")
    edges_export.to_file(str(edges_shp), driver="ESRI Shapefile")
    log.info(f"Shapefiles UTM exportados: {nodes_shp.name} · {edges_shp.name}")

    stats = {
        "nodos":         len(G_utm.nodes),
        "aristas":       len(G_utm.edges),
        "radio_m":       RADIO_M,
        "crs":           CRS_UTM,
        "velocidad_media_kph": round(
            np.mean([d.get("speed_kph", 30) for _, _, d in G_utm.edges(data=True)]), 1
        ),
        "longitud_total_km": round(
            sum(d.get("length", 0) for _, _, d in G_utm.edges(data=True)) / 1000, 1
        ),
    }
    with open(NETWORK_DIR / "red_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Estadísticas red: {stats}")

    return G_utm, nodes_export, edges_export


def geocodificar_farmacias():
    log.info("Generando localizaciones de farmacias y depósitos del modelo...")

    farmacias_raw = [
        {"id":"F01","nombre":"Farmacia Catedral","zona":"Centro","lat":37.3856,"lon":-5.9924,"prioridad_media":2.1},
        {"id":"F02","nombre":"Farmacia Alfalfa","zona":"Centro","lat":37.3876,"lon":-5.9901,"prioridad_media":2.0},
        {"id":"F03","nombre":"Farmacia Encarnación","zona":"Centro","lat":37.3894,"lon":-5.9932,"prioridad_media":1.9},
        {"id":"F04","nombre":"Farmacia Campana","zona":"Centro","lat":37.3888,"lon":-5.9943,"prioridad_media":2.2},
        {"id":"F05","nombre":"Farmacia San Vicente","zona":"Centro","lat":37.3912,"lon":-5.9978,"prioridad_media":2.0},
        {"id":"F06","nombre":"Farmacia San Lorenzo","zona":"Centro","lat":37.3921,"lon":-5.9985,"prioridad_media":1.8},
        {"id":"F07","nombre":"Farmacia Reyes Católicos","zona":"Centro","lat":37.3867,"lon":-5.9955,"prioridad_media":2.1},
        {"id":"F08","nombre":"Farmacia Feria","zona":"Centro","lat":37.3944,"lon":-5.9967,"prioridad_media":1.9},
        {"id":"F09","nombre":"Farmacia Sierpes","zona":"Centro","lat":37.3878,"lon":-5.9934,"prioridad_media":2.3},
        {"id":"F10","nombre":"Farmacia San Eloy","zona":"Centro","lat":37.3891,"lon":-5.9944,"prioridad_media":2.0},
        {"id":"F11","nombre":"Farmacia Nervión Centro","zona":"Nervión","lat":37.3844,"lon":-5.9745,"prioridad_media":2.0},
        {"id":"F12","nombre":"Farmacia Luis Montoto","zona":"Nervión","lat":37.3823,"lon":-5.9698,"prioridad_media":1.9},
        {"id":"F13","nombre":"Farmacia Ramón y Cajal","zona":"Nervión","lat":37.3851,"lon":-5.9714,"prioridad_media":2.1},
        {"id":"F14","nombre":"Farmacia Virgen de la Salud","zona":"Nervión","lat":37.3812,"lon":-5.9732,"prioridad_media":1.8},
        {"id":"F15","nombre":"Farmacia Amate","zona":"Este","lat":37.3765,"lon":-5.9612,"prioridad_media":2.2},
        {"id":"F16","nombre":"Farmacia Juan XXIII","zona":"Este","lat":37.3756,"lon":-5.9589,"prioridad_media":2.0},
        {"id":"F17","nombre":"Farmacia Polígono Norte","zona":"Norte","lat":37.4012,"lon":-5.9887,"prioridad_media":2.1},
        {"id":"F18","nombre":"Farmacia La Barzola","zona":"Nervión","lat":37.3865,"lon":-5.9723,"prioridad_media":1.9},
        {"id":"F19","nombre":"Farmacia San Bernardo","zona":"Nervión","lat":37.3834,"lon":-5.9756,"prioridad_media":2.0},
        {"id":"F20","nombre":"Farmacia Alcalde Isac","zona":"Este","lat":37.3778,"lon":-5.9601,"prioridad_media":1.8},
        {"id":"F21","nombre":"Farmacia Macarena","zona":"Norte","lat":37.4021,"lon":-5.9934,"prioridad_media":2.2},
        {"id":"F22","nombre":"Farmacia Martínez Montañés","zona":"Norte","lat":37.4045,"lon":-5.9956,"prioridad_media":2.0},
        {"id":"F23","nombre":"Farmacia Pino Montano","zona":"Norte","lat":37.4187,"lon":-5.9876,"prioridad_media":1.9},
        {"id":"F24","nombre":"Farmacia Bermejales Norte","zona":"Norte","lat":37.4065,"lon":-5.9945,"prioridad_media":2.1},
        {"id":"F25","nombre":"Farmacia San Gil","zona":"Norte","lat":37.3967,"lon":-5.9956,"prioridad_media":1.8},
        {"id":"F26","nombre":"Farmacia Doctor Fleming","zona":"Norte","lat":37.4034,"lon":-5.9912,"prioridad_media":2.0},
        {"id":"F27","nombre":"Farmacia Torneo","zona":"Norte","lat":37.3978,"lon":-5.9994,"prioridad_media":2.2},
        {"id":"F28","nombre":"Farmacia Ronda Norte","zona":"Norte","lat":37.4099,"lon":-5.9901,"prioridad_media":1.9},
        {"id":"F29","nombre":"Farmacia Tablada Norte","zona":"Norte","lat":37.4011,"lon":-5.9923,"prioridad_media":2.0},
        {"id":"F30","nombre":"Farmacia Santa Clara","zona":"Norte","lat":37.3992,"lon":-5.9978,"prioridad_media":1.8},
        {"id":"F31","nombre":"Farmacia Triana Centro","zona":"Triana","lat":37.3834,"lon":-6.0023,"prioridad_media":2.1},
        {"id":"F32","nombre":"Farmacia San Jacinto","zona":"Triana","lat":37.3821,"lon":-6.0045,"prioridad_media":2.0},
        {"id":"F33","nombre":"Farmacia Heliópolis","zona":"Sur","lat":37.3698,"lon":-5.9934,"prioridad_media":1.9},
        {"id":"F34","nombre":"Farmacia Los Bermejales","zona":"Sur","lat":37.3712,"lon":-5.9956,"prioridad_media":2.2},
        {"id":"F35","nombre":"Farmacia Bellavista","zona":"Sur","lat":37.3634,"lon":-5.9967,"prioridad_media":2.0},
        {"id":"F36","nombre":"Farmacia Pino Montano Sur","zona":"Sur","lat":37.3745,"lon":-5.9978,"prioridad_media":1.8},
        {"id":"F37","nombre":"Farmacia Palmera","zona":"Sur","lat":37.3678,"lon":-5.9945,"prioridad_media":2.1},
        {"id":"F38","nombre":"Farmacia Ciudad Jardín","zona":"Sur","lat":37.3756,"lon":-5.9823,"prioridad_media":1.9},
        {"id":"F39","nombre":"Farmacia Tablada Sur","zona":"Triana","lat":37.3767,"lon":-6.0012,"prioridad_media":2.0},
        {"id":"F40","nombre":"Farmacia Reina Mercedes","zona":"Sur","lat":37.3623,"lon":-5.9923,"prioridad_media":2.2},
        {"id":"F41","nombre":"Farmacia Camas","zona":"Aljarafe","lat":37.4012,"lon":-6.0234,"prioridad_media":2.0},
        {"id":"F42","nombre":"Farmacia San Juan de Aznalfarache","zona":"Aljarafe","lat":37.3712,"lon":-6.0345,"prioridad_media":1.9},
        {"id":"F43","nombre":"Farmacia Bormujos","zona":"Aljarafe","lat":37.3812,"lon":-6.0456,"prioridad_media":2.1},
        {"id":"F44","nombre":"Farmacia Tomares","zona":"Aljarafe","lat":37.3912,"lon":-6.0389,"prioridad_media":2.0},
        {"id":"F45","nombre":"Farmacia Mairena del Aljarafe","zona":"Aljarafe","lat":37.3614,"lon":-6.0612,"prioridad_media":1.8},
        {"id":"F46","nombre":"Farmacia Castilleja","zona":"Aljarafe","lat":37.3945,"lon":-6.0512,"prioridad_media":2.2},
        {"id":"F47","nombre":"Farmacia Dos Hermanas Norte","zona":"Sur","lat":37.3234,"lon":-5.9212,"prioridad_media":2.0},
        {"id":"F48","nombre":"Farmacia Alcalá de Guadaíra","zona":"Este","lat":37.3345,"lon":-5.8423,"prioridad_media":1.9},
        {"id":"F49","nombre":"Farmacia Pino Sur","zona":"Sur","lat":37.3589,"lon":-5.9812,"prioridad_media":2.1},
        {"id":"F50","nombre":"Farmacia La Rinconada","zona":"Norte","lat":37.4712,"lon":-5.9812,"prioridad_media":2.0},
    ]

    depositos_raw = [
        {
            "id": "D1",
            "nombre": "Almacén principal (modelo) — Polígono El Pino",
            "zona": "Depósito",
            "lat": 37.3826,
            "lon": -5.9970,
            "tipo": "principal",
            "prioridad_media": 0.0,
        },
        {
            "id": "D2",
            "nombre": "Almacén secundario (modelo) — Camas",
            "zona": "Depósito",
            "lat": 37.4012,
            "lon": -6.0198,
            "tipo": "secundario",
            "prioridad_media": 0.0,
        },
    ]

    df_farm = pd.DataFrame(farmacias_raw)
    df_dep  = pd.DataFrame(depositos_raw)
    df_farm["tipo"] = "farmacia"
    df_dep["tipo"]  = "deposito"
    df = pd.concat([df_dep, df_farm], ignore_index=True)

    transformer = Transformer.from_crs(CRS_GEO, CRS_UTM, always_xy=True)
    df["x_utm"], df["y_utm"] = transformer.transform(df["lon"].values, df["lat"].values)

    geometry = [Point(xy) for xy in zip(df["x_utm"], df["y_utm"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_UTM)

    csv_path = FARM_DIR / "farmacias_sevilla.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"CSV exportado: {csv_path} ({len(df_farm)} farmacias + {len(df_dep)} depósitos)")

    gpkg_path = FARM_DIR / "farmacias_sevilla.gpkg"
    gdf.to_file(str(gpkg_path), driver="GPKG")
    log.info(f"GeoPackage exportado: {gpkg_path}")

    return df, gdf


def snap_to_network(G, df_farmacias):
    log.info("Snapping farmacias y depósitos al nodo de red más cercano...")

    lats = df_farmacias["lat"].values
    lons = df_farmacias["lon"].values

    G_wgs = ox.projection.project_graph(G, to_crs=CRS_GEO)

    nearest = ox.nearest_nodes(G_wgs, X=lons, Y=lats)
    df_farmacias = df_farmacias.copy()
    df_farmacias["node_osm"] = nearest

    transformer = Transformer.from_crs(CRS_GEO, CRS_UTM, always_xy=True)
    snap_dist = []
    for i, row in df_farmacias.iterrows():
        node_data = G.nodes[row["node_osm"]]
        nx_coord  = (node_data["x"], node_data["y"])
        farm_utm  = transformer.transform(row["lon"], row["lat"])
        dist = np.sqrt((nx_coord[0]-farm_utm[0])**2 + (nx_coord[1]-farm_utm[1])**2)
        snap_dist.append(round(dist, 1))
    df_farmacias["snap_dist_m"] = snap_dist

    far = df_farmacias[df_farmacias["snap_dist_m"] > 500]
    if len(far) > 0:
        log.warning(f"{len(far)} farmacias a >500m del nodo más cercano — revisar coordenadas:")
        for _, r in far.iterrows():
            log.warning(f"  {r['id']} {r['nombre']}: {r['snap_dist_m']}m")
    else:
        log.info(f"Snap OK — distancia máxima: {df_farmacias['snap_dist_m'].max():.0f}m")

    csv_path = FARM_DIR / "farmacias_sevilla.csv"
    df_farmacias.to_csv(csv_path, index=False)
    log.info(f"CSV actualizado con snap: {csv_path}")

    return df_farmacias


def calcular_matriz_tiempos(G, df_farmacias, N_sample=None):
    nodes = df_farmacias["node_osm"].tolist()
    if N_sample:
        nodes = nodes[:N_sample]
    n = len(nodes)
    log.info(f"Calculando matriz de tiempos {n}×{n}...")

    matrix = np.full((n, n), np.inf)
    t0 = time.time()
    for i, orig in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, orig, weight="travel_time")
        for j, dest in enumerate(nodes):
            matrix[i, j] = lengths.get(dest, np.inf)

    elapsed = time.time() - t0
    log.info(f"Matriz calculada en {elapsed:.1f}s")

    max_finite = matrix[matrix < np.inf].max() if np.any(matrix < np.inf) else 0
    matrix[matrix == np.inf] = max_finite * 10
    matrix[matrix == 0] = 0

    matrix_path = FARM_DIR / "time_matrix.npy"
    np.save(str(matrix_path), matrix)

    off_diag = matrix[matrix > 0]
    log.info(f"Tiempos viaje — min: {off_diag.min():.0f}s  "
             f"media: {off_diag.mean():.0f}s  "
             f"max: {off_diag.max():.0f}s  "
             f"({off_diag.mean()/60:.1f} min media)")
    log.info(f"Matriz guardada: {matrix_path}")

    return matrix


def validar_datos(df_farmacias, G):
    checks = []

    missing = df_farmacias["node_osm"].isna().sum()
    checks.append(("node_osm asignado a todas",   missing == 0))

    invalid_nodes = [n for n in df_farmacias["node_osm"] if n not in G.nodes]
    checks.append(("Nodos válidos en red",          len(invalid_nodes) == 0))

    n_dep = len(df_farmacias[df_farmacias["tipo"] == "deposito"])
    checks.append(("Dos depósitos presentes",        n_dep == 2))

    n_farm = len(df_farmacias[df_farmacias["tipo"] == "farmacia"])
    checks.append((f"50 farmacias ({n_farm} encontradas)", n_farm == 50))

    sample_edge = list(G.edges(data=True))[0][2]
    has_tt = "travel_time" in sample_edge
    checks.append(("Red tiene travel_time",           has_tt))

    shp_ok = (NETWORK_DIR / "sevilla_nodes.shp").exists() and \
             (NETWORK_DIR / "sevilla_edges.shp").exists()
    checks.append(("Shapefiles UTM exportados",        shp_ok))

    print("\n" + "="*60)
    print("VALIDACIÓN DE DATOS")
    print("="*60)
    all_ok = True
    for label, ok in checks:
        sym = "✅" if ok else "❌"
        print(f"  {sym}  {label}")
        if not ok:
            all_ok = False
    print("="*60)
    if all_ok:
        print("Validación superada: datos listos")
    else:
        print("Validación incompleta: revisar los errores anteriores")
    print("="*60 + "\n")
    return all_ok


if __name__ == "__main__":
    print("\n" + "="*60)
    print("s1 — Red de Sevilla y localizaciones")
    print("="*60 + "\n")

    G_utm, nodes_gdf, edges_gdf = descargar_red_sevilla()

    df_farm, gdf_farm = geocodificar_farmacias()

    df_farm = snap_to_network(G_utm, df_farm)

    matrix = calcular_matriz_tiempos(G_utm, df_farm)

    hito1_ok = validar_datos(df_farm, G_utm)

    if hito1_ok:
        print("✅ Archivos generados en data/:")
        print("   data/network/sevilla_drive.graphml  → red para s1b y análisis de red")
        print("   data/network/sevilla_nodes.shp      → análisis GIS")
        print("   data/network/sevilla_edges.shp      → análisis GIS")
        print("   data/farmacias/farmacias_sevilla.csv  → todos los módulos")
        print("   data/farmacias/farmacias_sevilla.gpkg → QGIS visualización")
        print("   data/farmacias/time_matrix.npy       → instancia N=10 (s2)")
        print("\n→ Siguiente paso: python s1b_reubicar_deposito.py\n")
