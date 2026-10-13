import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

S4_CORREGIDO = "d3d515df104062ad"
S4_EQUIVALENTES = {S4_CORREGIDO, "c1eecb5ba4488c98", "f496876ef3a11d7a"}
CANONICO = {25: ("20260615_200957", "20260615_201043_s4"), 50: ("20260615_205432", "20260615_205516_s4")}
D = Path("data")
PUNTEROS = [D / "forecast" / "latest_run.txt", D / "integration" / "latest_integration.txt"]


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def run(cmd, log):
    t = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"falló: {' '.join(cmd)} (ver {log})")
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=5, help="réplicas por tamaño")
    ap.add_argument("--n", default="25,50")
    ap.add_argument("--forzar-entorno", action="store_true")
    a = ap.parse_args()
    if not Path("s4_integration.py").exists():
        sys.exit("ERROR: ejecuta desde la carpeta del pipeline")
    if os.environ.get("CONDA_DEFAULT_ENV") != "tfm_env" and not a.forzar_entorno:
        sys.exit("ERROR: activa el entorno con 'conda activate tfm_env'")
    if sha16("s4_integration.py") not in S4_EQUIVALENTES:
        sys.exit("ERROR: s4_integration.py no es la versión publicada (recurso por ruta): su huella no coincide")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = D / "replicas_vss" / stamp
    (out / "forecast").mkdir(parents=True); (out / "integration").mkdir(parents=True)
    log = out / "replicas.log"
    guard = {p: p.read_text() for p in PUNTEROS if p.exists()}
    filas = []
    try:
        for n in [int(x) for x in a.n.split(",")]:
            for r in range(1, a.r + 1):
                t3 = run([sys.executable, "s3_forecast_pipeline.py", "--n", str(n), "--k", "20", "--seed", "42"], log)
                s3 = (D / "forecast" / "latest_run.txt").read_text().strip()
                t4 = run([sys.executable, "s4_integration.py", "--n", str(n), "--run", s3], log)
                s4 = (D / "integration" / "latest_integration.txt").read_text().strip()
                sm = json.load(open(D / "integration" / s4 / "s4_summary.json", encoding="utf-8"))
                fm = json.load(open(D / "forecast" / s3 / "forecast_metrics.json", encoding="utf-8"))
                ee = list(csv.DictReader(open(D / "integration" / s4 / "s4_eev_eval.csv", encoding="utf-8")))
                v = sm["VSS"]["VSS_pct"]
                filas.append({"N": n, "replica": r, "run_s3": s3, "run_s4": s4,
                              "vss_f1": v["f1"], "vss_f2": v["f2"], "vss_f3": v["f3"],
                              "viajes_extra": sum(int(x["n_extra_trips"]) for x in ee),
                              "cobertura": fm["calibration"]["empirical_coverage_pct"],
                              "epocas": fm["bilstm"]["n_epochs_trained"],
                              "nll": fm["bilstm"]["val_nll_loss"], "t_s": round(t3 + t4, 1)})
                shutil.move(str(D / "forecast" / s3), out / "forecast" / s3)
                shutil.move(str(D / "integration" / s4), out / "integration" / s4)
                f = filas[-1]
                print(f"N={n} réplica {r}/{a.r}: VSS {f['vss_f1']:.2f} / {f['vss_f2']:.2f} / {f['vss_f3']:.2f} % · "
                      f"viajes extra {f['viajes_extra']} · cobertura {f['cobertura']} % · {f['t_s']:.0f} s", flush=True)
    finally:
        for p, txt in guard.items():
            p.write_text(txt)
        print("Punteros latest_*.txt restaurados:", ", ".join(t.strip() for t in guard.values()))

    with open(out / "replicas_vss.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)

    resumen = {"fecha": stamp, "replicas_por_tamano": a.r, "semilla": 42, "K": 20,
               "fuente_de_variacion": "entrenamiento no determinista del BiLSTM (TensorFlow); misma semilla y código",
               "s4_huella": sha16("s4_integration.py"), "por_tamano": {}}
    for n in sorted({f["N"] for f in filas}):
        fs = [f for f in filas if f["N"] == n]
        s3c, s4c = CANONICO[n]
        canon = {"run_s4": s4c}
        rc = D / "integration" / s4c / "recompute_eev_recurso.json"
        if rc.exists():
            canon["vss_pct"] = json.load(open(rc, encoding="utf-8"))["VSS_pct_corregido"]
        resumen["por_tamano"][str(n)] = {
            "canonico": canon,
            **{f"vss_{o}": {"min": min(f[f"vss_{o}"] for f in fs), "max": max(f[f"vss_{o}"] for f in fs),
                            "media": round(sum(f[f"vss_{o}"] for f in fs) / len(fs), 2)} for o in ("f1", "f2", "f3")},
            "H1_cumple (vss_f1 > 5)": f"{sum(f['vss_f1'] > 5 for f in fs)}/{len(fs)}",
            "H2_cumple (vss_f2 > 10)": f"{sum(f['vss_f2'] > 10 for f in fs)}/{len(fs)}",
            "orden_f2>f3>f1": f"{sum(f['vss_f2'] > f['vss_f3'] > f['vss_f1'] for f in fs)}/{len(fs)}",
            "cobertura": {"min": min(f["cobertura"] for f in fs), "max": max(f["cobertura"] for f in fs)},
        }
    with open(out / "replicas_vss_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\nResumen: {out / 'replicas_vss_resumen.json'}")
    for n, r in resumen["por_tamano"].items():
        print(f"  N={n}: f1 {r['vss_f1']['min']}–{r['vss_f1']['max']} % · f2 {r['vss_f2']['min']}–{r['vss_f2']['max']} % · "
              f"f3 {r['vss_f3']['min']}–{r['vss_f3']['max']} % · H1 {r['H1_cumple (vss_f1 > 5)']} · H2 {r['H2_cumple (vss_f2 > 10)']}")


if __name__ == "__main__":
    main()
