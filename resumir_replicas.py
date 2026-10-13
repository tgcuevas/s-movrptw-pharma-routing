import csv
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

BASE = Path("data/replicas_vss")
S4_EQUIVALENTES = {"d3d515df104062ad", "c1eecb5ba4488c98", "f496876ef3a11d7a"}
CANONICO = {25: "20260615_201043_s4", 50: "20260615_205516_s4"}
UMBRAL = {"f1": 5.0, "f2": 10.0}


def main():
    conjuntos, filas = [], []
    for d in sorted(p for p in BASE.iterdir() if p.is_dir()):
        r = d / "replicas_vss_resumen.json"
        if not r.exists():
            continue
        meta = json.load(open(r, encoding="utf-8"))
        if meta.get("s4_huella") not in S4_EQUIVALENTES:
            print(f"  se omite {d.name}: generado con otra versión de s4")
            continue
        conjuntos.append(d.name)
        filas += [dict(x, conjunto=d.name) for x in csv.DictReader(open(d / "replicas_vss.csv", encoding="utf-8"))]
    if not filas:
        sys.exit("ERROR: no hay conjuntos de réplicas válidos en data/replicas_vss/")

    out = {"fecha": datetime.now().isoformat(timespec="seconds"), "conjuntos": conjuntos,
           "semilla": 42, "K": 20, "s4_huellas_admitidas": sorted(S4_EQUIVALENTES),
           "fuente_de_variacion": "entrenamiento no determinista del BiLSTM (TensorFlow); misma semilla y código",
           "por_tamano": {}}
    for n in sorted({int(f["N"]) for f in filas}):
        fs = [f for f in filas if int(f["N"]) == n]
        canon = {}
        rc = Path("data/integration") / CANONICO.get(n, "") / "recompute_eev_recurso.json"
        if rc.is_file():
            canon = json.load(open(rc, encoding="utf-8"))["VSS_pct_corregido"]
        res = {"n_replicas": len(fs), "canonico_vss_pct": canon}
        for o in ("f1", "f2", "f3"):
            v = sorted(float(f[f"vss_{o}"]) for f in fs)
            res[f"vss_{o}"] = {"min": v[0], "mediana": round(st.median(v), 2), "media": round(st.mean(v), 2),
                               "max": v[-1], "positivos": f"{sum(x > 0 for x in v)}/{len(v)}"}
            if o in UMBRAL:
                res[f"vss_{o}"][f"sobre_umbral_{UMBRAL[o]:g}"] = f"{sum(x > UMBRAL[o] for x in v)}/{len(v)}"
        res["orden_f2>f3>f1"] = f"{sum(float(f['vss_f2']) > float(f['vss_f3']) > float(f['vss_f1']) for f in fs)}/{len(fs)}"
        res["viajes_extra"] = sorted(int(f["viajes_extra"]) for f in fs)
        res["cobertura_pct"] = {"min": min(float(f["cobertura"]) for f in fs), "max": max(float(f["cobertura"]) for f in fs)}
        out["por_tamano"][str(n)] = res
        print(f"N={n}: {len(fs)} réplicas · f1 mediana {res['vss_f1']['mediana']} % ({res['vss_f1']['min']}–{res['vss_f1']['max']}), "
              f"> 5 %: {res['vss_f1']['sobre_umbral_5']} · f2 mediana {res['vss_f2']['mediana']} % "
              f"({res['vss_f2']['min']}–{res['vss_f2']['max']}), > 10 %: {res['vss_f2']['sobre_umbral_10']}")
    with open(BASE / "replicas_vss_resumen_global.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Guardado: {BASE / 'replicas_vss_resumen_global.json'} ({len(conjuntos)} conjuntos)")


if __name__ == "__main__":
    main()
