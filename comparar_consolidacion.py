import json, sys
from pathlib import Path
import numpy as np, pandas as pd
A, B = Path(sys.argv[1]), Path(sys.argv[2])
def igual(x, y, ruta=""):
    if isinstance(x, dict) and isinstance(y, dict):
        ks = (set(x) | set(y)) - {"fecha"}
        return [d for k in sorted(ks) for d in (igual(x.get(k), y.get(k), f"{ruta}.{k}") if k in x and k in y else [f"{ruta}.{k}: falta en uno"])]
    if isinstance(x, list) and isinstance(y, list):
        return [f"{ruta}: longitud {len(x)} vs {len(y)}"] if len(x) != len(y) else [d for i, (a, b) in enumerate(zip(x, y)) for d in igual(a, b, f"{ruta}[{i}]")]
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return [] if np.isclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True) else [f"{ruta}: {x} vs {y}"]
    return [] if x == y else [f"{ruta}: {x!r} vs {y!r}"]
fa = sorted(p.relative_to(A) for p in A.rglob("*") if p.is_file() and "replicas_parcial" not in p.parts)
fb = {p.relative_to(B) for p in B.rglob("*") if p.is_file() and "replicas_parcial" not in p.parts}
total_dif = 0
for rel in fa:
    if rel not in fb:
        print(f"FALTA   {rel}"); total_dif += 1; continue
    if rel.suffix == ".json":
        d = igual(json.load(open(A / rel, encoding="utf-8")), json.load(open(B / rel, encoding="utf-8")))
    elif rel.suffix == ".csv":
        x, y = pd.read_csv(A / rel), pd.read_csv(B / rel)
        num = x.select_dtypes("number").columns
        d = [] if x.shape == y.shape and list(x.columns) == list(y.columns) and np.allclose(x[num].to_numpy(float), y[num].to_numpy(float), rtol=1e-9, equal_nan=True) \
             and x.drop(columns=num).equals(y.drop(columns=num)) else ["contenido distinto"]
    else:
        continue
    print(f"{'IDÉNTICO' if not d else 'DISTINTO'} {rel}" + ("" if not d else f" · {len(d)} diferencias, p. ej. {d[:3]}")); total_dif += bool(d)
print(f"\nRESULTADO: {'las dos ejecuciones son idénticas' if total_dif == 0 else f'{total_dif} ficheros con diferencias'}")
