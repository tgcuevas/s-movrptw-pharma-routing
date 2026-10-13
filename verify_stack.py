#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import importlib
from importlib import metadata as ilmd

_USE_COLOR = sys.stdout.isatty()
def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def red(t):    return _c("31", t)
def bold(t):   return _c("1", t)

OK   = green("[  OK  ]")
WARN = yellow("[ WARN ]")
FAIL = red("[ FAIL ]")

def _parse(v):
    parts = []
    for token in str(v).split("."):
        num = ""
        for ch in token:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return parts

def _cmp(a, b):
    pa, pb = _parse(a), _parse(b)
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)

def version_ok(found, op, expected):
    if op is None or expected is None:
        return True
    c = _cmp(found, expected)
    if op == "==":
        return _parse(found)[:len(_parse(expected))] == _parse(expected)
    if op == ">=":
        return c >= 0
    return True

STACK = [
    ("pymoo",           "pymoo",           "pymoo",           "==", "0.6.1.3", "core"),
    ("numpy",           "numpy",           "numpy",           ">=", "2.0",     "core"),
    ("pandas",          "pandas",          "pandas",          ">=", "2.0",     "core"),
    ("scipy",           "scipy",           "scipy",           "==", "1.17.1",  "core"),
    ("sklearn",         "scikit-learn",    "scikit-learn",    "==", "1.8.0",   "core"),
    ("xgboost",         "xgboost",         "xgboost",         "==", "3.2.0",   "core"),
    ("tensorflow",      "tensorflow",      "tensorflow",      ">=", "2.16",    "core"),
    ("keras",           "keras",           "keras",           "==", "3.13.2",  "core"),
    ("osmnx",           "osmnx",           "osmnx",           "==", "2.1.0",   "core"),
    ("networkx",        "networkx",        "networkx",        ">=", "3.0",     "core"),
    ("geopandas",       "geopandas",       "geopandas",       ">=", "1.0",     "core"),
    ("shapely",         "shapely",         "shapely",         ">=", "2.0",     "core"),
    ("pyproj",          "pyproj",          "pyproj",          ">=", "3.6",     "core"),
    ("fiona",           "fiona",           "fiona",           None, None,      "opt"),
    ("SALib",           "SALib",           "SALib",           ">=", "1.5.1",   "core"),
    ("scikit_posthocs", "scikit-posthocs", "scikit-posthocs", "==", "0.12.0",  "core"),
    ("ortools",         "ortools",         "ortools",         "==", "9.15",    "core"),
    ("statsmodels",     "statsmodels",     "statsmodels",     "==", "0.14.6",  "core"),
    ("matplotlib",      "matplotlib",      "matplotlib",      "==", "3.10.8",  "core"),
    ("seaborn",         "seaborn",         "seaborn",         ">=", "0.13.2",  "opt"),
    ("tqdm",            "tqdm",            "tqdm",            ">=", "4.67",     "opt"),
    ("joblib",          "joblib",          "joblib",          ">=", "1.5",     "opt"),
    ("yaml",            "PyYAML",          "pyyaml",          ">=", "6.0",     "opt"),
]

def get_version(module, dist):
    try:
        return ilmd.version(dist)
    except Exception:
        try:
            return getattr(importlib.import_module(module), "__version__", "¿?")
        except Exception:
            return "¿?"

def check_python():
    v = sys.version_info
    found = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        print(f"  {OK}   Python {found}")
        return 0, 0
    else:
        print(f"  {WARN} Python {found}  (se esperaba 3.11.x)")
        return 0, 1

def check_stack():
    n_fail = n_warn = 0
    width = max(len(s[2]) for s in STACK)
    for module, dist, name, op, expected, level in STACK:
        try:
            importlib.import_module(module)
        except Exception as e:
            if level == "core":
                print(f"  {FAIL} {name.ljust(width)}  NO INSTALADO  ({type(e).__name__})")
                n_fail += 1
            else:
                print(f"  {WARN} {name.ljust(width)}  no instalado (opcional)")
                n_warn += 1
            continue
        found = get_version(module, dist)
        spec = f"{op} {expected}" if op else "(sin fijar)"
        if version_ok(found, op, expected):
            print(f"  {OK}   {name.ljust(width)}  {str(found).ljust(12)} {spec}")
        else:
            print(f"  {WARN} {name.ljust(width)}  {str(found).ljust(12)} (se esperaba {spec})")
            n_warn += 1
    return n_fail, n_warn

def _try(desc, fn):
    try:
        fn()
        print(f"  {OK}   {desc}")
        return 0
    except Exception as e:
        print(f"  {WARN} {desc}  →  {type(e).__name__}: {e}")
        return 1

def check_functional():
    warns = 0
    def f_pymoo():
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.operators.crossover.ox import OrderCrossover
        from pymoo.indicators.hv import HV
        from pymoo.indicators.igd import IGD
    warns += _try("pymoo: NSGA2/NSGA3/MOEA-D, OrderCrossover, HV, IGD", f_pymoo)

    def f_osmnx():
        import osmnx as ox
        assert _cmp(getattr(ox, "__version__", "0"), "2.0") >= 0, "osmnx < 2.x"
        assert hasattr(ox, "add_edge_bearings") or hasattr(getattr(ox, "bearing", object), "add_edge_bearings")
    warns += _try("osmnx 2.x: API de grafo/bearings disponible (s1)", f_osmnx)

    def f_salib():
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    warns += _try("SALib: muestreo y análisis de Sobol (s5)", f_salib)

    def f_posthoc():
        import scikit_posthocs as sp
        assert hasattr(sp, "posthoc_nemenyi_friedman")
    warns += _try("scikit-posthocs: Nemenyi post-hoc de Friedman (s5b)", f_posthoc)

    def f_ortools():
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    warns += _try("OR-Tools: routing CVRPTW (s7)", f_ortools)

    def f_dl():
        import tensorflow as tf
        from tensorflow import keras
        keras.layers.Bidirectional(keras.layers.LSTM(4))
    warns += _try("TensorFlow/Keras: capa BiLSTM (s3)", f_dl)

    def f_xgb():
        import xgboost as xgb
        assert hasattr(xgb, "train")
    warns += _try("XGBoost: API de entrenamiento por cuantiles (s3)", f_xgb)

    return warns

def main():
    print("=" * 60)
    print(bold("  Verificación del entorno tfm_env"))
    print("=" * 60)

    print("\n" + bold("Intérprete de Python"))
    _, pwarn = check_python()

    print("\n" + bold("Paquetes del stack (requirements.txt)"))
    n_fail, n_warn = check_stack()

    print("\n" + bold("Comprobaciones funcionales del pipeline"))
    fwarn = check_functional()

    total_warn = n_warn + pwarn + fwarn
    print("\n" + "=" * 60)
    if n_fail == 0:
        msg = green("  Stack verificado: todos los paquetes críticos disponibles.")
        if total_warn:
            msg += yellow(f"  ({total_warn} aviso(s) — revisar versiones marcadas)")
        print(msg)
        print("=" * 60)
        print("\n  Listo. Recuerda activar el entorno antes de trabajar:")
        print("    conda activate tfm_env")
        sys.exit(0)
    else:
        print(red(f"  Faltan {n_fail} paquete(s) crítico(s). El pipeline NO puede ejecutarse."))
        print("=" * 60)
        print("\n  Sugerencia: reinstala dentro del entorno con")
        print("    conda activate tfm_env && pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
