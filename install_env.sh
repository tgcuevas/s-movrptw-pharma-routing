#!/bin/bash
# ============================================================
# install_env.sh
# Crea y configura el entorno conda tfm_env desde cero, instala
# todo el stack (incluido ortools para s7) y crea la estructura
# de directorios de salida. No requiere intervención manual.
#
# Uso (una sola vez):
#   bash install_env.sh
#
# Después, para trabajar en el TFM:
#   conda activate tfm_env
# ============================================================

set -e

echo "======================================================"
echo "  TFM — Creando entorno conda tfm_env"
echo "======================================================"

# 1. Crear entorno Python 3.11
echo ""
echo "[1/5] Creando entorno conda tfm_env (Python 3.11)..."
conda create -n tfm_env python=3.11 -y

# 2. Activar entorno (. en lugar de source — compatible con sh y bash)
echo ""
echo "[2/5] Activando tfm_env..."
CONDA_BASE=$(conda info --base)
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate tfm_env

# 3. Instalar paquetes GIS con conda-forge
echo ""
echo "[3/5] Instalando paquetes GIS con conda-forge..."
conda install -c conda-forge numpy pandas geopandas shapely fiona pyproj networkx -y

# 4. Instalar el resto con pip
echo ""
echo "[4/5] Instalando resto del stack con pip..."
echo "      (incluye scikit-posthocs para el Nemenyi y ortools para s7)"
pip install -r requirements.txt

# 5. Crear la estructura de directorios de salida (idempotente)
echo ""
echo "[5/5] Creando estructura de directorios de datos..."
mkdir -p \
    data/farmacias \
    data/network \
    data/forecast \
    data/integration \
    data/sensitivity/solomon100 \
    data/scale_n50 \
    data/solver \
    data/compare_ortools \
    docs
echo "      Estructura creada bajo ./data/ (+ docs/)."

echo ""
echo "======================================================"
echo "  Instalación completada."
echo "  Verificando stack..."
echo "======================================================"
python verify_stack.py

echo ""
echo "Para trabajar en el TFM, ejecuta siempre:"
echo "  conda activate tfm_env"
