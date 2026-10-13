#!/bin/bash
# ============================================================
# run_pipeline.sh
# Ejecuta el pipeline completo de extremo a extremo y el
# post-procesado (Nemenyi + comparación OR-Tools). No requiere
# intervención manual. Las réplicas de variabilidad del VSS se
# lanzan aparte: python replicas_vss.py
#
# Requisito previo (una sola vez):
#   bash install_env.sh        # crea el entorno y la estructura
#
# Uso:
#   bash run_pipeline.sh                 # pipeline completo + post-procesado
#   bash run_pipeline.sh --skip-s7       # omite la comparación OR-Tools (s7)
#   bash run_pipeline.sh --quick         # corridas reducidas (verificación)
# ============================================================

set -euo pipefail

RUN_S7=1
QUICK=""
RED_DEPOSITADA=0
for arg in "$@"; do
  case "$arg" in
    --skip-s7) RUN_S7=0 ;;
    --quick)   QUICK="--quick" ;;
    --red-depositada) RED_DEPOSITADA=1 ;;
    *) echo "[AVISO] argumento no reconocido: $arg" ;;
  esac
done

# --- Activar el entorno conda tfm_env ---
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda activate tfm_env 2>/dev/null; then
  echo "[ERROR] El entorno 'tfm_env' no existe. Ejecuta primero: bash install_env.sh"
  exit 1
fi
echo "Entorno activo: $(python -c 'import sys; print(sys.prefix)')"

# --- 1. Pipeline principal (s1 -> s6 + s5b) ---
echo ""
echo "======================================================"
echo "  [1/2] Pipeline principal (s1 -> s1b -> s2 -> s2b -> s3 -> s6)"
echo "======================================================"
if [ "$RED_DEPOSITADA" = "1" ]; then
  # Usa la red y la matriz de tiempos incluidas en el depósito (reproducen exactamente la
  # instancia N=10 y la Figura 4.1). Sin esta opción, s1 descarga la red actual de OpenStreetMap.
  for f in data/network/sevilla_drive.graphml data/farmacias/time_matrix.npy data/farmacias/farmacias_sevilla.csv; do
    [ -f "$f" ] || { echo "[ERROR] --red-depositada: falta $f"; exit 1; }
  done
  echo "  s1 y s1b omitidos: se usa la red depositada"
else
  python s1_red_sevilla.py
  python s1b_reubicar_deposito.py   # depósito en El Pino y componente fuertemente conexa
fi
python s2_nsga2_solver.py
python s2b_figura41.py
python s3_forecast_pipeline.py
python s4_integration.py
python s5_sensitivity.py   $QUICK
python s5b_solomon100.py   $QUICK
python s6_scale_n50.py     $QUICK

# --- 2. Post-procesado: Nemenyi y (opcional) comparación OR-Tools ---
echo ""
echo "======================================================"
echo "  [2/2] Post-procesado (Nemenyi" $([ "$RUN_S7" -eq 1 ] && echo "+ OR-Tools")")"
echo "======================================================"
# Post-hoc de Nemenyi a partir de los resultados guardados de s5b
python recompute_nemenyi.py
# Comparación empírica OR-Tools vs NSGA-II sobre la instancia real (s7)
if [ "$RUN_S7" -eq 1 ]; then
  python s7_ortools_compare.py --from-data data --n 25 --seed 42 --time-limit 30
else
  echo "  (s7 omitido por --skip-s7)"
fi

echo ""
echo "======================================================"
echo "  Pipeline completo finalizado correctamente."
echo "  Resultados en ./data/  ·  documentos en ./docs/"
echo "======================================================"
