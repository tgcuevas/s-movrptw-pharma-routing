# Optimización multiobjetivo estocástica de rutas de distribución farmacéutica urgente

Versión 1.0.0 del depósito *A hybrid stochastic multi-objective framework for urban emergency pharmaceutical routing: source code, synthetic data and results (S-MOVRPTW)*, que contiene el código y los datos del Trabajo Fin de Máster **«Optimización Multiobjetivo de Rutas de Emergencia Sanitaria en Entornos Urbanos»**, de Antonio González Cuevas (Máster en Modelización y Simulación del Transporte y la Movilidad, Universidad Pablo de Olavide, Sevilla, 2026).

El trabajo plantea un problema de rutas con capacidad y ventanas temporales, estocástico y con tres objetivos (tiempo de ruta, tardanza ponderada por prioridad clínica y un proxy de emisiones), para el reparto urgente de medicamentos a farmacias en Sevilla. Integra la predicción probabilística de la demanda (BiLSTM y XGBoost), la generación de escenarios correlacionados con SAA, la optimización con NSGA-II y su comparación con NSGA-III y MOEA/D, y el análisis de sensibilidad global de Sobol. Incluye además las pruebas de consolidación de la memoria (§4.7): pruebas placebo y controles negativos de las medidas del valor de la solución estocástica, y ocho preguntas contrastadas con criterios fijados antes de ejecutarlas (`protocolo_consolidacion.md`).

La instancia N=10 se construye sobre la red viaria real de OpenStreetMap; las de N=25 y N=50 usan una matriz de tiempos sintética con órdenes de magnitud coherentes con esa red. Las localizaciones de las farmacias son plausibles, definidas a mano, y la demanda es **sintética**. El trabajo no utiliza datos operacionales ni propietarios de ninguna entidad: la distribución farmacéutica urbana es solo el contexto de referencia.

## Summary (English)

Code and data for a master's thesis on stochastic tri-objective vehicle routing with time windows (route time, clinical-priority-weighted lateness and a CO₂ proxy) for urgent urban pharmaceutical distribution in Seville. The pipeline combines probabilistic demand forecasting (BiLSTM and XGBoost), correlated SAA scenarios, NSGA-II (compared with NSGA-III and MOEA/D) and Sobol global sensitivity analysis. The N=10 instance uses the real OpenStreetMap road network, the N=25 and N=50 instances a synthetic travel-time matrix, and demand is synthetic; no operational or proprietary data are used. The repository also contains the consolidation tests of the thesis: placebo tests and negative controls of the usual value-of-the-stochastic-solution measures, and eight questions tested against decision criteria fixed before running them. Documentation is in Spanish.

## Contenido

| Script | Función |
|---|---|
| `s1_red_sevilla.py` | Descarga la red viaria de Sevilla con OSMnx 2.1.0 y geocodifica 50 farmacias y 2 depósitos |
| `s1b_reubicar_deposito.py` | Sitúa el depósito en el Polígono Industrial El Pino, proyecta las localizaciones sobre la mayor componente fuertemente conexa y calcula la matriz de tiempos (Dijkstra dirigido) |
| `s2_nsga2_solver.py` | Motor NSGA-II (permutación, OX, inversión, Split de Prins): estabilidad sobre Solomon C101-50 y R101-50 (30 réplicas) e instancia Sevilla N=10 |
| `s2b_figura41.py` | Frente de la instancia N=10 (Figura 4.1) |
| `s3_forecast_pipeline.py` | Demanda sintética, BiLSTM probabilístico, XGBoost cuantílico, ensemble y K=20 escenarios SAA con Cholesky |
| `s4_integration.py` | Integración SAA: frentes EV y por escenario, EEV con recurso por ruta, VSS por objetivo y solución robusta TOPSIS |
| `s5_sensitivity.py` | Módulos A–F: Solomon N=25, Sobol, VSS_HV frente a K, cadena WS ≤ RP ≤ EEV, estabilidad y ablación de la predicción (véanse las «Notas sobre los datos») |
| `s5b_solomon100.py` | Benchmark Solomon de 100 clientes (6 familias, 30 réplicas) en su configuración original, que no es equiparable entre algoritmos (véase «Datos publicados») |
| `s6_scale_n50.py` | Escalado a N=50 y comparación con N=25 |
| `s7_ortools_compare.py` | Comparación con OR-Tools (búsqueda local guiada) sobre la instancia N=25 |
| `s8_mapa_instancia.py`, `s8_mapa_rutas.py` | Mapas de la instancia y de rutas sobre la red real |
| `recompute_eev_recurso.py`, `recompute_s5_recurso.py` | Recalculan el EEV con recurso por ruta sobre la ejecución de referencia (ver «Datos publicados») |
| `recompute_forecast_eval.py` | Evalúa la predicción frente a la demanda verdadera del día previsto (conocida por ser sintética): cobertura de [Q10, Q90] y skill del BiLSTM, XGBoost y el ensemble |
| `recompute_nemenyi.py`, `recompute_vss_tests.py` | Post-hoc de Nemenyi de s5b y test pareado del VSS a partir de resultados guardados |
| `replicas_vss.py`, `resumir_replicas.py` | Réplicas controladas de s3 → s4 para medir la variabilidad del VSS entre ejecuciones, y su resumen conjunto |
| `robustez_vss.py`, `protocolo_robustez.md` | Pruebas de robustez del VSS de §4.4.7 (plan único, simulación con desvíos al depósito, evaluación fuera de muestra frente a la predicción y frente a la demanda verdadera, réplicas) y el protocolo con los criterios fijados antes de ejecutarlas |
| `recompute_wilcoxon_holm.py` | Comparaciones por pares de Solomon 100 con Wilcoxon y corrección de Holm |
| `figuras_memoria.py` | Regenera las Figuras 4.2, 4.4, 4.5 y 4.6 de la memoria a partir de los datos publicados |
| `protocolo_consolidacion.md`, `protocolo_consolidacion.sha256` | Protocolo de las pruebas de consolidación (§4.7), con sus enmiendas, y las sumas de los scripts registradas antes de cada ejecución |
| `consolidacion_pruebas.py` | Pruebas placebo, control negativo de Sobol y convergencia (P1), comparación equilibrada (C1), problema de recurso con el modelo de §4.3.3 (C2) y descriptivos de la predicción (P5) |
| `s5b_equiparable.py` | Comparación de NSGA-II, NSGA-III y MOEA/D en condiciones equiparables (C3) |
| `consolidacion_complementarias.py` | Problema de recurso con la simulación con desvíos (C4), mecanismo de los desvíos (C5) y calidad de la predicción en las réplicas (C7) |
| `s5b_sevilla.py`, `s5b_presupuesto.py` | Comparación de algoritmos en los tamaños de Sevilla (C6) y regla del presupuesto por cliente (C8) |
| `comparar_consolidacion.py` | Compara dos ejecuciones de las pruebas de consolidación, fichero a fichero |
| `verify_stack.py` | Comprueba el entorno y las versiones instaladas |

## Instalación

Requiere Anaconda o Miniconda. No necesita GPU.

```bash
bash install_env.sh          # crea el entorno conda tfm_env (Python 3.11) e instala requirements.txt
conda activate tfm_env
python verify_stack.py       # comprueba versiones y funcionalidades
```

Las versiones están fijadas en `requirements.txt` (pymoo 0.6.1.3, OSMnx 2.1.0, XGBoost 3.2.0, Keras 3.13.2, OR-Tools 9.15, statsmodels 0.14.6, entre otras).

## Ejecución

```bash
bash run_pipeline.sh           # pipeline completo y post-procesado
bash run_pipeline.sh --quick   # verificación rápida con corridas reducidas
bash run_pipeline.sh --skip-s7 # sin la comparación con OR-Tools
bash run_pipeline.sh --red-depositada  # usa la red depositada en lugar de descargarla
```

Por defecto, s1 descarga la red viaria actual de OpenStreetMap, que se edita continuamente: los tiempos de viaje pueden variar unos segundos respecto a la red de junio de 2026 y, con ellos, la instancia N=10 de s2 y la Figura 4.1. Con `--red-depositada` se omiten s1 y s1b y se usan la red y la matriz de tiempos incluidas en el depósito, que reproducen exactamente lo publicado. El resto del pipeline (s3 a s7) no depende de la red descargada, porque usa la matriz calibrada de s4.

El orden es s1 → s1b → s2 → s2b → s3 → s4 → s5 → s5b → s6, seguido de `recompute_nemenyi.py` y `s7_ortools_compare.py`. Cada script escribe en `data/<módulo>/<identificador de ejecución>/` y los siguientes localizan la última ejecución mediante los ficheros `latest_*.txt`. En la máquina del autor, s5 tarda unos 40 minutos, s5b unos 25 y el resto pocos minutos cada uno; son tiempos orientativos.

Las réplicas de variabilidad no forman parte del pipeline y se lanzan aparte: `python replicas_vss.py` (5 réplicas por tamaño; restaura los punteros `latest_*.txt` al terminar) y `python resumir_replicas.py` para agregar todos los conjuntos. Las pruebas de robustez se ejecutan después con `python robustez_vss.py` (unos 15 minutos) y `python recompute_wilcoxon_holm.py`. La evaluación de la predicción frente a la demanda verdadera se obtiene con `python recompute_forecast_eval.py` (unos segundos).

Las pruebas de consolidación se lanzan al final: `python consolidacion_pruebas.py`, `python s5b_equiparable.py`, `python consolidacion_complementarias.py`, `python s5b_sevilla.py` y `python s5b_presupuesto.py`. Las cuatro últimas aceptan `--procesos N` para repartir el trabajo entre núcleos, y todas tienen un modo `--prueba` que comprueba el funcionamiento sin calcular ningún veredicto. En una máquina de 32 núcleos tardan, en conjunto, alrededor de una hora; la más lenta es `consolidacion_pruebas.py`, que usa un solo núcleo durante unos 45 minutos.

## Datos publicados y ejecución de referencia

Todas las cifras, tablas y figuras de la memoria proceden de la ejecución de referencia del 15 de junio de 2026 (semilla 42, K=20, n_gen=200 en s4):

| Módulo | Identificador |
|---|---|
| s3, predicción N=25 / N=50 | `20260615_200957` / `20260615_205432` |
| s4, integración N=25 / N=50 | `20260615_201043_s4` / `20260615_205516_s4` |
| s5, sensibilidad y validación | `20260615_201114_s5` |
| s6, escalado | `20260615_205431_s6` |

**EEV y VSS.** Los valores de EEV y de la diferencia EEV − RP de la Tabla 4.5, y los de VSS_HV de la Tabla 4.9, se obtienen con el recurso por ruta: cuando una ruta del plan determinista supera la capacidad en un escenario, se añade un viaje de vuelta al depósito. Están en `recompute_eev_recurso.json` (carpetas de s4) y en `recompute_s5_recurso.json`, `recompute_s5_mod_c_vsshv.csv` y `recompute_s5_mod_f_ablation.csv` (carpeta de s5). Ambos scripts de recálculo comprueban antes que reproducen exactamente los resultados guardados. Los ficheros originales de la ejecución (`s4_eev_eval.csv`, `s5_mod_c_vsshv.csv`, `s5_mod_d_evpi.json`, `s5_mod_f_ablation.csv`) se conservan, pero sus valores de EEV no son los que utiliza la memoria. Si se vuelve a ejecutar `recompute_s5_recurso.py`, los resultados nuevos se guardan con sufijo de fecha y los publicados no se tocan: los módulos C y D se reproducen exactamente, pero la fila BiLSTM+XGB del módulo F cambia en cada entrenamiento (9,8 % en el recálculo publicado, y 6,0 %, 13,7 % y 18,3 % en tres repeticiones en entornos nuevos, siempre por debajo del 20,3 % de la referencia naive). La memoria presenta la diferencia EEV − RP como una cota superior del valor de la solución estocástica y no usa el VSS_HV ni la cadena WS ≤ RP ≤ EEV para contrastar hipótesis, porque las pruebas placebo de `data/consolidacion/` muestran que estas medidas atribuyen a la incertidumbre ganancias que existen sin ella.

**Predicción.** `forecast_eval_verdad.json`, en las carpetas de s3 de N=25 y N=50, contiene la evaluación de la predicción frente a la demanda verdadera del día previsto (`python recompute_forecast_eval.py`). El script regenera primero el histórico sintético y comprueba que coincide con el publicado.

**Solomon 100.** `data/sensitivity/solomon100/` contiene la ejecución original de s5b, reproducible bit a bit. Su comparación entre algoritmos no es equiparable: MOEA/D trabaja con 15 individuos, NSGA-III se evalúa solo con sus soluciones representativas y la tardanza usa ventanas fijas por clase. Además, por un error en el generador, las familias RC1 y RC2 no son mixtas. La memoria usa la comparación equiparable de `data/consolidacion/` (C3).

**Consolidación.** `data/consolidacion/` contiene los resultados de §4.7: placebos, control negativo y convergencia (`P1_placebos_convergencia.json`), descriptivos de la predicción (`P5_descriptivos.json`), la comparación equilibrada y el problema de recurso (`referencia_N25.json`, `referencia_N50.json`, `replicas.json` y `decision_C1_C2.json`) y la comparación equiparable de algoritmos (`P4_*`). En `complementarias/` están las preguntas C4 a C8. Cada fichero de decisión recoge el veredicto según el protocolo, y `resultados_consolidacion.sha256` registra la suma de los 20 ficheros de resultados.

**Réplicas.** `data/replicas_vss/` contiene dos conjuntos de cinco réplicas de s3 → s4 por tamaño, generados con el código de esta versión y la misma semilla que la ejecución de referencia, a la que complementan sin sustituirla. `replicas_vss_resumen_global.json` los agrega y es el fichero que cita §5.3.4 de la memoria.

**Robustez.** `data/robustez/` contiene los resultados de §4.4.7: la validación de la simulación (`validacion.json`), la ejecución de referencia (`referencia_N25.json`, `referencia_N50.json` y las diferencias por escenario en los CSV), las réplicas (`replicas.json`) y la decisión según el protocolo (`decision.json`).

**Figuras.** `figuras_memoria/` contiene las figuras tal como aparecen en la memoria. Las figuras generadas automáticamente dentro de las carpetas de ejecución no se publican.

**Verificación.** `MANIFEST.sha256` recoge la suma SHA-256 de cada fichero (`sha256sum -c MANIFEST.sha256`), `TRAZABILIDAD.md` relaciona cada tabla y figura de la memoria con sus ficheros y `SANITIZACION.md` enumera los ficheros publicados con alguna diferencia respecto a la carpeta de trabajo.

**Red viaria.** Descargada de OpenStreetMap el 15 de junio de 2026 con un radio de 18 km: 34.962 nodos y 75.254 tramos (`data/network/red_stats.json`).

## Reproducibilidad

Son reproducibles bit a bit con semilla fija la red y la matriz de tiempos (dada la red publicada), el motor NSGA-II de s2, los módulos A–E de s5, el benchmark s5b y los recálculos, salvo la fila del ensemble de la ablación (módulo F), que entrena un BiLSTM. No lo son las magnitudes que dependen del entrenamiento del BiLSTM en TensorFlow, que no es determinista: predicción, escenarios SAA y, a partir de ellos, VSS, soluciones EV y robusta (s3, s4, s6 y la fila BiLSTM+XGB de la ablación). La comparación con OR-Tools usa un límite de tiempo, por lo que su resultado puede variar con la velocidad de la máquina.

Una ejecución nueva del pipeline produce por tanto cifras distintas de las de la memoria en esos módulos. Para reproducir exactamente la memoria hay que partir de los datos publicados. La variabilidad entre ejecuciones está medida en `data/replicas_vss/`.

Las pruebas de consolidación son reproducibles con las semillas fijadas en el protocolo: una segunda ejecución completa en una carpeta limpia dio los mismos 20 ficheros de resultados, comprobados con `comparar_consolidacion.py`, y los veredictos se verificaron además con un recálculo independiente.

## Resultados principales

| Resultado | Valor | Fichero |
|---|---|---|
| Diferencia EEV − RP, N=25 (f₁ / f₂ / f₃): cota superior | 6,3 % / 16,7 % / 7,7 %; sin incertidumbre, el mismo cálculo da 3,4 % / 15,1 % / 3,7 % | `integration/20260615_201043_s4/recompute_eev_recurso.json` · `consolidacion/P1_placebos_convergencia.json` |
| Diferencia EEV − RP, N=50 (f₁ / f₂ / f₃): cota superior | 7,2 % / 14,9 % / 10,1 %; sin incertidumbre, 4,6 % / 15,4 % / 4,8 % | `integration/20260615_205516_s4/recompute_eev_recurso.json` · ídem |
| Viajes de recurso del plan determinista | 6 en 6 de 20 escenarios (N=25) · 15 en 11 de 20 (N=50) | `recompute_eev_recurso.json` |
| Reducción de la tardanza con planes generados con escenarios, fuera de muestra y con el mismo número de candidatos (C1, confirmada) | 8,9 % (N=25) · 22,3 % (N=50), frente a la demanda verdadera | `consolidacion/decision_C1_C2.json` |
| Desvíos al depósito por ronda, plan determinista → plan con escenarios (C5, confirmada) | 0,37 → 0,08 (N=25) · 1,16 → 0,66 (N=50) | `consolidacion/complementarias/decision_C4_C5_C7.json` |
| Plan del problema de recurso: modelo de §4.3.3 (C2) y simulación con desvíos (C4) | C2 refutada · C4 no concluyente | `decision_C1_C2.json` · `complementarias/decision_C4_C5_C7.json` |
| Predicción N=25, frente a la demanda verdadera del día previsto | cobertura de [Q10, Q90] del BiLSTM 84,4 % (nominal 80 %) · skill frente a naive 21,8 % (máximo posible ese día, 22,3 %) · exceso sobre el error mínimo inferior al 2 % en las 20 réplicas (C7, confirmada) | `forecast/20260615_200957/forecast_eval_verdad.json` · `consolidacion/P5_descriptivos.json` · `complementarias/C7_prediccion_replicas.json` |
| VSS_HV con y sin incertidumbre | 5,5–12,1 % con incertidumbre · 10,7–18,2 % sin ella: no aísla el valor de la solución estocástica | `sensitivity/20260615_201114_s5/recompute_s5_mod_c_vsshv.csv` · `consolidacion/P1_placebos_convergencia.json` |
| Sobol S₁ sobre HV (288 evaluaciones) | n_gen 0,49 ± 0,47 · parámetros sin efecto en el modelo 0,14 ± 0,44 (control negativo): precisión insuficiente para ordenar | `sensitivity/20260615_201114_s5/s5_mod_b_sobol.json` |
| Convergencia de NSGA-II | el hipervolumen sigue creciendo un 60 % (N=25) y un 110 % (N=50) de 200 a 1.600 generaciones | `consolidacion/P1_placebos_convergencia.json` |
| CV del HV de NSGA-II (30 réplicas) | 9,5 % (N=25) · 10,7 % (N=50) · motor de s2: 35,8 % (C101-50) y 28,2 % (R101-50) | `s5_mod_e_stability.csv` · `s6_n50_stability.csv` · `solver/solomon_benchmark.csv` |
| Comparación equiparable de algoritmos (C3, C6 y C8: refutadas) | con 20.000 evaluaciones, NSGA-II es mejor en las 7 instancias de 25 clientes y MOEA/D en las de 50 y 100, con frentes de 1 a 3 soluciones; en Sevilla-50, NSGA-II solo iguala a MOEA/D con 80.000 evaluaciones | `consolidacion/P4_*` · `complementarias/C6_*` · `complementarias/C8_*` |
| OR-Tools frente a NSGA-II (f₁) | 7,13 h frente a 4,86 h, soluciones mutuamente no dominadas | `compare_ortools/ortools_vs_nsga2.json` |
| Robustez con plan único, frente a la demanda verdadera (§4.4.7) | Tardanza: N=25 17,6 % (réplicas: mediana 12,1 %), N=50 26,8 % (17,5 %), positiva en las 20 réplicas · Tiempo, plan más rápido: N=25 4,1 % (5,6 %), N=50 1,1 % (1,4 %) · Decisión del protocolo: no concluyente en ambas afirmaciones | `robustez/referencia_N25.json`, `robustez/referencia_N50.json`, `robustez/replicas.json`, `robustez/decision.json` |

## Notas sobre los datos

- `scale_n50/20260615_205431_s6/s6_summary.json` y `s6_comparison_n25_n50.csv` no son fuente de la memoria. El primero incluye un campo `n25_reference` con valores fijos que escribía una versión anterior de s6; el segundo recoge el VSS de f₁ con la aproximación de recurso anterior, el proxy de skill del BiLSTM y un CV del hipervolumen con desviación poblacional. La Tabla 4.11 y la Figura 4.6 se construyen a partir de los ficheros que indica `TRAZABILIDAD.md`.
- En modo `--quick` el pipeline solo verifica que todo se ejecuta: usa instancias y corridas reducidas, y sus resultados no son comparables con los de la memoria.
- `s4_eev_eval.csv` cuenta los viajes extra con una aproximación agregada anterior. Los valores que usa la memoria están en `recompute_eev_recurso.json`, que incluye también el test pareado de H1 con el recurso por ruta. `recompute_vss_tests.json` contiene los tests pareados de f₂ (H2) y f₃, que no dependen del recurso; su entrada de f₁ usa la aproximación anterior. La memoria no usa estos tests como evidencia, porque comparan cada escenario con su propio óptimo por objetivo (§4.3.3).
- Las comparaciones entre algoritmos de los módulos A y E de s5 (`s5_mod_a_*` y `s5_mod_e_wilcoxon.json`) usan la configuración de s5b, que no es equiparable, y el IGD del módulo E se calcula con un frente de referencia construido solo con NSGA-II. Del módulo E, la memoria usa solo la estabilidad de NSGA-II.
- En el análisis de Sobol (`s5_mod_b_sobol.json`), `p_cross`, `p_mut` y `k_saa` no intervienen en el modelo evaluado, por lo que sus índices funcionan como control negativo.
- En `forecast_metrics.json`, los campos `rmse_q50_approx` y `skill_q50_vs_naive` del bloque `bilstm` son un proxy heurístico derivado del error de XGBoost, no una medida del BiLSTM. La calidad de la predicción se evalúa en `forecast_eval_verdad.json`, frente a la demanda verdadera del día previsto. El campo `calibration.empirical_coverage_pct` no es la cobertura de [Q10, Q90]: mide una banda ensanchada en una desviación típica histórica por cada lado sobre los últimos 20 días del histórico, y la memoria no lo usa.
- El plan determinista y los escenarios SAA de s4 usan la media y la desviación predichas por el BiLSTM (`demand_q50`, `mu_real`, `sigma_real` de `demand_forecast.csv`). Las columnas `ens_q10`, `ens_q50` y `ens_q90` del ensemble se calculan y se publican, pero no alimentan la optimización.
- `s8_mapa_rutas.py` toma la demanda y las prioridades del run de s3 que indique `data/forecast/latest_run.txt`. El mapa ilustrativo de la presentación de la defensa usa el run de referencia de N=25: `python s8_mapa_rutas.py --run 20260615_200957`.
- `s7_ortools_compare.py` resuelve con OR-Tools la misma instancia que el problema EV de s4, con la mediana del BiLSTM (`demand_q50`) como demanda, y evalúa ambas soluciones con las mismas fórmulas que s4.

## Cita

Los metadatos de cita están en `CITATION.cff`. Para citar el trabajo:

```bibtex
@mastersthesis{gonzalezcuevas2026tfm,
  author = {González Cuevas, Antonio},
  title  = {Optimización Multiobjetivo de Rutas de Emergencia Sanitaria en Entornos Urbanos},
  school = {Universidad Pablo de Olavide},
  year   = {2026},
  type   = {Trabajo Fin de Máster}
}
```

Código y datos (versión 1.0.0): Zenodo, https://doi.org/10.5281/zenodo.20714222

## Licencia

El código se publica bajo licencia MIT (`LICENSE`). Los datos sintéticos y los resultados, bajo CC BY 4.0. Los ficheros de red derivados de OpenStreetMap (`data/network/`, matriz de tiempos) se rigen por la Open Database License (ODbL) 1.0, © colaboradores de OpenStreetMap.

## Contacto

Antonio González Cuevas · tgcuevas@gmail.com · ORCID [0009-0001-1381-9061](https://orcid.org/0009-0001-1381-9061)
# s-movrptw-pharma-routing
