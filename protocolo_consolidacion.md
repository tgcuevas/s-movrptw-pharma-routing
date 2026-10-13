# Protocolo de pruebas de consolidación

**Versión 1.0 · 3 de octubre de 2026 · Aprobado por el autor y fijado antes de ejecutar ninguna prueba**

Este protocolo se congela antes de ejecutar nada: se registra su SHA-256 y el de los scripts, y se incorpora al repositorio con fecha. Cualquier cambio posterior se documenta como enmienda en el apartado 11, con su motivo, igual que en `protocolo_robustez.md`.

## 1. Objetivo

Una verificación del código frente a la memoria v6.27 mostró que varias métricas no miden lo que la memoria afirma. El VSS de §4.3.3 compara el compromiso del plan determinista con el mínimo de cada objetivo por escenario. El VSS_HV compara la unión de K frentes con una sola solución. El EVPI compara dos ejecuciones equivalentes. Y la comparación de algoritmos de §4.4.1 y del Anexo E no es equiparable. Este protocolo responde a cuatro preguntas:

- **C1.** ¿La mejora en tardanza (f₂) de §4.4.7 resiste cuando ambos lados eligen su plan entre el mismo número de candidatos?
- **C2.** ¿Mejora la tardanza un plan del problema de recurso propiamente dicho, que minimiza el coste esperado sobre los escenarios, frente al plan determinista?
- **C3.** ¿Obtiene NSGA-II mejor rendimiento que NSGA-III y MOEA/D cuando los tres trabajan en condiciones equiparables? Es la hipótesis H3.
- **C4.** ¿Cuánto vale cada métrica cuando no hay incertidumbre, y cuánto falta para que NSGA-II converja? Esta pregunta es descriptiva, sin criterio de decisión.

**Transparencia sobre lo ya conocido.** Antes de este protocolo se observaron, de forma exploratoria y sobre la ejecución de referencia N=25, tres resultados:

- el VSS de s4 con escenarios idénticos: 3,4 %, 15,1 % y 3,7 %;
- el VSS_HV con escenarios idénticos: 13,8 % (K=8), 17,2 % (K=20) y 18,2 % (K=30);
- el aumento del hipervolumen del problema EV entre 200 y 800 generaciones: un 74 %, con la semilla 42.

Esos resultados motivan las pruebas P1. Se repiten aquí con scripts que se publicarán; si los valores difieren, se informa. **No se ha observado ningún resultado de C1, C2 ni C3.**

## 2. Datos

- **Ejecución de referencia:** la del 15 de junio de 2026, N=25 (s3 `20260615_200957`, s4 `20260615_201043_s4`) y N=50 (s3 `20260615_205432`, s4 `20260615_205516_s4`).
- **Réplicas:** las 10 por tamaño de `data/replicas_vss/`, igual que en `protocolo_robustez.md`.
- **Escenarios SAA:** los 20 de cada ejecución, que se reproducen y se verifican.
- **Evaluación fuera de muestra:** la misma de `protocolo_robustez.md`:
  - 1.000 escenarios de la distribución de la predicción, con semilla 20260925;
  - 1.000 realizaciones de la demanda verdadera del día previsto, con semilla 20260926.
- **Código de partida:** el del depósito (`~/deposito_prueba`), no la carpeta de trabajo `t_1_0`, que tiene versiones anteriores de `s7_ortools_compare.py` y de un mensaje de `s5_sensitivity.py`.

## 3. Planes, selección y medida (C1 y C2)

Se mantienen las definiciones de `protocolo_robustez.md`:

- **Plan:** una permutación con sus rutas cortadas por capacidad según su demanda de diseño.
- **Selección,** con los escenarios de la muestra SAA y la misma regla en ambos lados: **(B)** el mejor plan para cada objetivo o **(A)** el compromiso TOPSIS.
- **Medida:** VSS_o = (E[f_o(plan determinista)] − E[f_o(plan comparado)]) / E[f_o(plan determinista)] × 100, con intervalo de confianza al 95 % de la diferencia pareada por escenario.
- **Evaluación:** con la simulación S, que es la referencia, y con el modelo M como comparación, ambos ya validados en `protocolo_robustez.md` (pruebas V0 a V5).

## 4. Pruebas

### P1. Pruebas placebo y convergencia (C4, descriptiva)

- **P1a. VSS de s4 sin incertidumbre.** Las funciones de `s4_integration.py`, sin modificar, con los 20 escenarios iguales a la demanda Q50, en las ejecuciones de referencia N=25 y N=50. Se informa del VSS por objetivo y de su diferencia con el VSS publicado.
- **P1b. VSS_HV sin incertidumbre.** El módulo C de `recompute_s5_recurso.py`, sin modificar, con los escenarios iguales a la media, para K ∈ {3, 5, 8, 10, 15, 20, 25, 30} y 5 repeticiones.
- **P1c. EVPI sin incertidumbre.** El módulo D de `recompute_s5_recurso.py` con escenarios idénticos. Si el EVPI no es nulo, mide la diferencia entre ejecuciones.
- **P1d. Control negativo de Sobol.** Se informa, a partir de los datos publicados, de los índices de `p_cross`, `p_mut` y `k_saa`, que no intervienen en el modelo y deberían ser nulos.
- **P1e. Convergencia de NSGA-II.** El problema EV de N=25 y N=50 con n_gen ∈ {200, 400, 800, 1.600} y semillas 42 a 46. El hipervolumen se calcula con un punto de referencia común por tamaño. Se considera convergencia práctica un aumento del hipervolumen inferior al 1 % entre 800 y 1.600 generaciones.

### P2. Comparación equilibrada (C1)

- **Lado determinista:** 20 ejecuciones del problema EV (demanda Q50) con semillas 1042 a 1061. Son distintas de las de los escenarios SAA (42 a 61), con n_gen = 200 y población 100, como en s4. Los candidatos son la unión de los 20 frentes.
- **Lado estocástico:** los 20 frentes SAA de cada ejecución, reproducidos.
- **Selección, medida y evaluación:** las del apartado 3, con las mismas reglas en ambos lados.
- **Estadísticos de decisión:** E1q (VSS de f₂ con la regla B), E2q (f₂ con la regla A), E3q (f₁ con la regla B) y E4q (f₁ con la regla A).
- **Alcance:** la ejecución de referencia y las 10 réplicas de cada tamaño.

### P3. Plan del problema de recurso (C2)

- **Problema:** NSGA-II sobre la permutación, con las rutas cortadas según la demanda Q50, como un plan de primera etapa. Su objetivo es la media, sobre los 20 escenarios SAA, de los tres objetivos evaluados con el modelo M y recurso por ruta. Es el problema de recurso min_x E[F(x, ξ)] en su aproximación SAA.
- **Comparación simétrica:** una ejecución del problema de recurso (semilla 2042) frente a una del problema EV (la de s4, semilla 42), con n_gen = 200 y población 100 en ambas.
- **Selección, medida y evaluación:** las del apartado 3.
- **Estadísticos de decisión:** E1r, E2r, E3r y E4r, definidos como en P2.
- **Alcance:** la ejecución de referencia y las 10 réplicas de cada tamaño.

### P4. Repetición equiparable de la comparación de algoritmos (C3)

- **Instancias:** las dos del módulo A (C101-25 y R101-25) y las seis familias de Solomon con 100 clientes del Anexo E.
- **Presupuesto igual:** 20.000 evaluaciones para cada algoritmo, con terminación por número de evaluaciones.
- **Configuración:** 91 direcciones de referencia (das-dennis, 12 particiones) para NSGA-III y MOEA/D, y población 92 para NSGA-II y NSGA-III.
- **Conjunto evaluado:** el frente no dominado de la población final, sin duplicados, para los tres algoritmos. Para NSGA-III se usa la población, no solo sus representantes.
- **Instancias por réplica:** la demanda se muestrea en cada réplica (semilla 30000 + r), y los tres algoritmos se ejecutan sobre la misma instancia (semilla 40000 + r). La tardanza usa las ventanas de cada cliente generadas para su familia, no las ventanas fijas por clase.
- **Indicadores:**
  - **normalización:** con el ideal y el nadir de la unión de los tres frentes en cada réplica;
  - **hipervolumen:** con el punto de referencia (1,1; 1,1; 1,1);
  - **IGD:** frente a esa unión no dominada;
  - **Spacing:** sobre los frentes normalizados y sin duplicados.
- **Contrastes:** en cada instancia o familia, Friedman con el post-hoc de Nemenyi sobre el hipervolumen de 30 réplicas pareadas, y Wilcoxon por pares con la corrección de Holm (α = 0,05).

### P5. Cálculos pendientes (descriptivos)

- **P5a. Skill máximo posible.** El skill de un predictor que acierta la media verdadera del día previsto frente a la referencia naive, sobre las 10.000 realizaciones de `recompute_forecast_eval.py`. Sirve para leer el 21,8 % del BiLSTM (punto 29).
- **P5b. Variabilidad de los escenarios.** El CV de la demanda total según el modelo de escenarios, comparado con el de la muestra de 20 escenarios de la Tabla 4.4 (punto 32).

## 5. Validaciones previas

El script se detiene si alguna falla, antes de calcular ningún estadístico de decisión.

- **W0.** Las pruebas V0 a V5 de `protocolo_robustez.md` se repiten sin cambios.
- **W1.** Cada ejecución EV de P2, repetida con la misma semilla, da el mismo frente.
- **W2.** El objetivo del problema de recurso de P3, evaluado en el plan de compromiso EV, coincide con el EEV publicado (`recompute_eev_recurso.json`), con una tolerancia de 10⁻⁶.
- **W3.** En P1a, con escenarios idénticos, el EEV coincide exactamente con los valores del compromiso EV, porque no hay viajes extra.
- **W4.** En P4, cada ejecución registra 20.000 evaluaciones, con una tolerancia de una población.
- **W5.** En P4, el generador reproducido da exactamente la instancia original con su semilla (enmienda 1.1).

## 6. Criterios de decisión

Los umbrales de 8 y 6 réplicas son la misma convención de `protocolo_robustez.md`, y su interpretación es la misma.

| Pregunta | Confirmada si… | Refutada si… | No concluyente |
|---|---|---|---|
| **C1. La mejora en tardanza resiste la asimetría** | En la ejecución de referencia, el extremo inferior del IC 95 % de E1q y de E2q es > 0 en ambos tamaños y en los dos tipos de evaluación, **y** E1q > 0 en al menos 8 de las 10 réplicas de cada tamaño, en los dos tipos | La estimación puntual de E1q es ≤ 0 en la ejecución de referencia de algún tamaño o tipo, o E1q > 0 en menos de 6 de las 10 réplicas | En cualquier otro caso |
| **C2. El plan del problema de recurso mejora la tardanza** | Lo mismo con E1r y E2r | Lo mismo con E1r | En cualquier otro caso |
| **C3. NSGA-II supera a NSGA-III y MOEA/D (H3)** | NSGA-II tiene el mejor rango medio de hipervolumen en el conjunto de las 8 instancias o familias **y** supera de forma significativa a cada competidor (Wilcoxon-Holm) en al menos 5 de las 8 | Otro algoritmo tiene el mejor rango medio y supera de forma significativa a NSGA-II en al menos 5 de las 8 | En cualquier otro caso |

**Magnitud, sin valor de decisión:** para C1 y C2 se informa de si el extremo inferior del IC de E1 supera el 10 % (umbral de H2), y del valor de E3 frente al 5 % (umbral de H1). E4q y E4r se informan como descripción.

## 7. Réplicas

Se aplican las reglas del apartado 7 de `protocolo_robustez.md`: cada réplica debe reproducir sus frentes; si una no lo hace, se excluye y se informa; y se publican la mediana, el rango y el recuento de todas, sin seleccionar ninguna.

## 8. Reproducibilidad

- **Semillas:**
  - EV de P2: 1042 a 1061;
  - problema de recurso de P3: 2042;
  - P4: demanda 30000 + r, algoritmos 40000 + r;
  - P1e: 42 a 46;
  - fuera de muestra: 20260925 y 20260926.
- **Scripts:** `consolidacion_pruebas.py` (P1, P2, P3 y P5) y `s5b_equiparable.py` (P4). Su SHA-256 se registra en el apartado 11 antes de ejecutar.
- **Salidas:** `data/consolidacion/`, con un fichero JSON por prueba y un `decision.json` con el veredicto de C1, C2 y C3.
- Los resultados se publican sea cual sea el veredicto.

## 9. Consecuencias para la memoria

| Resultado | Consecuencia |
|---|---|
| C1 y C2 confirmadas | **H2: verificada en el signo.** Pasa a «verificada» si, además, el extremo inferior del IC de E1 supera el 10 % en la referencia de ambos tamaños y tipos y en al menos 8 de las 10 réplicas. La mejora en tardanza es el resultado principal del trabajo |
| Solo C1 o solo C2 confirmada | H2 queda «verificada en el signo» con la salvedad correspondiente, que se declara |
| C1 y C2 no concluyentes | H2 queda «no concluyente»; el signo se informa como tendencia |
| C1 o C2 refutada | H2 queda «no verificada». La memoria informa de que la ventaja en tardanza no resiste la prueba correspondiente |
| C3 confirmada | H3 queda «verificada en condiciones equiparables» y se sustituyen la Tabla 4.6, la Figura 4.2 y el Anexo E |
| C3 refutada o no concluyente | H3 queda «no verificada» o «no concluyente» y se informa del algoritmo con mejor rango |
| H1 | Se evalúa con E3q y E3r frente al 5 %. Si el extremo inferior del IC no supera el 5 %, queda «no verificada» |
| H5 | Queda «no evaluable», porque la métrica no es válida (P1b). La curva frente a K pasa a ilustrar el sesgo |
| H6 | Se mantiene solo entre los parámetros efectivos, y la dominancia de n_gen se interpreta como falta de convergencia (P1d y P1e) |

## 10. Lo que este protocolo no decide

No cambia la formulación de ninguna hipótesis. No sustituye el VSS de §4.3.3, que pasa a presentarse como cota superior con su descomposición (P1a). Tampoco repite el análisis de Sobol con los parámetros efectivos y las generaciones convergidas, que queda como trabajo futuro.

## 11. Registro de fijación y enmiendas

- **Fijación del protocolo:** 3 de octubre de 2026, aprobado por el autor sin cambios sobre el borrador. Su SHA-256 se calcula sobre este fichero con esta línea incluida y se publica junto a él en `protocolo_consolidacion.sha256`.
- **Fijación de los scripts:** su SHA-256 se registra en `protocolo_consolidacion.sha256` antes de la primera ejecución.
- **Análisis de resultados no concluyentes:** cualquier análisis posterior al veredicto se presenta como exploratorio. Una prueba nueva solo se añade como enmienda fechada antes de ejecutarla.
- **Enmienda 1.1 (3 de octubre de 2026, antes de cualquier ejecución).** Al escribir el script de P4 se comprobó que los generadores de instancias (`gen_solomon_100` en s5b y `gen_solomon_instance` en s5) solo calculan un ancho de ventana por familia (`tw_width`) y no crean ventanas por cliente. Además, generan la geometría y la demanda con el mismo generador aleatorio. Para aplicar el apartado 4 (P4) sin ambigüedad se fija lo siguiente. Ningún criterio de decisión cambia.
  - **Ventana de cada cliente:** [0, bᵢ], con bᵢ = τ(0, i) + ancho de su familia, donde τ(0, i) es el tiempo de viaje directo desde el depósito. La tardanza es f₂ = Σᵢ α_p(i) · max(0, cᵢ − bᵢ), con cᵢ el fin de servicio y los pesos de prioridad de siempre (10.000, 5.000 y 1.000). Así se conserva la diferencia entre familias de ventanas ajustadas y amplias que pretendía el generador.
  - **Instancias:** la geometría, las demandas medias μᵢ y las prioridades de cada instancia son las originales, generadas con sus semillas de siempre: SEED + crc32(familia) mod 1000 en Solomon-100, y SEED y SEED + 1 en C101-25 y R101-25. Solo la demanda realizada se muestrea en cada réplica, NegBin(μᵢ, φ = 4) acotada en 0,5 kg, con semilla 30000 + r. El script reproduce el generador original y comprueba, con su semilla, que da exactamente la misma instancia (validación W5).
  - **Vecindad de MOEA/D:** 20 vecinos, el valor por defecto de pymoo, porque el protocolo no lo fijaba. La configuración anterior usaba 5.
  - **Tolerancia de W2:** `recompute_eev_recurso.json` publica el EEV redondeado y solo recalcula f₁ con recurso por ruta; f₂ y f₃ los toma de s4, que escala f₃ de otra forma. W2 compara f₁ y f₂ con una tolerancia de 0,01 en unidades del objetivo, y la diferencia en f₃ se informa sin detener el script.

- **Enmienda 1.2 (4 de octubre de 2026, después de conocer los veredictos de C1 a C3 y antes de calcular nada de C4 a C7).** Se añaden cuatro preguntas complementarias. **No modifican H1 a H6 ni los veredictos de C1 a C3.** Se formulan a raíz de esos resultados, y la memoria lo dirá así. Ninguno de sus estadísticos de decisión se ha observado.
  - **C4. Problema de recurso con un modelo realista (complementa H2).** ¿Mejora la tardanza un plan del problema de recurso cuyo objetivo es la media, sobre los 20 escenarios SAA, de la **simulación S**, que sí recoge los retrasos de los desvíos al depósito? Es igual que P3, salvo por el objetivo, y usa la semilla 3042. La comparación es simétrica, una ejecución frente a la del problema EV (semilla 42), con la selección, la medida y la evaluación del apartado 3 y el alcance de P3. Estadísticos E1s a E4s. Criterios: los de C2.
  - **C5. Mecanismo de C1.** ¿Provocan los planes elegidos en P2 entre los frentes SAA menos desvíos al depósito fuera de muestra que los elegidos entre los frentes EV? Se mide Δ = desvíos medios por ronda del plan determinista − desvíos del plan SAA, con IC 95 % pareado por escenario, para la selección de E1 (regla B en f₂) y la de E2 (regla A), con la simulación S. **Confirmada** si, en la ejecución de referencia, el extremo inferior del IC de Δ es > 0 para ambas selecciones, en ambos tamaños y en los dos tipos de evaluación, y Δ(E1) > 0 en al menos 8 de las 10 réplicas de cada tamaño y tipo. **Refutada** si la estimación puntual de Δ(E1) es ≤ 0 en la referencia de algún tamaño o tipo, o si Δ(E1) > 0 en menos de 6 de las 10 réplicas. **No concluyente** en otro caso.
  - **C6. NSGA-II en los tamaños de Sevilla (complementa H3).** ¿Supera NSGA-II a MOEA/D, con presupuesto igual, en las instancias del problema real y en instancias nuevas de 25 clientes? Instancias: el problema EV de Sevilla con N=25 y N=50 (s4, demanda Q50, matriz de Sevilla y ventanas por clase) y cuatro instancias nuevas del generador de s5, C101 con semillas 1001 y 1002 y R101 con 1003 y 1004, con las ventanas y la demanda por réplica de la enmienda 1.1. La configuración, el presupuesto, los indicadores y las 30 réplicas son los de P4; en Sevilla, la demanda es la Q50 en todas las réplicas. Contraste: Wilcoxon pareado del hipervolumen entre NSGA-II y MOEA/D en cada instancia, con corrección de Holm sobre las seis. **Confirmada** si NSGA-II tiene mayor hipervolumen mediano con diferencia significativa en las dos instancias de Sevilla y en al menos 3 de las 4 nuevas. **Refutada** si MOEA/D es significativamente mejor en alguna instancia de Sevilla o en al menos 3 de las 4 nuevas. **No concluyente** en otro caso. NSGA-III se informa como descripción. Se conocen los resultados de P4 en C101-25 y R101-25, donde NSGA-II superó a MOEA/D; las seis instancias de C6 no se han ejecutado nunca.
  - **C7. Calidad de la predicción en las réplicas (complementa H4).** Para cada una de las 10 réplicas de cada tamaño, con el procedimiento de P5a (10.000 realizaciones de la demanda verdadera, semilla 20261003), se calcula el exceso del RMSE del BiLSTM sobre el error irreducible, que es el RMSE del predictor que conoce la media verdadera: exceso = RMSE_BiLSTM / RMSE_oráculo − 1. **Confirmada** si el exceso es < 2 % en al menos 8 de las 10 réplicas de cada tamaño. **Refutada** si se cumple en menos de 6 de las 10 en algún tamaño. **No concluyente** en otro caso. **Justificación del umbral:** con demanda Binomial Negativa de φ = 4 domina el ruido irreducible, y un exceso inferior al 2 % indica un error prácticamente igual al mínimo posible. **Transparencia:** en la ejecución de referencia, el exceso calculado a partir de P5a es de un 0,9 % en N=25 y en N=50; las réplicas no se han observado. El ensemble se informa como descripción.
  - **Consecuencias.** H1 a H6 conservan sus veredictos.
    - C4 confirmada: la memoria presenta un VSS de tardanza bien definido y positivo con un modelo realista, como resultado complementario a H2.
    - C5 confirmada: el mecanismo de C1 queda establecido.
    - C6 confirmada: la elección de NSGA-II para Sevilla queda justificada con evidencia confirmatoria, sin cambiar H3, que se refiere a la superioridad global.
    - C7 confirmada: la calidad de la predicción es robusta frente al entrenamiento.
    - Cualquier resultado refutado o no concluyente se informa igual.
  - **Scripts y semillas:** `consolidacion_complementarias.py` (C4, C5 y C7) y `s5b_sevilla.py` (C6), con las semillas indicadas. Su SHA-256 se registra en `protocolo_consolidacion.sha256` antes de ejecutar, y las salidas van a `data/consolidacion/complementarias/`.
- **Enmienda 1.3 (4 de octubre de 2026, después de conocer los veredictos de C1 a C7 y antes de ejecutar nada de C8).** Se añade una pregunta complementaria. No modifica H1 a H6 ni los veredictos anteriores.
  - **C8. Regla del presupuesto por cliente (complementa H3, C3 y C6).** En C3 y C6, con 20.000 evaluaciones, NSGA-II superó a MOEA/D en las siete instancias de 25 clientes (800 evaluaciones por cliente), y MOEA/D superó a NSGA-II en las de 50 y 100 clientes (400 y 200 por cliente). C8 contrasta si la ventaja depende del presupuesto por cliente. Para ello se ejecutan, con presupuestos que no se han usado nunca, la instancia de Sevilla con 50 farmacias (s4, demanda Q50) y la familia R1 de 100 clientes (s5b, demanda por réplica y ventana por cliente, enmienda 1.1):
    - **condiciones de decisión**, con 800 evaluaciones por cliente: Sevilla-50 con 40.000 y R1-100 con 80.000;
    - **condiciones descriptivas:** Sevilla-50 con 80.000 (1.600 por cliente) y R1-100 con 40.000 (400 por cliente).
  - **Configuración:** la de P4, con los tres algoritmos, 30 réplicas, semillas 40000 + r para los algoritmos y 30000 + r para la demanda de R1-100, e indicadores normalizados con la unión de los tres frentes de cada réplica. NSGA-III se informa como descripción.
  - **Contraste:** Wilcoxon pareado del hipervolumen entre NSGA-II y MOEA/D en cada condición, con corrección de Holm sobre las dos condiciones de decisión.
  - **Criterios.** **Confirmada** si, en las dos condiciones de decisión, NSGA-II tiene mayor hipervolumen mediano que MOEA/D con diferencia significativa. **Refutada** si MOEA/D es significativamente mejor en las dos. **No concluyente** en cualquier otro caso. En las cuatro condiciones se informa también del número de réplicas en que MOEA/D domina el frente entero de NSGA-II y del tamaño de los frentes.
  - **Consecuencias.** C8 confirmada: la memoria indica cuándo conviene cada algoritmo, NSGA-II a partir de unas 800 evaluaciones por cliente y MOEA/D por debajo. Refutada o no concluyente: el criterio se presenta con el alcance que muestren los datos.
  - **Script:** `s5b_presupuesto.py`. Su SHA-256 se registra en `protocolo_consolidacion.sha256` antes de ejecutar, y las salidas van a `data/consolidacion/complementarias/`.