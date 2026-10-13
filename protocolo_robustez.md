# Protocolo de pruebas de robustez del VSS

**Versión 1.2 · 25 de septiembre de 2026 · Fijado antes de calcular ningún estimando**

**Enmienda 1.2.** Al ejecutar las validaciones, la prueba V4 de la versión 1.1 falló y el script se detuvo antes de calcular ningún estimando. El diagnóstico mostró que el código era correcto y que el error estaba en el protocolo. V4 exigía que f₁ y f₂ no disminuyeran al aumentar la demanda un 10 %, y esa propiedad no se cumple con recurso de desvío al depósito. Con más demanda, el punto en que se agota la carga puede pasar a una farmacia más próxima al depósito, cuya ida y vuelta es más corta. En el caso observado, el desvío pasó de la farmacia 5 a la 6, su coste bajó de 1.739 a 1.559 s y f₁ bajó de 19.745 a 19.565 s. Es una propiedad conocida del VRP con demanda estocástica. V4 se sustituye por tres comprobaciones que sí son propiedades del modelo. Ningún criterio de decisión ha cambiado.

Cambios respecto a la 1.0, también anteriores a cualquier ejecución: criterio más estricto en la ejecución de referencia (extremo inferior del IC por encima del umbral), tolerancia numérica en V5, prueba V0 de reproducción del proceso verdadero, segundo tipo de evaluación fuera de muestra frente a la demanda verdadera y declaración del carácter convencional del recuento de réplicas.

Antonio González Cuevas.

## 1. Objetivo

Decidir, antes de modificar la memoria, si se sostienen tres afirmaciones:

- **A1.** El valor de la solución estocástica (VSS) es sólido en la tardanza clínica (f₂).
- **A2.** El VSS es modesto en el tiempo de ruta (f₁).
- **A3.** Este patrón no depende del entrenamiento de la predicción.

**Transparencia sobre lo ya conocido.** Antes de este protocolo se obtuvieron resultados exploratorios con el modelo de evaluación de la memoria, sobre la ejecución de referencia:

- VSS de plan único fuera de muestra: f₁ 4,4 % (N=25) y 2,5 % (N=50); f₂ 17,3 % y 26,4 %.
- Comparaciones por pares de Solomon 100 (Wilcoxon con corrección de Holm).

Esos resultados motivan este protocolo, pero no sirven para decidir. Las decisiones se toman solo con las pruebas de los apartados 4 a 6, que aún no se han ejecutado.

## 2. Datos

- **Ejecución de referencia:**
  - N=25: s3 `20260615_200957`, s4 `20260615_201043_s4`;
  - N=50: s3 `20260615_205432`, s4 `20260615_205516_s4`.
- **Réplicas:** las 10 por tamaño de `data/replicas_vss/`, con el mismo código y la misma semilla.
- **Escenarios:**
  - dentro de la muestra, los K=20 escenarios SAA de cada ejecución;
  - **fuera de muestra (modelo):** 1.000 escenarios nuevos del mismo generador de escenarios de s3 (distribución de la predicción), con semilla 20260925. Detecta el optimismo de evaluar con la propia muestra;
  - **fuera de muestra (demanda verdadera):** 1.000 realizaciones de la demanda verdadera del día previsto (18/12/2024, índice 200 del generador sintético de s3), con semilla 20260926. Se muestrea del proceso que generó los datos: Binomial Negativa con φ = 4, media μᵢ·s_t·w_t (tendencia extrapolada a ese día y efecto de miércoles), independiente entre farmacias y acotada en 0,5 kg. Mide el rendimiento del pipeline completo, errores de predicción incluidos.
- No se modifica ningún fichero existente; los resultados se escriben en ficheros nuevos.

## 3. Planes, selección y medida

- **Plan:** una permutación con sus rutas cortadas por capacidad según su demanda de diseño. Los planes del frente EV usan la demanda Q50; los de cada frente SAA, la demanda de su escenario.
- **Lado determinista:** los planes del frente EV. **Lado estocástico:** los planes de los 20 frentes SAA.
- **Selección**, con los escenarios de la muestra SAA y la misma regla en ambos lados:
  - **(B) Mejor plan por objetivo:** el plan de menor coste esperado en ese objetivo.
  - **(A) Compromiso:** TOPSIS sobre los vectores de coste esperado no dominados. Como sensibilidad se aplican también otras tres reglas: x̄ de la memoria, suma ponderada normalizada y punto de codo.
- **Medida:** VSS_o = (E[f_o(plan determinista)] − E[f_o(plan estocástico)]) / E[f_o(plan determinista)] × 100, con intervalo de confianza al 95 % de la diferencia pareada por escenario (media ± 1,96·s/√K).

## 4. Evaluación de los planes

- **Modelo M (el de la memoria, §4.3.3):**
  - rutas fijas;
  - f₁ más 1.800 s por cada ruta desbordada;
  - f₂ fija;
  - f₃ escalada con la carga total del escenario.
- **Simulación S (nueva):** cada ruta se recorre en su orden.
  - El vehículo sale lleno (carga Q), como en el VRP con demanda estocástica clásico, porque la demanda de cada farmacia solo se conoce al llegar.
  - Si en una farmacia no le queda carga suficiente, entrega lo que lleva, vuelve al depósito, recarga y regresa a esa farmacia para completar la entrega, tantas veces como haga falta.
  - Los tiempos de llegada de las farmacias siguientes se desplazan en consecuencia.
  - La tardanza se mide al terminar el servicio.
  - f₁ es la duración total de las rutas, con los desvíos. f₃ = β·Σ_r T_r·(L_r/Q), con L_r la carga entregada en la ruta en ese escenario, como en la ecuación (4.3).

La simulación S es la evaluación principal. El modelo M se informa al lado para comparar con la memoria.

## 5. Validación de la simulación

La simulación solo se usa si supera las cinco pruebas siguientes. El script se detiene si alguna falla.

| Prueba | Condición |
|---|---|
| V0. Proceso verdadero | El generador sintético reconstruido reproduce exactamente el histórico de la ejecución (`demand_synthetic.csv`), valor a valor |
| V1. Equivalencia en el escenario de diseño | Cada plan evaluado con su propia demanda de diseño reproduce su f₁, f₂ y f₃ del frente (tolerancia relativa 10⁻⁶) |
| V2. Equivalencia sin desbordamiento | En los escenarios donde ninguna ruta se desborda, f₁ y f₂ coinciden con el modelo M |
| V3. Caso calculado a mano | Una ruta de tres farmacias que se desborda en la segunda: tiempos, tardanza y f₁ coinciden con el cálculo manual documentado en el propio test |
| V4. Coherencia | (a) El número de desvíos de cada ruta es igual a ceil(L_r/Q) − 1. (b) Al multiplicar la demanda por 1,1, el número de desvíos no disminuye en ninguno de 200 pares plan-escenario al azar. (c) En cada escenario, f₁ simulado es igual al tiempo de las rutas sin desvíos más la suma de las idas y vueltas al depósito en los puntos de desbordamiento, calculados de forma independiente a partir de la demanda acumulada (tolerancia absoluta 10⁻⁶ s) |
| V5. Contraste independiente | El VSS de f₁ de la memoria calculado con el f₁ simulado de x̄ difiere como máximo en 0,05 puntos del obtenido por otra vía con la variante «desvío real 2·t_i0» (6,12 % en N=25). Si la matriz de tiempos no es simétrica, se informa de su asimetría máxima |

## 6. Criterios de decisión

Los estimandos principales se miden con la simulación S, en los dos tipos de evaluación fuera de muestra (modelo y demanda verdadera), para N=25 y N=50:

- **E1:** VSS de f₂ con la regla (B).
- **E2:** VSS de f₂ con la regla (A).
- **E3:** VSS de f₁ con la regla (B).
- **E4:** VSS de f₁ con la regla (A).

| Afirmación | Se confirma si… | Se refuta si… | En otro caso |
|---|---|---|---|
| **A1. Tardanza sólida** | En la ejecución de referencia, el extremo inferior del IC 95 % de E1 y de E2 es > 10 % (umbral de H2), en ambos tamaños y en los dos tipos de evaluación, **y** E1 > 10 % en al menos 8 de las 10 réplicas de cada tamaño, también en los dos tipos | La estimación puntual de E1 es < 10 % en la ejecución de referencia de algún tamaño o tipo, o E1 > 10 % en menos de 6 de las 10 réplicas | No concluyente, por ejemplo si la estimación puntual supera el 10 % pero el extremo inferior del IC no |
| **A2. Tiempo modesto** | La mediana de E3 entre réplicas es < 5 % y el extremo superior del IC 95 % de E3 en la ejecución de referencia es < 5 %, en ambos tamaños y en los dos tipos | La mediana de E3 es ≥ 5 % en ambos tamaños y en los dos tipos (H1 se reforzaría) | No concluyente |
| **A3. Independencia del entrenamiento** | A1 y A2 se confirman con el criterio de réplicas indicado | — | Se informa del número de réplicas que cumplen |

E4 y las reglas alternativas de compromiso se informan como descripción, sin valor de decisión: la proporción de réplicas con E4 ≤ 0 y el rango de E4 según la regla.

## 7. Réplicas

- Cada réplica se procesa con el mismo código y el mismo protocolo que la ejecución de referencia.
- Cada réplica debe reproducir exactamente su frente EV y sus frentes SAA. Si una no lo hace, se excluye y se informa.
- Se informa de la mediana, el rango y el número de réplicas que cumplen cada criterio, sin seleccionar ninguna.
- Los recuentos de 8 y 6 réplicas son una convención, no un contraste estadístico: con 10 réplicas, 8 de 10 es compatible con proporciones verdaderas de entre el 44 y el 97,5 % (intervalo de Clopper-Pearson al 95 %). Describen la robustez frente al entrenamiento; la inferencia principal descansa en la ejecución de referencia y sus intervalos.

## 8. Reproducibilidad

- Todas las semillas están fijadas.
- El análisis se ejecuta dos veces aquí; los resultados deben ser idénticos byte a byte.
- Después se ejecuta en h1, y los resultados deben coincidir con los de aquí. Solo los resultados de h1 entran en el depósito.
- **Salidas:**
  - `robustez_vss.json` y `robustez_vss_oos.csv` en cada carpeta s4 de referencia;
  - `robustez_replicas.json` en `data/replicas_vss/`;
  - el registro de las pruebas V1–V5.

## 9. Consecuencias para la memoria

| Resultado | Cambio en la memoria |
|---|---|
| A1 y A2 confirmadas | H1 pasa a «parcialmente verificada» y H2 queda reforzada. Resumen y conclusiones presentan la tardanza como beneficio principal y el tiempo como beneficio modesto. Nueva sección de robustez en §4.4 con sus tablas |
| A1 confirmada, A2 no concluyente | H2 queda reforzada; H1 se mantiene como «verificada en la ejecución de referencia, no robusta entre réplicas» y se añade el rango de E3 |
| A2 refutada | H1 queda reforzada con el VSS de plan único |
| A1 refutada o no concluyente | No se cambia el mensaje principal; se informa del resultado y H2 se matiza |

En todos los casos se añade a §4.3.3 la cota del estimador de la memoria (VSS ≤ EEV − RP_ideal ≤ VSS + EVPI). Los resultados de Wilcoxon con Holm se incorporan a H3 y a §4.2.1.

## 10. Lo que este protocolo no decide

- La demanda es sintética.
- El recurso es un modelo, aunque más realista en la simulación S.
- La matriz de tiempos de s4 es una aproximación calibrada.

Estas cuestiones son limitaciones que se declaran en la memoria, no dudas que se resuelvan con pruebas.
