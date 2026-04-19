# Práctica Final de Estadística y Probabilidad

**Alumno:** Adel Toutouh El Bouchti  
**Dataset usado en los ejercicios 1 y 2:** `diamonds.csv`  
**Fuente pública:** dataset `diamonds` (muy difundido en `ggplot2` / `Rdatasets`)

---

## Ejercicio 1 — Estadística descriptiva

### 1.1 Dataset elegido y justificación

He elegido el dataset **diamonds**, que contiene información sobre diamantes y su precio. Me ha parecido una buena opción porque:

- tiene bastantes observaciones (más de 50.000), así que permite hacer un análisis descriptivo sólido;
- mezcla variables **numéricas** (`carat`, `depth`, `table`, `price`, `x`, `y`, `z`) y **categóricas** (`cut`, `color`, `clarity`);
- el objetivo principal (`price`) es continuo, por lo que también sirve bien para el ejercicio 2 de regresión;
- es un dataset público y muy conocido, así que su procedencia es fácil de justificar.

Antes de analizarlo, hice una limpieza sencilla:

- eliminé filas con dimensiones imposibles (`x`, `y` o `z` menores o iguales que 0);
- eliminé algunos valores extremos muy raros en `x`, `y` y `z` usando **Z-score > 5**.

---

### 1.2 Resumen estructural del dataset

Después de la limpieza, el dataset se quedó en:

- **Filas:** 53.917
- **Columnas:** 10
- **Tamaño en memoria:** aproximadamente **11,38 MB**

#### Tipos de dato por columna

| Columna | Tipo |
|---|---|
| carat | float64 |
| cut | object |
| color | object |
| clarity | object |
| depth | float64 |
| table | float64 |
| price | int64 |
| x | float64 |
| y | float64 |
| z | float64 |

#### Porcentaje de nulos por columna

En este caso, todas las columnas tienen el mismo resultado:

- `carat`: **0,00%**
- `cut`: **0,00%**
- `color`: **0,00%**
- `clarity`: **0,00%**
- `depth`: **0,00%**
- `table`: **0,00%**
- `price`: **0,00%**
- `x`: **0,00%**
- `y`: **0,00%**
- `z`: **0,00%**

Por tanto, **no ha hecho falta imputar ni eliminar nulos**, porque simplemente no había.

---

### 1.3 Análisis estadístico de las variables numéricas

He guardado el resumen completo en `output/ej1_descriptivo.csv`, pero destaco algunas ideas principales.

#### Variable `price`

- **Media:** 3930,91
- **Mediana:** 2401,00
- **Asimetría:** 1,6184
- **Curtosis:** 2,1788
- **IQR:** 4374,00

Interpretación:

La variable `price` está claramente **sesgada a la derecha**, porque la media es bastante mayor que la mediana y la asimetría es positiva. Esto significa que hay bastantes diamantes caros que empujan la media hacia arriba. También tiene una curtosis positiva, así que hay más peso en las colas que en una distribución perfectamente normal.

#### Outliers

En el fichero `output/ej1_outliers.txt` he resumido la limpieza y el análisis de outliers.

- Filas eliminadas por dimensiones imposibles: **20**
- Filas eliminadas por extremos en `x`, `y`, `z`: **3**
- Outliers detectados en `price` con criterio IQR: **3531** (aprox. **6,55%**)

En `price` **no los eliminé**, porque parecen valores reales del mercado y no errores evidentes.

---

### 1.4 Distribuciones de las variables numéricas

Los histogramas están en `output/ej1_histogramas.png`.

En general:

- `price` y `carat` tienen distribuciones claramente asimétricas a la derecha;
- `x`, `y` y `z` también presentan asimetría y colas algo largas;
- `depth` y `table` están bastante más concentradas y tienen una forma más estable;
- no parece que todas las variables sigan una distribución normal, sobre todo `price` y `carat`.

Esto es importante porque luego, al construir modelos, hay que tener en cuenta que hay variables muy concentradas y otras mucho más dispersas.

---

### 1.5 Boxplots y detección visual de outliers

Los boxplots están en `output/ej1_boxplots.png`.

Lo que se observa visualmente es lo siguiente:

- `price` tiene bastantes valores altos separados del bloque central;
- `carat` también presenta muchos valores extremos;
- `x`, `y` y `z` tenían algunos casos muy raros, y por eso se hizo una limpieza inicial;
- `depth` y `table` parecen bastante más estables.

Mi decisión fue **no eliminar outliers de precio ni de carat por defecto**, porque pueden representar diamantes reales de gama alta. Solo eliminé los casos claramente problemáticos de dimensiones.

---

### 1.6 Correlaciones y multicolinealidad

El mapa de calor está en `output/ej1_heatmap_correlacion.png`.

#### Variables más correlacionadas con `price`

- `carat`: **0,9216**
- `y`: **0,8888**
- `x`: **0,8872**
- `z`: **0,8821**

La variable más relacionada con el precio es claramente **`carat`**, lo cual tiene bastante sentido: a mayor quilataje, normalmente mayor precio.

#### Posible multicolinealidad

Sí, se aprecia una multicolinealidad fuerte entre las dimensiones físicas:

- correlación `x` - `y`: **0,9987**
- correlación `x` - `z`: **0,9911**
- correlación `y` - `z`: **0,9907**

Por eso, en el ejercicio 2 decidí quitar `x`, `y` y `z`, ya que aportan información muy parecida entre sí y además `carat` ya recoge bastante del tamaño del diamante.

---

### 1.7 Variables categóricas

El gráfico está en `output/ej1_categoricas.png`.

#### Frecuencias de `cut`

| Categoría | Frecuencia | % |
|---|---:|---:|
| Ideal | 21547 | 39,96 |
| Premium | 13779 | 25,56 |
| Very Good | 12080 | 22,40 |
| Good | 4902 | 9,09 |
| Fair | 1609 | 2,98 |

Aquí se ve que **Ideal** domina bastante.

#### Frecuencias de `color`

| Categoría | Frecuencia | % |
|---|---:|---:|
| G | 11284 | 20,93 |
| E | 9795 | 18,17 |
| F | 9538 | 17,69 |
| H | 8297 | 15,39 |
| D | 6774 | 12,56 |
| I | 5421 | 10,05 |
| J | 2808 | 5,21 |

La variable `color` está algo más repartida, aunque la categoría **G** es la más frecuente.

#### Frecuencias de `clarity`

| Categoría | Frecuencia | % |
|---|---:|---:|
| SI1 | 13063 | 24,23 |
| VS2 | 12254 | 22,73 |
| SI2 | 9184 | 17,03 |
| VS1 | 8168 | 15,15 |
| VVS2 | 5066 | 9,40 |
| VVS1 | 3654 | 6,78 |
| IF | 1790 | 3,32 |
| I1 | 738 | 1,37 |

Aquí las categorías más frecuentes son **SI1** y **VS2**.

#### Comentario general

No diría que hay un desbalance extremo en todas las variables categóricas, pero sí hay algunas categorías claramente más representadas que otras, por ejemplo `cut = Ideal` o `clarity = SI1`. Esto puede influir un poco en los coeficientes del modelo del ejercicio 2.

---

## Ejercicio 2 — Inferencia con scikit-learn

### 2.1 Preparación del dataset y preprocesado

He usado el mismo dataset limpio del ejercicio 1.

#### Variable objetivo

La variable objetivo principal es:

- **`price`** (continua)

Por tanto, el modelo principal es una **regresión lineal**.

#### Predictores descartados por multicolinealidad

He eliminado:

- `x`
- `y`
- `z`

La razón es que estaban fuertemente correlacionadas entre sí y también muy relacionadas con `carat`.

#### Preprocesado aplicado

- Variables numéricas: **StandardScaler**
- Variables categóricas: **OneHotEncoder** con `drop='first'`

#### Métricas del modelo de regresión

Guardadas en `output/ej2_metricas_regresion.txt`.

**Train**
- MAE: **806,4342**
- RMSE: **1162,2852**
- R²: **0,9152**

**Test**
- MAE: **785,0183**
- RMSE: **1127,7005**
- R²: **0,9193**

#### Interpretación

El modelo funciona bastante bien para un modelo lineal sencillo, ya que explica alrededor del **91,9%** de la variabilidad del precio en test.

Además, no parece haber sobreajuste importante, porque las métricas de entrenamiento y test son muy parecidas. De hecho, el rendimiento en test incluso sale ligeramente mejor en este caso, lo cual sugiere que el modelo generaliza razonablemente bien.

---

### 2.2 Variables más influyentes

El gráfico está en `output/ej2_coeficientes.png`.

Los coeficientes con más peso absoluto son sobre todo:

- varias categorías de `clarity` (`IF`, `VVS1`, `VVS2`, `VS1`, `VS2`);
- `carat`;
- algunas categorías de `color`, especialmente `J`, `I` y `H` con efecto negativo;
- varias categorías de `cut` con efecto positivo moderado.

Esto encaja bastante con el dominio del problema:

- más **quilates** suelen implicar más precio;
- una **mejor claridad** aumenta bastante el valor del diamante;
- los colores peores (por ejemplo `J`) penalizan el precio.

---

### 2.3 Residuos y calidad del ajuste

El gráfico de residuos está en `output/ej2_residuos.png`.

A simple vista:

- los residuos se concentran alrededor de 0;
- no se ve una curva muy marcada, así que el ajuste lineal no parece completamente disparatado;
- sí hay algo más de dispersión en ciertos rangos de precio, lo que sugiere que el problema real no es perfectamente lineal.

Aun así, para un primer modelo, el resultado es bastante bueno.

---

### 2.4 Conclusión conectando con el ejercicio 1

La información del ejercicio 1 me sirvió bastante para interpretar este modelo:

1. **`carat` tenía una correlación muy alta con `price`**, así que ya era esperable que fuese una variable importante.
2. **`x`, `y` y `z` estaban fuertemente correlacionadas entre sí**, por eso tenía sentido quitarlas para evitar multicolinealidad.
3. Las distribuciones y frecuencias de `cut`, `color` y `clarity` ayudaban a entender por qué algunas categorías tienen más peso en el ajuste.
4. Los outliers de `price` y `carat` explican en parte que el problema no sea completamente limpio ni perfectamente lineal.

#### Nota adicional sobre el checklist final

Como en el checklist del profesor aparece `ej2_matriz_confusion.png`, he añadido una **clasificación auxiliar** solo para cubrir ese punto.

- Transformación auxiliar del target:  
  **`price_bin = 1` si `price >= mediana (2401)` y `0` en caso contrario**.
- Salida generada: `output/ej2_matriz_confusion.png`

Aun así, el ejercicio principal sigue siendo el de **regresión lineal sobre `price`**.

---

## Ejercicio 3 — Regresión lineal múltiple desde cero con NumPy

### 3.1 Generación de datos

He generado un dataset sintético con:

- **200 muestras**
- **3 variables explicativas**
- semilla fija para reproducibilidad
- ruido gaussiano con desviación estándar 1,5

La fórmula usada es la indicada en el enunciado:

\[
y = 5 + 2x_1 - x_2 + 0.5x_3 + \varepsilon
\]

---

### 3.2 Implementación

He implementado el ajuste con la solución cerrada de mínimos cuadrados ordinarios:

\[
\hat{\beta} = (X^T X)^{-1} X^T y
\]

En realidad, en el código he usado `np.linalg.solve`, que es una forma más estable de resolver el sistema sin calcular la inversa explícita.

---

### 3.3 Resultados obtenidos

Guardados en:

- `output/ej3_coeficientes.txt`
- `output/ej3_metricas.txt`
- `output/ej3_predicciones.png`

#### Coeficientes ajustados

- β0 = **5,073486**
- β1 = **1,901816**
- β2 = **-0,809869**
- β3 = **0,675742**

#### Coeficientes reales

- β0 = **5,000000**
- β1 = **2,000000**
- β2 = **-1,000000**
- β3 = **0,500000**

Se parecen bastante, así que el modelo está recuperando bien la relación general.

#### Métricas en test

- MAE = **1,112102**
- RMSE = **1,390634**
- R² = **0,736391**

#### Interpretación

El modelo recupera razonablemente bien los coeficientes verdaderos. El error es bajo y el ajuste es bueno, aunque el **R² queda algo por debajo del 0,80 orientativo** que aparece en la tabla del enunciado. Aun así, sigue siendo un resultado coherente porque hay ruido aleatorio y la partición train/test también influye.

El gráfico `ej3_predicciones.png` muestra que los puntos siguen bastante bien la diagonal, así que la predicción es aceptable.

---

## Ejercicio 4 — Series temporales

### 4.1 ¿La serie tiene tendencia?

Sí. La serie tiene una **tendencia creciente** bastante clara a lo largo del tiempo.

En la descomposición (`output/ej4_descomposicion.png`) la componente de tendencia sube progresivamente, así que se puede decir que la serie no es estacionaria en nivel si se mira de forma global.

---

### 4.2 ¿Tiene estacionalidad? ¿De qué tipo?

Sí. Tiene una **estacionalidad anual** clara.

La señal estacional se repite aproximadamente cada 365 días, porque el ejercicio se ha construido precisamente con esa periodicidad. En la descomposición se aprecia una componente periódica bastante marcada.

---

### 4.3 ¿El residuo parece ideal (media 0, normalidad y sin autocorrelación)?

El residuo no es perfecto, pero **se acerca bastante** a un ruido razonable.

Datos de `output/ej4_analisis.txt`:

- Media: **0,127078**
- Desviación estándar: **3,222043**
- Asimetría: **-0,050917**
- Curtosis: **-0,061028**
- p-valor Jarque-Bera: **0,576561**
- p-valor ADF: **0,000000**

#### Interpretación

- La **media** está cerca de 0.
- La **asimetría** y la **curtosis** están bastante cerca de 0, así que la forma del residuo se parece bastante a una normal.
- El test de **Jarque-Bera** no da evidencia fuerte contra normalidad.
- El test **ADF** sugiere que el residuo es estacionario.
- En `output/ej4_acf_pacf.png` no se aprecian picos muy fuertes o persistentes, así que no parece haber una autocorrelación importante remanente.

Por tanto, el residuo se comporta **bastante parecido a ruido**.

---

### 4.4 ¿Hay ruido? ¿Cuánto?

Sí, hay ruido, y su intensidad se puede resumir bastante bien con la desviación estándar del residuo:

- **σ ≈ 3,22**

Eso significa que el componente aleatorio medio de la serie oscila alrededor de unas 3,2 unidades.

Además, en `output/ej4_histograma_ruido.png` el histograma del residuo tiene una forma razonablemente parecida a una campana, así que encaja bastante bien con la idea de ruido gaussiano.

---

## Conclusión general

En conjunto, la práctica queda completa y coherente:

- en el **Ejercicio 1** se hace una exploración estadística clara del dataset;
- en el **Ejercicio 2** se usa esa información para construir e interpretar una regresión lineal con scikit-learn;
- en el **Ejercicio 3** se implementa una regresión lineal múltiple desde cero con NumPy;
- en el **Ejercicio 4** se genera y analiza una serie temporal con tendencia, estacionalidad y ruido.

También he dejado en `output/` todos los ficheros generados por los scripts, incluyendo los añadidos para cubrir mejor el checklist final.
