# Práctica Final de Estadística y Probabilidad

Este repositorio contiene mi práctica final de la asignatura de **Estadística y Probabilidad**, desarrollada con Python.

El objetivo principal del proyecto es aplicar distintos conceptos estadísticos y de análisis de datos mediante ejercicios prácticos, trabajando tanto con un dataset real como con datos sintéticos generados para el análisis.

## Autor

**Adel Toutouh El Bouchti**

## Descripción del proyecto

La práctica está dividida en cuatro ejercicios principales:

1. **Estadística descriptiva**
2. **Inferencia y regresión con scikit-learn**
3. **Regresión lineal múltiple desde cero con NumPy**
4. **Análisis de series temporales**

En los dos primeros ejercicios se utiliza el dataset `diamonds.csv`, un conjunto de datos público muy conocido que contiene información sobre diamantes, sus características y su precio.

En los ejercicios 3 y 4 se trabajan datos sintéticos para implementar una regresión múltiple desde cero y analizar una serie temporal con tendencia, estacionalidad y ruido.

## Estructura del repositorio

```text
practica_final_-Toutouh_El_Bouchti_Adel-/
│
├── data/
│   └── diamonds.csv
│
├── output/
│   ├── gráficos generados
│   ├── métricas
│   └── ficheros de resultados
│
├── ejercicio1_descriptivo.py
├── ejercicio2_inferencia.py
├── ejercicio3_regresion_multiple.py
├── ejercicio4_series_temporales.py
├── Respuestas.md
└── README.md
```

## Ejercicio 1: Estadística descriptiva

En este ejercicio se realiza un análisis descriptivo del dataset `diamonds.csv`.

Se trabajan aspectos como:

* carga y limpieza del dataset;
* revisión de tipos de datos;
* análisis de valores nulos;
* cálculo de medidas estadísticas;
* detección de outliers;
* histogramas;
* boxplots;
* análisis de correlaciones;
* estudio de variables categóricas.

También se genera una carpeta `output/` con los resultados y gráficos obtenidos durante el análisis.

## Ejercicio 2: Inferencia y regresión con scikit-learn

En este ejercicio se utiliza el mismo dataset limpio del ejercicio anterior para construir un modelo de regresión lineal.

El objetivo es predecir el precio de los diamantes a partir de diferentes variables explicativas.

Se aplican técnicas como:

* separación entre variables predictoras y variable objetivo;
* preprocesado de variables numéricas y categóricas;
* escalado de variables numéricas;
* codificación One-Hot para variables categóricas;
* entrenamiento de un modelo de regresión lineal;
* evaluación mediante MAE, RMSE y R²;
* análisis de residuos;
* interpretación de coeficientes.

También se incluye una clasificación auxiliar para generar una matriz de confusión, tal como se solicita en el checklist de la práctica.

## Ejercicio 3: Regresión lineal múltiple desde cero

En este ejercicio se implementa una regresión lineal múltiple usando NumPy, sin utilizar directamente un modelo de scikit-learn para el ajuste principal.

Se genera un dataset sintético con tres variables explicativas y una variable objetivo.

El modelo se ajusta mediante la solución de mínimos cuadrados ordinarios:

```text
β = (XᵀX)^(-1)Xᵀy
```

En el código se utiliza `np.linalg.solve` para resolver el sistema de forma más estable.

Se calculan métricas como:

* MAE;
* RMSE;
* R².

Además, se comparan los coeficientes reales con los coeficientes estimados por el modelo.

## Ejercicio 4: Series temporales

En este ejercicio se genera y analiza una serie temporal sintética.

La serie incluye:

* tendencia;
* estacionalidad anual;
* componente cíclica;
* ruido aleatorio.

El análisis incluye:

* descomposición de la serie temporal;
* estudio de la tendencia;
* análisis de la estacionalidad;
* análisis del residuo;
* test de normalidad Jarque-Bera;
* test de estacionariedad ADF;
* gráficos ACF y PACF;
* histograma del ruido.

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* scikit-learn
* Statsmodels

## Cómo ejecutar el proyecto

Primero, clonar el repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
```

Entrar en la carpeta del proyecto:

```bash
cd practica_final_-Toutouh_El_Bouchti_Adel-
```

Instalar las librerías necesarias:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

Ejecutar los scripts:

```bash
python ejercicio1_descriptivo.py
python ejercicio2_inferencia.py
python ejercicio3_regresion_multiple.py
python ejercicio4_series_temporales.py
```

Los resultados se guardarán en la carpeta `output/`.

## Resultados principales

Algunas conclusiones generales del proyecto son:

* El dataset `diamonds.csv` permite trabajar tanto variables numéricas como categóricas.
* La variable `price` presenta una distribución asimétrica a la derecha, con presencia de valores altos.
* La variable `carat` está muy relacionada con el precio del diamante.
* Las dimensiones físicas `x`, `y` y `z` presentan una fuerte multicolinealidad.
* El modelo de regresión lineal consigue un buen rendimiento para una primera aproximación.
* La regresión múltiple implementada desde cero permite entender mejor el funcionamiento interno de los mínimos cuadrados.
* La serie temporal generada muestra tendencia, estacionalidad y ruido, lo que permite practicar técnicas básicas de análisis temporal.

## Archivo de respuestas

El archivo `Respuestas.md` contiene la explicación desarrollada de la práctica, incluyendo interpretaciones, resultados y conclusiones de cada ejercicio.

## Conclusión

Esta práctica me ha servido para reforzar conceptos importantes de estadística, probabilidad, regresión y análisis de series temporales.

Además, me ha permitido seguir practicando el uso de Python, la organización de proyectos y la documentación de resultados en GitHub.
