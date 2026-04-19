"""Ejercicio 2 - Inferencia con scikit-learn.

La parte principal del ejercicio es una regresión lineal para predecir el
precio. Como el checklist final menciona una matriz de confusión, también se
incluye una pequeña clasificación auxiliar binarizando el target respecto a la
mediana del precio, solo para cubrir ese punto del checklist.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diamonds.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def cargar_y_limpiar_datos(ruta_csv: Path) -> pd.DataFrame:
    """Carga el dataset y elimina errores evidentes.

    Parameters
    ----------
    ruta_csv : Path
        Ruta al fichero CSV del dataset.

    Returns
    -------
    pd.DataFrame
        Dataset limpio listo para modelado.
    """
    df = pd.read_csv(ruta_csv)
    df = df[(df[["x", "y", "z"]] > 0).all(axis=1)].copy()
    zscores = np.abs(zscore(df[["x", "y", "z"]]))
    df = df[(zscores <= 5).all(axis=1)].copy()
    return df


def construir_preprocesado(X: pd.DataFrame) -> ColumnTransformer:
    """Construye el transformador para variables numéricas y categóricas.

    Parameters
    ----------
    X : pd.DataFrame
        Matriz de predictores.

    Returns
    -------
    ColumnTransformer
        Transformador de escalado y codificación.
    """
    categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numericas = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocesado = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricas),
        ]
    )
    return preprocesado


def entrenar_regresion_lineal(df: pd.DataFrame):
    """Entrena la regresión lineal y calcula métricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    tuple
        Modelo entrenado, y_test, predicciones de test, métricas, columnas
        eliminadas y el split usado.
    """
    columnas_quitadas = ["x", "y", "z"]
    X = df.drop(columns=["price"] + columnas_quitadas)
    y = df["price"]

    preprocesado = construir_preprocesado(X)
    modelo = Pipeline(
        steps=[
            ("preprocesado", preprocesado),
            ("regresion", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo.fit(X_train, y_train)
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    metricas = {
        "train": {
            "mae": mean_absolute_error(y_train, y_pred_train),
            "rmse": mean_squared_error(y_train, y_pred_train) ** 0.5,
            "r2": r2_score(y_train, y_pred_train),
        },
        "test": {
            "mae": mean_absolute_error(y_test, y_pred_test),
            "rmse": mean_squared_error(y_test, y_pred_test) ** 0.5,
            "r2": r2_score(y_test, y_pred_test),
        },
    }

    return modelo, y_test, y_pred_test, metricas, columnas_quitadas, (X_train, X_test)


def guardar_metricas(metricas: dict, columnas_quitadas: list[str]) -> None:
    """Guarda las métricas de regresión lineal en un TXT.

    Parameters
    ----------
    metricas : dict
        Diccionario con métricas de train y test.
    columnas_quitadas : list[str]
        Lista de columnas retiradas por multicolinealidad.

    Returns
    -------
    None
        No devuelve nada. Guarda un fichero de texto en output/.
    """
    with open(OUTPUT_DIR / "ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
        f.write("Ejercicio 2 - Métricas de la regresión lineal\n")
        f.write("=" * 50 + "\n")
        f.write(f"Columnas eliminadas por multicolinealidad: {', '.join(columnas_quitadas)}\n\n")
        f.write("Conjunto de entrenamiento\n")
        f.write(f"MAE  : {metricas['train']['mae']:.4f}\n")
        f.write(f"RMSE : {metricas['train']['rmse']:.4f}\n")
        f.write(f"R²   : {metricas['train']['r2']:.4f}\n\n")
        f.write("Conjunto de test\n")
        f.write(f"MAE  : {metricas['test']['mae']:.4f}\n")
        f.write(f"RMSE : {metricas['test']['rmse']:.4f}\n")
        f.write(f"R²   : {metricas['test']['r2']:.4f}\n")


def graficar_residuos(y_test: pd.Series, y_pred_test: np.ndarray) -> None:
    """Genera el gráfico de residuos frente a valores predichos.

    Parameters
    ----------
    y_test : pd.Series
        Valores reales del conjunto de test.
    y_pred_test : np.ndarray
        Predicciones del modelo sobre test.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    residuos = y_test - y_pred_test

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_test, residuos, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.title("Residuos frente a valores predichos")
    plt.xlabel("Precio predicho")
    plt.ylabel("Residuo")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej2_residuos.png", dpi=180, bbox_inches="tight")
    plt.close()


def graficar_coeficientes(modelo: Pipeline) -> None:
    """Dibuja los coeficientes más influyentes del modelo lineal.

    Parameters
    ----------
    modelo : Pipeline
        Pipeline ya entrenado con preprocesado y regresión lineal.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    nombres = modelo.named_steps["preprocesado"].get_feature_names_out()
    coefs = pd.Series(modelo.named_steps["regresion"].coef_, index=nombres)
    coefs = coefs.sort_values(key=lambda s: np.abs(s), ascending=False).head(12)
    coefs = coefs.sort_values()

    plt.figure(figsize=(10, 7))
    plt.barh(coefs.index, coefs.values)
    plt.title("Coeficientes más influyentes del modelo")
    plt.xlabel("Valor del coeficiente")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej2_coeficientes.png", dpi=180, bbox_inches="tight")
    plt.close()


def graficar_matriz_confusion_auxiliar(df: pd.DataFrame) -> None:
    """Genera una matriz de confusión auxiliar para cubrir el checklist.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    columnas_quitadas = ["x", "y", "z"]
    X = df.drop(columns=["price"] + columnas_quitadas)
    y_bin = (df["price"] >= df["price"].median()).astype(int)

    preprocesado = construir_preprocesado(X)
    modelo = Pipeline(
        steps=[
            ("preprocesado", preprocesado),
            ("logistica", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    modelo.fit(X_train, y_train)

    disp = ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test)
    disp.ax_.set_title("Matriz de confusión auxiliar (price >= mediana)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej2_matriz_confusion.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Ejecuta todo el flujo del ejercicio 2.

    Parameters
    ----------
    None
        Esta función no recibe argumentos.

    Returns
    -------
    None
        No devuelve nada. Genera ficheros de salida en output/.
    """
    df = cargar_y_limpiar_datos(DATA_PATH)
    modelo, y_test, y_pred_test, metricas, columnas_quitadas, _ = entrenar_regresion_lineal(df)
    guardar_metricas(metricas, columnas_quitadas)
    graficar_residuos(y_test, y_pred_test)
    graficar_coeficientes(modelo)
    graficar_matriz_confusion_auxiliar(df)

    print("Ejercicio 2 completado.")
    print(f"R² test: {metricas['test']['r2']:.4f}")
    print("Ficheros generados en output/:")
    print(" - ej2_metricas_regresion.txt")
    print(" - ej2_residuos.png")
    print(" - ej2_coeficientes.png")
    print(" - ej2_matriz_confusion.png")


if __name__ == "__main__":
    main()
