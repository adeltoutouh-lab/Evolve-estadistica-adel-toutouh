"""PRÁCTICA FINAL — EJERCICIO 3.

Regresión lineal múltiple implementada desde cero con NumPy.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def regresion_lineal_multiple(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta una regresión lineal múltiple usando la solución OLS.

    Parameters
    ----------
    X_train : np.ndarray
        Matriz de entrenamiento sin columna de unos.
    y_train : np.ndarray
        Vector objetivo de entrenamiento.
    X_test : np.ndarray
        Matriz de prueba sin columna de unos.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Coeficientes ajustados y predicciones sobre el test set.
    """
    X_train_b = np.column_stack([np.ones(X_train.shape[0]), X_train])
    xtx = X_train_b.T @ X_train_b
    xty = X_train_b.T @ y_train
    coefs = np.linalg.solve(xtx, xty)

    X_test_b = np.column_stack([np.ones(X_test.shape[0]), X_test])
    y_pred = X_test_b @ coefs
    return coefs, y_pred


def calcular_mae(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el error absoluto medio.

    Parameters
    ----------
    y_real : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.

    Returns
    -------
    float
        Valor del MAE.
    """
    return float(np.mean(np.abs(y_real - y_pred)))


def calcular_rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la raíz del error cuadrático medio.

    Parameters
    ----------
    y_real : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.

    Returns
    -------
    float
        Valor del RMSE.
    """
    return float(np.sqrt(np.mean((y_real - y_pred) ** 2)))


def calcular_r2(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el coeficiente de determinación R².

    Parameters
    ----------
    y_real : np.ndarray
        Valores reales.
    y_pred : np.ndarray
        Valores predichos.

    Returns
    -------
    float
        Valor de R².
    """
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - (ss_res / ss_tot))


def graficar_real_vs_predicho(y_real: np.ndarray, y_pred: np.ndarray, ruta_salida: Path = OUTPUT_DIR / "ej3_predicciones.png") -> None:
    """Genera el gráfico de valores reales frente a predichos.

    Parameters
    ----------
    y_real : np.ndarray
        Valores reales del conjunto de test.
    y_pred : np.ndarray
        Valores predichos por el modelo.
    ruta_salida : Path, optional
        Ruta donde se guardará la imagen.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_real, y_pred, alpha=0.7)

    minimo = min(y_real.min(), y_pred.min())
    maximo = max(y_real.max(), y_pred.max())
    plt.plot([minimo, maximo], [minimo, maximo], linestyle="--")

    plt.title("Valores reales vs. valores predichos")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=180, bbox_inches="tight")
    plt.close()


def dividir_train_test(X: np.ndarray, y: np.ndarray, semilla: int = 42, proporcion_train: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide los datos en train y test de forma reproducible.

    Parameters
    ----------
    X : np.ndarray
        Matriz de características.
    y : np.ndarray
        Vector objetivo.
    semilla : int, optional
        Semilla usada en la partición.
    proporcion_train : float, optional
        Proporción destinada a entrenamiento.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test.
    """
    test_size = 1 - proporcion_train
    return train_test_split(X, y, test_size=test_size, random_state=semilla)


if __name__ == "__main__":
    SEMILLA = 42
    np.random.seed(SEMILLA)

    n_muestras = 200
    n_features = 3

    X = np.random.randn(n_muestras, n_features)
    coefs_reales = np.array([5.0, 2.0, -1.0, 0.5])
    ruido = np.random.normal(0, 1.5, n_muestras)
    y = coefs_reales[0] + X @ coefs_reales[1:] + ruido

    X_train, X_test, y_train, y_test = dividir_train_test(X, y, semilla=SEMILLA, proporcion_train=0.8)

    coefs, y_pred = regresion_lineal_multiple(X_train, y_train, X_test)

    mae = calcular_mae(y_test, y_pred)
    rmse = calcular_rmse(y_test, y_pred)
    r2 = calcular_r2(y_test, y_pred)

    print("=" * 50)
    print("RESULTADOS — Regresión Lineal Múltiple (NumPy)")
    print("=" * 50)
    print(f"\nCoeficientes reales:   {coefs_reales}")
    print(f"Coeficientes ajustados: {coefs}")
    print("\nMétricas sobre test set:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  R²   = {r2:.4f}")

    with open(OUTPUT_DIR / "ej3_coeficientes.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Coeficientes ajustados\n")
        f.write("=" * 50 + "\n")
        nombres = ["Intercepto (β0)"] + [f"β{i+1} (feature {i+1})" for i in range(n_features)]
        for nombre, valor in zip(nombres, coefs):
            f.write(f"{nombre}: {valor:.6f}\n")
        f.write("\nCoeficientes reales de referencia\n")
        for nombre, valor in zip(nombres, coefs_reales):
            f.write(f"{nombre}: {valor:.6f}\n")

    with open(OUTPUT_DIR / "ej3_metricas.txt", "w", encoding="utf-8") as f:
        f.write("Regresión Lineal Múltiple — Métricas\n")
        f.write("=" * 50 + "\n")
        f.write(f"MAE  : {mae:.6f}\n")
        f.write(f"RMSE : {rmse:.6f}\n")
        f.write(f"R²   : {r2:.6f}\n")

    graficar_real_vs_predicho(y_test, y_pred)

    print("\nSalidas guardadas en output/")
    print("  → output/ej3_coeficientes.txt")
    print("  → output/ej3_metricas.txt")
    print("  → output/ej3_predicciones.png")
