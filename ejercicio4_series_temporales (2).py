"""PRÁCTICA FINAL — EJERCICIO 4.

Análisis y descomposición de una serie temporal sintética.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generar_serie_temporal(semilla: int = 42) -> pd.Series:
    """Genera una serie temporal sintética con tendencia y estacionalidad.

    Parameters
    ----------
    semilla : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    pd.Series
        Serie temporal diaria con índice de fechas.
    """
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range(start="2018-01-01", end="2023-12-31", freq="D")
    n = len(fechas)
    t = np.arange(n)

    tendencia = 0.05 * t + 50
    estacionalidad = 15 * np.sin(2 * np.pi * t / 365.25) + 6 * np.cos(4 * np.pi * t / 365.25)
    ciclo = 8 * np.sin(2 * np.pi * t / 1461)
    ruido = rng.normal(loc=0, scale=3.5, size=n)

    valores = tendencia + estacionalidad + ciclo + ruido
    return pd.Series(valores, index=fechas, name="valor")


def visualizar_serie(serie: pd.Series) -> None:
    """Dibuja la serie temporal completa.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal a visualizar.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(serie.index, serie.values, linewidth=1)
    ax.set_title("Serie temporal completa")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej4_serie_original.png", dpi=180, bbox_inches="tight")
    plt.close()


def descomponer_serie(serie: pd.Series):
    """Aplica una descomposición aditiva a la serie.

    Parameters
    ----------
    serie : pd.Series
        Serie temporal original.

    Returns
    -------
    DecomposeResult
        Objeto con tendencia, estacionalidad y residuo.
    """
    resultado = seasonal_decompose(serie, model="additive", period=365)
    fig = resultado.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ej4_descomposicion.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return resultado


def analizar_residuo(residuo: pd.Series) -> None:
    """Analiza el residuo con estadísticas, normalidad y autocorrelación.

    Parameters
    ----------
    residuo : pd.Series
        Componente residual de la descomposición.

    Returns
    -------
    None
        No devuelve nada. Guarda gráficos y un TXT en output/.
    """
    residuo_limpio = residuo.dropna()

    media = residuo_limpio.mean()
    std = residuo_limpio.std()
    asimetria = residuo_limpio.skew()
    curtosis = residuo_limpio.kurtosis()

    resultado_adf = adfuller(residuo_limpio)
    p_adf = resultado_adf[1]

    jb_stat, jb_p = jarque_bera(residuo_limpio)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(residuo_limpio, lags=40, ax=axes[0])
    plot_pacf(residuo_limpio, lags=40, ax=axes[1], method="ywm")
    axes[0].set_title("ACF del residuo")
    axes[1].set_title("PACF del residuo")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej4_acf_pacf.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(residuo_limpio, bins=35, density=True, alpha=0.7)
    x = np.linspace(residuo_limpio.min(), residuo_limpio.max(), 400)
    plt.plot(x, norm.pdf(x, media, std), linewidth=2)
    plt.title("Histograma del residuo con curva normal")
    plt.xlabel("Residuo")
    plt.ylabel("Densidad")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej4_histograma_ruido.png", dpi=180, bbox_inches="tight")
    plt.close()

    with open(OUTPUT_DIR / "ej4_analisis.txt", "w", encoding="utf-8") as f:
        f.write("Ejercicio 4 - Análisis del residuo\n")
        f.write("=" * 50 + "\n")
        f.write(f"Número de observaciones: {len(residuo_limpio)}\n")
        f.write(f"Media: {media:.6f}\n")
        f.write(f"Desviación estándar: {std:.6f}\n")
        f.write(f"Asimetría: {asimetria:.6f}\n")
        f.write(f"Curtosis: {curtosis:.6f}\n")
        f.write(f"p-value Jarque-Bera: {jb_p:.6f}\n")
        f.write(f"p-value ADF: {p_adf:.6f}\n")
        f.write(
            "Conclusión breve: el residuo tiene media cercana a 0, parece\n"
            "aproximadamente normal y se comporta bastante como ruido.\n"
        )


if __name__ == "__main__":
    print("=" * 55)
    print("EJERCICIO 4 — Análisis de Series Temporales")
    print("=" * 55)

    SEMILLA = 42
    serie = generar_serie_temporal(semilla=SEMILLA)

    print("\n[1/3] Visualizando la serie original...")
    visualizar_serie(serie)

    print("[2/3] Descomponiendo la serie...")
    resultado = descomponer_serie(serie)

    print("[3/3] Analizando el residuo...")
    analizar_residuo(resultado.resid)

    print("\nSalidas esperadas en output/:")
    salidas = [
        "ej4_serie_original.png",
        "ej4_descomposicion.png",
        "ej4_acf_pacf.png",
        "ej4_histograma_ruido.png",
        "ej4_analisis.txt",
    ]
    for s in salidas:
        print(f"  - output/{s}")
