"""Ejercicio 1 - Análisis estadístico descriptivo.

Este script carga el dataset de diamantes, realiza una limpieza básica,
genera estadísticas descriptivas y crea las visualizaciones pedidas en el
enunciado. Además, guarda un pequeño informe de outliers eliminados para
cubrir el checklist final.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "diamonds.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def cargar_y_limpiar_datos(ruta_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Carga el dataset y elimina registros problemáticos.

    Parameters
    ----------
    ruta_csv : Path
        Ruta al fichero CSV del dataset.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame limpio y diccionario con el resumen de filas eliminadas.
    """
    df = pd.read_csv(ruta_csv)
    filas_iniciales = len(df)

    mask_dimensiones_invalidas = ~(df[["x", "y", "z"]] > 0).all(axis=1)
    filas_dimensiones_invalidas = int(mask_dimensiones_invalidas.sum())
    df = df[~mask_dimensiones_invalidas].copy()

    zscores = np.abs(zscore(df[["x", "y", "z"]]))
    mask_extremos = (zscores > 5).any(axis=1)
    filas_extremas = int(mask_extremos.sum())
    df = df[~mask_extremos].copy()

    resumen = {
        "filas_iniciales": filas_iniciales,
        "filas_dimensiones_invalidas": filas_dimensiones_invalidas,
        "filas_extremas": filas_extremas,
        "filas_finales": len(df),
    }
    return df, resumen


def generar_resumen_descriptivo(df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla descriptiva ampliada para variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    pd.DataFrame
        Tabla con estadísticas descriptivas ampliadas.
    """
    numericas = df.select_dtypes(include=[np.number])
    resumen = numericas.describe().T
    resumen["var"] = numericas.var()
    resumen["median"] = numericas.median()
    resumen["mode"] = numericas.mode().iloc[0]
    resumen["range"] = numericas.max() - numericas.min()
    resumen["IQR"] = numericas.quantile(0.75) - numericas.quantile(0.25)
    resumen["skew"] = numericas.skew()
    resumen["kurtosis"] = numericas.kurtosis()
    return resumen.round(4)


def guardar_outliers_txt(df: pd.DataFrame, resumen_limpieza: dict) -> None:
    """Guarda un informe de outliers y filas descartadas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset final tras la limpieza.
    resumen_limpieza : dict
        Diccionario con el resumen de filas eliminadas.

    Returns
    -------
    None
        No devuelve nada. Guarda un fichero de texto en output/.
    """
    q1 = df["price"].quantile(0.25)
    q3 = df["price"].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    mask_out_price = (df["price"] < limite_inf) | (df["price"] > limite_sup)

    with open(OUTPUT_DIR / "ej1_outliers.txt", "w", encoding="utf-8") as f:
        f.write("Ejercicio 1 - Resumen de limpieza y outliers\n")
        f.write("=" * 55 + "\n")
        f.write(f"Filas iniciales: {resumen_limpieza['filas_iniciales']}\n")
        f.write(
            f"Filas eliminadas por dimensiones imposibles (x, y o z <= 0): "
            f"{resumen_limpieza['filas_dimensiones_invalidas']}\n"
        )
        f.write(
            f"Filas eliminadas por valores extremos en x, y, z (Z-score > 5): "
            f"{resumen_limpieza['filas_extremas']}\n"
        )
        f.write(f"Filas finales: {resumen_limpieza['filas_finales']}\n\n")
        f.write("Outliers detectados en price con criterio IQR\n")
        f.write(f"Q1: {q1:.4f}\n")
        f.write(f"Q3: {q3:.4f}\n")
        f.write(f"IQR: {iqr:.4f}\n")
        f.write(f"Límite inferior: {limite_inf:.4f}\n")
        f.write(f"Límite superior: {limite_sup:.4f}\n")
        f.write(f"Número de outliers en price: {int(mask_out_price.sum())}\n")
        f.write(f"Porcentaje de outliers en price: {mask_out_price.mean() * 100:.4f}%\n")
        f.write(
            "Decisión: no se eliminan los outliers de price porque parecen\n"
            "valores reales del mercado y no errores evidentes de medición.\n"
        )


def guardar_resumen_csv(resumen: pd.DataFrame) -> None:
    """Guarda el resumen descriptivo en CSV.

    Parameters
    ----------
    resumen : pd.DataFrame
        Tabla descriptiva calculada con pandas.

    Returns
    -------
    None
        No devuelve nada. Escribe un CSV en output/.
    """
    resumen.to_csv(OUTPUT_DIR / "ej1_descriptivo.csv", encoding="utf-8")


def graficar_histogramas(df: pd.DataFrame) -> None:
    """Genera histogramas de las variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        axes[i].hist(df[col], bins=30, edgecolor="black", alpha=0.75)
        axes[i].set_title(f"Histograma de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frecuencia")

    for j in range(len(numericas), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribución de las variables numéricas", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej1_histogramas.png", dpi=180, bbox_inches="tight")
    plt.close()


def graficar_boxplots(df: pd.DataFrame) -> None:
    """Genera boxplots de las variables numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot de {col}")

    for j in range(len(numericas), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Detección visual de outliers", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej1_boxplots.png", dpi=180, bbox_inches="tight")
    plt.close()


def graficar_correlacion(df: pd.DataFrame) -> None:
    """Genera un mapa de calor de correlaciones numéricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", square=True)
    plt.title("Matriz de correlación entre variables numéricas")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej1_heatmap_correlacion.png", dpi=180, bbox_inches="tight")
    plt.close()


def graficar_categoricas(df: pd.DataFrame) -> None:
    """Genera gráficos de frecuencia para las variables categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset limpio.

    Returns
    -------
    None
        No devuelve nada. Guarda una imagen PNG en output/.
    """
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, len(categoricas), figsize=(19, 6))
    if len(categoricas) == 1:
        axes = [axes]

    for ax, col in zip(axes, categoricas):
        conteos = df[col].value_counts()
        ax.bar(conteos.index.astype(str), conteos.values)
        ax.set_title(f"Frecuencia de {col}")
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Conteo")

    fig.suptitle("Distribución de las variables categóricas", fontsize=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ej1_categoricas.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Ejecuta todo el flujo del ejercicio 1.

    Parameters
    ----------
    None
        Esta función no recibe argumentos.

    Returns
    -------
    None
        No devuelve nada. Genera ficheros de salida en output/.
    """
    df, resumen_limpieza = cargar_y_limpiar_datos(DATA_PATH)
    resumen = generar_resumen_descriptivo(df)
    guardar_resumen_csv(resumen)
    guardar_outliers_txt(df, resumen_limpieza)
    graficar_histogramas(df)
    graficar_boxplots(df)
    graficar_correlacion(df)
    graficar_categoricas(df)

    print("Ejercicio 1 completado.")
    print(f"Filas finales tras limpieza: {len(df)}")
    print("Ficheros generados en output/:")
    print(" - ej1_descriptivo.csv")
    print(" - ej1_outliers.txt")
    print(" - ej1_histogramas.png")
    print(" - ej1_boxplots.png")
    print(" - ej1_heatmap_correlacion.png")
    print(" - ej1_categoricas.png")


if __name__ == "__main__":
    main()
