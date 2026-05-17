import numpy as np
import pandas as pd


def segmentar_pacientes(df):
    """Limpia un DataFrame clínico y segmenta pacientes por riesgo.

    Elimina duplicados y nulos, asigna grupo de riesgo basado en
    glucosa y presión arterial, y ordena por edad ascendente.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas ["edad", "glucosa", "presion_arterial",
        "imc", "consultas_previas"].

    Returns
    -------
    pd.DataFrame
        DataFrame limpio con columna "grupo_riesgo" agregada.
    """
    df_clean = df.drop_duplicates().dropna().copy()

    conditions = [
        (df_clean["glucosa"] >= 140) | (df_clean["presion_arterial"] >= 140),
        (df_clean["glucosa"] >= 100) | (df_clean["presion_arterial"] >= 120),
    ]
    choices = ["alto", "medio"]
    df_clean["grupo_riesgo"] = np.select(conditions, choices, default="bajo")

    return df_clean.sort_values("edad").reset_index(drop=True)
