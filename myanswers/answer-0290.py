import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def predecir_ciclo_asimetrico(df: pd.DataFrame, target_col: str, peso_subestimacion: float) -> dict:
    """
    Entrena un modelo de regresión y calcula un error cuadrático asimétrico ponderado.
    """
    # 1. Separar características y variable objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. División train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Escalado
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # 4. Modelo
    modelo = GradientBoostingRegressor(random_state=42)
    modelo.fit(X_tr_s, y_tr)

    # 5. Predicción
    y_pred = modelo.predict(X_te_s)
    y_te_np = y_te.to_numpy()

    # 6. Error asimétrico vectorizado
    subestimaciones = y_pred < y_te_np
    errores = np.where(
        subestimaciones,
        peso_subestimacion * (y_te_np - y_pred) ** 2,
        (y_pred - y_te_np) ** 2
    )

    wmse = round(float(np.mean(errores)), 4)
    n_subestimaciones = int(np.sum(subestimaciones))

    return {
        "modelo": modelo,
        "wmse": wmse,
        "n_subestimaciones": n_subestimaciones
    }