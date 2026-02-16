"""Generador de casos de uso para la función comparar_regresores."""

import random

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_comparar_regresores():
    """Genera un caso de prueba aleatorio para comparar_regresores.

    Construye una matriz X y un vector y de regresión con parámetros
    aleatorios, y calcula el resultado esperado de la comparación.

    Returns
    -------
    tuple
        (input_data, output_data) donde input_data es un dict con
        las claves 'X', 'y', 'n_folds' y output_data es un dict
        con las métricas de comparación.
    """

    # 1. Configuración aleatoria de dimensiones
    n_samples = random.randint(100, 300)
    n_features = random.randint(3, 8)
    noise = random.uniform(5.0, 50.0)
    generation_seed = random.randint(0, 9999)

    # 2. Generar datos de regresión con sklearn
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=generation_seed,
    )

    # 3. Definir número de folds aleatorio
    n_folds = random.randint(3, 7)

    # ---------------------------------------------------------
    # 4. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "n_folds": n_folds,
    }

    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener comparar_regresores
    # ---------------------------------------------------------

    # A. Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # B. Evaluar LinearRegression con cross_val_score
    lr_model = LinearRegression()
    lr_scores = cross_val_score(
        lr_model, X_scaled, y, cv=n_folds, scoring="r2",
    )

    # C. Evaluar Ridge con cross_val_score
    ridge_model = Ridge(alpha=1.0)
    ridge_scores = cross_val_score(
        ridge_model, X_scaled, y, cv=n_folds, scoring="r2",
    )

    # D. Determinar mejor modelo
    lr_mean = lr_scores.mean()
    ridge_mean = ridge_scores.mean()
    best_model = (
        "Ridge" if ridge_mean > lr_mean else "LinearRegression"
    )

    output_data = {
        "linear_mean_r2": lr_mean,
        "linear_std_r2": lr_scores.std(),
        "ridge_mean_r2": ridge_mean,
        "ridge_std_r2": ridge_scores.std(),
        "mejor_modelo": best_model,
    }

    return input_data, output_data


# --- Ejemplo de uso y validación ---
if __name__ == "__main__":
    # Función solución para validación
    def comparar_regresores(X, y, n_folds):
        """Solución de referencia para validar el generador."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_scores = cross_val_score(
            LinearRegression(), X_scaled, y,
            cv=n_folds, scoring="r2",
        )
        ridge_scores = cross_val_score(
            Ridge(alpha=1.0), X_scaled, y,
            cv=n_folds, scoring="r2",
        )

        lr_mean = lr_scores.mean()
        ridge_mean = ridge_scores.mean()

        return {
            "linear_mean_r2": lr_mean,
            "linear_std_r2": lr_scores.std(),
            "ridge_mean_r2": ridge_mean,
            "ridge_std_r2": ridge_scores.std(),
            "mejor_modelo": (
                "Ridge" if ridge_mean > lr_mean
                else "LinearRegression"
            ),
        }

    # Verificar aleatoriedad
    print("=== Verificando aleatoriedad ===")
    for i in range(3):
        test_input, expected_output = (
            generar_caso_de_uso_comparar_regresores()
        )
        print(f"\nCaso {i + 1}:")
        print(f"  - Shape X: {test_input['X'].shape}")
        print(f"  - n_folds: {test_input['n_folds']}")
        print(f"  - LR mean R2: {expected_output['linear_mean_r2']:.4f}")
        print(f"  - Ridge mean R2: {expected_output['ridge_mean_r2']:.4f}")
        print(f"  - Mejor modelo: {expected_output['mejor_modelo']}")

    # Validar con 10 casos de prueba
    print("\n=== Validando 10 casos de prueba ===")
    for i in range(10):
        test_input, expected_output = (
            generar_caso_de_uso_comparar_regresores()
        )
        result = comparar_regresores(**test_input)

        for key in expected_output:
            if isinstance(expected_output[key], float):
                assert abs(result[key] - expected_output[key]) < 1e-10, (
                    f"Fallo en {key}: "
                    f"{result[key]} != {expected_output[key]}"
                )
            else:
                assert result[key] == expected_output[key], (
                    f"Fallo en {key}: "
                    f"{result[key]} != {expected_output[key]}"
                )
        print(f"  Caso {i + 1}: OK")

    # Probando consistencia en decisión de mejor modelo
    print("\n=== Probando casos con modelos similares ===")
    for i in range(20):
        test_input, expected_output = (
            generar_caso_de_uso_comparar_regresores()
        )
        result = comparar_regresores(**test_input)

        # Verificar consistencia en la decisión del mejor modelo
        lr_mean = result["linear_mean_r2"]
        ridge_mean = result["ridge_mean_r2"]
        expected_best = (
            "Ridge" if ridge_mean > lr_mean else "LinearRegression"
        )

        assert result["mejor_modelo"] == expected_best, (
            f"Inconsistencia en mejor_modelo: "
            f"{result['mejor_modelo']} vs esperado {expected_best}"
        )

        if abs(lr_mean - ridge_mean) < 0.001:
            print(
                f"  Caso {i + 1}: Modelos muy cercanos "
                f"(diff={abs(lr_mean - ridge_mean):.6f})"
            )

    print("  20 casos de consistencia OK")

    print("\nTodos los casos pasaron correctamente.")
