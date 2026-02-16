"""Generador de casos de uso para la función evaluar_clasificador_fraude."""

import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def generar_caso_de_uso_evaluar_clasificador_fraude():
    """Genera un caso de prueba aleatorio para evaluar_clasificador_fraude.

    Construye una matriz X y un vector y de clasificación binaria
    con parámetros aleatorios, y calcula las métricas esperadas.

    Returns
    -------
    tuple
        (input_data, output_data) donde input_data es un dict con
        las claves 'X', 'y', 'test_size', 'random_state' y
        output_data es un dict con las métricas esperadas.
    """

    # 1. Configuración aleatoria de dimensiones
    n_samples = random.randint(100, 300)
    n_features = random.randint(3, 8)
    n_informative = random.randint(2, min(n_features, 5))
    n_redundant = random.randint(0, n_features - n_informative)

    # 2. Generar datos de clasificación binaria con sklearn
    generation_seed = random.randint(0, 9999)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=generation_seed,
    )

    # 3. Definir hiperparámetros aleatorios
    test_size = round(random.uniform(0.15, 0.4), 2)
    random_state = random.randint(0, 9999)

    # ---------------------------------------------------------
    # 4. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "test_size": test_size,
        "random_state": random_state,
    }

    # ---------------------------------------------------------
    # 5. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener
    #    evaluar_clasificador_fraude
    # ---------------------------------------------------------

    # A. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # B. Escalar características (fit solo sobre train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # C. Entrenar DecisionTreeClassifier con max_depth=5
    clf = DecisionTreeClassifier(
        max_depth=5, random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)

    # D. Predecir y calcular métricas
    y_pred = clf.predict(X_test_scaled)

    output_data = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    return input_data, output_data


# --- Ejemplo de uso y validación ---
if __name__ == "__main__":
    # Función solución para validación
    def evaluar_clasificador_fraude(X, y, test_size, random_state):
        """Solución de referencia para validar el generador."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = DecisionTreeClassifier(
            max_depth=5, random_state=random_state,
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0,
            ),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }

    # Verificar aleatoriedad
    print("=== Verificando aleatoriedad ===")
    for i in range(3):
        test_input, expected_output = (
            generar_caso_de_uso_evaluar_clasificador_fraude()
        )
        print(f"\nCaso {i + 1}:")
        print(f"  - Shape X: {test_input['X'].shape}")
        print(f"  - test_size: {test_input['test_size']}")
        for metric, value in expected_output.items():
            print(f"  - {metric}: {value:.4f}")

    # Validar con 10 casos de prueba
    print("\n=== Validando 10 casos de prueba ===")
    for i in range(10):
        test_input, expected_output = (
            generar_caso_de_uso_evaluar_clasificador_fraude()
        )
        result = evaluar_clasificador_fraude(**test_input)

        for key in expected_output:
            assert abs(result[key] - expected_output[key]) < 1e-10, (
                f"Fallo en {key}: {result[key]} != {expected_output[key]}"
            )
        print(f"  Caso {i + 1}: OK")

    # Prueba de stress: 50 casos
    print("\n=== Prueba de stress: 50 casos ===")
    for i in range(50):
        test_input, expected_output = (
            generar_caso_de_uso_evaluar_clasificador_fraude()
        )
        result = evaluar_clasificador_fraude(**test_input)

        for key in expected_output:
            diff = abs(result[key] - expected_output[key])
            assert diff < 1e-10, (
                f"Caso {i + 1} falló en {key}: diff={diff}"
            )

        if (i + 1) % 10 == 0:
            print(f"  {i + 1} casos OK")

    print("\nTodos los casos pasaron correctamente.")
