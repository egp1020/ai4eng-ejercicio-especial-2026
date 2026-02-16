"""Generador de casos de uso para la función resumen_ventas_por_region."""

import random

import numpy as np
import pandas as pd


def generar_caso_de_uso_resumen_ventas_por_region():
    """Genera un caso de prueba aleatorio para resumen_ventas_por_region.

    Construye un DataFrame de ventas con productos, regiones,
    cantidades, precios y descuentos aleatorios, y calcula el
    resumen esperado por región.

    Returns
    -------
    tuple
        (input_data, output_data) donde input_data es un dict con
        la clave 'df' y output_data es el DataFrame resumen esperado.
    """

    # 1. Configuración aleatoria de dimensiones
    n_rows = random.randint(20, 60)

    # 2. Definir catálogos de productos y regiones
    product_catalog = [
        "Laptop", "Mouse", "Teclado", "Monitor", "Auriculares",
        "Webcam", "Impresora", "Tablet", "Disco SSD", "Cable HDMI",
    ]
    region_catalog = ["Norte", "Sur", "Centro", "Oriente", "Occidente"]

    # Usar al menos 3 regiones para que no sea trivial
    n_regions = random.randint(3, len(region_catalog))
    active_regions = random.sample(region_catalog, n_regions)

    # 3. Generar datos aleatorios de ventas
    products = [random.choice(product_catalog) for _ in range(n_rows)]
    regions = [random.choice(active_regions) for _ in range(n_rows)]
    quantities = np.random.randint(1, 20, size=n_rows)
    unit_prices = np.round(np.random.uniform(10.0, 2000.0, size=n_rows), 2)
    discounts = np.round(np.random.uniform(0.0, 0.3, size=n_rows), 2)

    # 4. Construir el DataFrame de entrada
    df = pd.DataFrame({
        "producto": products,
        "region": regions,
        "cantidad": quantities,
        "precio_unitario": unit_prices,
        "descuento": discounts,
    })

    # ---------------------------------------------------------
    # 5. Construir el objeto INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
    }

    # ---------------------------------------------------------
    # 6. Calcular el OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica que debería tener
    #    resumen_ventas_por_region
    # ---------------------------------------------------------

    # A. Calcular ingreso neto por transacción
    df["ingreso_neto"] = (
        df["cantidad"] * df["precio_unitario"] * (1 - df["descuento"])
    )

    # B. Agrupar por región y calcular métricas
    summary = df.groupby("region").agg(
        total_ingresos=("ingreso_neto", "sum"),
        promedio_descuento=("descuento", "mean"),
        num_transacciones=("region", "count"),
    ).reset_index()

    # C. Calcular porcentaje de ingresos por región
    total_general = summary["total_ingresos"].sum()
    summary["porcentaje_ingresos"] = np.round(
        summary["total_ingresos"] / total_general * 100, 2,
    )

    # D. Ordenar por total_ingresos descendente y reiniciar índice
    summary = summary.sort_values(
        "total_ingresos", ascending=False,
    ).reset_index(drop=True)

    output_data = summary

    return input_data, output_data


# --- Ejemplo de uso y validación ---
if __name__ == "__main__":
    # Función solución para validación
    def resumen_ventas_por_region(df):
        """Solución de referencia para validar el generador."""
        df = df.copy()
        df["ingreso_neto"] = (
            df["cantidad"] * df["precio_unitario"] * (1 - df["descuento"])
        )

        summary = df.groupby("region").agg(
            total_ingresos=("ingreso_neto", "sum"),
            promedio_descuento=("descuento", "mean"),
            num_transacciones=("region", "count"),
        ).reset_index()

        total_general = summary["total_ingresos"].sum()
        summary["porcentaje_ingresos"] = np.round(
            summary["total_ingresos"] / total_general * 100, 2,
        )

        return summary.sort_values(
            "total_ingresos", ascending=False,
        ).reset_index(drop=True)

    # Verificar aleatoriedad
    print("=== Verificando aleatoriedad ===")
    for i in range(3):
        test_input, expected_output = (
            generar_caso_de_uso_resumen_ventas_por_region()
        )
        print(f"\nCaso {i + 1}:")
        print(f"  - Filas en input: {len(test_input['df'])}")
        print(f"  - Regiones: {expected_output['region'].tolist()}")
        print(
            f"  - Suma porcentajes: "
            f"{expected_output['porcentaje_ingresos'].sum():.2f}%"
        )

    # Validar con 10 casos de prueba
    print("\n=== Validando 10 casos de prueba ===")
    for i in range(10):
        test_input, expected_output = (
            generar_caso_de_uso_resumen_ventas_por_region()
        )
        result = resumen_ventas_por_region(**test_input)
        pd.testing.assert_frame_equal(result, expected_output)
        print(f"  Caso {i + 1}: OK")

    # Probando casos extremos (porcentajes y consistencia)
    print("\n=== Probando casos extremos ===")
    for i in range(20):
        test_input, expected_output = (
            generar_caso_de_uso_resumen_ventas_por_region()
        )
        result = resumen_ventas_por_region(**test_input)

        # Verificar que los porcentajes sumen aproximadamente 100
        pct_sum = result["porcentaje_ingresos"].sum()
        assert 99.5 <= pct_sum <= 100.5, (
            f"Suma de porcentajes anormal: {pct_sum}"
        )

        pd.testing.assert_frame_equal(result, expected_output)

    print("  20 casos extremos OK")

    print("\nTodos los casos pasaron correctamente.")
