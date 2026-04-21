from __future__ import annotations

import argparse
import random
from time import perf_counter


def generate_matrices(size: int, seed: int = 2026) -> tuple[list[list[float]], list[list[float]]]:
    """
    Genera dos matrices cuadradas con valores aleatorios.
    
    Args:
        size: Dimensión de las matrices cuadradas (size x size)
        seed: Seed para reproducibilidad de números aleatorios
    
    Returns:
        Tupla con dos matrices cuadradas de tamaño (size x size)
    """
    rng = random.Random(seed)
    matrix_a = [[rng.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[rng.random() for _ in range(size)] for _ in range(size)]
    return matrix_a, matrix_b


def matrix_multiply(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """
    Realiza la multiplicación de dos matrices de forma secuencial.
    
    Args:
        matrix_a: Primera matriz
        matrix_b: Segunda matriz
    
    Returns:
        Matriz resultado de la multiplicación
    """
    size = len(matrix_a)
    result = [[0.0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices secuencial")
    parser.add_argument("--complejidad", type=int, default=512, help="Dimensión de la matriz cuadrada")
    parser.add_argument("--workers", type=int, default=1, help="Parámetro informativo para mantener misma interfaz")
    args = parser.parse_args()

    # Generar matrices
    matrix_a, matrix_b = generate_matrices(args.complejidad)

    # Realizar multiplicación y medir tiempo
    start = perf_counter()
    result = matrix_multiply(matrix_a, matrix_b)
    elapsed = perf_counter() - start

    # Calcular promedio
    flat_result = [val for row in result for val in row]
    avg_val = sum(flat_result) / len(flat_result)

    # Mostrar resultados
    print(f"Multiplicación de matrices secuencial")
    print(f"Dimensión: {args.complejidad}x{args.complejidad}")
    print(f"Workers: {args.workers}")
    print(f"Tiempo elapsed: {elapsed:.6f} segundos")
    print(f"Resultado (promedio): {avg_val:.6f}")


if __name__ == "__main__":
    main()
