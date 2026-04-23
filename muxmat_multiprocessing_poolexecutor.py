from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import argparse
import random


def generate_matrices(size: int, seed: int = 2026):
    """
    Genera dos matrices cuadradas con valores aleatorios.
    """
    rng = random.Random(seed)
    matrix_a = [[rng.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[rng.random() for _ in range(size)] for _ in range(size)]
    return matrix_a, matrix_b


def multiply_row(args: tuple) -> tuple[int, list[float]]:
    """Calcula una fila de la multiplicación"""
    row_index, matrix_a_row, matrix_b = args
    size = len(matrix_b)
    result_row = [0.0] * size
    
    for j in range(size):
        for k in range(size):
            result_row[j] += matrix_a_row[k] * matrix_b[k][j]
    
    return row_index, result_row


def calcular_promedio(result: list[list[float]]) -> float:
    """Calcula el promedio de todos los elementos de la matriz resultado."""
    total = sum(sum(row) for row in result)
    count = len(result) * len(result[0])
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices con ProcessPoolExecutor")
    parser.add_argument("--complejidad", type=int, default=512, help="Dimensión de la matriz")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de procesos")
    args = parser.parse_args()

    matrix_a, matrix_b = generate_matrices(args.complejidad)
    size = len(matrix_a)
    result = [[0.0 for _ in range(size)] for _ in range(size)]

    # Preparar tareas
    tasks = [(i, row, matrix_b) for i, row in enumerate(matrix_a)]

    start = perf_counter()
    
    # Usar ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # map() ejecuta y espera automáticamente
        for row_index, result_row in executor.map(multiply_row, tasks):
            result[row_index] = result_row

    elapsed = perf_counter() - start
    promedio = calcular_promedio(result)

    print(f"Multiplicación de matrices con ProcessPoolExecutor")
    print(f"Dimensión: {args.complejidad}x{args.complejidad}")
    print(f"Procesos: {args.workers}")
    print(f"Tiempo elapsed: {elapsed:.6f} segundos")
    print(f"Resultado (promedio): {promedio:.6f}")


if __name__ == "__main__":
    main()
