from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import argparse
import random
from numba import jit, prange
import numpy as np

def generate_matrices(size: int, seed: int = 2026):
    rng = random.Random(seed)
    matrix_a = [[rng.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[rng.random() for _ in range(size)] for _ in range(size)]
    return matrix_a, matrix_b

@jit(nopython=True,parallel=True) # Compilar con Numba - esto optimiza significativamente
def multiply_row_numba(matrix_a_row, matrix_b, size: int) -> list:
    """Calcula una fila de la multiplicación - compilada con Numba"""
    result_row = [0.0] * size
    
    for j in prange(size):
        for k in range(size):
            result_row[j] += matrix_a_row[k] * matrix_b[k][j]
    
    return result_row

def multiply_row(args: tuple) -> tuple[int, list]:
    """Wrapper para usar con ThreadPoolExecutor"""
    row_index, matrix_a_row, matrix_b = args
    size = len(matrix_b)
    result_row = multiply_row_numba(matrix_a_row, matrix_b, size)
    return row_index, result_row

def calcular_promedio(result: list[list[float]]) -> float:
    total = sum(sum(row) for row in result)
    count = len(result) * len(result[0])
    return total / count

def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices con ThreadPoolExecutor + Numba")
    parser.add_argument("--complejidad", type=int, default=512, help="Dimensión de la matriz")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de threads")
    args = parser.parse_args()

    matrix_a, matrix_b = generate_matrices(args.complejidad)
    size = len(matrix_a)
    result = [[0.0 for _ in range(size)] for _ in range(size)]

    # Se necesita convertir las listas a tuplas de numpy para Numba
    matrix_a_np = [np.array(row, dtype=np.float64) for row in matrix_a]
    matrix_b_np = np.array(matrix_b, dtype=np.float64)

    # Preparar tareas
    tasks = [(i, matrix_a_np[i], matrix_b_np) for i in range(size)]

    # Warmup: primera compilación de Numba
    print("Compilando con Numba (primera ejecución)...")
    multiply_row_numba(matrix_a_np[0], matrix_b_np, size)

    start = perf_counter()
    
    # Usar ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row_index, result_row in executor.map(multiply_row, tasks):
            result[row_index] = list(result_row)

    elapsed = perf_counter() - start
    promedio = calcular_promedio(result)

    print(f"\nMultiplicación de matrices con ThreadPoolExecutor + Numba")
    print(f"Dimensión: {args.complejidad}x{args.complejidad}")
    print(f"Workers: {args.workers}")
    print(f"Tiempo: {elapsed:.6f} segundos")
    print(f"Promedio: {promedio:.6f}")

if __name__ == "__main__":
    main()