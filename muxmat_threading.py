from __future__ import annotations

import argparse
import random
from time import perf_counter
from queue import Queue
import threading

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


def worker(task_queue: Queue, result: list[list[float]], matrix_b: list[list[float]]) -> None:
    while True:
        item = task_queue.get()
        if item is None:
            break

        row_index, matrix_a_row = item
        size = len(matrix_b)
        # Calcula esta fila del resultado
        for j in range(size):
            for k in range(size):
                result[row_index][j] += matrix_a_row[k] * matrix_b[k][j]
        
        task_queue.task_done()

def calcular_promedio(result: list[list[float]]) -> float: 
    total = sum(sum(row) for row in result)
    count = len(result) * len(result[0])
    return total / count

def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices secuencial con transpuesta")
    parser.add_argument("--complejidad", type=int, default=512, help="Dimensión de la matriz cuadrada")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de hilos")
    args = parser.parse_args()

    matrix_a, matrix_b = generate_matrices(args.complejidad)
    size = len(matrix_a)
    
    # Inicializar resultado con ceros
    result: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
    task_queue: Queue = Queue()

    threads: list[threading.Thread] = []
    for _ in range(args.workers):
        thread = threading.Thread(target=worker, args=(task_queue, result, matrix_b))
        thread.start()
        threads.append(thread)

    start = perf_counter()
    for row_index, row in enumerate(matrix_a):
        task_queue.put((row_index, row))

    task_queue.join()

    for _ in threads:
        task_queue.put(None)

    for thread in threads:
        thread.join()

    promedio = calcular_promedio(result)
    elapsed = perf_counter() - start

    # Mostrar resultados
    print(f"Multiplicación de matrices secuencial")
    print(f"Dimensión: {args.complejidad}x{args.complejidad}")
    print(f"Workers: {args.workers}")
    print(f"Tiempo elapsed: {elapsed:.6f} segundos")
    print(f"Resultado (promedio): {promedio:.6f}")


if __name__ == "__main__":
    main()
