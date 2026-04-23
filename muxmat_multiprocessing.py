from __future__ import annotations

import argparse
import random
from time import perf_counter
from multiprocessing import Process, Queue, Manager


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


def worker(task_queue: Queue, result_queue: Queue, matrix_b: list[list[float]]) -> None:
    """
    Worker que calcula filas asignadas de la multiplicación.
    """
    while True:
        item = task_queue.get()
        if item is None:
            break

        row_index, matrix_a_row = item
        size = len(matrix_b)
        result_row = [0.0] * size
        
        # Calcula esta fila del resultado
        for j in range(size):
            for k in range(size):
                result_row[j] += matrix_a_row[k] * matrix_b[k][j]
        
        result_queue.put((row_index, result_row))


def calcular_promedio(result: list[list[float]]) -> float:
    """Calcula el promedio de todos los elementos de la matriz resultado."""
    total = sum(sum(row) for row in result)
    count = len(result) * len(result[0])
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplicación de matrices con multiprocessing")
    parser.add_argument("--complejidad", type=int, default=512, help="Dimensión de la matriz cuadrada")
    parser.add_argument("--workers", type=int, default=4, help="Cantidad de procesos")
    args = parser.parse_args()

    matrix_a, matrix_b = generate_matrices(args.complejidad)
    size = len(matrix_a)
    
    # Inicializar resultado
    result: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
    
    # Crear colas
    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    # Crear procesos
    processes: list[Process] = []
    for _ in range(args.workers):
        process = Process(target=worker, args=(task_queue, result_queue, matrix_b))
        process.start()
        processes.append(process)

    start = perf_counter()
    
    # Enviar tareas (cada fila de A)
    for row_index, row in enumerate(matrix_a):
        task_queue.put((row_index, row))

    # Recopilar resultados
    for _ in range(size):
        row_index, result_row = result_queue.get()
        result[row_index] = result_row

    # Detener workers
    for _ in processes:
        task_queue.put(None)

    for process in processes:
        process.join()

    elapsed = perf_counter() - start
    promedio = calcular_promedio(result)

    print(f"Multiplicación de matrices con multiprocessing")
    print(f"Dimensión: {args.complejidad}x{args.complejidad}")
    print(f"Procesos: {args.workers}")
    print(f"Tiempo elapsed: {elapsed:.6f} segundos")
    print(f"Resultado (promedio): {promedio:.6f}")


if __name__ == "__main__":
    main()
