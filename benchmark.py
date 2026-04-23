#!/usr/bin/env python3
"""
Ejecuta todos los scripts muxmat* y genera una tabla comparativa con tiempos,
speed-ups y eficiencias respecto al script secuencial base.

Scripts secuenciales : se corren con --complejidad (arg, default 512).
Scripts paralelos    : se corren siempre con complejidad 512 y 1024.
Speed-up             : t_secuencial(misma complejidad) / t_script
Eficiencia           : speed_up / workers
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

# Complejidades fijas para los scripts paralelos
PARALLEL_COMPLEJIDADES = [512, 1024]

# (nombre_script, es_secuencial, workers_fijo)
# workers_fijo=None significa que se usarán los workers del argumento --workers
SCRIPTS: list[tuple[str, bool, int | None]] = [
    ("muxmat_secuencial.py",                    True,  1),
    ("muxmat_threadpoolexecutor_numba.py",       False, None),
    ("muxmat_multiprocessing_numba.py",          False, None),
]

# Regex que cubre todas las variantes de salida de los scripts
_RE_TIME = re.compile(r"Tiempo(?:\s+elapsed)?:\s*([\d.]+)\s*segundos", re.IGNORECASE)
_RE_AVG  = re.compile(r"(?:Resultado\s*\(promedio\)|Promedio):\s*([\d.eE+\-]+)", re.IGNORECASE)


def parse_output(text: str) -> tuple[float | None, float | None]:
    t = _RE_TIME.search(text)
    a = _RE_AVG.search(text)
    return (float(t.group(1)) if t else None,
            float(a.group(1)) if a else None)


def run_once(script: Path, complejidad: int, workers: int) -> tuple[float | None, float | None, str]:
    cmd = [sys.executable, str(script), "--complejidad", str(complejidad), "--workers", str(workers)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, None, "timeout"
    if result.returncode != 0:
        return None, None, result.stderr.strip()
    t, avg = parse_output(result.stdout)
    return t, avg, ""


def run_script(script: Path, complejidad: int, workers: int, repeticiones: int) \
        -> tuple[float | None, float | None, str]:
    times, avgs = [], []
    for _ in range(repeticiones):
        t, avg, err = run_once(script, complejidad, workers)
        if t is None:
            return None, None, err
        times.append(t)
        if avg is not None:
            avgs.append(avg)
    return (statistics.mean(times),
            statistics.mean(avgs) if avgs else None,
            "")


# ── Formato de tabla ──────────────────────────────────────────────────────────

def col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return widths


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = col_widths(headers, rows)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo de scripts muxmat*"
    )
    parser.add_argument(
        "--complejidad-secuencial", type=int, default=512,
        dest="complejidad_seq",
        help="Dimensión para los scripts secuenciales (default: 512)"
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[2, 4],
        metavar="W",
        help="Lista de workers para los scripts paralelos (default: 2 4)"
    )
    parser.add_argument(
        "--repeticiones", type=int, default=3,
        help="Repeticiones por configuración para promediar tiempos (default: 3)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Complejidades que necesitan un baseline secuencial propio
    complejidades_baseline = sorted(set([args.complejidad_seq] + PARALLEL_COMPLEJIDADES))

    print(f"\nBenchmark muxmat*")
    print(f"  Complejidad secuencial : {args.complejidad_seq}x{args.complejidad_seq}")
    print(f"  Complejidades paralelos: {PARALLEL_COMPLEJIDADES}")
    print(f"  Workers                : {args.workers}")
    print(f"  Repeticiones           : {args.repeticiones}\n")

    # Calcular un baseline por cada complejidad que se va a usar
    seq_script = base_dir / "muxmat_secuencial.py"
    baselines: dict[int, float] = {}
    for c in complejidades_baseline:
        print(f"Ejecutando baseline {c}x{c} ({seq_script.name}) ...", flush=True)
        t_base, _, err = run_script(seq_script, c, 1, args.repeticiones)
        if t_base is None:
            print(f"  ERROR en baseline {c}x{c}: {err}")
            sys.exit(1)
        baselines[c] = t_base
        print(f"  Tiempo base {c}x{c}: {t_base:.6f} s")
    print()

    headers = ["Script", "Complejidad", "Workers", "Tiempo (s)", "Promedio", "Speed-up", "Eficiencia"]
    rows: list[list[str]] = []

    for script_name, is_sequential, fixed_workers in SCRIPTS:
        script_path = base_dir / script_name
        if not script_path.exists():
            print(f"  AVISO: {script_name} no encontrado, se omite.")
            continue

        label = script_name.replace("muxmat_", "").replace(".py", "")

        if is_sequential:
            complejidad_list = [args.complejidad_seq]
            worker_list      = [fixed_workers]
        else:
            complejidad_list = PARALLEL_COMPLEJIDADES
            worker_list      = args.workers

        for c in complejidad_list:
            t_base = baselines[c]
            for w in worker_list:
                print(f"  {label:45s} {c}x{c}  workers={w} ...", end=" ", flush=True)

                t, avg, err = run_script(script_path, c, w, args.repeticiones)

                if t is None:
                    print("ERROR")
                    rows.append([label, f"{c}x{c}", str(w), "ERROR", "-", "-", "-"])
                    continue

                speed_up   = t_base / t
                efficiency = speed_up / w

                rows.append([
                    label, f"{c}x{c}", str(w),
                    f"{t:.6f}",
                    f"{avg:.6f}" if avg is not None else "-",
                    f"{speed_up:.4f}",
                    f"{efficiency:.4f}",
                ])
                print(f"{t:.6f} s  speed-up={speed_up:.4f}  efic={efficiency:.4f}")

    print()
    print_table(headers, rows)
    print()


if __name__ == "__main__":
    main()
