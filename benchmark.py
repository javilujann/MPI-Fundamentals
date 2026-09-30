#!/usr/bin/env python3
"""Measure serial against MPI execution times and draw the charts of the report.

This is a helper for the lab report, not part of the delivery: it just runs
``serial-proteins.py`` and ``mpi-proteins.py`` as subprocesses, collects the
execution times they print and turns them into a CSV table and three charts.

Usage:
    python benchmark.py [-f proteins.csv] [-p ABCD ...] [-n 1 2 4 ...] [-r 3]
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes

HERE = Path(__file__).parent
"""Directory holding the two matchers, so the script runs from anywhere."""

TIME = re.compile(r"Execution time(?: with \d+ processes)?: ([0-9.]+) s")
"""Matches the execution time line printed by both matchers."""

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
"""Categorical colours, assigned to patterns in this fixed order."""

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
"""Text and grid colours: the marks carry the identity, the text never does."""

IDEAL = "Ideal"
"""Name of the reference curve, drawn apart from the measured ones."""


def warm(path: Path) -> None:
    """Read ``path`` once so that every measurement starts with a warm cache."""
    with path.open("rb") as data:
        while data.read(1 << 24):
            pass


def measure(command: list[str], repeats: int) -> float:
    """Run ``command`` ``repeats`` times and return the shortest time reported.

    The minimum is used instead of the mean because it is the run least polluted
    by the rest of the system.

    Args:
        command: Matcher invocation, which must print an execution time.
        repeats: Number of times the command is run.

    Returns:
        The shortest execution time, in seconds.
    """
    times: list[float] = []
    for _ in range(repeats):
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        found = TIME.search(result.stdout)
        if found is None:
            message = f"no execution time in the output of {' '.join(command)}\n{result.stderr}"
            raise RuntimeError(message)
        times.append(float(found[1]))
        print(f"    {times[-1]:.3f} s")
    return min(times)


def style(axes: Axes, title: str, ylabel: str) -> None:
    """Apply the common look of the three charts: recessive grid and axes."""
    axes.set_title(title, color=INK, fontsize=12, loc="left")
    axes.set_xlabel("MPI processes", color=MUTED)
    axes.set_ylabel(ylabel, color=MUTED)
    axes.grid(visible=True, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.legend(frameon=False, labelcolor=MUTED)


def chart(
    path: Path, title: str, ylabel: str, procs: list[int], curves: dict[str, list[float]]
) -> None:
    """Draw one line per curve and write the chart to ``path``.

    Args:
        path: PNG file to write.
        title: Chart title.
        ylabel: Label of the Y axis.
        procs: Process counts, the X axis.
        curves: Values to draw, keyed by pattern. The ``IDEAL`` key, if
            present, is drawn as a dashed grey reference line.
    """
    figure, axes = plt.subplots(figsize=(7, 4.2), layout="constrained")
    measured = [value for name, values in curves.items() if name != IDEAL for value in values]
    span = max(measured) - min(measured)
    labelled: list[float] = []
    colours = iter(SERIES)
    for name, values in curves.items():
        if name == IDEAL:
            axes.plot(procs, values, color=MUTED, linewidth=1.2, linestyle="--", label=name)
            continue
        axes.plot(
            procs, values, color=next(colours), linewidth=2, marker="o", markersize=5, label=name
        )
        # Direct label on the last point, dropped when two curves end so close
        # that the labels would overlap. The legend still gives the identity.
        if any(abs(values[-1] - other) <= span / 25 for other in labelled):
            continue
        labelled.append(values[-1])
        axes.annotate(
            f"{values[-1]:.2f}",
            (procs[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=MUTED,
            fontsize=9,
        )
    axes.set_xticks(procs)
    axes.margins(x=0.08)  # room for the direct labels of the last point
    style(axes, title, ylabel)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"  wrote {path}")


def parse_args() -> argparse.Namespace:
    """Parse the command line. Every argument is optional."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-f", "--file", type=Path, default=Path("proteins.csv"))
    parser.add_argument("-p", "--patterns", nargs="+", default=["ABCD", "CDEFGH", "GH"])
    parser.add_argument("-n", "--procs", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10, 12])
    parser.add_argument("-r", "--repeats", type=int, default=3)
    parser.add_argument("-o", "--out", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    """Measure both versions, write the CSV table and draw the charts."""
    args = parse_args()
    path: Path = args.file
    patterns: list[str] = args.patterns
    procs: list[int] = args.procs
    out: Path = args.out
    out.mkdir(exist_ok=True)

    print(f"Warming the page cache with {path} ({path.stat().st_size / 1e6:.0f} MB)")
    warm(path)

    serial: dict[str, float] = {}
    parallel: dict[str, list[float]] = {}
    for pattern in patterns:
        common = ["-f", str(path), "-p", pattern, "--no-plot"]
        print(f"'{pattern}': serial")
        serial[pattern] = measure(
            [sys.executable, str(HERE / "serial-proteins.py"), *common], args.repeats
        )
        parallel[pattern] = []
        for nprocs in procs:
            print(f"'{pattern}': MPI with {nprocs} processes")
            command = ["mpiexec", "-n", str(nprocs), sys.executable, str(HERE / "mpi-proteins.py")]
            parallel[pattern].append(measure([*command, *common], args.repeats))

    table = out / "benchmark.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pattern", "version", "processes", "time_s", "speedup", "efficiency"])
        for pattern in patterns:
            writer.writerow([pattern, "serial", 1, f"{serial[pattern]:.3f}", "1.00", "1.00"])
            for nprocs, elapsed in zip(procs, parallel[pattern], strict=True):
                speedup = serial[pattern] / elapsed
                row = [pattern, "mpi", nprocs, f"{elapsed:.3f}"]
                writer.writerow([*row, f"{speedup:.2f}", f"{speedup / nprocs:.2f}"])
    print(f"  wrote {table}")

    speedups = {p: [serial[p] / t for t in parallel[p]] for p in patterns}
    efficiency = {p: [s / n for s, n in zip(speedups[p], procs, strict=True)] for p in patterns}
    chart(out / "time.png", "Execution time", "Seconds", procs, parallel)
    chart(
        out / "speedup.png",
        "Speedup over the serial version",
        "Speedup",
        procs,
        {IDEAL: [float(n) for n in procs], **speedups},
    )
    chart(
        out / "efficiency.png",
        "Parallel efficiency",
        "Efficiency",
        procs,
        {IDEAL: [1.0] * len(procs), **efficiency},
    )


if __name__ == "__main__":
    main()
