#!/usr/bin/env python3
"""Parallel protein pattern matcher with MPI (Lab 2, part two).

Same search as ``serial-proteins.py``, split between MPI processes:

* **Decomposition.** Each protein of the data set is an independent task.
* **Assignation.** The file is cut into ``nprocs`` contiguous byte ranges and
  every process reads and searches its own range, straight from disk.
* **Orchestration.** The pattern is broadcast from the first process, the
  counters are reduced and the ten best matches of each process are gathered.
* **Mapping.** Left to ``mpiexec``, one process per core by default.

Usage:
    mpiexec -n 8 python mpi-proteins.py [-f proteins.csv] [-p PATTERN]
"""

import argparse
import heapq
from itertools import chain
from pathlib import Path

from mpi4py import MPI

TOP_N = 10
"""Number of proteins shown in the bar chart."""

BLOCK_SIZE = 8 << 20
"""Bytes read from the data set per I/O operation (8 MiB)."""

BAR, MUTED, GRID = "#2a78d6", "#52514e", "#dcdbd6"
"""Colours of the bar chart: one bar colour, and muted text and grid lines."""

type Match = tuple[int, int, int]
"""A match, as ``(occurrences, hydrofob, protid)``.

Comparing two of these tuples already implements the ranking the lab asks for:
more occurrences first, ties broken by the highest hydrofob (and finally by the
protein id, which only makes the output deterministic).
"""


def scan(path: Path, pattern: bytes, start: int, end: int) -> tuple[int, int, list[Match]]:
    """Look for ``pattern`` in the proteins stored in the bytes ``[start, end)``.

    A protein belongs to the range when its line *starts* inside it, so the
    ranges of the MPI processes cover every line of the file exactly once
    without the processes having to agree on where the lines are.

    Args:
        path: Data set in the format written by ``proteins-generator.py``.
        pattern: Uppercase pattern to look for, encoded as bytes.
        start: First byte of the range.
        end: First byte after the range.

    Returns:
        The number of proteins with at least one occurrence, the total number of
        occurrences, and the ``TOP_N`` best matches of the range as a heap.
    """
    proteins = 0
    occurrences = 0
    best: list[Match] = []
    with path.open("rb") as data:
        data.seek(start)
        # Drops the CSV header in the first range and, for any other range, the
        # tail of the line that the preceding range already owns.
        data.readline()
        while (position := data.tell()) < end:
            block = data.read(min(BLOCK_SIZE, end - position))
            block += data.readline()  # complete the line the block cut in two
            # The pattern is searched over the whole block instead of line by
            # line: the lines that do not match -- most of them -- are never
            # built as Python objects, and only the line around each hit is
            # rebuilt and split into fields.
            cursor = 0
            while (hit := block.find(pattern, cursor)) >= 0:
                stop = block.find(b"\n", hit)
                if stop < 0:
                    stop = len(block)  # last line of a file with no final "\n"
                line = block[block.rfind(b"\n", 0, hit) + 1 : stop]
                cursor = stop + 1  # every hit of this line is counted at once
                protid, _enzyme, hydrofob, sequence = line.split(b",")
                count = sequence.count(pattern)
                if not count:
                    continue  # the hit was in a field other than the sequence
                proteins += 1
                occurrences += count
                match = (count, int(hydrofob), int(protid))
                # Only the TOP_N best matches are kept, which also keeps the
                # message that this process sends to the root a fixed size.
                if len(best) < TOP_N:
                    heapq.heappush(best, match)
                elif match > best[0]:
                    heapq.heapreplace(best, match)
    return proteins, occurrences, best


def report(pattern: str, proteins: int, occurrences: int, best: list[Match]) -> None:
    """Print the ranking of ``best`` and the protein with most occurrences.

    Args:
        pattern: Pattern that was searched, in uppercase.
        proteins: Number of proteins with at least one occurrence.
        occurrences: Total number of occurrences in the data set.
        best: Best matches, already sorted from best to worst.
    """
    if not best:
        print(f"No protein contains '{pattern}'.")
        return
    print(f"Proteins containing '{pattern}': {proteins} ({occurrences} occurrences)")
    print(f"Top {len(best)} proteins (protid, occurrences, hydrofob):")
    for count, hydrofob, protid in best:
        print(f"  {protid:>9} {count:>4} {hydrofob:>4}")
    count, hydrofob, protid = best[0]
    print(
        f"Protein with max occurrences: protid {protid} "
        f"with {count} occurrences (hydrofob {hydrofob})"
    )


def plot(pattern: str, best: list[Match], save: Path | None) -> None:
    """Draw a bar chart of the best matches, protein id against occurrences.

    Args:
        pattern: Pattern that was searched, in uppercase.
        best: Best matches, already sorted from best to worst.
        save: File to write the chart to; when ``None`` the chart is shown.
    """
    # Imported here and not at the top of the file so that only the process that
    # draws the chart pays for the import.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not best:
        return
    figure, axes = plt.subplots(figsize=(9, 5), layout="constrained")
    bars = axes.bar(
        [str(protid) for *_, protid in best],
        [count for count, *_ in best],
        width=0.62,
        color=BAR,
    )
    axes.bar_label(bars, color=MUTED, padding=2)
    axes.set_title(f"Top {len(best)} proteins matching '{pattern}'", loc="left")
    axes.set_xlabel("Protein id", color=MUTED)
    axes.set_ylabel("Occurrences", color=MUTED)
    # Recessive axes and grid: the bars are what has to be read, not the frame.
    axes.tick_params(colors=MUTED)
    axes.grid(visible=True, axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    if save is None:
        plt.show()
    else:
        figure.savefig(save, dpi=150)
        print(f"Bar chart saved to {save}")


def parse_args() -> argparse.Namespace:
    """Parse the command line. Every argument is optional."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-f", "--file", type=Path, default=Path("proteins.csv"), help="data set to search"
    )
    parser.add_argument("-p", "--pattern", help="pattern to search, instead of asking for it")
    parser.add_argument("--save", type=Path, help="write the bar chart here instead of showing it")
    parser.add_argument("--no-plot", action="store_true", help="skip the bar chart")
    parser.add_argument(
        "-s",
        "--serial-time",
        type=float,
        help="execution time of serial-proteins.py, to show the speedup",
    )
    return parser.parse_args()


def main() -> None:
    """Run the parallel matcher."""
    comm = MPI.COMM_WORLD  # global communicator
    rank = comm.Get_rank()  # id of this process
    nprocs = comm.Get_size()  # total number of processes

    args = parse_args()
    path: Path = args.file

    # 1 and 2: only the first process reads the pattern from the keyboard and
    # changes it to uppercase, and then broadcasts it to all the others.
    pattern: str | None = None
    if rank == 0:
        given: str | None = args.pattern
        typed = given if given is not None else input("Pattern to search: ")
        pattern = typed.strip().upper() or None
    pattern = comm.bcast(pattern, root=0)
    if pattern is None:
        message = "the pattern cannot be empty"
        raise SystemExit(message)

    # 3: the barrier makes every process start measuring at the same instant.
    comm.Barrier()
    start = MPI.Wtime()

    # 4 to 6: every process searches its own range of the file, on its own.
    file_size = path.stat().st_size
    proteins, occurrences, best = scan(
        path,
        pattern.encode(),
        rank * file_size // nprocs,
        (rank + 1) * file_size // nprocs,
    )

    # The two counters are reduced with a sum, and every process sends its own
    # TOP_N candidates: a protein in the global top TOP_N is necessarily in the
    # top TOP_N of the process that found it, so the union holds all of them.
    total_proteins = comm.reduce(proteins, op=MPI.SUM, root=0)
    total_occurrences = comm.reduce(occurrences, op=MPI.SUM, root=0)
    candidates: list[list[Match]] | None = comm.gather(best, root=0)

    # 11 and 12: the root process stops the clock and prints the results.
    if rank != 0 or candidates is None:
        return
    best = heapq.nlargest(TOP_N, chain.from_iterable(candidates))
    elapsed = MPI.Wtime() - start
    print(f"Execution time with {nprocs} processes: {elapsed:.3f} s")
    serial_time: float | None = args.serial_time
    if serial_time is not None:
        print(f"Speedup over the serial version ({serial_time:.3f} s): {serial_time / elapsed:.2f}")
    report(pattern, total_proteins or 0, total_occurrences or 0, best)
    if not args.no_plot:
        plot(pattern, best, args.save)


if __name__ == "__main__":
    main()
