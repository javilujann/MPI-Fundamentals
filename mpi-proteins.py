#!/usr/bin/env python3

import argparse
import heapq
from itertools import chain
from pathlib import Path

from mpi4py import MPI

# Colours of the bar chart
BAR, MUTED, GRID = "#2a78d6", "#52514e", "#dcdbd6"

type Match = tuple[int, int, int]
# Comparing two of these tuples already implements the ranking the lab asks for:
# more occurrences first, ties broken by the highest hydrofob.


def scan(path: Path, pattern: bytes, start: int, end: int) -> tuple[int, int, list[tuple]]:
    proteins = 0
    occurrences = 0
    best = []

    with path.open("rb") as data:
        data.seek(start)
        
        # Skip the CSV header (if at byte 0) or the partial line 
        # that belongs to the previous worker's chunk
        data.readline()

        # Keep reading line by line until we pass the assigned end byte
        while data.tell() < end:
            line = data.readline()
            if not line:
                break  # End of file reached

            protid, _enzyme, hydrofob, sequence = line.split(b",")
           
            count = sequence.count(pattern)
            if count > 0:
                proteins += 1
                occurrences += count
                match = (count, int(hydrofob), int(protid))

                best.append(match)
                best.sort(reverse=True)  # Sorts descending so the highest counts are first
                best = best[:10]         # Keep only the first 10 elements

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


def plot(pattern: str, best: list[Match]) -> None:
    """Draw a bar chart of the best matches, protein id against occurrences.

    Args:
        pattern: Pattern that was searched, in uppercase.
        best: Best matches, already sorted from best to worst.
    """
    import matplotlib.pyplot as plt

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
    axes.tick_params(colors=MUTED)
    axes.grid(visible=True, axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    plt.show()


def main() -> None:
    """Run the parallel matcher."""
    comm = MPI.COMM_WORLD  # global communicator
    rank = comm.Get_rank()  # id of this process
    nprocs = comm.Get_size()  # total number of processes
    path = Path("proteins.csv") # Path to the file

    # 1 and 2: only the first process reads the pattern from the keyboard and
    # changes it to uppercase, and then broadcasts it to all the others.
    pattern: str | None = None
    if rank == 0:
        pattern = input("Pattern to search: ").strip().upper()
        
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
    
    flatten_best = []
    for best_list in candidates:
        flatten_best.extend(best_list)
    flatten_best.sort(reverse=True)
    best = flatten_best[:10]
    
    elapsed = MPI.Wtime() - start
    print(f"Execution time with {nprocs} processes: {elapsed:.3f} s")

    report(pattern, total_proteins or 0, total_occurrences or 0, best)
    plot(pattern, best)


if __name__ == "__main__":
    main()
