#!/usr/bin/env python3
"""Serial protein pattern matcher (Lab 2, part one).

Reads a pattern from the keyboard, counts how many times it occurs inside every
protein sequence of ``proteins.csv``, prints the execution time, draws a bar
chart with the ten best-matching proteins and reports the best one of all.

Usage:
    python serial-proteins.py [-f proteins.csv] [-p PATTERN] [--save FILE]
"""

import argparse
import heapq
import time
from pathlib import Path

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

    A protein belongs to the range when its line *starts* inside it, so several
    processes splitting a file into consecutive ranges cover every line exactly
    once without having to agree on where the lines are.

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
        # Drops the CSV header when start == 0 and, for any other range, the
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
                # Only the TOP_N best matches are kept, so the memory used does
                # not depend on how many proteins match.
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
    # Imported here and not at the top of the file so that the runs that draw
    # no chart -- and, in the MPI version, the processes that never draw one --
    # do not pay for the import.
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
    return parser.parse_args()


def main() -> None:
    """Run the serial matcher."""
    args = parse_args()
    given: str | None = args.pattern
    # 1 and 2: read the pattern from the keyboard and change it to uppercase.
    pattern = (given if given is not None else input("Pattern to search: ")).strip().upper()
    if not pattern:
        parser_error = "the pattern cannot be empty"
        raise SystemExit(parser_error)
    path: Path = args.file

    # 3 to 6: time the search of the pattern over the whole data set.
    start = time.perf_counter()
    proteins, occurrences, best = scan(path, pattern.encode(), 0, path.stat().st_size)
    best = heapq.nlargest(TOP_N, best)
    elapsed = time.perf_counter() - start

    # 7 to 9: execution time, ranking, bar chart and the best protein.
    print(f"Execution time: {elapsed:.3f} s")
    report(pattern, proteins, occurrences, best)
    if not args.no_plot:
        plot(pattern, best, args.save)


if __name__ == "__main__":
    main()
