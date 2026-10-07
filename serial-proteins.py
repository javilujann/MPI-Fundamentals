#!/usr/bin/env python3

import time
from heapq import heappush, heapreplace
from pathlib import Path

# Colours of the bar chart
BAR, MUTED, GRID = "#2a78d6", "#52514e", "#dcdbd6"

# Proteins the lab asks to rank, to print and to draw.
TOP_N = 10

# Fields of every CSV row: protid, enzyme, hydrofob, sequence.
FIELDS = 4

type Match = tuple[int, int, int]
# Comparing two of these tuples already implements the ranking the lab asks for:
# more occurrences first, ties broken by the highest hydrofob.


def scan(path: Path, pattern: bytes, start: int, end: int) -> tuple[int, int, list[Match]]:
    """Count the occurrences of ``pattern`` in the lines owned by one chunk.

    A chunk owns every line whose *first* byte falls inside ``[start, end)``, so
    splitting the file into consecutive chunks and scanning each one covers
    every line exactly once, with no line read twice and none left out.

    Occurrences are counted without overlapping, the way ``bytes.count`` does:
    ``b"ABABA"`` holds one ``b"ABA"``, not two.

    Args:
        path: Data set to search.
        pattern: Pattern to search, in uppercase and already encoded.
        start: First byte of the chunk.
        end: First byte after the chunk.

    Returns:
        The number of proteins with at least one occurrence, the total number of
        occurrences, and the best ``TOP_N`` matches sorted from best to worst.
    """
    proteins = 0
    occurrences = 0
    # A min-heap of at most TOP_N matches, so best[0] is the worst candidate
    # kept so far: exactly the one a new match has to beat. A heap instead of a
    # sorted list turns the work done per match from a sort into a comparison.
    best: list[Match] = []

    with path.open("rb") as data:
        if start == 0:
            data.readline()  # Skip the CSV header
        else:
            # One byte back on purpose. The readline below then swallows the
            # rest of the line that straddles `start`, which belongs to the
            # previous chunk; and when `start` already is a line start it
            # swallows only the newline before it, so that line is not lost.
            data.seek(start - 1)
            data.readline()

        # First byte of the next line, kept by hand instead of asking the file
        # for it with data.tell() once per line.
        position = data.tell()
        while position < end:
            line = data.readline()
            if not line:
                break  # End of file reached
            position += len(line)

            # maxsplit=3 stops splitting as soon as the four fields are known,
            # and keeps a comma inside the sequence from breaking the unpacking.
            fields = line.split(b",", 3)
            if len(fields) != FIELDS:
                message = f"{path}: malformed line at byte {position - len(line)}: {line!r}"
                raise ValueError(message)
            protid, _enzyme, hydrofob, sequence = fields

            count = sequence.count(pattern)
            if count == 0:
                continue

            proteins += 1
            occurrences += count

            if len(best) < TOP_N:
                heappush(best, (count, int(hydrofob), int(protid)))
            elif count >= best[0][0]:
                # Only worth building the tuple here: a protein with fewer
                # occurrences than the worst candidate cannot enter the ranking,
                # and most of them have fewer.
                match = (count, int(hydrofob), int(protid))
                if match > best[0]:
                    heapreplace(best, match)

    best.sort(reverse=True)  # Sorts descending so the highest counts are first
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
    # Imported here and not at the top of the file: the chart is the last
    # optional step, so a run that never gets to it does not pay for
    # matplotlib's import time.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if not best:
        return
    _figure, axes = plt.subplots(figsize=(9, 5), layout="constrained")
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
    """Run the serial matcher."""
    # 1 and 2: read the pattern from the keyboard and change it to uppercase.
    pattern = input("Pattern to search: ").strip().upper()
    if not pattern:
        message = "the pattern cannot be empty"
        raise SystemExit(message)

    path = Path("proteins.csv")
    if not path.is_file():
        message = f"{path} not found; create it with proteins-generator.py"
        raise SystemExit(message)

    # 3 to 6: time the search of the pattern over the whole data set.
    start = time.perf_counter()
    proteins, occurrences, best = scan(path, pattern.encode(), 0, path.stat().st_size)
    elapsed = time.perf_counter() - start

    # 7 to 9: execution time, ranking, bar chart and the best protein.
    print(f"Execution time: {elapsed:.3f} s")
    report(pattern, proteins, occurrences, best)
    plot(pattern, best)


if __name__ == "__main__":
    main()
