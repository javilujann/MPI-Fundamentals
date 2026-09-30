#!/usr/bin/env python3

import time
from pathlib import Path

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
    """Run the serial matcher."""
    
    # 1 and 2: read the pattern from the keyboard and change it to uppercase.
    pattern = input("Pattern to search: ").strip().upper()
    if not pattern:
        parser_error = "the pattern cannot be empty"
        raise SystemExit(parser_error)
    path = Path("proteins.csv")

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
