#!/usr/bin/env python3

"""Measure the serial and the parallel matcher over both data sets.

Every experiment is run ``RUNS`` times and only the fastest run is kept, which
is what the lab asks for: the three repetitions absorb the noise of the page
cache and of the operating system, and the lowest time is the one that
describes the machine.

The experiments are the serial program and the MPI program with one process,
two processes, and so on up to the highest number of processes ``mpiexec``
accepts on this device. They are repeated for the two data sets named in
``DATA_SETS``, and each data set gets its own results file holding the output
printed by its experiments.

The benchmark takes no arguments: both CSV files must sit in the project
directory under the names hard-coded below. Run it with the interpreter of the
virtual environment, the one that has mpi4py and matplotlib installed:

    .venv/bin/python benchmark.py

Neither ``serial-proteins.py`` nor ``mpi-proteins.py`` is touched or imported,
so the programs that are measured are exactly the programs that are delivered.
"""

import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Pattern fed to the keyboard prompt of both programs. The generator plants
# "ABCD" on purpose in part of the sequences, so this pattern fills the ranking
# of the ten best proteins instead of leaving it empty.
PATTERN = "ABCD"

# Repetitions of every experiment; the fastest one is the one that is kept.
RUNS = 3

# The two programs under test.
SERIAL = Path("serial-proteins.py")
PARALLEL = Path("mpi-proteins.py")

# Both of them read this exact name from the current directory, and they must
# not be modified, so the benchmark points this name at one data set at a time
# with a symbolic link instead of copying gigabytes around.
DATA = Path("proteins.csv")

# Data set to measure and file that collects its results.
DATA_SETS: tuple[tuple[Path, Path], ...] = (
    (Path("proteins_5000000.csv"), Path("results_5000000_proteins.txt")),
    (Path("proteins_20000000.csv"), Path("results_20000000_proteins.txt")),
)

# The time both programs print, the one measurement they agree on: the serial
# one prints "Execution time: 1.234 s" and the parallel one prints
# "Execution time with 4 processes: 1.234 s". Not anchored to the start of a
# line on purpose: the keyboard prompt is echoed without a newline, so the
# first of those two lines comes out behind "Pattern to search: ".
TIME = re.compile(r"Execution time(?: with \d+ processes)?: ([0-9.]+) s")

# MPLBACKEND keeps plot() from opening a window that would block the benchmark
# until somebody closes it; with the Agg backend plt.show() does nothing.
# PYTHONUNBUFFERED makes a crashing run still show what it had printed.
ENVIRONMENT = os.environ | {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}

RULE = "=" * 78

type Experiment = tuple[str, list[str]]
# Label of the experiment and the command line that runs it.


def executable(name: str) -> str:
    """Return the absolute path of ``name``.

    Args:
        name: Program to look for in PATH.

    Returns:
        The absolute path of the program.
    """
    found = shutil.which(name)
    if found is None:
        message = f"{name} not found in PATH; an MPI runtime is needed"
        raise SystemExit(message)
    return found


def allowed(mpiexec: str) -> int:
    """Return the highest number of processes ``mpiexec`` starts on this device.

    Asked by trying, not guessed. An MPI runtime decides on its own how many
    processes it is willing to place: Open MPI counts one slot per core and
    refuses to go further, so on a machine with twelve cores and sixteen
    hardware threads ``-n 13`` already fails even though the operating system
    offers sixteen processors. Running the experiments the runtime cannot place
    is not possible, and asking it with ``--oversubscribe`` to place them anyway
    would measure processes fighting over a core instead of the machine.

    The probe starts a Python that does nothing, which is enough: a runtime that
    will not place the processes fails before it runs anything. The number of
    processors is the ceiling, because a runtime willing to oversubscribe would
    otherwise accept any number.

    Args:
        mpiexec: Absolute path of the MPI launcher.

    Returns:
        The highest usable number of processes.
    """
    ceiling = os.process_cpu_count() or 1
    highest = 0
    for processes in range(1, ceiling + 1):
        probe = subprocess.run(  # noqa: S603
            [mpiexec, "-n", str(processes), sys.executable, "-c", ""],
            capture_output=True,
            text=True,
            env=ENVIRONMENT,
            check=False,
        )
        if probe.returncode != 0:
            break
        highest = processes
    if highest == 0:
        message = f"{mpiexec} could not start even a single process"
        raise SystemExit(message)
    return highest


def point_to(data: Path) -> None:
    """Make ``proteins.csv`` the name of ``data``.

    A regular file already sitting there is never destroyed: the benchmark stops
    instead, because that file is somebody's data set.

    Args:
        data: Data set the programs have to read.
    """
    if DATA.is_symlink():
        DATA.unlink()
    elif DATA.exists():
        message = (
            f"{DATA} exists and is not a symbolic link. The two programs read that exact "
            f"name, so the benchmark needs it; move the file out of the way and run again."
        )
        raise SystemExit(message)
    DATA.symlink_to(data.name)


def release() -> None:
    """Remove the symbolic link the benchmark created, if it is still there."""
    if DATA.is_symlink():
        DATA.unlink()


def execute(command: list[str]) -> tuple[float, str]:
    """Run one experiment once.

    Args:
        command: Command line to run.

    Returns:
        The execution time the program printed and everything it printed on
        standard output.

    Raises:
        RuntimeError: The program failed, or printed no execution time.
    """
    finished = subprocess.run(  # noqa: S603
        command,
        input=f"{PATTERN}\n",
        capture_output=True,
        text=True,
        env=ENVIRONMENT,
        check=False,
    )
    elapsed = TIME.search(finished.stdout)
    if finished.returncode != 0 or elapsed is None:
        reason = (
            f"exit code {finished.returncode}"
            if finished.returncode != 0
            else "no execution time printed"
        )
        message = (
            f"{' '.join(command)} failed ({reason})\n"
            f"{finished.stdout.strip()}\n{finished.stderr.strip()}"
        ).strip()
        raise RuntimeError(message)
    return float(elapsed[1]), finished.stdout


def best_of(command: list[str], label: str) -> tuple[float, str, list[float]]:
    """Run one experiment ``RUNS`` times and keep the fastest run.

    Args:
        command: Command line to run.
        label: Name of the experiment, to show the progress on the screen.

    Returns:
        The lowest execution time, the output of that very run, and the times of
        all the runs in the order they were measured.
    """
    times: list[float] = []
    fastest = float("inf")
    output = ""
    for run in range(1, RUNS + 1):
        print(f"  {label}: run {run}/{RUNS} ... ", end="", flush=True)
        elapsed, printed = execute(command)
        print(f"{elapsed:.3f} s")
        if elapsed < fastest:
            fastest, output = elapsed, printed
        times.append(elapsed)
    return fastest, output, times


def experiments(mpiexec: str, maximum: int) -> list[Experiment]:
    """Build the list of experiments to run, in the order they are run.

    Args:
        mpiexec: Absolute path of the MPI launcher.
        maximum: Highest number of processes to measure.

    Returns:
        The serial experiment, followed by the parallel one with one process,
        two processes, and so on up to ``maximum``.
    """
    planned: list[Experiment] = [("serial", [sys.executable, str(SERIAL)])]
    planned += [
        (
            f"mpiexec -n {processes}",
            [mpiexec, "-n", str(processes), sys.executable, str(PARALLEL)],
        )
        for processes in range(1, maximum + 1)
    ]
    return planned


def summarise(measured: list[tuple[str, float]]) -> str:
    """Build the table of times and speedups that closes a results file.

    The speedup is measured against the serial program, which is the first
    experiment of every file and the baseline requirement 12 of the lab asks
    for. It is left out when the serial experiment itself failed.

    Args:
        measured: Label and best time of every experiment that succeeded.

    Returns:
        The table, ready to be written.
    """
    serial = next((time for label, time in measured if label == "serial"), None)
    lines = [RULE, f"Summary — best of {RUNS} runs", RULE, f"{'experiment':<20}{'time (s)':>12}"]
    if serial is not None:
        lines[-1] += f"{'speedup':>12}"
    for label, time in measured:
        line = f"{label:<20}{time:>12.3f}"
        if serial is not None:
            line += f"{serial / time:>12.2f}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def benchmark(data: Path, results: Path, mpiexec: str, maximum: int) -> None:
    """Run every experiment for one data set and write its results file.

    The file is written as the experiments finish, not at the end, so that the
    results already measured survive an interruption.

    Args:
        data: Data set to measure.
        results: File that collects the output of the experiments.
        mpiexec: Absolute path of the MPI launcher.
        maximum: Highest number of processes to measure.
    """
    print(f"\n{data} ({data.stat().st_size} bytes) -> {results}")
    point_to(data)
    measured: list[tuple[str, float]] = []
    failed: list[str] = []
    with results.open("w", encoding="utf-8") as report:
        report.write(
            f"{RULE}\n"
            f"Data set: {data}, {data.stat().st_size} bytes\n"
            f"Pattern: {PATTERN}\n"
            f"Repetitions per experiment: {RUNS}, fastest one kept\n"
            f"Processes: 1 to {maximum}, the most mpiexec starts on this device\n"
            f"Date: {datetime.now(UTC).astimezone():%Y-%m-%d %H:%M:%S %z}\n"
            f"{RULE}\n"
        )
        report.flush()
        for label, command in experiments(mpiexec, maximum):
            try:
                fastest, output, times = best_of(command, label)
            except RuntimeError as error:
                print(f"  {label}: FAILED")
                failed.append(label)
                report.write(f"\n{RULE}\n{label} — failed\n{RULE}\n{error}\n")
                report.flush()
                continue
            runs = ", ".join(f"{time:.3f}" for time in times)
            measured.append((label, fastest))
            report.write(
                f"\n{RULE}\n"
                f"{label} — best of {RUNS}: {fastest:.3f} s (runs: {runs} s)\n"
                f"{RULE}\n"
                f"{output}"
            )
            report.flush()
        report.write(f"\n{summarise(measured)}")
    if failed:
        print(f"  {len(failed)} experiment(s) failed; the reason is in {results}")


def main() -> None:
    """Run the whole benchmark."""
    for program in (SERIAL, PARALLEL):
        if not program.is_file():
            message = f"{program} not found; run the benchmark from the project directory"
            raise SystemExit(message)

    mpiexec = executable("mpiexec")
    maximum = allowed(mpiexec)
    print(
        f"Pattern '{PATTERN}', {RUNS} runs per experiment, "
        f"mpiexec from -n 1 to -n {maximum}, the most it starts here"
    )

    missing = [data for data, _ in DATA_SETS if not data.is_file()]
    try:
        for data, results in DATA_SETS:
            if data in missing:
                print(f"\n{data} not found, skipped; create it with proteins-generator.py")
                continue
            benchmark(data, results, mpiexec, maximum)
    finally:
        release()

    print("\nDone.")
    for data, results in DATA_SETS:
        absent = f" (not written: {data} is missing)" if data in missing else ""
        print(f"  {results}{absent}")


if __name__ == "__main__":
    main()
