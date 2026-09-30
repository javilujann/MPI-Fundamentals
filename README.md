# Lab 2 — Distributed protein matching in Python

Technological Fundamentals in the Big Data World — Master in Big Data.

Searches a pattern typed on the keyboard inside the `sequence` field of every
protein of `proteins.csv`, and reports the ten proteins with most occurrences.
There are two versions of the same program: a serial one and a parallel one
written with MPI (`mpi4py`).

## Layout

The project is flat: the programs sit at the repository root, because the two
that are delivered have to be stand-alone scripts.

```
.
├── proteins-generator.py   # given by the teaching staff, delivered untouched
├── serial-proteins.py      # delivered — part one
├── mpi-proteins.py         # delivered — part two
├── benchmark.py            # helper: measures both versions and draws the charts
├── authors.txt             # delivered
├── report.typ              # source of report.pdf, which is delivered
├── results/                # benchmark.csv and the charts of the report
└── pyproject.toml          # Ruff and Pyright configuration
```

`serial-proteins.py` and `mpi-proteins.py` repeat the `scan`, `report` and
`plot` functions instead of importing them from a common module. That is
deliberate: the delivery accepts only these two scripts, so each one has to run
on its own.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --group lab --group dev
pre-commit install          # needs a git repository; run `git init` first if there is none
```

An MPI runtime is needed as well (Open MPI or MPICH); `mpi4py` binds to it.

## Creating the data sets

Always with **42** as the seed, as the lab requires:

```bash
python proteins-generator.py 50000 42     # development
python proteins-generator.py 5000000 42   # tests and delivery
```

The generator always writes `proteins.csv`. Rename the small one out of the way
(`mv proteins.csv proteins-dev.csv`) if you want to keep both.

## Running the programs

```bash
python serial-proteins.py                        # asks for the pattern
mpiexec -n 12 python mpi-proteins.py             # same, in parallel
mpiexec -n 12 python mpi-proteins.py -s 1.268    # ...and print the speedup
```

Both accept the same optional arguments, which exist to script the measurements
and are not needed for normal use:

| Argument            | Meaning                                              |
| ------------------- | ---------------------------------------------------- |
| `-f`, `--file`      | data set to search (default `proteins.csv`)          |
| `-p`, `--pattern`   | pattern to search, instead of asking for it          |
| `--save FILE`       | write the bar chart to `FILE` instead of showing it  |
| `--no-plot`         | skip the bar chart                                   |
| `-s`, `--serial-time` | serial time, to print the speedup (MPI version only) |

Pass more processes than cores only with `mpiexec --oversubscribe`.

## Measuring

```bash
python benchmark.py -r 3
```

Runs both versions for several patterns and process counts and writes
`results/benchmark.csv` plus the three charts used in the report.

## Building the report

```bash
typst compile report.typ report.pdf
```

The report reads the author names from `authors.txt` and every timing in it from
`results/benchmark.csv`, so neither is written down twice.

## Packaging the delivery

Fill in `authors.txt` first, rebuild the report, and then, with the NIA of the
group:

```bash
zip 100052132_lab2_2026.zip report.pdf authors.txt serial-proteins.py mpi-proteins.py
```

## Everyday commands

| Task             | Command                      |
| ---------------- | ---------------------------- |
| Lint             | `ruff check .`               |
| Lint and autofix | `ruff check --fix .`         |
| Format           | `ruff format .`              |
| Type-check       | `pyright`                    |
| Run every hook   | `pre-commit run --all-files` |
