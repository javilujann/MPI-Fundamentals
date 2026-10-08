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
├── authors.txt             # delivered
└── pyproject.toml          # Ruff and Pyright configuration
```

`serial-proteins.py` and `mpi-proteins.py` repeat the `scan`, `report` and
`plot` functions instead of importing them from a common module. That is
deliberate: the delivery accepts only these two scripts, so each one has to run
on its own. The two copies of `scan` are kept byte-identical, so a fix to one is
a fix to the other.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --group dev
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

Both programs take no arguments: they ask for the pattern on the keyboard and
read `proteins.csv` from the current directory.

```bash
python serial-proteins.py                 # asks for the pattern
mpiexec -n 12 python mpi-proteins.py      # same, in parallel
```

Each one prints its own execution time, the ranking of the ten best proteins and
the protein with most occurrences, and then draws the bar chart. Pass more
processes than cores only with `mpiexec --oversubscribe`.

## Measuring

The measurements of the report are taken by hand: run the serial version once,
run the parallel one for each process count and each machine under test, and
compute the speedup from the times both of them print.

## Everyday commands

| Task             | Command                      |
| ---------------- | ---------------------------- |
| Lint             | `ruff check .`               |
| Lint and autofix | `ruff check --fix .`         |
| Format           | `ruff format .`              |
| Type-check       | `pyright`                    |
| Run every hook   | `pre-commit run --all-files` |
