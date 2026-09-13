# quops-implementations

Companion implementations for the paper **"Benchmarking the computational power of quantum computers"**. This repository demonstrates how to implement the QUOPS benchmark using
[Guppy](https://docs.quantinuum.com/guppy/) and
[pytket](https://docs.quantinuum.com/tket/).

## The QUOPS benchmark

The quantum universal operation performance system (QUOPS) is a benchmark for
comparing computational capability across quantum computers and tracking progress
toward quantum scientific utility. It quantifies the size of the largest
computationally relevant quantum circuits a machine can execute successfully. By providing a common measure across
different qubit technologies and physical- and logical-qubit architectures,
QUOPS enables fair cross-platform comparisons and puts hardware capability in
the context of the resources needed for utility-scale challenge problems.

The benchmark uses random circuits of of independently variable width and "size" (approximately, the number of gates in the circuit), with
mirror-circuit experiments used to efficiently determine whether or not they can be executed successfully.
See the [paper](#accompanying-paper-and-experiments) for the full benchmark definition, methodology, and
its application to assessing progress toward utility-scale challenge problems.

## Accompanying Paper and Experiments

**Benchmarking the computational power of quantum computers**

**Paper link: [To be added when the paper is available online.]**

The paper is the reference for the benchmark definition, experimental protocols,
and reported results. These notebooks are explanatory implementations, not an
exact reproduction of every experiment in the paper. In particular, demonstration
parameters and statistical procedures may differ from those used for the
reported results. Notably, estimating a device's QUOPS score requires the full statistical
procedure described in the paper, rather than simply testing individual circuit
shapes.

## Repository contents

| File | Description |
| --- | --- |
| [Guppy notebook](guppy/quops_in_guppy.ipynb) | A detailed introduction to physical-level QUOPS and its implementation in Guppy. This covers runtime circuit randomness, Pauli twirling, Clifford frame randomisation, mirrored circuits, and polarisation estimation with confidence bounds. |
| [pytket notebook](pytket/quops-with-pytket.ipynb) | A circuit-construction walkthrough covering QUOPS circuit construction, and pytket passes to implement Pauli and Clifford frame randomisation. Leakage detection is also discussed. |

Note that the Guppy notebook gives an extensive explanation of the
benchmarking method, while the pytket notebook focuses on circuit-based implementation.

## Getting started

The repository uses [uv](https://docs.astral.sh/uv/) for Python environment and
dependency management. Install `uv` and have `make` available, then run from the
repository root:

```sh
make setup
```

This creates separate environments in `guppy/.venv` and `pytket/.venv` and registers a Jupyter kernel for each.

| Notebook | Kernel to use |
| --- | --- |
| `guppy/quops_in_guppy.ipynb` | `QUOPS Guppy (Python 3.13.7)` |
| `pytket/quops-with-pytket.ipynb` | `QUOPS pytket (Python 3.13.7)` |
