## MBUndo

MBUndo is a tool that takes in a circuit using unitary uncomputation to uncompute temporary values and transforms that uncomputation into measurement-based uncomputation soundly, efficiently and automatically.
MBUndo implements endpoint (see Sec. 4.1 of the accompanying paper) and interval (see Sec. 4.2 of the accompanying paper) measurement-based uncomputation for Qiskit circuits. The `examples/` directory contains benchmark circuits synthesized by a state-of-the-art quantum circuit uncomputation synthesizer called [Reqomp](https://github.com/eth-sri/Reqomp), with several implementation bugs fixed. 

## Build MBUndo

Install uv, then run the following command from the project root. [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv sync --locked
```

This creates `.venv` with Python 3.10 or later, Qiskit 2.5.2 and the local MBUndo package. Verify the installation by importing both packages.

```sh
uv run python -c "import MBUndo, qiskit; print('Qiskit', qiskit.__version__)"
```

The expected output is `Qiskit 2.5.2`. To build a wheel and source archive in `dist/`, run `uv build`.

## Run MBUndo

From the project root, run MBUndo on a QPY circuit from `examples/`. Use `--mode` to select `interval` or `endpoint`, and `--rccx-on` or `--rccx-off` to enable or disable RCCX post-processing (see Sec. 5 of the accompanying paper) on the output of that pass. The default is `--mode interval --rccx-on`.

```sh
uv run python run_mbundo.py examples/adder.qpy --mode interval --rccx-on  -o outputs/adder_interval_rccx_on.qpy
```

The required `-o` (or `--output`) argument sets the path of the transformed QPY file. This example writes `outputs/adder_interval_rccx_on.qpy` relative to the project root. MBUndo creates the directory if needed and prints the output path.

The full artifact for reproducing all the data reported in the paper will be released on Zenodo.
