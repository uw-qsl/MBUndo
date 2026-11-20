"""Apply measurement-based uncomputation to a Qiskit QPY circuit."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from qiskit import QuantumCircuit, qpy
from qiskit.exceptions import QiskitError

from MBUndo import apply_rccx_rewrites, optimize_cleanup_endpoints, optimize_cleanup_intervals


def ancilla_indices(circuit: QuantumCircuit) -> list[int]:
    """Return the qubit indices declared as ancillas in the QPY metadata."""
    if "ancillas" not in circuit.metadata:
        raise ValueError("Set circuit.metadata['ancillas'] to the list of ancilla qubit indices before saving the input.")
    indices = circuit.metadata["ancillas"]
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < circuit.num_qubits for i in indices):
        raise ValueError("metadata['ancillas'] must list integer indices within the circuit's qubit range.")
    if len(set(indices)) != len(indices):
        raise ValueError("metadata['ancillas'] contains duplicate qubit indices.")
    return indices


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python run_mbundo.py examples/mcx.qpy --mode interval --rccx-on -o mcx_interval.qpy",
    )
    parser.add_argument("input", type=Path, help="QPY containing one circuit with declared ancillas")
    parser.add_argument("-o", "--output", type=Path, required=True, help="destination QPY for the transformed circuit")
    parser.add_argument("--mode", choices=("interval", "endpoint"), default="interval", help="MBU pass (default: interval)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rccx-on", dest="rccx", action="store_true", help="enable RCCX post-processing (default)")
    group.add_argument("--rccx-off", dest="rccx", action="store_false", help="skip RCCX post-processing")
    parser.set_defaults(rccx=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose an output path distinct from the input QPY.")

    try:
        with args.input.open("rb") as stream:
            circuits = qpy.load(stream)
        if len(circuits) != 1:
            raise ValueError(f"Expected one input circuit, found {len(circuits)}.")
        circuit = circuits[0]
        indices = ancilla_indices(circuit)
        ancillas = [circuit.qubits[i] for i in indices]
        optimize = optimize_cleanup_intervals if args.mode == "interval" else optimize_cleanup_endpoints
        result = optimize(circuit, ancillas)
        if args.rccx:
            result = apply_rccx_rewrites(result, ancillas)
        producer = {"tool": "MBUndo", "mode": args.mode, "rccx_post": args.rccx}
        if "producer" in circuit.metadata:
            producer["input_producer"] = circuit.metadata["producer"]
        result.metadata = {**circuit.metadata, "format": "mbu-circuit-qpy", "version": 1,
                           "ancillas": indices, "producer": producer}

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as directory:
            pending = Path(directory) / "circuit.qpy"
            with pending.open("w+b") as stream:
                qpy.dump(result, stream)
                # QPY can write some wide controlled gates that fail when read back.
                stream.seek(0)
                qpy.load(stream)
            pending.replace(args.output)
    except (OSError, ValueError, QiskitError, OverflowError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
