"""Replace eligible Toffoli gates with relative-phase Toffoli gates after MBU."""

from __future__ import annotations
from collections.abc import Iterable
from math import pi
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.library import CCXGate, RCCXGate
from .symbolic_execution import BasisValues


def apply_rccx_rewrites(circuit: QuantumCircuit, known_zero_qubits: Iterable[Qubit], 
                        enable_pair_rewrite: bool = True, enable_constant_target_rewrite: bool = True) -> QuantumCircuit:
    """Rewrite top-level CCX gates with two positive controls.

    Track basis values forwards, assuming each qubit in ``known_zero_qubits``
    starts in |0>. There are two ways to use RCCX here.

    1. If both controls and the target have the same values just after one
       CCX as just before another, replace both. Their relative phases cancel.
    2. For an unpaired CCX whose target is known to be 0 or 1, use one RCCX
       and a Clifford correction for that target value.
    """
    if not (enable_pair_rewrite or enable_constant_target_rewrite):
        return circuit
    pair_positions, target_constants = _find_rccx_rewrites(circuit, frozenset(known_zero_qubits), 
                                                           enable_pair_rewrite, enable_constant_target_rewrite)
    return _emit_rccx_rewrites(circuit, pair_positions, target_constants)


def _find_rccx_rewrites(circuit: QuantumCircuit, known_zero_qubits: frozenset[Qubit], 
                        enable_pairs: bool, enable_constants: bool) -> tuple[frozenset[int], dict[int, int]]:
    values = BasisValues(circuit.qubits, known_zero_qubits)
    unmatched_by_output_triple = {}
    pair_positions = set()
    target_constants = {}
    for position, instruction in enumerate(circuit.data):
        operation = instruction.operation
        qargs = tuple(instruction.qubits)
        closes_pair = False
        if _is_positive_ccx(operation):
            if enable_constants:
                constant = values.constant(qargs[-1])
                if constant is not None:
                    target_constants[position] = constant
            if enable_pairs:
                input_triple = tuple(values.value[qubit] for qubit in qargs)
                partners = unmatched_by_output_triple.get(input_triple)
                if partners:
                    pair_positions.update((partners.pop(), position))
                    closes_pair = True
        # An existing RCCX updates the basis labels just as a CCX would.
        values.advance(CCXGate() if isinstance(operation, RCCXGate) else operation, qargs)
        if enable_pairs and _is_positive_ccx(operation) and not closes_pair:
            output_triple = tuple(values.value[qubit] for qubit in qargs)
            unmatched_by_output_triple.setdefault(output_triple, []).append(position)
    return frozenset(pair_positions), target_constants


def _emit_rccx_rewrites(circuit: QuantumCircuit, pair_positions: frozenset[int], 
                        target_constants: dict[int, int]) -> QuantumCircuit:
    rewritten = circuit.copy_empty_like()
    rewritten.global_phase = circuit.global_phase
    for position, instruction in enumerate(circuit.data):
        if position in pair_positions:
            _emit_exact_rccx_rewrite(rewritten, instruction.qubits, None)
        elif position in target_constants:
            _emit_exact_rccx_rewrite(rewritten, instruction.qubits, target_constants[position])
        else:
            rewritten.append(instruction.operation, instruction.qubits, instruction.clbits)
    return rewritten


def _emit_exact_rccx_rewrite(circuit: QuantumCircuit, qargs, target_constant: int | None) -> None:
    circuit.append(RCCXGate(), qargs)
    if target_constant is None:
        return
    target = qargs[-1]
    if target_constant == 0:
        circuit.sdg(target)
    else:
        circuit.s(target)
        circuit.z(qargs[0])
        # For a target that starts at 1, these gates leave a global factor of i.
        circuit.global_phase -= pi / 2


def _is_positive_ccx(operation) -> bool:
    return isinstance(operation, CCXGate) and operation.ctrl_state == 3
