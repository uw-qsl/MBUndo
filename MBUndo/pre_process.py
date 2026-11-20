"""Prepare XOR and relative-phase Toffoli gates for cleanup analysis."""

from __future__ import annotations
from qiskit.circuit import QuantumCircuit, Qubit
from qiskit.circuit.library import CCXGate, CSwapGate, DCXGate, MCMTGate, RCCXGate, SwapGate, XGate
from .symbolic_execution import BasisValues


def expand_multi_target_x_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Expand SWAP, CSwap, DCX and X-base MCMT into single-target X-family gates.

    This lets the analysis follow each XOR write and keep track of its target.
    """
    if not any(_is_multi_target_x_gate(instruction.operation) for instruction in circuit.data):
        return circuit
    expanded = circuit.copy_empty_like()
    for instruction in circuit.data:
        _append_expanded(expanded, instruction.operation, 
                         tuple(instruction.qubits), tuple(instruction.clbits))
    return expanded


def _append_expanded(circuit: QuantumCircuit, 
                     operation, qargs: tuple[Qubit, ...], cargs: tuple) -> None:
    if not _is_multi_target_x_gate(operation):
        circuit.append(operation, qargs, cargs)
        return
    definition = operation.definition
    circuit.global_phase += definition.global_phase
    for inner in definition.data:
        _append_expanded(circuit, inner.operation, 
                         tuple(qargs[definition.find_bit(qubit).index] for qubit in inner.qubits), ())


def _is_multi_target_x_gate(operation) -> bool:
    if operation.base_class in (SwapGate, CSwapGate, DCXGate):
        return True
    return isinstance(operation, MCMTGate) and operation.base_gate.base_class is XGate


def restore_rccx_pairs(circuit: QuantumCircuit, ancillas: frozenset[Qubit]) -> QuantumCircuit:
    """Replace two RCCX gates with CCX gates when their relative phases cancel.

    Work backwards from the ancillas' final zeros. The values just after the
    first RCCX must match those just before the second on both controls and
    the target. Control order matters because the phase on basis state
    (a, b, t) is ``i**(a*b + 2*a*t)``.
    """
    if not any(instruction.operation.base_class is RCCXGate for instruction in circuit.data):
        return circuit
    values = BasisValues(circuit.qubits, ancillas)
    unmatched_by_input_triple = {}
    paired = set()
    for position in reversed(range(len(circuit.data))):
        instruction = circuit.data[position]
        operation = instruction.operation
        qargs = tuple(instruction.qubits)
        if operation.base_class is not RCCXGate:
            values.advance(operation, qargs, backwards=True)
            continue
        output_triple = tuple(values.value[qubit] for qubit in qargs)
        # RCCX flips the same basis bit as CCX.
        values.advance(CCXGate(), qargs, backwards=True)
        input_triple = tuple(values.value[qubit] for qubit in qargs)
        partners = unmatched_by_input_triple.get(output_triple)
        if partners:
            paired.update((partners.pop(), position))
        else:
            unmatched_by_input_triple.setdefault(input_triple, []).append(position)
    if not paired:
        return circuit
    restored = circuit.copy_empty_like()
    for position, instruction in enumerate(circuit.data):
        operation = CCXGate(label=instruction.operation.label) if position in paired else instruction.operation
        restored.append(operation, instruction.qubits, instruction.clbits)
    return restored
