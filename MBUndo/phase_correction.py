"""Phase corrections for X-basis measurements in endpoint and interval MBU."""

from __future__ import annotations
from collections.abc import Sequence
from math import pi
from qiskit.circuit import ClassicalRegister, Clbit, Gate, QuantumCircuit, Qubit
from qiskit.circuit.library import MCPhaseGate
from .symbolic_execution import active_low_controls


def measurement_register(circuit: QuantumCircuit, size: int) -> ClassicalRegister:
    """Add a fresh classical register with one bit per measurement."""
    used_register_names = {register.name for register in (*circuit.qregs, *circuit.cregs)}
    name = "mbu_m0"
    while name in used_register_names:
        name += "_"
    register = ClassicalRegister(size, name)
    circuit.add_register(register)
    return register


def emit_xor_phase_correction(circuit: QuantumCircuit, operation: Gate, 
                              controls: Sequence[Qubit], measurement_bit: Clbit) -> None:
    """Correct the phase on the XOR gate's controls when the measurement is 1.

    An unconditional X contributes only a global phase within that branch.
    """
    if not controls:
        return
    active_low = active_low_controls(operation, controls)
    with circuit.if_test((measurement_bit, 1)):
        for control in active_low:
            circuit.x(control)
        emit_semantic_phase_factor(circuit, controls)
        for control in reversed(active_low):
            circuit.x(control)


def emit_semantic_phase_factor(circuit: QuantumCircuit, factor_qubits: Sequence[Qubit]) -> None:
    """For a nonempty set of qubits, flip the phase of the all-ones basis state."""
    degree = len(factor_qubits)
    if degree == 0:
        return
    if degree == 1:
        circuit.z(factor_qubits[0])
        return
    if degree == 2:
        circuit.cz(factor_qubits[0], factor_qubits[1])
        return
    circuit.append(MCPhaseGate(pi, degree - 1), factor_qubits)
