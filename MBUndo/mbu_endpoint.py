"""Endpoint MBU replaces the final XOR in an ancilla lifetime with a measurement."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit, Qubit
from .phase_correction import emit_xor_phase_correction, measurement_register
from .pre_process import expand_multi_target_x_gates, restore_rccx_pairs
from .symbolic_execution import BasisValues, has_t_count, is_exact_xor, validate_input


@dataclass(frozen=True, slots=True)
class CleanupEndpoint:
    """An ancilla and the index of its cleanup gate in ``circuit.data``."""

    target: Qubit
    position: int


def optimize_cleanup_endpoints(circuit: QuantumCircuit, ancillas: Iterable[Qubit], 
                               enable_rccx_pair_restoration: bool = True, 
                               enable_multi_target_expansion: bool = True) -> QuantumCircuit:
    """Return a circuit with eligible cleanup gates replaced by X-basis measurements.

    Each marked ancilla must start and finish in |0>. A final X or CX stays
    in the circuit because replacing it would save no T gates.
    """
    ancillas = frozenset(ancillas)
    if enable_multi_target_expansion:
        circuit = expand_multi_target_x_gates(circuit)
    if enable_rccx_pair_restoration:
        circuit = restore_rccx_pairs(circuit, ancillas)
    endpoints = tuple(endpoint for endpoint in find_cleanup_endpoints(circuit, ancillas) 
                      if has_t_count(circuit.data[endpoint.position].operation))
    endpoint_by_position = {endpoint.position: endpoint for endpoint in endpoints}
    result = circuit.copy_empty_like()
    measurement_by_endpoint = {}
    if endpoints:
        register = measurement_register(result, len(endpoints))
        measurement_by_endpoint = dict(zip(endpoints, register, strict=True))
    for position, instruction in enumerate(circuit.data):
        endpoint = endpoint_by_position.get(position)
        if endpoint is None:
            result.append(instruction.operation, instruction.qubits, instruction.clbits)
            continue
        measurement = measurement_by_endpoint[endpoint]
        result.h(endpoint.target)
        result.measure(endpoint.target, measurement)
        result.reset(endpoint.target)
        emit_xor_phase_correction(result, instruction.operation, 
                                  tuple(instruction.qubits[:-1]), measurement)
    return result


def find_cleanup_endpoints(circuit: QuantumCircuit, ancillas: Iterable[Qubit]) -> tuple[CleanupEndpoint, ...]:
    """Find XOR gates that leave a marked ancilla at zero.

    Work backwards from the known final zeros. If an ancilla is reused,
    the scan can find more than one cleanup gate on that wire. 
    """
    marked = validate_input(circuit, ancillas)
    values = BasisValues(circuit.qubits, marked)
    endpoints = []
    for position in reversed(range(len(circuit.data))):
        instruction = circuit.data[position]
        operation = instruction.operation
        qargs = tuple(instruction.qubits)
        target = qargs[-1] if is_exact_xor(operation) else None
        if target in marked and values.is_proved_zero(target):
            endpoints.append(CleanupEndpoint(target, position))
        values.advance(operation, qargs, backwards=True)
    return tuple(sorted(endpoints, key=lambda endpoint: endpoint.position))
