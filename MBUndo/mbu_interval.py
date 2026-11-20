"""Interval MBU uses one measurement for a group of XOR writes to an ancilla.

Measure and reset the ancilla before the first write. Apply each phase
correction where its XOR was, so it uses the same control values.
"""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit, Qubit
from .phase_correction import emit_xor_phase_correction, measurement_register
from .pre_process import expand_multi_target_x_gates, restore_rccx_pairs
from .symbolic_execution import BasisValues, has_t_count, is_exact_xor, validate_input


@dataclass(frozen=True, slots=True)
class CleanupInterval:
    """An ancilla and the indices of its XOR writes, in circuit order."""

    target: Qubit
    positions: tuple[int, ...]


def optimize_cleanup_intervals(circuit: QuantumCircuit, ancillas: Iterable[Qubit], 
                               enable_rccx_pair_restoration: bool = True, 
                               enable_multi_target_expansion: bool = True) -> QuantumCircuit:
    """Replace each selected group of cleanup XORs with a measurement and phase corrections.

    Each marked ancilla must start and finish in |0>. An interval containing
    just one X or CX stays in the circuit because replacing it would save no T gates.
    """
    ancillas = frozenset(ancillas)
    if enable_multi_target_expansion:
        circuit = expand_multi_target_x_gates(circuit)
    if enable_rccx_pair_restoration:
        circuit = restore_rccx_pairs(circuit, ancillas)
    intervals = tuple(interval for interval in find_cleanup_intervals(circuit, ancillas) 
                      if len(interval.positions) > 1 or has_t_count(circuit.data[interval.positions[0]].operation))
    return _emit_cleanup_intervals(circuit, intervals)


def find_cleanup_intervals(circuit: QuantumCircuit, 
                           ancillas: Iterable[Qubit]) -> tuple[CleanupInterval, ...]:
    """Find groups of XOR writes that end with a marked ancilla at zero.

    Scan backwards from the final zeros. When an XOR leaves its target at
    zero, collect earlier XOR writes to that target. Stop at any other use
    of the target or when the analysis proves it was already zero before a write.
    """
    marked = validate_input(circuit, ancillas)
    values = BasisValues(circuit.qubits, marked)
    pending_writes = {qubit: [] for qubit in circuit.qubits if qubit in marked}
    intervals = []

    def close_interval(target: Qubit) -> None:
        writes = pending_writes[target]
        if writes:
            intervals.append(CleanupInterval(target, tuple(reversed(writes))))
            writes.clear()

    for position in reversed(range(len(circuit.data))):
        instruction = circuit.data[position]
        operation = instruction.operation
        qargs = tuple(instruction.qubits)
        target = qargs[-1] if is_exact_xor(operation) else None
        is_marked_target_xor = target in pending_writes
        for qubit in qargs:
            if qubit in pending_writes and qubit != target:
                close_interval(qubit)
        if is_marked_target_xor:
            extends_open_interval = bool(pending_writes[target])
            opens_at_zero_cut = values.is_proved_zero(target)
            if extends_open_interval or opens_at_zero_cut:
                pending_writes[target].append(position)
        values.advance(operation, qargs, backwards=True)
        if is_marked_target_xor and values.is_proved_zero(target):
            close_interval(target)
    for qubit in pending_writes:
        close_interval(qubit)
    return tuple(sorted(intervals, key=lambda interval: interval.positions[0]))


def _emit_cleanup_intervals(circuit: QuantumCircuit, 
                            intervals: tuple[CleanupInterval, ...]) -> QuantumCircuit:
    interval_by_position = {position: interval for interval in intervals for position in interval.positions}
    result = circuit.copy_empty_like()
    measurement_by_interval = {}
    if intervals:
        register = measurement_register(result, len(intervals))
        measurement_by_interval = dict(zip(intervals, register, strict=True))
    for position, instruction in enumerate(circuit.data):
        interval = interval_by_position.get(position)
        if interval is None:
            result.append(instruction.operation, instruction.qubits, instruction.clbits)
            continue
        measurement = measurement_by_interval[interval]
        if position == interval.positions[0]:
            result.h(interval.target)
            result.measure(interval.target, measurement)
            result.reset(interval.target)
        emit_xor_phase_correction(result, instruction.operation, tuple(instruction.qubits[:-1]), measurement)
    return result
