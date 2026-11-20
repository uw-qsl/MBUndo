"""Track which qubits have equal or constant computational-basis values.

Each qubit has a symbolic label. Three rules update these labels (Sec. 3 of the paper). 
Xor combines the target label with the gate's control condition by XOR. 
For a CX with control c and target t, the new label is t XOR c. 
Diag keeps the labels because diagonal gates change only phases.
Forget assigns a fresh symbol to each qubit written by any other gate.
Matching labels prove that two values agree on every computational-basis path.

The cleanup passes and RCCX pair restoration work backwards from the ancillas'
final zeros. RCCX post-processing works forwards from their initial zeros.
An XOR undoes itself, so the same update works in either direction.

A measurement makes the measured qubit's label unknown. At an if/else,
the analysis forgets the label of any qubit that either branch might write.
A reset gives a known zero going forwards. Going backwards, the value
before the reset is unknown.
"""

from __future__ import annotations
from collections.abc import Iterable, Sequence
from qiskit.circuit import Barrier, ControlledGate, Delay, Gate, IfElseOp, Measure, QuantumCircuit, Qubit, Reset
from qiskit.circuit.library import C3XGate, C4XGate, CCXGate, CXGate, DiagonalGate, IGate, MCXGate, PhaseGate, RCCXGate, RZGate, RZZGate, SGate, SdgGate, TGate, TdgGate, U1Gate, XGate, ZGate


def validate_input(circuit: QuantumCircuit, ancillas: Iterable[Qubit]) -> frozenset[Qubit]:
    """Check the ancilla list and the top-level instructions.

    An if/else block is accepted as a whole. Other top-level control flow,
    including loops and switches, is rejected.
    """
    marked = frozenset(ancillas)
    if not marked.issubset(circuit.qubits):
        raise ValueError("all marked ancillas must belong to the input circuit")
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, (Measure, IfElseOp)):
            continue
        if instruction.clbits or not isinstance(operation, (Gate, Barrier, Reset)):
            raise ValueError(f"MBU accepts unitary gates, barriers, measurements, resets and if/else blocks, not {operation.name}")
    return marked


class BasisValues:
    """Represent each qubit's basis value as an XOR of symbolic atoms.

    The empty set means zero. Atom 0 represents the constant one, so toggling
    that atom negates the value. Equal sets prove that two values agree on
    every computational-basis path. Different sets can still represent equal values.
    """

    def __init__(self, qubits: Iterable[Qubit], zero: frozenset[Qubit]) -> None:
        self.one = frozenset((0,))
        self.next_atom = 1
        self.products = {}
        self.value = {qubit: frozenset() if qubit in zero else self._fresh() for qubit in qubits}

    def is_proved_zero(self, qubit: Qubit) -> bool:
        return not self.value[qubit]

    def constant(self, qubit: Qubit) -> int | None:
        """Return 0 or 1 if known, otherwise None."""
        value = self.value[qubit]
        if not value:
            return 0
        if value == self.one:
            return 1
        return None

    def advance(self, operation, qargs: tuple[Qubit, ...], backwards: bool = False) -> None:
        """Update the labels across one instruction.

        With ``backwards=True``, use the labels after the instruction to infer
        the labels before it.
        """
        if is_exact_xor(operation):
            controls, target = qargs[:-1], qargs[-1]
            if controls:
                active_low = frozenset(active_low_controls(operation, controls))
                factors = tuple(self.value[control] ^ self.one 
                                if control in active_low else self.value[control] for control in controls)
                predicate = self._product(factors)
            else:
                predicate = self.one
            self.value[target] ^= predicate
        elif isinstance(operation, Reset):
            # After a reset the bit is zero. Its previous value could have been anything.
            self.value[qargs[0]] = self._fresh() if backwards else frozenset()
        elif isinstance(operation, IfElseOp):
            # Either branch can run, so forget any qubit a branch might write.
            for qubit in block_written_qubits(operation, qargs):
                self.value[qubit] = self._fresh()
        elif not preserves_basis_labels(operation):
            # Even an unknown controlled gate preserves its control bits.
            written = qargs[operation.num_ctrl_qubits:] if isinstance(operation, ControlledGate) else qargs
            for qubit in written:
                self.value[qubit] = self._fresh()

    def _product(self, factors: tuple[frozenset[int], ...]) -> frozenset[int]:
        """Simplify the AND of the factors and reuse the label for repeated products."""
        if any(not factor for factor in factors):
            return frozenset()
        operands = frozenset(factor for factor in factors if factor != self.one)
        if not operands:
            return self.one
        if len(operands) == 1:
            return next(iter(operands))
        if operands not in self.products:
            self.products[operands] = self._pair_product(*operands) if len(operands) == 2 else self._fresh()
        return self.products[operands]

    def _pair_product(self, left: frozenset[int], right: frozenset[int]) -> frozenset[int]:
        """Multiply two labels using (U + e)(T + d) = UT + eT + dU + ed.

        Here + means XOR, and e and d are the constant bits. Pulling them
        out lets positive and negative controls share the same product UT.
        """
        core_left, core_right = left - self.one, right - self.one
        if core_left == core_right:
            # A Boolean value ANDed with itself is the same value.
            core = core_left
        else:
            key = frozenset((core_left, core_right))
            if key not in self.products:
                self.products[key] = self._fresh()
            core = self.products[key]
        value = core
        if 0 in left:
            value ^= core_right
        if 0 in right:
            value ^= core_left
        if 0 in left and 0 in right:
            value ^= self.one
        return value

    def _fresh(self) -> frozenset[int]:
        atom = self.next_atom
        self.next_atom += 1
        return frozenset((atom,))


def is_exact_xor(operation) -> bool:
    """Recognize standard single-target X gates with no relative phase."""
    if operation.base_class is XGate:
        return True
    if operation.base_class not in (CXGate, CCXGate, C3XGate, C4XGate, MCXGate):
        return False
    return operation.num_qubits == operation.num_ctrl_qubits + 1


def has_t_count(operation) -> bool:
    """For an exact XOR gate, return whether its decomposition needs T gates."""
    return bool(getattr(operation, "num_ctrl_qubits", 0) >= 2)


def active_low_controls(operation: ControlledGate, controls: Sequence[Qubit]) -> tuple[Qubit, ...]:
    """Return the controls whose corresponding bit in ``ctrl_state`` is zero."""
    return tuple(control for position, control in enumerate(controls) if not (operation.ctrl_state >> position) & 1)


def preserves_basis_labels(operation) -> bool:
    """Recognize diagonal gates (including controlled ones), barriers and delays."""
    if isinstance(operation, (Barrier, Delay, IGate, DiagonalGate, PhaseGate, 
                              U1Gate, RZGate, RZZGate, ZGate, SGate, SdgGate, TGate, TdgGate)):
        return True
    return isinstance(operation, ControlledGate) and preserves_basis_labels(operation.base_gate)


def block_written_qubits(operation: IfElseOp, qargs: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
    """Return the qubits in the enclosing circuit that either branch may change."""
    positions = set()
    for block in operation.blocks:
        for instruction in block.data:
            for qubit in _written_qubits(instruction.operation, tuple(instruction.qubits)):
                positions.add(block.find_bit(qubit).index)
    return tuple(qargs[position] for position in sorted(positions))


def _written_qubits(operation, qargs: tuple[Qubit, ...]) -> tuple[Qubit, ...]:
    if is_exact_xor(operation) or isinstance(operation, RCCXGate):
        return qargs[-1:]
    if isinstance(operation, IfElseOp):
        return block_written_qubits(operation, qargs)
    if preserves_basis_labels(operation):
        return ()
    if isinstance(operation, ControlledGate):
        return qargs[operation.num_ctrl_qubits:]
    return qargs
