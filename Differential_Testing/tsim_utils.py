"""
Translates Qiskit circuits into Tsim syntax and runs them through Tsim sampler

Tsim gateset used: {H, S, T, CNOT}
Qiskit transpilation: ['h', 's', 't', 'cx']

R <q> - reset qubit to |0>
H <q> - Hadamard
S <q> - Phase gate
T <q> - T gate
CNOT <c> <t> - controlled X
M <q>... - measure qubits
"""

from __future__ import annotations
from typing import Sequence
from qiskit import QuantumCircuit, transpile

TSIM_GATESET: list[str] = ["h", "s", "t", "cx"]
_QUBIT_MAP: dict[str, str] = {
    "h": "H",
    "s": "S",
    "t": "T",
    "x": "X",
    "y": "Y",
    "z": "Z",
}

def qiskit_to_tsim_program(qc: QuantumCircuit, *, initial_state: Sequence[int] | None = None, measure_all: bool = True, optimise: bool = False,
) -> str:
    """Translate a Qiskit circuit into a string with the syntax for a Tsim program

    qc: Qiskit circuit being translated
    initial_state: sequence of 0/1 values (default all zero)
    measure_all: if true append M <q0> <q1> … for every qubit at the end of the program
    optimise: if true pass optimisation_level=3 to Qiskit transpiler, otherwise optimisation_level=0

    Returns: str in format comatible with Tsim
    """
    n = qc.num_qubits
    if initial_state is None:
        initial_state = [0] * n
    if len(initial_state) != n:
        raise ValueError(f"initial_state length {len(initial_state)} not equal to num_qubits {n}")

    opt_level = 3 if optimise else 0
    transpiled = transpile(qc, basis_gates=TSIM_GATESET, optimization_level=opt_level,)
    lines: list[str] = []

    # Reset all qubits (required by Tsim)
    lines.append("# INITIALISATION")
    reset_line = "R " + " ".join(str(i) for i in range(n))
    lines.append(reset_line)
    # Apply X if qubit initialised to |1>.
    ones = [str(i) for i, b in enumerate(initial_state) if b]
    if ones:
        lines.append("X " + " ".join(ones))
    lines.append("# CIRCUIT")
    for instr in transpiled.data: # Walk transpiled instructions
        name = instr.operation.name
        q_indices = [transpiled.find_bit(q).index for q in instr.qubits]
        if name == "cx":
            ctrl, tgt = q_indices
            lines.append(f"CNOT {ctrl} {tgt}")
        elif name in _QUBIT_MAP:
            tsim_name = _QUBIT_MAP[name]
            for qi in q_indices:
                lines.append(f"{tsim_name} {qi}")
        elif name in ("barrier", "id"):
            pass
        elif name == "measure":
            pass # M added at end instead
        else:
            raise ValueError(
                f"Unsupported gate '{name}' encountered during Tsim "
                f"translation.  Transpile to {TSIM_GATESET} first."
            )
    if measure_all:
        lines.append("# MEASURE")
        measure_line = "M " + " ".join(str(i) for i in range(n))
        lines.append(measure_line)
    return "\n".join(lines) + "\n"

def run_tsim(
    qc: QuantumCircuit, *, initial_state: Sequence[int] | None = None, shots: int = 4096, seed: int = 0, optimise: bool = False,
) -> int | None:
    """Run a circuit through Tsim and return dominant output integer.
    Classical  circuits always have same exact result (dominant output has probability 1)
    Superposition circuits: dominant outcome returned (non-definitive answer)

    qc: input qiskit circuit
    initial_state: per-qubit initial value (defaults to all zeros)
    shots: Number of samples to draw.
    seed: random seed passed to compile_sampler
    optimise: whether to optimise qiskit transpilation.

    Returns int or None - Dominant output as an integer (little-endian bit order) or None if tsim not installed
    """
    try:
        import tsim
        import numpy as np
    except ImportError:
        return None
    n = qc.num_qubits
    program_text = qiskit_to_tsim_program(qc, initial_state=initial_state, measure_all=True, optimise=optimise,)
    circuit = tsim.Circuit(program_text)
    sampler = circuit.compile_sampler(seed=seed)
    raw = sampler.sample(shots=shots)
    powers = np.array([1 << i for i in range(n)], dtype=np.int64)
    integers = (raw.astype(np.int64) * powers).sum(axis=1)
    values, counts = np.unique(integers, return_counts=True)
    dominant_int = int(values[counts.argmax()])
    return dominant_int

TSIM_LABEL = "Tsim [{H,S,T,CX}]"
