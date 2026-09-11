from pytket import Circuit, OpType
from pytket.passes import CustomPass


def _remove_resets_transform(circ: Circuit) -> Circuit:
    new_circ = Circuit()

    for q in circ.qubits:
        new_circ.add_qubit(q)
    for b in circ.bits:
        new_circ.add_bit(b)

    if circ.name:
        new_circ.name = circ.name

    for cmd in circ:
        if cmd.op.type != OpType.Reset:
            new_circ.add_gate(cmd.op, cmd.args)

    return new_circ


RemoveResets = CustomPass(_remove_resets_transform)
