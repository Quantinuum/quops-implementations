from pytket.passes import CustomPass
from pytket.circuit import Circuit, OpType
import numpy as np
from enum import Enum, auto
from pytket.circuit import BitRegister


LEAKAGE_DETECTION_BIT_NAME = "leakage_detection_bit"
LEAKAGE_DETECTION_QUBIT_NAME = "leakage_detection_qubit"

SHOT_COUNT_REGISTER_SIZE = 32
RANDOM_NUMBER_REGISTER_SIZE = 32
BOUND_REGISTER_SIZE = 32
SEED_REGISTER_SIZE = 64
SIGN_SAFE_RANDOM_BITS = 31

# Seed layout: 32 bits of shot number, 31 payload bits, and one sign bit left unused.
assert SEED_REGISTER_SIZE == SHOT_COUNT_REGISTER_SIZE + SIGN_SAFE_RANDOM_BITS + 1

def _copy_command_with_barriers(src_circuit: Circuit, dst_circuit: Circuit, cmd) -> None:
    if cmd.op.type == OpType.Barrier:
        dst_circuit.add_barrier(cmd.args)
    else:
        dst_circuit.add_gate(Op=cmd.op, args=cmd.args)


class RandomnessType(Enum):
    """Supported randomness lifetimes for the randomisation passes."""

    PER_SHOT = auto()
    NO_CLASSICAL = auto()


def int_to_bit_values(value: int, size: int = RANDOM_NUMBER_REGISTER_SIZE) -> list[bool]:
    """Return the little-endian bit expansion of ``value`` with fixed width ``size``."""

    return [bool((value >> bit_index) & 1) for bit_index in range(size)]


def initialise_rng_seed(
    circuit: Circuit,
    rng: np.random.Generator,
    seed_register_name: str,
) -> None:
    """Create and populate the seed register used by pytket shot-level RNG ops.

    The first 32 bits are filled from the job shot number. The remaining
    31 payload bits are filled from compile-time randomness, leaving the final
    sign bit unset to avoid backend issues with signed classical values.
    """

    seed_register = circuit.add_c_register(name=seed_register_name, size=SEED_REGISTER_SIZE)
    circuit.get_job_shot_num(
        creg=BitRegister(name=seed_register_name, size=SHOT_COUNT_REGISTER_SIZE)
    )
    circuit.add_c_setbits(
        values=[bool(value) for value in rng.choice(a=[True, False], size=SIGN_SAFE_RANDOM_BITS)],
        args=seed_register.to_list()[
            SHOT_COUNT_REGISTER_SIZE : SHOT_COUNT_REGISTER_SIZE + SIGN_SAFE_RANDOM_BITS
        ],
    )
    circuit.set_rng_seed(creg=seed_register)


def configure_rng_bound(
    circuit: Circuit,
    bound_register_name: str,
    upper_bound: int,
) -> None:
    """Create the bound register consumed by bounded shot-level RNG generation."""

    bound_register = circuit.add_c_register(name=bound_register_name, size=BOUND_REGISTER_SIZE)
    circuit.add_c_setbits(
        values=int_to_bit_values(upper_bound, BOUND_REGISTER_SIZE),
        args=bound_register.to_list(),
    )
    circuit.set_rng_bound(creg=bound_register)


def populate_random_register(
    register: BitRegister,
    rng: np.random.Generator,
    upper_bound: int | None = None,
) -> list[bool]:
    """Generate compile-time bit values for ``register``.

    When ``upper_bound`` is omitted, each bit is sampled independently.
    When ``upper_bound`` is provided, a single integer in ``[0, upper_bound)``
    is drawn and expanded into the register width.
    """

    bit_count = len(register.to_list())
    if upper_bound is None:
        return [bool(value) for value in rng.choice(a=[True, False], size=bit_count)]

    return int_to_bit_values(int(rng.integers(upper_bound)), bit_count)

def FrameRandomisation(
    rng: np.random.Generator = np.random.default_rng(),
    randomness_type: RandomnessType = RandomnessType.PER_SHOT,
) -> CustomPass:
    """Build a pass that wraps a circuit in randomly chosen single-qubit Clifford gates.

    The pass samples one value in ``[0, 6)`` per qubit either per shot by using the
    circuit RNG, or once per compiled circuit by drawing compile-time values from
    ``rng``. It applies the corresponding Clifford gates before the
    input circuit, appends the original circuit, and then applies the inverse
    frame afterwards.

    This preserves the overall unitary only when the wrapped circuit has
    identity input-output wire mapping and is used in a mirror/identity setting
    (e.g. forward-plus-dagger benchmarks). In particular, the pass now rejects
    circuits with implicit wire swaps.

    :param rng: Random number generator used to seed the per-shot
        device RNG state embedded into the output circuit.
    :param randomness_type: Whether randomness changes per shot, is fixed per
        compiled circuit, or is sampled with no extra classical registers
        (``RandomnessType.NO_CLASSICAL``), defaults to RandomnessType.PER_SHOT
    :returns: A pytket ``CustomPass`` that transforms a measurement-free circuit
        into an equivalent frame-randomised circuit.
    :raises ValueError: If the input circuit already contains measurements, has
        implicit wire swaps, or uses classical register names reserved by this pass.
    """

    def transform(raw_circuit: Circuit) -> Circuit:
        """Insert the frame-randomisation scaffold around ``raw_circuit``.

        The transformed circuit preserves the original qubits and classical bits,
        allocates the RNG registers needed for frame selection, applies the chosen
        frame before ``raw_circuit``, and uncomputes that frame afterwards.
        The intended use is when ``raw_circuit`` is logically identity on the
        measured wires.

        :param raw_circuit: The circuit to frame-randomise. It must not contain
            measurements or implicit wire swaps because the pass appends
            extra operations after the original body and assumes identity wire
            mapping for cancellation.
        :returns: A new circuit containing the original circuit wrapped by
            conditional frame and inverse-frame operations.
        """

        if any(command.op.type == OpType.Measure for command in raw_circuit.get_commands()):
            raise ValueError("FrameRandomisation expects a circuit with no measurements.")

        if raw_circuit.has_implicit_wireswaps:
            raise ValueError(
                "FrameRandomisation does not support circuits with implicit wire swaps."
            )

        if any(
            register.name == "frame_seed"
            or register.name.startswith("frame_rng_")
            or register.name.startswith("frame_commuted_pauli_")
            for register in raw_circuit.c_registers
        ):
            raise ValueError(
                'Registers named "frame_seed", or starting with "frame_rng_" or "frame_commuted_pauli_" '
                "are reserved for FrameRandomisation."
            )
        # Add all the qubits and bits from the original circuit.
        frame_circuit = Circuit()
        for qubit in raw_circuit.qubits:
            frame_circuit.add_qubit(qubit)
        for bit in raw_circuit.bits:
            frame_circuit.add_bit(bit)

        if randomness_type == RandomnessType.PER_SHOT:
            initialise_rng_seed(
                circuit=frame_circuit,
                rng=rng,
                seed_register_name="frame_seed",
            )
            configure_rng_bound(
                circuit=frame_circuit,
                bound_register_name="frame_rng_bound",
                upper_bound=6,
            )

        def _apply_frame(qubit, frame_index: int, *, condition_bits=None) -> None:
            if frame_index in [1, 3, 5]:
                if condition_bits is None:
                    frame_circuit.X(qubit)
                else:
                    frame_circuit.X(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )
            if frame_index in [2, 3, 4, 5]:
                if condition_bits is None:
                    frame_circuit.H(qubit)
                else:
                    frame_circuit.H(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )
            if frame_index in [4, 5]:
                if condition_bits is None:
                    frame_circuit.S(qubit)
                else:
                    frame_circuit.S(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )

        def _apply_inverse_frame(qubit, frame_index: int, *, condition_bits=None) -> None:
            if frame_index in [4, 5]:
                if condition_bits is None:
                    frame_circuit.Sdg(qubit)
                else:
                    frame_circuit.Sdg(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )
            if frame_index in [2, 3, 4, 5]:
                if condition_bits is None:
                    frame_circuit.H(qubit)
                else:
                    frame_circuit.H(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )
            if frame_index in [1, 3, 5]:
                if condition_bits is None:
                    frame_circuit.X(qubit)
                else:
                    frame_circuit.X(
                        qubit,
                        condition_bits=condition_bits,
                        condition_value=frame_index,
                    )

        qubit_rng_reg = {}
        qubit_frame_index = {}
        if randomness_type == RandomnessType.NO_CLASSICAL:
            qubit_frame_index = {
                qubit: int(rng.integers(6))
                for qubit in raw_circuit.qubits
            }
            for qubit, frame_index in qubit_frame_index.items():
                _apply_frame(qubit, frame_index)
        else:
            # Generate one register per qubit which will be used to store the random numbers which determine the frame changes.
            qubit_rng_reg = {
                qubit: frame_circuit.add_c_register(
                    name=f"frame_rng_qubit_{qubit}",
                    size=RANDOM_NUMBER_REGISTER_SIZE,
                )
                for qubit in raw_circuit.qubits
            }

            # For each register...
            for qubit, rng_reg in qubit_rng_reg.items():
                # Populate the register with random numbers.
                if randomness_type == RandomnessType.PER_SHOT:
                    frame_circuit.get_rng_num(creg=rng_reg)
                else:
                    frame_circuit.add_c_setbits(
                        values=populate_random_register(
                            register=rng_reg,
                            rng=rng,
                            upper_bound=6,
                        ),
                        args=rng_reg.to_list(),
                    )

                for frame_index in range(1, 6):
                    _apply_frame(qubit, frame_index, condition_bits=rng_reg.to_list())

        # Append the original circuit.
        frame_circuit.append(raw_circuit)

        # Undo the frame changes with the inverse gates.
        if randomness_type == RandomnessType.NO_CLASSICAL:
            for qubit, frame_index in qubit_frame_index.items():
                _apply_inverse_frame(qubit, frame_index)
        else:
            for qubit, rng_reg in qubit_rng_reg.items():
                for frame_index in range(1, 6):
                    _apply_inverse_frame(
                        qubit,
                        frame_index,
                        condition_bits=rng_reg.to_list(),
                    )

        return frame_circuit 
    
    return CustomPass(transform=transform)

def RandomXPass(
    rng: np.random.Generator = np.random.default_rng(),
    randomness_type: RandomnessType = RandomnessType.PER_SHOT,
) -> CustomPass:
    """Build a pass that appends random X gates to each qubit and corrects measurements.

    With probability 0.5, independently per eligible measured qubit, an X gate
    is inserted just before the final measurements of the circuit. For every
    eligible measurement, the corresponding measurement bit is XOR-ed with the
    same random bit that conditioned the X gate, so that the net logical effect
    on all randomized measured outputs is the identity.

    Measurements targeting the Quantinuum leakage-detection registers are
    preserved but excluded from randomization and classical correction.

    Random bits can either vary per shot using the circuit RNG or be fixed per
    compiled circuit from compile-time sampling.

    :param rng: Random number generator used to produce the compile-time seed
        component.
    :param randomness_type: Whether randomness varies per shot, is fixed per
        compiled circuit, or is sampled with no extra classical registers
        (``RandomnessType.NO_CLASSICAL``), defaults to RandomnessType.PER_SHOT.
    :returns: A pytket ``CustomPass`` that wraps the circuit with per-qubit
        random X gates and the corresponding classical correction.
    :raises ValueError: If the input circuit uses register names reserved by
        this pass.
    """

    def transform(raw_circuit: Circuit) -> Circuit:
        """Insert random X gates and classical corrections into ``raw_circuit``.

        The command stream is replayed in order. When a data measurement is
        encountered, a conditional X gate is inserted immediately before it and
        a classical correction immediately after. All other commands are
        replayed verbatim, preserving the relative ordering between data
        measurements and any subsequent leakage-detection ancilla operations
        on the same qubit.

        :param raw_circuit: Circuit to transform.
        :returns: Transformed circuit with random X gates and classical
            measurement correction.
        """

        if any(
            register.name == "rx_seed"
            or register.name.startswith("rx_rng_")
            for register in raw_circuit.c_registers
        ):
            raise ValueError(
                'Registers named "rx_seed", or starting with '
                '"rx_rng_" are reserved for RandomXPass.'
            )

        # Collect data measurements up front to allocate RNG registers before
        # replaying the command stream.  A "data measurement" is any Measure
        # whose qubit and target bit are not in the leakage-detection registers.
        # A qubit measured for leakage after its data measurement (ancilla reuse)
        # is not counted again.
        data_measurements = []
        data_measured_qubits = set()
        for cmd in raw_circuit.get_commands():
            if cmd.op.type != OpType.Measure:
                continue
            qubit, bit = cmd.args[0], cmd.args[1]
            if (
                bit.reg_name == LEAKAGE_DETECTION_BIT_NAME
                or qubit.reg_name == LEAKAGE_DETECTION_QUBIT_NAME
            ):
                continue
            if qubit in data_measured_qubits:
                raise ValueError(
                    "RandomXPass does not support measuring the same qubit more than once."
                )
            data_measured_qubits.add(qubit)
            data_measurements.append((qubit, bit))

        # Reconstruct the circuit preserving all original qubits and bits.
        new_circuit = Circuit()
        for qubit in raw_circuit.qubits:
            new_circuit.add_qubit(qubit)
        for bit in raw_circuit.bits:
            new_circuit.add_bit(bit)

        if randomness_type == RandomnessType.PER_SHOT and data_measurements:
            initialise_rng_seed(
                circuit=new_circuit,
                rng=rng,
                seed_register_name="rx_seed",
            )

        qubit_random_bit: dict = {}
        qubit_apply_x: dict = {}
        if randomness_type == RandomnessType.NO_CLASSICAL:
            qubit_apply_x = {
                qubit: bool(rng.integers(2))
                for qubit, _ in data_measurements
            }
        else:
            # Populate pooled 32-bit RNG registers and use one bit per qubit.
            # Ceiling division so any partial block of qubits still gets one RNG register.
            random_bit_list = []
            n_rng_registers = (
                len(data_measurements) + RANDOM_NUMBER_REGISTER_SIZE - 1
            ) // RANDOM_NUMBER_REGISTER_SIZE
            for i in range(n_rng_registers):
                random_bit_register = new_circuit.add_c_register(
                    name=f"rx_rng_{i}",
                    size=RANDOM_NUMBER_REGISTER_SIZE,
                )
                random_bit_list += random_bit_register.to_list()
                if randomness_type == RandomnessType.PER_SHOT:
                    new_circuit.get_rng_num(creg=random_bit_register)
                else:
                    new_circuit.add_c_setbits(
                        values=populate_random_register(
                            register=random_bit_register,
                            rng=rng,
                        ),
                        args=random_bit_register.to_list(),
                    )

            qubit_random_bit = {
                qubit: random_bit_list[i]
                for i, (qubit, _) in enumerate(data_measurements)
            }

        # Replay the command stream in order, inserting X and correction inline
        # around each data measurement.
        for cmd in raw_circuit.get_commands():
            is_data_measure = (
                cmd.op.type == OpType.Measure
                and cmd.args[1].reg_name != LEAKAGE_DETECTION_BIT_NAME
                and cmd.args[0].reg_name != LEAKAGE_DETECTION_QUBIT_NAME
            )
            if is_data_measure:
                qubit = cmd.args[0]
                bit = cmd.args[1]
                # ── Conditional X immediately before measurement ───────────────
                if randomness_type == RandomnessType.NO_CLASSICAL:
                    if qubit_apply_x[qubit]:
                        new_circuit.X(qubit)
                else:
                    new_circuit.X(qubit, condition=qubit_random_bit[qubit])
                new_circuit.Measure(qubit, bit)
                # ── Classical correction immediately after measurement ─────────
                # measurement_bit XOR random_bit restores the original value.
                if randomness_type == RandomnessType.NO_CLASSICAL:
                    if qubit_apply_x[qubit]:
                        new_circuit.add_clexpr_from_logicexp(bit ^ 1, [bit])
                else:
                    new_circuit.add_clexpr_from_logicexp(
                        bit ^ qubit_random_bit[qubit], [bit]
                    )
            elif cmd.op.type == OpType.Barrier:
                new_circuit.add_barrier(cmd.args)
            else:
                new_circuit.add_gate(Op=cmd.op, args=cmd.args)

        return new_circuit

    return CustomPass(transform=transform)


def ZZMaxRandomCompilation(
    rng: np.random.Generator = np.random.default_rng(),
    randomness_type: RandomnessType = RandomnessType.PER_SHOT,
) -> CustomPass:
    """ A pass which performs randomised compiling by inserting random 
    Pauli gates before and after each ZZMax gate in the circuit.

    This pass implements Pauli twirling of the ZZMax gate by inserting
    random Pauli gates before and after each ZZMax gate in the circuit.
    The random Pauli gates are generated either per shot by using the
    circuit RNG, or once per circuit by drawing compile-time values from
    ``rng`` and embedding them in the circuit.

    :param rng: Random number generator, defaults to np.random.default_rng()
    :type rng: np.random.Generator, optional
    :param randomness_type: Whether randomness changes per shot, is fixed per
        compiled circuit, or is sampled with no extra classical registers
        (``RandomnessType.NO_CLASSICAL``), defaults to RandomnessType.PER_SHOT
    :type randomness_type: RandomnessType, optional
    :return: Custom pass implementing randomised compiling.
    :rtype: CustomPass
    """

    def transform(raw_circuit: Circuit) -> Circuit:
        """Transforms given circuit by Pauli twirling the ZZMax gates.

        :param raw_circuit: Circuit to be transformed.
        :raises ValueError: If a reserved register name is used.
        :return: Transformed circuit with Pauli twirled ZZMax gates.
        """

        if any(
            register.name == "rc_seed"
            or register.name.startswith("rc_rng_")
            or register.name.startswith("rc_commuted_pauli_")
            for register in raw_circuit.c_registers
        ):
            raise ValueError(
                'Registers named "rc_seed" or starting with "rc_rng_" or "rc_commuted_pauli_" '
                "are reserved for ZZMaxRandomCompilation."
            )
        # Add all the qubits and bits from the original circuit.
        rc_circuit = Circuit()
        for qubit in raw_circuit.qubits:
            rc_circuit.add_qubit(qubit)
        for bit in raw_circuit.bits:
            rc_circuit.add_bit(bit)

        if randomness_type == RandomnessType.PER_SHOT:
            initialise_rng_seed(
                circuit=rc_circuit,
                rng=rng,
                seed_register_name="rc_seed",
            )

        if randomness_type == RandomnessType.NO_CLASSICAL:
            for cmd in raw_circuit.get_commands():
                if cmd.op.type == OpType.ZZMax:
                    gate_random_bits = np.array(
                        [
                            [bool(rng.integers(2)), bool(rng.integers(2))],
                            [bool(rng.integers(2)), bool(rng.integers(2))],
                        ]
                    )
                    gate_commuted_bits = np.array(
                        [
                            [
                                gate_random_bits[0][0],
                                gate_random_bits[0][0]
                                ^ gate_random_bits[0][1]
                                ^ gate_random_bits[1][0],
                            ],
                            [
                                gate_random_bits[1][0],
                                gate_random_bits[1][0]
                                ^ gate_random_bits[1][1]
                                ^ gate_random_bits[0][0],
                            ],
                        ]
                    )

                    for qubit_index in [0, 1]:
                        qubit = cmd.args[qubit_index]
                        if gate_random_bits[qubit_index][0]:
                            rc_circuit.X(qubit)
                        if gate_random_bits[qubit_index][1]:
                            rc_circuit.Z(qubit)

                    rc_circuit.ZZMax(qubit0=cmd.args[0], qubit1=cmd.args[1])

                    for qubit_index in [0, 1]:
                        qubit = cmd.args[qubit_index]
                        if gate_commuted_bits[qubit_index][1]:
                            rc_circuit.Z(qubit)
                        if gate_commuted_bits[qubit_index][0]:
                            rc_circuit.X(qubit)
                else:
                    _copy_command_with_barriers(raw_circuit, rc_circuit, cmd)

            return rc_circuit

        # Create a list of bits containing random values.
        random_bit_list = []
        # Create a list of bits which will be used to determine which Pauli
        # gates to apply after the ZZMax gates.
        commuted_pauli_bit_list = []
        # Populate a sufficient number of bits with randomness.
        for i in range(
            (
                (4 * raw_circuit.n_gates_of_type(type=OpType.ZZMax))
                // RANDOM_NUMBER_REGISTER_SIZE
            )
            + 1
        ):
            random_bit_register = rc_circuit.add_c_register(
                name=f"rc_rng_{i}",
                size=RANDOM_NUMBER_REGISTER_SIZE,
            )
            random_bit_list += random_bit_register.to_list()
            if randomness_type == RandomnessType.PER_SHOT:
                rc_circuit.get_rng_num(creg=random_bit_register)
            else:
                rc_circuit.add_c_setbits(
                    values=populate_random_register(
                        register=random_bit_register,
                        rng=rng,
                    ),
                    args=random_bit_register.to_list(),
                )

            commuted_pauli_bit_list += rc_circuit.add_c_register(
                name=f"rc_commuted_pauli_{i}",
                size=RANDOM_NUMBER_REGISTER_SIZE,
            ).to_list()

        # Each ZZMax gate needs 4 random bits (2 for each qubit) to determine which Pauli gates to apply.
        # This list comprehension reshapes the list of random bits into a list of lists,
        # where each sublist contains the 4 random bits needed for one ZZMax gate.
        # Those 4 bits are then further reshaped into 2 length 2 lists,
        # with one list for each qubit. The first bit in each list determines
        # whether to apply an X gate, and the second bit determines whether to apply a Z gate.
        gate_random_bit_list = np.reshape(random_bit_list, (-1, 2, 2))
        gate_commuted_pauli_bit_list = np.reshape(commuted_pauli_bit_list, (-1, 2, 2))

        # Now we calculate the conditions for which Pauli gates to apply after the ZZMax gates.
        # First we iterate through each set of random bits for each gate.
        for gate_random_bit, gate_commuted_pauli_bit in zip(gate_random_bit_list, gate_commuted_pauli_bit_list):

            # Then we iterate through each qubit in the gate.
            # We will also need to refer to the random bits for the other qubit in the gate
            # as a Z gate is sent through the ZZMax gate if there is an X gate on the other qubit.
            for qubit_index, other_index in [(0,1), (1,0)]:
            
                qubit_x = gate_random_bit[qubit_index][0]
                qubit_z = gate_random_bit[qubit_index][1]
                other_x = gate_random_bit[other_index][0]

                # Here we are actually just copying the x condition from before the gate to after the gate.
                # This is because an X gate before the ZZMax gate will commute through the gate as an X gate after the ZZMax gate.
                rc_circuit.add_clexpr_from_logicexp(
                    qubit_x | gate_commuted_pauli_bit[qubit_index][0], 
                    [gate_commuted_pauli_bit[qubit_index][0]]
                )
                # X gates will also cause Z gates to be sent through the ZZMax gate to both qubits.
                # Z will also commute through the gate as a Z gate after the ZZMax gate.
                rc_circuit.add_clexpr_from_logicexp(
                    qubit_x ^ qubit_z ^ other_x,
                    [gate_commuted_pauli_bit[qubit_index][1]]
                )

        # At this point you may wish to add a barrier and reset all of the qubits.
        # If the reset is very good this will mean the memory error added while
        # waiting for the classical calculations is removed.
        # At the time of writing I'm not sure if this is actually beneficial in practice.

        zzmax_count = 0
        for cmd in raw_circuit.get_commands():

            # If we find a ZZMax...
            if cmd.op.type == OpType.ZZMax:

                # We gather the random bits we need for each qubit.
                qubit_random_bit_list = gate_random_bit_list[zzmax_count]
                qubit_commuted_pauli_bit_list = gate_commuted_pauli_bit_list[zzmax_count]

                # For each qubit acted on by the ZZMax gate.
                for qubit_index in [0,1]:
                    
                    qubit = cmd.args[qubit_index]
                    # We use one of the random bits for the x condition...
                    rc_circuit.X(qubit, condition=qubit_random_bit_list[qubit_index][0])
                    # ... and one for the z condition
                    rc_circuit.Z(qubit, condition=qubit_random_bit_list[qubit_index][1])

                # We have added the Pauli gates before the ZZMax.
                # Now we can add the gate itself.
                rc_circuit.ZZMax(qubit0=cmd.args[0], qubit1=cmd.args[1])

                # Now we add the Pauli gates needed after the ZZMax.
                for qubit_index in [0,1]:
                    
                    qubit = cmd.args[qubit_index]
                    rc_circuit.Z(qubit, condition=qubit_commuted_pauli_bit_list[qubit_index][1])
                    rc_circuit.X(qubit, condition=qubit_commuted_pauli_bit_list[qubit_index][0])
        
                zzmax_count += 1

            # If the gate is not ZZMax then we add it back in.
            else:
                _copy_command_with_barriers(raw_circuit, rc_circuit, cmd)

        return rc_circuit

    return CustomPass(transform=transform)
