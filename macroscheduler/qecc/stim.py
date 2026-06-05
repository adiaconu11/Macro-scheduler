"""
This is the module that implements the stim circuit according to the
specified scheduling.
"""

from .css import CSS_Code

def generate_stim_str(code: CSS_Code, rounds, obs_type, **kwargs) -> str:
    """
    Generate a stim circuit string for the specified schedule and rounds.

    Args:
        code (CSS_Code): Code to generate the syndrome-extraction circuit for.
            The schedule must already be defined and validated.
        rounds (int): Number of syndrome-extraction rounds.
        obs_type (str): Observable type to target. Must be `"X"` or `"Z"`.
        **kwargs: Optional noise configuration. Supported keys:
            `p_2q_channel` (str | None): Two-qubit error channel applied after
                each CNOT gate.
            `p_idle_channel` (str | None): One-qubit error channel applied to
                idling qubits.
            `p_meas` (float | None): Measurement error probability.

    Returns:
        str: Stim circuit source string.

    Raises:
        ValueError: If unknown keyword arguments are provided.
        AssertionError: If argument types or `obs_type` are invalid.
    """

    # --- Gather the arguments from the keywords and safety checks
    valid = {'p_2q_channel', 'p_idle_channel', 'p_meas'}
    p_2q_channel = kwargs.get('p_2q_channel', None)
    assert p_2q_channel is None or isinstance(p_2q_channel, str), "p_2q_channel must be a string representing the error channel."
    p_idle_channel = kwargs.get('p_idle_channel', None)
    assert p_idle_channel is None or isinstance(p_idle_channel, str), "p_idle_channel must be a string representing the error channel."
    p_meas = kwargs.get('p_meas', None)

    unknown_args = set(kwargs.keys()) - valid
    if unknown_args:
        raise ValueError(f"Unknown keyword arguments: {unknown_args}")
    
    code.validate_se_schedule()
    assert obs_type in ['X', 'Z'], f"Invalid obs_type: {obs_type}. Must be 'X' or 'Z'."

    # --- Define the qubits and their indicators

    num_data_q = code.num_data_q
    x_begin = num_data_q
    x_end = x_begin + code.num_x_check
    z_begin = x_end
    z_end = z_begin + code.num_z_check
    all_end = z_end
    num_x_check = code.num_x_check
    num_z_check = code.num_z_check

    data_q = ""
    for i in range(num_data_q):
        data_q += f" {i}"
    x_check_q = ""
    for i in range(x_begin, x_end):
        x_check_q += f" {i}"
    z_check_q = ""
    for i in range(z_begin, z_end):
        z_check_q += f" {i}"
    
    # --- Define the schedule for lattice operations
    # control is on ancilla, target on data!
    schedule = [[] for _ in range(code.depth)]
    for i in range(num_x_check):
        for j, q in enumerate(code.x_checks[i]):
            schedule[code.s_x_check[i][j]].append((i+x_begin, q))
    for i in range(num_z_check):
        for j, q in enumerate(code.z_checks[i]):
            schedule[code.s_z_check[i][j]].append((q, i+z_begin))
    with open("schedule.txt", "w") as f:
        for layer in schedule:
            f.write(str(layer) + "\n")
    def lattice_with_noise(indentation='    ') -> str:
        ans = ""
        for layer in schedule:
            qubit_roles = [0 for _ in range(z_end)]
            cx_qubits_str = ""
            for (src, dest) in layer:
                cx_qubits_str += f" {src} {dest}"
                qubit_roles[src] = 1
                qubit_roles[dest] = 1
            idling_str = ""
            for i in range(all_end):
                if qubit_roles[i] == 0:
                    idling_str += f" {i}"
            ans += indentation + "CX" + cx_qubits_str + "\n"
            if p_2q_channel is not None and cx_qubits_str:
                ans += indentation + p_2q_channel + cx_qubits_str + "\n"
            if idling_str and p_idle_channel is not None:
                ans += indentation + p_idle_channel + idling_str + "\n"
            ans += indentation + "TICK\n\n"
        return ans

    # --- Define the functions to generate the different parts of the circuit    

    def stabilizers_with_noise(indentation='    ') -> str:
        ans = lattice_with_noise(indentation)
        ans += indentation + f"Z_ERROR({p_meas})" + x_check_q + "\n"
        ans += indentation +  "MRX" + x_check_q + "\n"
        ans += indentation + f"X_ERROR({p_meas})" + z_check_q + "\n"
        ans += indentation + "MR" + z_check_q + "\n"
        ans += indentation + "TICK\n\n"
        return ans

    def init() -> str:
        ans = ""
        if obs_type == 'X':
            ans += "RX" + data_q + x_check_q + "\n"
            ans += f"Z_ERROR({p_meas})" + data_q + x_check_q + "\n"
            ans += "R" + z_check_q + "\n"
            ans += f"X_ERROR({p_meas})" + z_check_q + "\n"
        else:
            ans += "R" + data_q + z_check_q + "\n"
            ans += f"X_ERROR({p_meas})" + data_q + z_check_q + "\n"
            ans += "RX" + x_check_q + "\n"
            ans += f"Z_ERROR({p_meas})" + x_check_q + "\n"
        ans += "TICK\n\n"
        ans += stabilizers_with_noise('')
        if obs_type == 'X':
            for i in range(num_x_check):
                ans += f"DETECTOR({x_begin + i},0) rec[{-num_z_check-num_x_check+i}]\n"
        else:
            for i in range(num_z_check):
                ans += f"DETECTOR({z_begin + i},0) rec[{-num_z_check+i}]\n"
        return ans

    def rounds_steps(indentation='    ') -> str:
        ans = f"REPEAT {rounds-1} {{\n\n"
        ans += stabilizers_with_noise(indentation)
        ans += indentation + 'SHIFT_COORDS(0, 1)\n'
        if obs_type == 'X':
            for i in range(num_x_check):
                ans += indentation + f"DETECTOR({x_begin + i},0) " +\
                        f"rec[{-num_z_check-num_x_check+i}] " +\
                        f"rec[{-num_z_check-num_x_check+i - num_x_check - num_z_check}]\n"
        else:
            for i in range(num_z_check):
                ans += indentation + f"DETECTOR({z_begin + i},0) " +\
                        f"rec[{-num_z_check+i}] " +\
                        f"rec[{-num_z_check+i - num_x_check - num_z_check}]\n"
        ans += "}\n\n"
        return ans

    def final_step() -> str:
        ans = "SHIFT_COORDS(0, 1)\n"
        if obs_type == 'X':
            ans += f"Z_ERROR({p_meas})" + data_q + "\n"
            ans += 'MX' + data_q + "\n"
            for i in range(num_x_check):
                ans += f"DETECTOR({x_begin + i},0) rec[{-num_z_check-num_x_check+i - num_data_q}]"
                for q in code.x_checks[i]:
                    ans += f" rec[{-num_data_q + q}]"
                ans += "\n"
            for i, obs in enumerate(code.x_logicals):
                ans += f"OBSERVABLE_INCLUDE({i})"
                for q in obs:
                    ans += f" rec[{-num_data_q + q}]"
                ans += '\n'
        else:
            ans += f"X_ERROR({p_meas})" + data_q + '\n'
            ans += "M" + data_q + '\n'
            for i in range(num_z_check):
                ans += f"DETECTOR({z_begin + i},0) rec[{-num_z_check+i - num_data_q}]"
                for q in code.z_checks[i]:
                    ans += f" rec[{-num_data_q + q}]"
                ans += '\n'
            for i, obs in enumerate(code.z_logicals):
                ans += f"OBSERVABLE_INCLUDE({i})"
                for q in obs:
                    ans += f" rec[{-num_data_q + q}]"
                ans += '\n'
        return ans

    return init() + rounds_steps() + final_step()
