from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import stim

from macroscheduler import scheduling
from macroscheduler.qecc.stim import generate_stim_str
from macroscheduler.qecc.qalp import (
    AlgebraicBlockCode,
    BB_Code,
    Block,
    LiftedMatrix,
    Perm,
    Polynomial,
    QALP,
    TT_Code,
    Term,
    construct_lifted_product_mats,
)
from macroscheduler.scheduling import all_schedules, color_protograph, random_schedule


def _validate_schedule(code, schedule):
    code.set_se_schedule_ir(*schedule)
    code.validate_se_schedule()


def _canonical_schedule(schedule):
    s_x_check, s_z_check = schedule
    return (
        tuple(tuple(row) for row in s_x_check),
        tuple(tuple(row) for row in s_z_check),
    )


def _perms():
    identity = Perm.identity(3)
    shift = Perm.right_shift(3)
    return identity, shift, shift ** 2


def _lifted_from_powers(powers, lift):
    shift = Perm.right_shift(lift)
    return LiftedMatrix(
        [[Block([shift ** (int(power) % lift)]) for power in row] for row in powers],
        lift,
    )


def _classical_lifted(check_mat):
    identity = Perm.identity(1)
    return LiftedMatrix(
        [
            [Block([identity]) if entry else Block([]) for entry in row]
            for row in check_mat
        ],
        1,
    )


def _nontrivial_2d_qalp():
    identity = Perm.identity(1)
    zero = Block([])
    a_matrix = LiftedMatrix(
        [
            [Block([identity]), Block([identity])],
            [Block([identity]), zero],
        ],
        1,
    )
    b_matrix = LiftedMatrix(
        [
            [Block([identity]), Block([identity])],
            [zero, Block([identity])],
        ],
        1,
    )
    return QALP([a_matrix, b_matrix])


def _quasi_cyclic_lp():
    lift = 30
    a_powers = [
        [0, 0, 0, 0, 0],
        [0, 2, 14, 24, 25],
        [0, 16, 11, 14, 13]
    ]
    b_powers = [[(lift - power) % lift for power in row] for row in a_powers]
    return QALP([
        _lifted_from_powers(a_powers, lift),
        _lifted_from_powers(b_powers, lift),
    ])


def _parallel_1x1_qalp(dim):
    identity, shift, shift2 = _perms()
    blocks = [
        Block([identity, shift]),
        Block([identity, shift2]),
        Block([shift, shift2]),
        Block([identity, shift]),
    ]
    return QALP([LiftedMatrix([[block]], 3) for block in blocks[:dim]])


def _assert_css_commutes(code):
    hx = np.array(code.Hx, dtype=int)
    hz = np.array(code.Hz, dtype=int)
    assert np.count_nonzero(hx @ hz.T % 2) == 0


def _poly_from_xy_powers(x_pows, y_pows) -> Polynomial:
    return Polynomial([Term([int(px), int(py)]) for px, py in zip(x_pows, y_pows)])


def _reference_bb_code():
    fixture_path = Path(__file__).with_name("bb_code_72_12_6.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        ref = json.load(f)

    return BB_Code(
        l=int(ref["l"]),
        m=int(ref["m"]),
        A_poly=_poly_from_xy_powers(ref["A_x_pows"], ref["A_y_pows"]),
        B_poly=_poly_from_xy_powers(ref["B_x_pows"], ref["B_y_pows"]),
        validate=True,
    )


def _reference_tt_code():
    return TT_Code(
        l=2,
        m=2,
        n=4,
        A_poly=[[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 2]],
        B_poly=[[0, 1, 2], [0, 1, 3]],
        C_poly=[[0, 1, 0], [1, 1, 1]],
        validate=True,
    )


def test_color_protograph_groups_parallel_edges_into_distinct_colors():
    code = _parallel_1x1_qalp(2)
    colors = color_protograph(code.lifted_check_mats[0])

    assert len(colors) == 2
    assert sorted(len(color_class) for color_class in colors) == [1, 1]
    assert sorted(edge for color_class in colors for edge in color_class) == [
        (0, 0, 0),
        (0, 0, 1),
    ]


def test_general_2d_qalp_schedules_validate_and_respect_cap():
    code = _nontrivial_2d_qalp()
    assert not isinstance(code, AlgebraicBlockCode)
    _assert_css_commutes(code)

    schedules = all_schedules(code, max_schedules=2)

    assert len(schedules) == 2
    for schedule in schedules:
        _validate_schedule(code, schedule)
        assert code.depth == 4


def test_random_schedule_supports_general_qalp():
    code = _nontrivial_2d_qalp()
    _assert_css_commutes(code)

    schedule = random_schedule(code, np.random.default_rng(3))

    _validate_schedule(code, schedule)
    assert code.depth == 4


def test_lifted_product_matches_paper_shaped_processor_code():
    code = _quasi_cyclic_lp()

    assert code.num_data_q == 1020
    assert code.num_x_check == code.num_z_check == 450
    assert min(code.degrees_x_check) == max(code.degrees_x_check) == 8
    assert min(code.degrees_z_check) == max(code.degrees_z_check) == 8
    _assert_css_commutes(code)


def test_quasi_cyclic_lp_example_builds_circuit_and_100_valid_schedules(tmp_path, monkeypatch):
    """Regression for example_codes.ipynb quasi-cyclic LP / https://doi.org/10.1038/s41567-024-02479-z

    Uses the B_{20}^{30} code from Xu, Q., Bonilla Ataides, J.P., Pattison, C.A. et al. 
    "Constant-overhead fault-tolerant quantum computation with reconfigurable atom arrays.", Eq. (6).
    """
    code = _quasi_cyclic_lp()

    schedules = all_schedules(code, max_schedules=100)

    assert len(schedules) == 100
    for schedule in schedules:
        _validate_schedule(code, schedule)

    code.set_se_schedule_ir(*schedules[0])
    monkeypatch.chdir(tmp_path)
    circuit = stim.Circuit(
        generate_stim_str(code, rounds=2, obs_type="Z", p_meas=0.0)
    )

    assert circuit.num_qubits == code.num_total_q


def test_lifted_product_hgp_when_lift_is_one():
    lifted_mats = [
        _classical_lifted([[1, 0, 1], [0, 1, 1]]),
        _classical_lifted([[1, 1], [0, 1], [1, 0]]),
    ]

    hx_lifted, hz_lifted = construct_lifted_product_mats(lifted_mats)
    code = QALP(lifted_mats)

    np.testing.assert_array_equal(hx_lifted, np.array(code.Hx, dtype=int))
    np.testing.assert_array_equal(hz_lifted, np.array(code.Hz, dtype=int))
    assert hx_lifted.shape == (4, 12)
    assert hz_lifted.shape == (9, 12)
    _assert_css_commutes(code)


def test_general_lifted_product_qalp_with_nontrivial_lift_schedules():
    lift = 3
    a_powers = [[0, 1], [2, 0]]
    b_powers = [[(lift - power) % lift for power in row] for row in a_powers]
    code = QALP([
        _lifted_from_powers(a_powers, lift),
        _lifted_from_powers(b_powers, lift),
    ])
    _assert_css_commutes(code)

    schedules = all_schedules(code, max_schedules=1)

    assert len(schedules) == 1
    _validate_schedule(code, schedules[0])
    assert code.depth == 4

    random_sched = random_schedule(code, np.random.default_rng(14))
    _validate_schedule(code, random_sched)
    assert code.depth == 4

def test_general_qalp_3d_parallel_smoke():
    dim, expected_depth = (3, 6)
    code = _parallel_1x1_qalp(dim)
    assert not isinstance(code, AlgebraicBlockCode)
    _assert_css_commutes(code)

    schedules = all_schedules(code, max_schedules=1)

    assert len(schedules) == 1
    _validate_schedule(code, schedules[0])
    assert code.depth == expected_depth

def test_algebraic_block_code_dispatches_to_legacy_scheduler():
    poly = Polynomial([Term([0]), Term([1])])
    code = AlgebraicBlockCode([3], [poly, poly])

    with (
        patch.object(scheduling, "_all_schedules_ABC", return_value=["abc"]) as abc,
        patch.object(scheduling, "_all_schedules_QALP", side_effect=AssertionError),
    ):
        assert all_schedules(code) == ["abc"]

    abc.assert_called_once_with(code, max_schedules=None)


def test_algebraic_block_code_and_qalp_paths_generate_same_schedules():
    poly_a = Polynomial([Term([0]), Term([1]), Term([2])])
    poly_b = Polynomial([Term([0]), Term([1]), Term([2])])
    code = AlgebraicBlockCode([3], [poly_a, poly_b])
    _assert_css_commutes(code)

    abc_schedules = scheduling._all_schedules_ABC(code)
    qalp_schedules = scheduling._all_schedules_QALP(code)

    assert len(abc_schedules) == len(qalp_schedules) == 288
    assert Counter(map(_canonical_schedule, abc_schedules)) == Counter(
        map(_canonical_schedule, qalp_schedules)
    )


def test_reference_bb_code_abc_and_qalp_paths_generate_same_schedules():
    code = _reference_bb_code()
    _assert_css_commutes(code)

    abc_schedules = scheduling._all_schedules_ABC(code)
    qalp_schedules = scheduling._all_schedules_QALP(code)

    assert len(abc_schedules) == len(qalp_schedules) == 288
    assert Counter(map(_canonical_schedule, abc_schedules)) == Counter(
        map(_canonical_schedule, qalp_schedules)
    )
    _validate_schedule(code, scheduling._random_schedule_QALP(code, np.random.default_rng(11)))


def test_reference_tt_code_abc_and_qalp_paths_generate_same_schedules():
    code = _reference_tt_code()
    _assert_css_commutes(code)

    abc_schedules = scheduling._all_schedules_ABC(code)
    qalp_schedules = scheduling._all_schedules_QALP(code)

    assert len(abc_schedules) == len(qalp_schedules) == 1728
    assert Counter(map(_canonical_schedule, abc_schedules)) == Counter(
        map(_canonical_schedule, qalp_schedules)
    )
    _validate_schedule(code, scheduling._random_schedule_QALP(code, np.random.default_rng(12)))
