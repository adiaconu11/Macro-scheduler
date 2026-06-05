import json
from pathlib import Path

import numpy as np

from macroscheduler.qecc.qalp import BB_Code, Polynomial, Term


def _poly_from_xy_powers(x_pows, y_pows) -> Polynomial:
    if len(x_pows) != len(y_pows):
        raise ValueError("x and y power arrays must have the same length")
    return Polynomial([Term([int(px), int(py)]) for px, py in zip(x_pows, y_pows)])


def test_bb_code_constructor_matches_reference_parity_checks():
    fixture_path = Path(__file__).with_name("bb_code_72_12_6.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        ref = json.load(f)

    l = int(ref["l"])
    m = int(ref["m"])

    a_poly = _poly_from_xy_powers(ref["A_x_pows"], ref["A_y_pows"])
    b_poly = _poly_from_xy_powers(ref["B_x_pows"], ref["B_y_pows"])

    code = BB_Code(l=l, m=m, A_poly=a_poly, B_poly=b_poly, validate=True)

    hx_expected = np.array(ref["Hx"], dtype=int)
    hz_expected = np.array(ref["Hz"], dtype=int)

    hx_got = np.array(code.Hx, dtype=int)
    hz_got = np.array(code.Hz, dtype=int)

    np.testing.assert_array_equal(hx_got, hx_expected)
    np.testing.assert_array_equal(hz_got, hz_expected)
