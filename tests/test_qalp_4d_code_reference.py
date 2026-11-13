from collections import Counter
from pathlib import Path

import numpy as np

from macroscheduler.qecc.qalp import AlgebraicBlockCode, Polynomial, Term


def _row_counter(mat: np.ndarray) -> Counter:
    return Counter(map(tuple, mat.tolist()))


def _poly_from_terms_list(t_list):
    return Polynomial([Term(t) for t in t_list])


def test_4d_code_matches_reference_up_to_row_permutation():
    moduli = [3, 3]
    a_terms = [[2, 1], [2, 2]]
    b_terms = [[0, 0], [1, 2]]
    c_terms = [[1, 0], [2, 1]]
    d_terms = [[1, 1], [2, 1]]

    a_poly = _poly_from_terms_list(a_terms)
    b_poly = _poly_from_terms_list(b_terms)
    c_poly = _poly_from_terms_list(c_terms)
    d_poly = _poly_from_terms_list(d_terms)

    code = AlgebraicBlockCode(moduli, [a_poly, b_poly, c_poly, d_poly])

    fixture = Path(__file__).with_name("4D_code_54_6_6.npy")
    data = np.load(fixture, allow_pickle=True).tolist()

    # Keep the same mapping used in the notebook.
    hz_ref = np.array(data["Hz"], dtype=int)
    hx_ref = np.array(data["Hx"], dtype=int)

    hx = np.array(code.Hx, dtype=int)
    hz = np.array(code.Hz, dtype=int)

    assert hx.shape == hx_ref.shape
    assert hz.shape == hz_ref.shape

    # Row order can differ; compare as multisets of rows.
    assert _row_counter(hx) == _row_counter(hx_ref)
    assert _row_counter(hz) == _row_counter(hz_ref)
