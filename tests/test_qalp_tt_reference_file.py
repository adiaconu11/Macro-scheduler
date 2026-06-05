from collections import Counter
from pathlib import Path

import numpy as np

from macroscheduler.qecc.qalp import Polynomial, TT_Code, Term


def _row_counter(mat: np.ndarray) -> Counter:
    return Counter(map(tuple, mat.tolist()))


def test_tt_code_matches_reference_file_up_to_row_permutation_with_swapped_labels():
    fixture = Path(__file__).with_name("TT_Code_unit_reference.npy")
    data = np.load(fixture, allow_pickle=True).tolist()

    # Per current convention check:
    # treat file labels as swapped
    hx_ref = np.array(data["Hz"], dtype=int)
    hz_ref = np.array(data["Hx"], dtype=int)

    code = TT_Code( # First row in Table III of Menon et al. Phys. Rev. X 16, 021014
        l=2,
        m=2,
        n=4,
        A_poly=[[0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 2]],
        B_poly=[[0, 1, 2], [0, 1, 3]],
        C_poly=[[0, 1, 0], [1, 1, 1]],
        validate=True,
    )

    hx = np.array(code.Hx, dtype=int)
    hz = np.array(code.Hz, dtype=int)

    assert hx.shape == hx_ref.shape
    assert hz.shape == hz_ref.shape

    # Row order is intentionally ignored.
    assert _row_counter(hx) == _row_counter(hx_ref)
    assert _row_counter(hz) == _row_counter(hz_ref)
