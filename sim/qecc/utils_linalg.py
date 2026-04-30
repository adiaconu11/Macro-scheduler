"""
This file contains some utilities for linear algebra that is used for CSS codes.
"""

from typing import List, Optional, Tuple
import numpy as np

def row_echelon(mat, reduced=False):
    r"""Convert a binary matrix to row-echelon (or reduced row-echelon) form.

    Unlike the `make_systematic` method, no column swaps are performed.

    Args:
        mat (ndarray): Binary matrix in `numpy.ndarray` format.
        reduced (bool, optional): If `True`, return reduced row-echelon form.
            Defaults to `False`.

    Returns:
        list: `[row_ech_form, rank, transform, pivot_cols]`, where:
            `row_ech_form` (ndarray): Row-echelon form of `mat`.
            `rank` (int): Matrix rank.
            `transform` (ndarray): Transformation matrix such that
                `transform @ mat == row_ech_form` (mod 2).
            `pivot_cols` (list[int]): Pivot column indices.
    """

    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]: # will evaluate to True if this column is not all-zero
            if not reduced: # clean entries below the pivot 
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:           # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:    
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m: # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]

def rank(mat):
    r"""Return the rank of a binary matrix.

    Args:
        mat (ndarray): Binary matrix in `numpy.ndarray` format.

    Returns:
        int: Matrix rank.
    """
    return row_echelon(mat)[1]

def kernel(mat):
    r"""Computes the kernel of the matrix M.
    All vectors x in the kernel of M satisfy the following condition::

        Mx=0 \forall x \in ker(M)

    Args:
        mat (ndarray): Binary matrix in `numpy.ndarray` format.

    Returns:
        tuple[ndarray, int, list[int]]:
            `ker`: Basis vectors spanning the kernel of `mat`.
            `rank`: Rank of `mat.T` (same as rank of `mat`).
            `pivot_cols`: Pivot indices of `mat.T` (useful for `row_basis`).
    
    Note
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM::

        P@M=ReM=[A,0]^T,
    
    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.
    """

    transpose = mat.T
    m, _ = transpose.shape
    _, rank, transform, pivot_cols = row_echelon(transpose)
    ker = transform[rank:m]
    return ker, rank, pivot_cols

def row_basis(mat):
    r"""Return a basis for the row space of a matrix.

    Args:
        mat (ndarray): Input matrix.

    Returns:
        ndarray: Matrix whose rows form a basis of the row space.
    """
    return mat[row_echelon(mat.T)[3]]

def compute_code_distance(mat, is_pcm=True, is_basis=False):
    r"""Compute the distance of a linear code from a parity-check or generator matrix.

    The code distance is the minimum Hamming weight of a nonzero codeword.

    Args:
        mat (ndarray): Parity-check matrix (default) or generator matrix.
        is_pcm (bool, optional): If `True`, interpret `mat` as a parity-check
            matrix. If `False`, interpret `mat` as a generator matrix.
            Defaults to `True`.
        is_basis (bool, optional): If `True`, treat `mat`/derived generator as
            a basis directly. If `False`, compute a row basis before evaluating
            weights. Defaults to `False`.

    Note:
        Runtime scales exponentially with block size. In practice, computing
        distance for block lengths above about `10` can be very slow.

    Returns:
        int: Code distance.
    """
    gen = mat
    if is_pcm:
        gen = kernel(mat)
    if len(gen)==0: return np.inf # infinite code distance
    cw = gen
    if not is_basis:
        cw = row_basis(gen) # nonzero codewords
    return np.min(np.sum(cw, axis=1))

def inverse(mat):
    r"""Compute the left inverse of a full-rank binary matrix.

    Args:
        mat (ndarray): Binary matrix to invert. Must be either square
            full-rank or rectangular with full-column rank.

    Returns:
        ndarray: Inverted binary matrix.
    
    Note
    -----
    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows::

        Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form::

        row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to::

        Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P"""

    m, n = mat.shape
    reduced_row_ech, rank, transform, _ = row_echelon(mat, reduced=True)
    if m == n and rank == m:
        return transform
    # compute the left-inverse
    elif m > rank and n == rank:  # left inverse
        return reduced_row_ech.T @ transform % 2
    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")

def classical_code_distance(parity_check: List[List[int]]) -> int:
    import gurobipy as gp
    from gurobipy import GRB
    min_distance = 0
    H = np.array(parity_check)
    r, n = H.shape
    model = gp.Model()
    # Codeword bits
    c = model.addVars(n, vtype=GRB.BINARY, name="c")
    # For each parity check: sum H[l,i] * c[i] ≡ 0 mod 2
    for l in range(r):
        expr = gp.LinExpr()
        for i in range(n):
            if H[l,i]:
                expr += c[i]
        # Introduce integer variable to linearize mod 2 equation
        z = model.addVar(vtype=GRB.INTEGER, name=f"z_{l}")
        model.addConstr(expr - 2*z == 0)
    # No all-zeros codeword
    model.addConstr(gp.quicksum(c[i] for i in range(n)) >= 1)
    # Minimize Hamming weight
    model.setObjective(gp.quicksum(c[i] for i in range(n)), GRB.MINIMIZE)
    model.Params.OutputFlag = 0  # Silence solver output
    model.optimize()
    if model.status == GRB.OPTIMAL:
        min_distance = int(model.objVal)
        print("Minimum distance:", min_distance)
    else:
        raise ValueError((
            "No nonzero codeword found "
            "(should not happen unless H = full rank or trivial)."))
    return min_distance

def sample_sparse_parity_check(
        n: int,
        r: int,
        w_r: int,
        w_c: int,
        max_tries: Optional[int] =1000) -> List[List[int]]:
    
    import numpy as np
    for attempt in range(max_tries):
        H = np.zeros((r, n), dtype=int)
        col_weights = np.zeros(n, dtype=int)
        # For each row
        for i in range(r):
            candidates = [j for j in range(n) if col_weights[j] < w_c]
            # If not enough candidates to fill the row, fail and restart.
            if len(candidates) < w_r:
                break
            pos = np.random.choice(candidates, size=w_r, replace=False)
            H[i, pos] = 1
            col_weights[pos] += 1
        # After all rows, check
        if np.any(H.sum(axis=0) == 0):
            continue  # zero column!
        if np.any(H.sum(axis=0) == w_c) and np.all(H.sum(axis=0) <= w_c) and np.any(H.sum(axis=1) == w_r) and np.all(H.sum(axis=1) <= w_r):
            if np.linalg.matrix_rank(H) == r:
                if np.any(H.sum(axis=0) == H.shape[0]):
                    continue  # all-1 column!
                if np.any(H.sum(axis=0) == 0):
                    continue  # zero column!
                if classical_code_distance(H.tolist()) >= 2:
                    return H
    raise RuntimeError("Failed to sample a valid matrix after many tries.")

def parity_check_validity(mat: List[List[int]]) -> Tuple[int, int]:
    # check the given list of list is a binary matrix
    width = len(mat[0])
    height = len(mat)
    for row in mat:
        if len(row) != width:
            raise ValueError('width not equal')
        for entry in row:
            if entry != 0 and entry != 1:
                raise ValueError('entry not 0 or 1')
    return (height, width)
