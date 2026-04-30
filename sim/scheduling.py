"""
This is the general file for the scheduling module.
"""

from typing import List, Optional, Tuple
import numpy as np
import itertools
import stim

from .qecc.qalp import construct_mats, compute_basis, Term
from .qecc.qalp import AlgebraicBlockCode
from .qecc.stim import generate_stim_str


def _split_poly_options(poly):
    """Return allowed (l_poly, g_poly) splits for one ordered polynomial."""
    n_terms = len(poly)
    cut_floor = n_terms // 2
    cut_ceil = (n_terms + 1) // 2
    if cut_floor == cut_ceil:
        return [(poly[:cut_floor], poly[cut_floor:])]
    return [
        (poly[:cut_floor], poly[cut_floor:]),
        (poly[:cut_ceil], poly[cut_ceil:]),
    ]

def _implement_perm_choice(code: AlgebraicBlockCode, stages, l_polys, g_polys, perm_l_polys, perm_g_polys,
                           center_mat, center_poly, get_q_pairs_x, get_q_pairs_z):
    x_depth = 0
    z_depth = 0
    code.init_se_schedule_ir()

    # Left side: greater terms on X, lesser terms on Z
    for mat_name in stages[:-1]:
        for term in g_polys[mat_name]:
            q_pairs = get_q_pairs_x(mat_name, term)
            for check_no, qubit_no in q_pairs:
                code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
            x_depth += 1
        
        for term in perm_l_polys[mat_name]:
            q_pairs = get_q_pairs_z(mat_name, term)
            for check_no, qubit_no in q_pairs:
                code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
            z_depth += 1
        x_depth = z_depth = max(x_depth, z_depth)

    # Center: full poly on both X and Z, same order
    for term in center_poly:
        q_pairs_x = get_q_pairs_x(center_mat, term)
        q_pairs_z = get_q_pairs_z(center_mat, term)
        for check_no, qubit_no in q_pairs_x:
            code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
        for check_no, qubit_no in q_pairs_z:
            code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
        x_depth += 1
        z_depth += 1

    # Right side: lesser terms on X, greater terms on Z
    for mat_name in reversed(stages[:-1]):
        for term in l_polys[mat_name]:
            q_pairs = get_q_pairs_x(mat_name, term)
            for check_no, qubit_no in q_pairs:
                code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
            x_depth += 1
        
        for term in perm_g_polys[mat_name]:
            q_pairs = get_q_pairs_z(mat_name, term)
            for check_no, qubit_no in q_pairs:
                code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
            z_depth += 1
        x_depth = z_depth = max(x_depth, z_depth)

    code.depth = int(max(x_depth, z_depth))
    code.validate_se_schedule()

def all_schdules(input_code: AlgebraicBlockCode):
    """
    Will compute all possible schedules according to Daniel's construction.
    Returns a list of tuples (s_x_check, s_z_check) for each schedule.
    """
    moduli = input_code.moduli
    polys = input_code.polys
    Dim = len(polys)
    
    monomial_basis = compute_basis(moduli)
    code = AlgebraicBlockCode(moduli, polys)
    l = int(np.array(moduli).prod())

    all_schedules = []
    num_schedules = 0

    def get_q_pairs_x(mat_name: int, term: Term):
        """
        Get the qubit pairs for the X checks from the given terms.

        Args:
            mat_name (int): Matrix index (`0` for A, `1` for B, `2` for C, `3` for D).
            term (Term): Monomial term to materialize into the selected matrix.

        Returns:
            list[tuple[int, int]]: `(check_idx, qubit_idx)` pairs where `Hx == 1`.
        """
        # Implementation for getting qubit pairs for X checks
        # mats = [np.zeros((l, l)), np.zeros((l, l)), np.zeros((l, l)), np.zeros((l, l))]
        mats = [np.zeros((l, l)) for _ in range(Dim)]
        mats[mat_name] = term.get_perm(moduli, monomial_basis).to_matrix()
        Hx, _ = construct_mats(
            [1]*Dim, # m_array
            [1]*Dim, # n_array
            mats
        )
        coords = np.where(Hx == 1)
        return [x for x in zip(coords[0], coords[1])]

    def get_q_pairs_z(mat_name: int, term: Term):
        """
        Get the qubit pairs for the Z checks from the given terms.

        Args:
            mat_name (int): Matrix index (`0` for A, `1` for B, `2` for C, `3` for D).
            term (Term): Monomial term to materialize into the selected matrix.

        Returns:
            list[tuple[int, int]]: `(check_idx, qubit_idx)` pairs where `Hz == 1`.
        """
        # Implementation for getting qubit pairs for Z checks
        # mats = [np.zeros((l, l)), np.zeros((l, l)), np.zeros((l, l)), np.zeros((l, l))]
        mats = [np.zeros((l, l)) for _ in range(Dim)]
        mats[mat_name] = term.get_perm(moduli, monomial_basis).to_matrix()
        _, Hz = construct_mats(
            [1]*Dim, # m_array
            [1]*Dim, # n_array
            mats
        )
        coords = np.where(Hz == 1)
        return [x for x in zip(coords[0], coords[1])]

    poly_ords = [list(itertools.permutations(range(len(poly)))) for poly in polys]

    stage_ords = list(itertools.permutations(range(Dim)))

    for p_ord in itertools.product(*poly_ords):
        new_polys = [[polys[j][i] for i in p_ord[j]] for j in range(Dim)]

        for stages in stage_ords:
            center_mat = stages[-1]

            # Baseline floor/ceil split; odd non-center mats can flip independently.
            l_polys_base = [new_polys[i][:len(new_polys[i])//2] for i in range(Dim)]
            g_polys_base = [new_polys[i][len(new_polys[i])//2:] for i in range(Dim)]

            split_sources = []
            split_targets = []
            for mat_name in stages[:-1]:
                split_opts = _split_poly_options(new_polys[mat_name])
                if len(split_opts) > 1:
                    split_sources.append(split_opts)
                    split_targets.append(mat_name)

            split_products = itertools.product(*split_sources) if split_sources else [()]

            for split_choice in split_products:
                l_polys = [list(p) for p in l_polys_base]
                g_polys = [list(p) for p in g_polys_base]
                for mat_name, (l_choice, g_choice) in zip(split_targets, split_choice):
                    l_polys[mat_name] = list(l_choice)
                    g_polys[mat_name] = list(g_choice)

                # Only permute mats that are actually used on left/right sides.
                # The center mat runs with new_polys[center_mat] in fixed order.
                perm_sources = []
                perm_targets = []
                for mat_name in stages[:-1]:
                    perm_sources.append(itertools.permutations(l_polys[mat_name]))
                    perm_targets.append(("l", mat_name))
                    perm_sources.append(itertools.permutations(g_polys[mat_name]))
                    perm_targets.append(("g", mat_name))

                for permuted_lists in itertools.product(
                    *perm_sources
                ):
                    # perm_l_polys = [list(inner) for inner in l_polys]
                    # perm_g_polys = [list(inner) for inner in g_polys]
                    perm_l_polys = [0] * Dim
                    perm_g_polys = [0] * Dim

                    for (side, mat_name), perm in zip(perm_targets, permuted_lists):
                        if side == "l":
                            perm_l_polys[mat_name] = list(perm)
                        else:
                            perm_g_polys[mat_name] = list(perm)

                    _implement_perm_choice(code, stages, l_polys, g_polys, perm_l_polys, perm_g_polys,
                                           center_mat, new_polys[center_mat], get_q_pairs_x, get_q_pairs_z)
                    all_schedules.append((code.s_x_check, code.s_z_check))
                    num_schedules += 1
                
    return all_schedules


def random_schedule(
    input_code: AlgebraicBlockCode,
    rng: Optional[np.random.Generator] = None,
):
    """
    Sample one random schedule from the same search space as
    `all_schdules`, without enumerating every schedule.

    Builds one sampled schedule directly (no retry loop).
    """
    if rng is None:
        rng = np.random.default_rng()

    moduli = input_code.moduli
    polys = input_code.polys
    Dim = len(polys)

    monomial_basis = compute_basis(moduli)
    code = AlgebraicBlockCode(moduli, polys)
    l = int(np.array(moduli).prod())

    def get_q_pairs_x(mat_name: int, term: Term) -> Tuple[int, int]:
        mats = [np.zeros((l, l)) for _ in range(Dim)]
        mats[mat_name] = term.get_perm(moduli, monomial_basis).to_matrix()
        Hx, _ = construct_mats(
            [1]*Dim,
            [1]*Dim,
            mats
        )
        coords = np.where(Hx == 1)
        return [x for x in zip(coords[0], coords[1])]

    def get_q_pairs_z(mat_name: int, term: Term) -> Tuple[int, int]:
        mats = [np.zeros((l, l)) for _ in range(Dim)]
        mats[mat_name] = term.get_perm(moduli, monomial_basis).to_matrix()
        _, Hz = construct_mats(
            [1]*Dim,
            [1]*Dim,
            mats
        )
        coords = np.where(Hz == 1)
        return [x for x in zip(coords[0], coords[1])]

    p_ord = [rng.permutation(len(poly)).tolist() for poly in polys]
    new_polys = [[polys[j][i] for i in p_ord[j]] for j in range(Dim)]
    l_polys = [new_polys[i][:len(new_polys[i])//2] for i in range(Dim)]
    g_polys = [new_polys[i][len(new_polys[i])//2:] for i in range(Dim)]

    stages = rng.permutation(len(polys)).tolist()
    center_mat = stages[-1]

    # Independent odd split choices only for non-center matrices.
    for mat_name in stages[:-1]:
        split_opts = _split_poly_options(new_polys[mat_name])
        if len(split_opts) > 1:
            choice_idx = int(rng.integers(len(split_opts)))
            l_choice, g_choice = split_opts[choice_idx]
            l_polys[mat_name] = list(l_choice)
            g_polys[mat_name] = list(g_choice)

    perm_l_polys = [list(inner) for inner in l_polys]
    perm_g_polys = [list(inner) for inner in g_polys]
    for mat_name in stages[:-1]:
        l_idx = rng.permutation(len(l_polys[mat_name]))
        g_idx = rng.permutation(len(g_polys[mat_name]))
        perm_l_polys[mat_name] = [l_polys[mat_name][i] for i in l_idx]
        perm_g_polys[mat_name] = [g_polys[mat_name][i] for i in g_idx]

    x_depth = 0
    z_depth = 0
    _implement_perm_choice(code, stages, l_polys, g_polys, perm_l_polys, perm_g_polys,
                            center_mat, new_polys[center_mat], get_q_pairs_x, get_q_pairs_z)
    return code.s_x_check.copy(), code.s_z_check.copy()
