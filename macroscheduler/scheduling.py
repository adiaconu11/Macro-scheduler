"""
This is the general file for the scheduling module.
"""

import copy
from typing import List, Optional, Tuple
import numpy as np
import itertools
import stim

from .qecc.qalp import construct_lifted_product_mats, compute_basis, Term, LiftedMatrix
from .qecc.qalp import Block
from .qecc.qalp import AlgebraicBlockCode, QALP
from .qecc.stim import generate_stim_str
from .qecc.utils_graphs import bipartite_edge_coloring


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

def _copy_schedule(code):
    return copy.deepcopy(code.s_x_check), copy.deepcopy(code.s_z_check)

def _single_term_lifted_mats(Dim: int, l: int, mat_name: int, perm) -> List[LiftedMatrix]:
    mats = [
        LiftedMatrix([[Block([])]], l, validate=False)
        for _ in range(Dim)
    ]
    mats[mat_name] = LiftedMatrix([[Block([perm])]], l, validate=False)
    return mats

def _all_schedules_ABC(input_code: AlgebraicBlockCode, max_schedules: Optional[int] = None):
    if max_schedules is not None:
        if max_schedules < 0:
            raise ValueError("max_schedules must be non-negative")
        if max_schedules == 0:
            return []

    moduli = input_code.moduli
    polys = input_code.polys
    Dim = len(polys)
    
    monomial_basis = compute_basis(moduli)
    code = AlgebraicBlockCode(moduli, polys)
    l = int(np.array(moduli).prod())

    all_schedules = []
    def get_q_pairs_x(mat_name: int, term: Term):
        """
        Get the qubit pairs for the X checks from the given terms.

        Args:
            mat_name (int): Matrix index (`0` for A, `1` for B, `2` for C, `3` for D).
            term (Term): Monomial term to materialize into the selected matrix.

        Returns:
            list[tuple[int, int]]: `(check_idx, qubit_idx)` pairs where `Hx == 1`.
        """
        mats = _single_term_lifted_mats(
            Dim,
            l,
            mat_name,
            term.get_perm(moduli, monomial_basis),
        )
        Hx, _ = construct_lifted_product_mats(mats)
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
        mats = _single_term_lifted_mats(
            Dim,
            l,
            mat_name,
            term.get_perm(moduli, monomial_basis),
        )
        _, Hz = construct_lifted_product_mats(mats)
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
                    all_schedules.append(_copy_schedule(code))
                    if max_schedules is not None and len(all_schedules) >= max_schedules:
                        return all_schedules
                
    return all_schedules

def color_protograph(M: LiftedMatrix) -> List[List[Tuple[int, int, int]]]:
    """
    Find a minimum coloring for the protograph described by mat.
    Each M_{i,j}^(k) is an edge of the multigraph.
    """
    M_tilde = M.M_tilde

    num_node_left = M.m # left is the checks
    num_node = M.m + M.n # right is the data sectors

    adj_left = [[] for _ in range(num_node_left)]
    edges = []
    edge_records = []

    for i in range(M.m):
        for j in range(M.n):
            for k, _ in enumerate(M_tilde[i][j]):
                rnode = j + num_node_left
                adj_left[i].append(rnode)
                edges.append((i, rnode))
                edge_records.append((i, j, k))

    edge_colors = bipartite_edge_coloring(num_node, num_node_left, adj_left, edges)
    if not edge_colors:
        return []

    color_classes = [[] for _ in range(max(edge_colors) + 1)]
    for edge_record, color in zip(edge_records, edge_colors):
        color_classes[color].append(edge_record)
    return color_classes

def _lifted_matrix_for_color_class(M: LiftedMatrix, color_class: List[Tuple[int, int, int]]) -> LiftedMatrix:
    blocks = [[Block([]) for _ in range(M.n)] for _ in range(M.m)]
    for i, j, k in color_class:
        blocks[i][j].append(M.M_tilde[i][j][k])
    return LiftedMatrix(blocks, M.l, validate=False)

def _zero_lifted_like(M: LiftedMatrix) -> LiftedMatrix:
    return LiftedMatrix(
        [[Block([]) for _ in range(M.n)] for _ in range(M.m)],
        M.l,
        validate=False,
    )

def _make_qalp_pair_getter(input_code: QALP, protograph_colors):
    color_mats = {}
    for mat_name, lifted_matrix in enumerate(input_code.lifted_check_mats):
        for color_id, color_class in enumerate(protograph_colors[mat_name]):
            color_mats[(mat_name, color_id)] = _lifted_matrix_for_color_class(
                lifted_matrix,
                color_class,
            )

    pair_cache = {}

    def get_q_pairs(side: str, mat_name: int, color_id: int) -> List[Tuple[int, int]]:
        key = (side, mat_name, color_id)
        if key not in pair_cache:
            mats = [
                _zero_lifted_like(lifted_matrix)
                for lifted_matrix in input_code.lifted_check_mats
            ]
            mats[mat_name] = color_mats[(mat_name, color_id)]
            Hx, Hz = construct_lifted_product_mats(mats)
            selected = Hx if side == 'X' else Hz
            coords = np.where(selected != 0)
            pair_cache[key] = [
                (int(check_no), int(qubit_no))
                for check_no, qubit_no in zip(coords[0], coords[1])
            ]
        return pair_cache[key]

    return get_q_pairs

def _assign_qalp_pairs(code: QALP, side: str, q_pairs: List[Tuple[int, int]], stage: int):
    schedule = code.s_x_check if side == 'X' else code.s_z_check
    for check_no, qubit_no in q_pairs:
        pos = code.qubit_pos_in_check(side, check_no, qubit_no)
        if pos is None:
            raise ValueError(f"{side}-check {check_no} does not contain qubit {qubit_no}")
        if schedule[check_no][pos] != -1:
            raise ValueError(
                f"{side}-check {check_no}, qubit {qubit_no} was scheduled twice"
            )
        schedule[check_no][pos] = stage

def _implement_qalp_choice(
        code: QALP,
        stages,
        l_colors,
        g_colors,
        perm_l_colors,
        perm_g_colors,
        center_mat,
        center_colors,
        get_q_pairs):
    x_depth = 0
    z_depth = 0
    code.init_se_schedule_ir()

    # Left side: lesser colors on X, greater colors on Z.
    for mat_name in stages[:-1]:
        for color_id in l_colors[mat_name]:
            _assign_qalp_pairs(
                code,
                'X',
                get_q_pairs('X', mat_name, color_id),
                x_depth,
            )
            x_depth += 1

        for color_id in perm_g_colors[mat_name]:
            _assign_qalp_pairs(
                code,
                'Z',
                get_q_pairs('Z', mat_name, color_id),
                z_depth,
            )
            z_depth += 1
        x_depth = z_depth = max(x_depth, z_depth)

    # Center: all colors on both X and Z in the same order.
    for color_id in center_colors:
        _assign_qalp_pairs(
            code,
            'X',
            get_q_pairs('X', center_mat, color_id),
            x_depth,
        )
        _assign_qalp_pairs(
            code,
            'Z',
            get_q_pairs('Z', center_mat, color_id),
            z_depth,
        )
        x_depth += 1
        z_depth += 1

    # Right side: greater colors on X, lesser colors on Z.
    for mat_name in reversed(stages[:-1]):
        for color_id in g_colors[mat_name]:
            _assign_qalp_pairs(
                code,
                'X',
                get_q_pairs('X', mat_name, color_id),
                x_depth,
            )
            x_depth += 1

        for color_id in perm_l_colors[mat_name]:
            _assign_qalp_pairs(
                code,
                'Z',
                get_q_pairs('Z', mat_name, color_id),
                z_depth,
            )
            z_depth += 1
        x_depth = z_depth = max(x_depth, z_depth)

    code.depth = int(max(x_depth, z_depth))
    code.validate_se_schedule()

def _validate_qalp_dimension(input_code: QALP):
    Dim = len(input_code.lifted_check_mats)
    if Dim not in (2, 3, 4):
        raise NotImplementedError("Only 2D and 3D QALP schedules are supported")

def _all_schedules_QALP(input_code: QALP, max_schedules: Optional[int] = None):
    if max_schedules is not None:
        if max_schedules < 0:
            raise ValueError("max_schedules must be non-negative")
        if max_schedules == 0:
            return []

    _validate_qalp_dimension(input_code)
    Dim = len(input_code.lifted_check_mats)
    protograph_colors = [
        color_protograph(lifted_matrix)
        for lifted_matrix in input_code.lifted_check_mats
    ]
    get_q_pairs = _make_qalp_pair_getter(input_code, protograph_colors)
    code = QALP(
        input_code.lifted_check_mats,
        validate=False,
    )
    all_schedules = []

    color_ords = [
        list(itertools.permutations(range(len(color_classes))))
        for color_classes in protograph_colors
    ]
    stage_ords = list(itertools.permutations(range(Dim)))

    for c_ord in itertools.product(*color_ords):
        new_colors = [list(c_ord[i]) for i in range(Dim)]

        for stages in stage_ords:
            center_mat = stages[-1]

            l_colors_base = [new_colors[i][:len(new_colors[i])//2] for i in range(Dim)]
            g_colors_base = [new_colors[i][len(new_colors[i])//2:] for i in range(Dim)]

            split_sources = []
            split_targets = []
            for mat_name in stages[:-1]:
                split_opts = _split_poly_options(new_colors[mat_name])
                if len(split_opts) > 1:
                    split_sources.append(split_opts)
                    split_targets.append(mat_name)

            split_products = itertools.product(*split_sources) if split_sources else [()]

            for split_choice in split_products:
                l_colors = [list(p) for p in l_colors_base]
                g_colors = [list(p) for p in g_colors_base]
                for mat_name, (l_choice, g_choice) in zip(split_targets, split_choice):
                    l_colors[mat_name] = list(l_choice)
                    g_colors[mat_name] = list(g_choice)

                perm_sources = []
                perm_targets = []
                for mat_name in stages[:-1]:
                    perm_sources.append(itertools.permutations(l_colors[mat_name]))
                    perm_targets.append(("l", mat_name))
                    perm_sources.append(itertools.permutations(g_colors[mat_name]))
                    perm_targets.append(("g", mat_name))

                perm_products = itertools.product(*perm_sources) if perm_sources else [()]
                for permuted_lists in perm_products:
                    perm_l_colors = [list(inner) for inner in l_colors]
                    perm_g_colors = [list(inner) for inner in g_colors]

                    for (side, mat_name), perm in zip(perm_targets, permuted_lists):
                        if side == "l":
                            perm_l_colors[mat_name] = list(perm)
                        else:
                            perm_g_colors[mat_name] = list(perm)

                    _implement_qalp_choice(
                        code,
                        stages,
                        l_colors,
                        g_colors,
                        perm_l_colors,
                        perm_g_colors,
                        center_mat,
                        new_colors[center_mat],
                        get_q_pairs,
                    )

                    all_schedules.append(_copy_schedule(code))
                    if max_schedules is not None and len(all_schedules) >= max_schedules:
                        return all_schedules

    return all_schedules

def all_schedules(input_code: AlgebraicBlockCode | QALP, max_schedules: Optional[int] = None):
    """
    Will compute all possible schedules according to Daniel's construction.
    Returns a list of tuples (s_x_check, s_z_check) for each schedule.
    """

    if isinstance(input_code, AlgebraicBlockCode):
        return _all_schedules_ABC(input_code, max_schedules=max_schedules)
    elif isinstance(input_code, QALP):
        return _all_schedules_QALP(input_code, max_schedules=max_schedules)
    raise TypeError("input_code must be an AlgebraicBlockCode or QALP")

def _random_schedule_ABC(
    input_code: AlgebraicBlockCode,
    rng: np.random.Generator,
):
    moduli = input_code.moduli
    polys = input_code.polys
    Dim = len(polys)

    monomial_basis = compute_basis(moduli)
    code = AlgebraicBlockCode(moduli, polys)
    l = int(np.array(moduli).prod())

    def get_q_pairs_x(mat_name: int, term: Term) -> List[Tuple[int, int]]:
        mats = _single_term_lifted_mats(
            Dim,
            l,
            mat_name,
            term.get_perm(moduli, monomial_basis),
        )
        Hx, _ = construct_lifted_product_mats(mats)
        coords = np.where(Hx == 1)
        return [x for x in zip(coords[0], coords[1])]

    def get_q_pairs_z(mat_name: int, term: Term) -> List[Tuple[int, int]]:
        mats = _single_term_lifted_mats(
            Dim,
            l,
            mat_name,
            term.get_perm(moduli, monomial_basis),
        )
        _, Hz = construct_lifted_product_mats(mats)
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

    _implement_perm_choice(code, stages, l_polys, g_polys, perm_l_polys, perm_g_polys,
                            center_mat, new_polys[center_mat], get_q_pairs_x, get_q_pairs_z)
    return _copy_schedule(code)

def _random_schedule_QALP(
    input_code: QALP,
    rng: np.random.Generator,
):
    _validate_qalp_dimension(input_code)
    Dim = len(input_code.lifted_check_mats)
    protograph_colors = [
        color_protograph(lifted_matrix)
        for lifted_matrix in input_code.lifted_check_mats
    ]
    get_q_pairs = _make_qalp_pair_getter(input_code, protograph_colors)

    for _ in range(1000):
        code = QALP(
            input_code.lifted_check_mats,
            validate=False,
        )
        new_colors = [
            rng.permutation(len(color_classes)).tolist()
            for color_classes in protograph_colors
        ]
        l_colors = [new_colors[i][:len(new_colors[i])//2] for i in range(Dim)]
        g_colors = [new_colors[i][len(new_colors[i])//2:] for i in range(Dim)]

        stages = rng.permutation(Dim).tolist()
        center_mat = stages[-1]

        for mat_name in stages[:-1]:
            split_opts = _split_poly_options(new_colors[mat_name])
            if len(split_opts) > 1:
                choice_idx = int(rng.integers(len(split_opts)))
                l_choice, g_choice = split_opts[choice_idx]
                l_colors[mat_name] = list(l_choice)
                g_colors[mat_name] = list(g_choice)

        perm_l_colors = [list(inner) for inner in l_colors]
        perm_g_colors = [list(inner) for inner in g_colors]
        for mat_name in stages[:-1]:
            l_idx = rng.permutation(len(l_colors[mat_name]))
            g_idx = rng.permutation(len(g_colors[mat_name]))
            perm_l_colors[mat_name] = [l_colors[mat_name][i] for i in l_idx]
            perm_g_colors[mat_name] = [g_colors[mat_name][i] for i in g_idx]

        _implement_qalp_choice(
            code,
            stages,
            l_colors,
            g_colors,
            perm_l_colors,
            perm_g_colors,
            center_mat,
            new_colors[center_mat],
            get_q_pairs,
        )

        return _copy_schedule(code)

    schedules = _all_schedules_QALP(input_code, max_schedules=1)
    if schedules:
        return schedules[0]
    raise ValueError("could not find a valid QALP schedule")

def random_schedule(
    input_code: AlgebraicBlockCode | QALP,
    rng: Optional[np.random.Generator] = None,
):
    """
    Sample one random schedule from the same search space as
    `all_schdules`, without enumerating every schedule.

    AlgebraicBlockCode schedules are built directly. General QALP schedules may
    resample macro choices until one passes the schedule validator.
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(input_code, AlgebraicBlockCode):
        return _random_schedule_ABC(input_code, rng)
    if isinstance(input_code, QALP):
        return _random_schedule_QALP(input_code, rng)
    raise TypeError("input_code must be an AlgebraicBlockCode or QALP")
