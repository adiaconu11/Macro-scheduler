from itertools import combinations, permutations
import random

import numpy as np
from typing import List, Optional

from .css import CSS_Code
from .qalp import Perm


def func(a: int, b: int, p: int) -> List[int]:
    return [(a*j + b) % p for j in range(p)]

class KasaiCode(CSS_Code):
    def __init__(self, L, J, P, f_params, g_params, validate=True):
        # f_params and g_params are tuples of (a, b) for the linear functions
        if validate:
            pass
        
        L_over_2 = L // 2
        assert(len(f_params)) == L_over_2
        F_mats = [Perm(func(a, b, P)).to_matrix() for a, b in f_params]
        G_mats = [Perm(func(a, b, P)).to_matrix() for a, b in g_params]

        self.Hx_left_blocks =  [[F_mats[(j-i + L_over_2)%L_over_2] for j in range(L_over_2)] for i in range(J)]
        self.Hx_right_blocks = [[G_mats[(j-i + L_over_2)%L_over_2] for j in range(L_over_2)] for i in range(J)]

        Hx_left_rows = [np.hstack(row) for row in self.Hx_left_blocks]
        Hx_right_rows = [np.hstack(row) for row in self.Hx_right_blocks]
        Hx_left = np.vstack(Hx_left_rows)
        Hx_right = np.vstack(Hx_right_rows)
        # Hx = np.block([Hx_left, Hx_right])
        Hx = np.hstack([Hx_left, Hx_right])

        # Check with Daniel if by inverting func, aren't you actually inverting the permutation?
        self.Hz_left_blocks =  [[G_mats[(i-j + L_over_2)%L_over_2].T for j in range(L_over_2)] for i in range(J)]
        self.Hz_right_blocks = [[F_mats[(i-j + L_over_2)%L_over_2].T for j in range(L_over_2)] for i in range(J)]
        Hz_left_rows = [np.hstack(row) for row in self.Hz_left_blocks]
        Hz_right_rows = [np.hstack(row) for row in self.Hz_right_blocks]
        Hz_left = np.vstack(Hz_left_rows)
        Hz_right = np.vstack(Hz_right_rows)
        Hz = np.hstack([Hz_left, Hz_right])

        self.J = J
        self.P = P
        self.L_over_2 = L_over_2
        self.F_mats = F_mats
        self.G_mats = G_mats

        super().__init__(Hx=Hx, Hz=Hz)

def get_qubit_pairs_kasai(code: KasaiCode, check_type:str, mat_type: str, mat_idx: int):
    base_col_offset = 0
    if check_type == 'X':
        if mat_type == 'F':
            mat = code.F_mats[mat_idx]
        elif mat_type == 'G':
            base_col_offset = code.L_over_2 * code.P
            mat = code.G_mats[mat_idx]
        else:
            raise ValueError("mat_type must be 'F' or 'G'")
    elif check_type == 'Z':
        if mat_type == 'G':
            mat = code.G_mats[mat_idx].T
        elif mat_type == 'F':
            base_col_offset = code.L_over_2 * code.P
            mat = code.F_mats[mat_idx].T
        else:
            raise ValueError("mat_type must be 'F' or 'G'")
    else:
        raise ValueError("check_type must be 'X' or 'Z'")

    raw_coords = np.where(mat == 1)
    ans = []
    for block_row in range(code.J):
        if check_type == 'X':
            block_col = (mat_idx + block_row) % code.L_over_2
        else:
            block_col = (block_row - mat_idx) % code.L_over_2

        row_offset = block_row * code.P
        col_offset = base_col_offset + block_col * code.P
        row_coords = raw_coords[0] + row_offset
        col_coords = raw_coords[1] + col_offset
        ans = ans + list(zip(row_coords, col_coords))
    return ans

def _implement_perm_kasai(code: KasaiCode, 
                          middle: str, mid_perm: List[int],
                          early_x: List[int], early_z: List[int],
                          late_x: List[int], late_z: List[int]):
    code.init_se_schedule_ir()
    x_depth = z_depth = 0
    if middle == 'F':
        outside = 'G'
    elif middle == 'G':
        outside = 'F'
    else:
        raise ValueError("Middle should be either F or G!")

    # Early
    for mat_idx in early_x:
        q_pairs = get_qubit_pairs_kasai(code, 'X', outside, mat_idx)
        for check_no, qubit_no in q_pairs:
            code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
        x_depth += 1
    for mat_idx in early_z:
        q_pairs = get_qubit_pairs_kasai(code, 'Z', outside, mat_idx)
        for check_no, qubit_no in q_pairs:
            code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
        z_depth += 1

    x_depth = z_depth = max(x_depth, z_depth) # in case the two sides aren't equal

    # Middle
    for mat_idx in mid_perm:
        q_pairs_x = get_qubit_pairs_kasai(code, 'X', middle, mat_idx)
        q_pairs_z = get_qubit_pairs_kasai(code, 'Z', middle, mat_idx)
        for check_no, qubit_no in q_pairs_x:
            code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
        for check_no, qubit_no in q_pairs_z:
            code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
        x_depth += 1
        z_depth += 1
    
    # Late
    for mat_idx in late_x:
        q_pairs = get_qubit_pairs_kasai(code, 'X', outside, mat_idx)
        for check_no, qubit_no in q_pairs:
            code.s_x_check[check_no][code.qubit_pos_in_check('X', check_no, qubit_no)] = x_depth
        x_depth += 1
    for mat_idx in late_z:
        q_pairs = get_qubit_pairs_kasai(code, 'Z', outside, mat_idx)
        for check_no, qubit_no in q_pairs:
            code.s_z_check[check_no][code.qubit_pos_in_check('Z', check_no, qubit_no)] = z_depth
        z_depth += 1
    
    code.depth = int(max(x_depth, z_depth))
    code.validate_se_schedule()

def all_schedules(code: KasaiCode, max_schedules: Optional[int] = None):
    if max_schedules is not None and max_schedules < 0:
        raise ValueError("max_schedules must be non-negative")

    schedules = []
    if max_schedules == 0:
        return schedules
    
    L_over_2 = code.L_over_2
    mid = L_over_2 // 2
    remainder = L_over_2 % 2

    def _perm_n_implement(sec_A: List[int], sec_B: List[int]) -> bool:
        for early_X in permutations(sec_A):
            for late_X in permutations(sec_B):
                for early_Z in permutations(sec_B):
                    for late_Z in permutations(sec_A):
                        for mid_perm in permutations(list(range(L_over_2))):
                            mid_perm = list(mid_perm)
                            early_X = list(early_X)
                            early_Z = list(early_Z)
                            late_X = list(late_X)
                            late_Z = list(late_Z)
                            _implement_perm_kasai(code, 'G', mid_perm, early_X, early_Z, late_X, late_Z)
                            schedules.append((code.s_x_check, code.s_z_check))
                            if max_schedules is not None and len(schedules) >= max_schedules:
                                return True

                            _implement_perm_kasai(code, 'F', mid_perm, early_X, early_Z, late_X, late_Z)
                            schedules.append((code.s_x_check, code.s_z_check))
                            if max_schedules is not None and len(schedules) >= max_schedules:
                                return True
        return False

    if remainder == 0: # Even case: equal split
        for comb in combinations(range(L_over_2), mid):
            sec_A = list(comb)
            sec_B = [i for i in range(L_over_2) if i not in sec_A]
            if _perm_n_implement(sec_A, sec_B):
                return schedules
    else: # Odd case: split with remainder
        # Case 1: early X is a little smaller
        for comb in combinations(range(L_over_2), mid):
            sec_A = list(comb)
            sec_B = [i for i in range(L_over_2) if i not in sec_A]
            if _perm_n_implement(sec_A, sec_B):
                return schedules
        # Case 2: early X is a little bigger
        for comb in combinations(range(L_over_2), mid + 1):
            sec_A = list(comb)
            sec_B = [i for i in range(L_over_2) if i not in sec_A]
            if _perm_n_implement(sec_A, sec_B):
                return schedules
    
    return schedules


def random_schedule(code: KasaiCode):
    L_over_2 = code.L_over_2
    mid = L_over_2 // 2
    remainder = L_over_2 % 2

    if remainder == 0:
        split_size = mid
    else:
        split_size = random.choice([mid, mid + 1])

    sec_A = list(random.choice(list(combinations(range(L_over_2), split_size))))
    sec_B = [i for i in range(L_over_2) if i not in sec_A]

    early_x = sec_A[:]
    late_x = sec_B[:]
    early_z = sec_B[:]
    late_z = sec_A[:]
    mid_perm = list(range(L_over_2))

    random.shuffle(early_x)
    random.shuffle(late_x)
    random.shuffle(early_z)
    random.shuffle(late_z)
    random.shuffle(mid_perm)

    middle = random.choice(['F', 'G'])
    _implement_perm_kasai(code, middle, mid_perm, early_x, early_z, late_x, late_z)

    return (code.s_x_check, code.s_z_check)

class Kasai_Code(CSS_Code):
    def __init__(self, twoL, J, P, f_a, f_b, g_a, g_b, Hx = 0, Hz = 0):
        L = twoL // 2
        assert L % 2 == 0, "L must be even"
        assert J >= 3, "J must be at least 3"
        assert J <= L // 2, "J must be at most L/2"
        assert len(f_a) == L, "f_a must have length L"
        assert len(f_b) == L, "f_b must have length L"
        assert len(g_a) == L, "g_a must have length L"
        assert len(g_b) == L, "g_b must have length L"
        def gcd(x, y): # Euclidean algorithm for greatest common divisor
            while y:
                x, y = y, x % y
            return x
        for a in f_a:
            assert gcd(a, P) == 1, "f_a elements must be coprime to P"
        for a in g_a:
            assert gcd(a, P) == 1, "g_a elements must be coprime to P"

       
        def mod_inv(a, P):
            def extended_gcd(a, b):
                """
                Returns (gcd, x, y) such that:
                a*x + b*y = gcd
                """
                if b == 0:
                    return a, 1, 0
                gcd, x1, y1 = extended_gcd(b, a % b)
                x = y1
                y = x1 - (a // b) * y1
                return gcd, x, y
            """
            Returns the modular inverse of a modulo p.
            Raises ValueError if inverse does not exist.
            """
            gcd, x, _ = extended_gcd(a, P)
            if gcd != 1:
                raise ValueError(f"No modular inverse exists for {a} mod {P}")
            return x % P
        
        f_a_inv = [mod_inv(a, P) % P for a in f_a]
        g_a_inv = [mod_inv(a, P) % P for a in g_a]
        f_b_inv = [(-f_a_inv[i] * f_b[i]) % P for i in range(L)]
        g_b_inv = [(-g_a_inv[i] * g_b[i]) % P for i in range(L)]

        x_checks = []
        s_x_checks = []
        for j in range(J):
            for p in range(P):
                x_check = []
                s_x_check = []
                for l in range(L):
                    f_id = (l-j) % L
                    x_check.append(
                        P*l + (f_a_inv[f_id] * p + f_b_inv[f_id]) % P
                    )
                    if f_id < L // 2:
                        s_x_check.append(f_id)
                    else:
                        s_x_check.append(f_id + L)
                for r in range(L):
                    g_id = (r-j) % L
                    x_check.append(
                        P*(L+r) + (g_a_inv[g_id] * p + g_b_inv[g_id]) % P
                    )
                    s_x_check.append(g_id + L // 2)
                x_checks.append(x_check)
                s_x_checks.append(s_x_check)
        
        z_checks = []
        s_z_checks = []
        for j in range(J):
            for p in range(P):
                z_check = []
                s_z_check = []
                for l in range(L):
                    g_id = (j-l) % L
                    z_check.append(
                        P*l + (g_a[g_id] * p + g_b[g_id]) % P
                    )
                    s_z_check.append(g_id + L // 2)
                for r in range(L):
                    f_id = (j-r) % L
                    z_check.append(
                        P*(L+r) + (f_a[f_id] * p + f_b[f_id]) % P
                    )
                    if f_id < L // 2:
                        s_z_check.append(f_id + 3 * (L // 2))
                    else:
                        s_z_check.append(f_id - L // 2)
                z_checks.append(z_check)
                s_z_checks.append(s_z_check)

        super().__init__(x_checks=x_checks, z_checks=z_checks)
        # super().__init__(Hx=Hx, Hz=Hz)
        # compare self.x_checks with x_checks and self.z_checks with z_checks to make sure they match
        # assert np.array_equal(self.x_checks, x_checks), "x_checks do not match"
        # check which check does not match
        for i, (a, b) in enumerate(zip(self.x_checks, x_checks)):
            if not np.array_equal(a, b):
                print(f"x_check {i} does not match")
                print(f"Expected: {b}")
                print(f"Got: {a}")
                break
        assert np.array_equal(self.z_checks, z_checks), "z_checks do not match"
        # self.set_se_schedule_ir(s_x_checks, s_z_checks, depth=L)
        self.set_se_schedule_ir(s_x_checks, s_z_checks)
        print(self.depth)
        self.validate_se_schedule()

# with open("macroscheduler/qecc/kasai_hx.npy", "rb") as f:
#     Hx = np.load(f, allow_pickle=True)
# with open("macroscheduler/qecc/kasai_hz.npy", "rb") as f:
#     Hz = np.load(f, allow_pickle=True)

if __name__ == "__main__":
    tmp = Kasai_Code(
        12, 3, 768,
        (763, 679, 397, 61, 697, 373),
        (435, 69, 330, 18, 612, 246),
        (289, 257, 625, 41, 193, 449),
        (496, 640, 200, 524, 672, 672),
        # Hx=Hx,
        # Hz=Hz
    )

# andrei = KasaiCode(
#     L=12,
#     J=3,
#     P=768,
#     f_params=[(763, 435), (679, 69), (397, 330), (61, 18), (697, 612), (373, 246)],
#     g_params=[(289, 496), (257, 640), (625, 200), (41, 524), (193, 672), (449, 672)]
# )

# assert np.array_equal(tmp.Hx, andrei.Hx), "Hx does not match"
# assert np.array_equal(tmp.Hz, andrei.Hz), "Hz does not match"

# tmp = Kasai_Code(
#     12, 3, 8,
#     (5, 5, 1, 5, 5, 1),
#     (7, 3, 6, 7, 3, 6),
#     (5, 5, 5, 5, 5, 5),
#     (7, 5, 7, 7, 5, 7)
# )
