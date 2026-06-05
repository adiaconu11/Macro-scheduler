from typing import List, Optional, Tuple, Dict, Any
import random
import copy
from .utils_graphs import bipartite_edge_coloring

class CSS_Code:
    """
    This is the main class for manipulating CSS codes.
    
    Attributes:
        num_data_q (int): number of data qubits
        num_x_check (int): number of X-type check qubits
        num_z_check (int): number of Z-type check qubits
        num_total_q (int): total number of qubits
        x_checks (List[List[int]]): list of X checks, each check is a list of data qubit indices
        z_checks (List[List[int]]): list of Z checks, each check is a list of data qubit indices
        Hx (List[List[int]]): parity-check matrix for X checks
        Hz (List[List[int]]): parity-check matrix for Z checks
        x_logicals (List[List[int]]): list of X logical operators, each is a list of data qubit indices
        z_logicals (List[List[int]]): list of Z logical operators, each is a list of data qubit indices
        s_x_check (List[List[int]]): s_x_check[i][j] is the stage of j-th data qubit interaction in i-th X check (0-indexed)
        s_z_check (List[List[int]]): s_z_check[i][j] is the stage of j-th data qubit interaction in i-th Z check (0-indexed)
        depth (int): total number of CNOT layers in the syndrome extraction schedule
    """
    
    def __init__(
            self,
            code_str: Optional[str] = None,
            if_logical: Optional[bool] = False,
            x_checks: Optional[List[List[int]]] = None,
            z_checks: Optional[List[List[int]]] = None,
            Hx: Optional[List[List[int]]] = None,
            Hz: Optional[List[List[int]]] = None,
            num_data_q: Optional[int] = None,
            x_logicals: Optional[List[List[int]]] = None,
            z_logicals: Optional[List[List[int]]] = None) -> None:
        """Load or construct a CSS code.

        Supported construction modes:
        1) From code string (legacy): CSS_Code(code_str, if_logical=True/False)
        2) From checks: CSS_Code(x_checks=..., z_checks=..., num_data_q=...)
        3) From parity-check matrices: CSS_Code(Hx=..., Hz=...)
        """
        self.x_checks = []
        self.z_checks = []
        self.x_logicals = []
        self.z_logicals = []

        if code_str is not None:
            self._init_from_code_str(code_str, if_logical=if_logical)
        else:
            self._init_from_components(
                x_checks=x_checks,
                z_checks=z_checks,
                Hx=Hx,
                Hz=Hz,
                num_data_q=num_data_q,
                x_logicals=x_logicals,
                z_logicals=z_logicals,
            )

        self.compute_degree()
        self.compute_edges()
        self.init_se_schedule_ir()

    def _init_from_code_str(self, code_str: str, if_logical: bool = False) -> None:
        text_lines = code_str.split('\n')
        line = text_lines[0].strip().split(' ')
        self.num_data_q = int(line[1])
        self.num_x_check = int(line[2])
        self.num_z_check = int(line[3])
        self.num_total_q = self.num_data_q + self.num_x_check + self.num_z_check
        self.Hx = [[0 for _ in range(self.num_data_q)] for _ in range(self.num_x_check)]
        self.Hz = [[0 for _ in range(self.num_data_q)] for _ in range(self.num_z_check)]

        if if_logical:
            self.num_logical_q = int(line[4])

        for text_line in text_lines[1:]:
            line = text_line.strip().split(' ')
            if len(line) == 1 and line[0] == '':
                continue
            if line[-1] == 'X':
                self.x_checks.append([int(num) for num in line[:-1]])
                for num in line[:-1]:
                    self.Hx[len(self.x_checks) - 1][int(num)] = 1
            if line[-1] == 'Z':
                self.z_checks.append([int(num) for num in line[:-1]])
                for num in line[:-1]:
                    self.Hz[len(self.z_checks) - 1][int(num)] = 1
            if if_logical and line[-1] == 'LX':
                self.x_logicals.append([int(num) for num in line[:-1]])
            if if_logical and line[-1] == 'LZ':
                self.z_logicals.append([int(num) for num in line[:-1]])

    def _init_from_components(
            self,
            x_checks: Optional[List[List[int]]],
            z_checks: Optional[List[List[int]]],
            Hx: Optional[List[List[int]]],
            Hz: Optional[List[List[int]]],
            num_data_q: Optional[int],
            x_logicals: Optional[List[List[int]]],
            z_logicals: Optional[List[List[int]]]) -> None:
        has_checks = x_checks is not None and z_checks is not None
        has_mats = Hx is not None and Hz is not None
        if not has_checks and not has_mats:
            raise ValueError("Provide either (x_checks, z_checks) or (Hx, Hz), or pass code_str")

        if has_mats:
            self.Hx = [list(map(int, row)) for row in Hx]
            self.Hz = [list(map(int, row)) for row in Hz]
            if self.Hx and self.Hz and len(self.Hx[0]) != len(self.Hz[0]):
                raise ValueError("Hx and Hz must have the same number of columns")
            self.num_x_check = len(self.Hx)
            self.num_z_check = len(self.Hz)
            self.num_data_q = len(self.Hx[0]) if self.Hx else (len(self.Hz[0]) if self.Hz else 0)
            self.compute_checks_from_H()

        if has_checks:
            self.x_checks = [[int(q) for q in check] for check in x_checks]
            self.z_checks = [[int(q) for q in check] for check in z_checks]
            self.num_x_check = len(self.x_checks)
            self.num_z_check = len(self.z_checks)
            if num_data_q is None:
                max_x = max((max(check) if check else -1 for check in self.x_checks), default=-1)
                max_z = max((max(check) if check else -1 for check in self.z_checks), default=-1)
                self.num_data_q = max(max_x, max_z) + 1
            else:
                self.num_data_q = num_data_q
            self.compute_H_from_checks()

            if has_mats:
                self.assert_consistent_representations(Hx=Hx, Hz=Hz)

        self.num_total_q = self.num_data_q + self.num_x_check + self.num_z_check
        self.x_logicals = [] if x_logicals is None else x_logicals
        self.z_logicals = [] if z_logicals is None else z_logicals

    def compute_H_from_checks(self) -> None:
        self.Hx = [[0 for _ in range(self.num_data_q)] for _ in range(self.num_x_check)]
        self.Hz = [[0 for _ in range(self.num_data_q)] for _ in range(self.num_z_check)]
        for i, check in enumerate(self.x_checks):
            for q in check:
                if q < 0 or q >= self.num_data_q:
                    raise ValueError(f"X-check index out of range: q={q}, num_data_q={self.num_data_q}")
                self.Hx[i][q] = 1
        for i, check in enumerate(self.z_checks):
            for q in check:
                if q < 0 or q >= self.num_data_q:
                    raise ValueError(f"Z-check index out of range: q={q}, num_data_q={self.num_data_q}")
                self.Hz[i][q] = 1

    def compute_checks_from_H(self) -> None:
        if self.Hx and any(len(row) != len(self.Hx[0]) for row in self.Hx):
            raise ValueError("Hx rows have inconsistent lengths")
        if self.Hz and any(len(row) != len(self.Hz[0]) for row in self.Hz):
            raise ValueError("Hz rows have inconsistent lengths")

        self.x_checks = []
        self.z_checks = []

        for row in self.Hx:
            for entry in row:
                if entry not in (0, 1):
                    raise ValueError("Hx must be binary")
            self.x_checks.append([j for j, entry in enumerate(row) if entry == 1])

        for row in self.Hz:
            for entry in row:
                if entry not in (0, 1):
                    raise ValueError("Hz must be binary")
            self.z_checks.append([j for j, entry in enumerate(row) if entry == 1])
    
    def qubit_pos_in_check(self, check_type: str, check_idx: int, qubit_idx: int) -> Optional[int]:
        """Return the position of qubit_idx in the specified check, or None if not present."""
        if check_type == 'X':
            check = self.x_checks[check_idx]
        elif check_type == 'Z':
            check = self.z_checks[check_idx]
        else:
            raise ValueError("check_type must be 'X' or 'Z'")
        
        try:
            return check.index(qubit_idx)
        except ValueError:
            return None

    def assert_consistent_representations(
            self,
            Hx: Optional[List[List[int]]] = None,
            Hz: Optional[List[List[int]]] = None) -> None:
        """Assert that the provided Hx and Hz are consistent with the x_checks and z_checks."""
        if Hx is not None:
            if [list(map(int, row)) for row in Hx] != self.Hx:
                raise ValueError("Provided Hx is inconsistent with x_checks")
        if Hz is not None:
            if [list(map(int, row)) for row in Hz] != self.Hz:
                raise ValueError("Provided Hz is inconsistent with z_checks")

    def compute_degree(self) -> float:
        """Computes the tanner graph degree for this code."""
        self.degrees_data_q = [0 for _ in range(self.num_data_q)]
        self.degrees_data_q_xpart = [0 for _ in range(self.num_data_q)]
        self.degrees_data_q_zpart = [0 for _ in range(self.num_data_q)]
        self.degrees_x_check = [len(x_check) for x_check in self.x_checks]
        self.degrees_z_check = [len(z_check) for z_check in self.z_checks]
        for check in self.x_checks:
            for q in check:
                self.degrees_data_q_xpart[q] += 1
        for check in self.z_checks:
            for q in check:
                self.degrees_data_q_zpart[q] += 1
        for q in range(self.num_data_q):
            self.degrees_data_q[q] = self.degrees_data_q_xpart[q] + self.degrees_data_q_zpart[q]
        self.maxdeg_data_q = max(self.degrees_data_q)
        self.maxdeg_x_check = max(self.degrees_x_check)
        self.maxdeg_z_check = max(self.degrees_z_check)
        self.maxdeg_x_graph = max(self.maxdeg_x_check, max(self.degrees_data_q_xpart))
        self.maxdeg_z_graph = max(self.maxdeg_z_check, max(self.degrees_data_q_zpart))
        self.maxdeg = max(
            self.maxdeg_data_q, self.maxdeg_x_check, self.maxdeg_z_check)
        self.density_mindepth = 2 * (sum(self.degrees_x_check) + sum(self.degrees_z_check)) / self.maxdeg / self.num_total_q

    def export_str(self, print_logical: Optional[bool] = False) -> str:
        """
        Export CSS code to string defined in __init__.
        specify print_logical=True to include logical operators in the export (if they exist)
        """
        code_str = ''
        if print_logical and hasattr(self, 'num_logical_q'):
            code_str += (
                f"qecc {self.num_data_q} {self.num_x_check} "
                f"{self.num_z_check} {self.num_logical_q}\n"
            )
        else:
            code_str += (
                f"qecc {self.num_data_q} {self.num_x_check} "
                f"{self.num_z_check}\n"
            )
        for check in self.x_checks:
            code_str += (" ".join(str(num) for num in check) + " X\n")
        for check in self.z_checks:
            code_str += (" ".join(str(num) for num in check) + " Z\n")
        if print_logical:
            for logical in getattr(self, "x_logicals", []):
                code_str += (" ".join(str(num) for num in logical) + " LX\n")
            for logical in getattr(self, "z_logicals", []):
                code_str += (" ".join(str(num) for num in logical) + " LZ\n")
        return code_str

    def compute_edges(self) -> None:
        """Compute the edges of the Tanner graph and their mapping to check/qubit indices."""
        self.edges = []
        self.edge2id = {}
        for i, check in enumerate(self.x_checks):
            for q in sorted(check):
                self.edges.append(('X', i, q))
                self.edge2id[('X', i, q)] = len(self.edges) - 1
        for i, check in enumerate(self.z_checks):
            for q in sorted(check):
                self.edges.append(('Z', i, q))
                self.edge2id[('Z', i, q)] = len(self.edges) - 1

    def init_se_schedule_ir(self, fill_value: int = -1) -> None:
        """Initialize syndrome-extraction schedule IR.

        IR is represented by:
        - s_x_check[i][j]: stage of j-th data qubit interaction in i-th X check
        - s_z_check[i][j]: stage of j-th data qubit interaction in i-th Z check
        - depth: total number of CNOT layers (initialized to -1, meaning unassigned)
        """
        self.s_x_check = [[fill_value for _ in check] for check in self.x_checks]
        self.s_z_check = [[fill_value for _ in check] for check in self.z_checks]
        self.depth = -1

    def set_se_schedule_ir(
            self,
            s_x_check: List[List[int]],
            s_z_check: List[List[int]],
            depth: Optional[int] = None) -> None:
        """
        Set syndrome-extraction schedule IR after validating its shape.
        This function will not validate the correctness of the schedule!
        Will also update the depth if provided, otherwise will infer it from the schedule.
        """
        if len(s_x_check) != len(self.x_checks):
            raise ValueError("s_x_check length does not match number of X checks")
        if len(s_z_check) != len(self.z_checks):
            raise ValueError("s_z_check length does not match number of Z checks")

        for i, row in enumerate(s_x_check):
            if len(row) != len(self.x_checks[i]):
                raise ValueError(f"s_x_check[{i}] length does not match x_checks[{i}]")
        for i, row in enumerate(s_z_check):
            if len(row) != len(self.z_checks[i]):
                raise ValueError(f"s_z_check[{i}] length does not match z_checks[{i}]")

        self.s_x_check = copy.deepcopy(s_x_check)
        self.s_z_check = copy.deepcopy(s_z_check)
        if depth is None:
            self.depth = self.infer_depth_from_se_schedule()
        else:
            self.depth = depth

    def infer_depth_from_se_schedule(self) -> int:
        """Infer depth from IR as 1 + max assigned stage, or -1 if empty/unassigned."""
        all_stages = []
        for row in self.s_x_check:
            all_stages.extend([int(v) for v in row if int(v) >= 0])
        for row in self.s_z_check:
            all_stages.extend([int(v) for v in row if int(v) >= 0])
        if not all_stages:
            return -1
        return max(all_stages) + 1
    
    def validate_se_schedule(self) -> None:
        """
        Validate that the speficied SE schedule is fully assigned and correct!
        This function will raise an error if schedule is invalid, otherwise it does nothing.
        Will also check that the depth is correct!
        """
        if not self._has_complete_se_schedule():
            raise ValueError("SE schedule is not fully assigned")
        
        if not self._qubits_one_check_per_stage():
            raise ValueError("SE schedule has multiple qubits interacting in the same check at the same stage")
    
        self._check_consistent_ordering()
        for i in range(self.num_x_check):
            min_stage = min(self.s_x_check[i])
            max_stage = max(self.s_x_check[i])
            if min_stage < 0 or max_stage >= self.depth:
                raise ValueError(f"s_x_check[{i}] has invalid stages with min {min_stage} and max {max_stage} with depth {self.depth}")
        for i in range(self.num_z_check):
            min_stage = min(self.s_z_check[i])
            max_stage = max(self.s_z_check[i])
            if min_stage < 0 or max_stage >= self.depth:
                raise ValueError(f"s_z_check[{i}] has invalid stages with min {min_stage} and max {max_stage} with depth {self.depth}")

    def _check_consistent_ordering(self) -> None:
        """
        Check that X-checks and Z-checks are consistently ordered.
        This function implements the condition from 4.1, last bullet point.
        """
        for i, x_check in enumerate(self.x_checks):
            for ii, z_check in enumerate(self.z_checks):
                parity = 1
                for j, q in enumerate(x_check):
                    for jj, qq in enumerate(z_check):
                        if q == qq:
                            parity *= self.s_x_check[i][j]-self.s_z_check[ii][jj]
                if parity < 0:
                    raise ValueError(f"Incosistent ordering between X check {i} and Z check {ii}")

        # for i, x_check in enumerate(self.x_checks):
        #     for ii, z_check in enumerate(self.z_checks):
        #         common_qubits = list(set(x_check) & set(z_check))
        #         n_common = len(common_qubits)
        #         if n_common <= 1:
        #             continue
        #         for a in range(n_common):
        #             for b in range(a + 1, n_common):
        #                 j = common_qubits[a]
        #                 jj = common_qubits[b]
        #                 if (self.s_x_check[i][j] - self.s_z_check[ii][jj]) * (self.s_x_check[i][jj] - self.s_z_check[ii][j]) < 0:
        #                     raise ValueError(
        #                         f"Inconsistent ordering between X check {i} and Z check {ii} on qubits {j} and {jj}"
        #                     )

    def _qubits_one_check_per_stage(self) -> bool:
        """Return True iff each qubit interacts with at most one check per stage."""
        # For each stage, count how many times each qubit appears in any check
        stage_qubit_counts = {}
        for i, row in enumerate(self.s_x_check):
            for j, stage in enumerate(row):
                q = self.x_checks[i][j]
                if stage not in stage_qubit_counts:
                    stage_qubit_counts[stage] = {}
                if q not in stage_qubit_counts[stage]:
                    stage_qubit_counts[stage][q] = 0
                stage_qubit_counts[stage][q] += 1
        for i, row in enumerate(self.s_z_check):
            for j, stage in enumerate(row):
                q = self.z_checks[i][j]
                if stage not in stage_qubit_counts:
                    stage_qubit_counts[stage] = {}
                if q not in stage_qubit_counts[stage]:
                    stage_qubit_counts[stage][q] = 0
                stage_qubit_counts[stage][q] += 1

        # If any qubit appears more than once in any stage, return False
        for _, qubits in stage_qubit_counts.items():
            if any(count > 1 for count in qubits.values()):
                return False
        return True

    def _has_complete_se_schedule(self) -> bool:
        """Return True iff all schedule entries are assigned (>=0)."""
        for row in self.s_x_check:
            if any(int(v) < 0 for v in row):
                return False
        for row in self.s_z_check:
            if any(int(v) < 0 for v in row):
                return False
        return True

    def compute_logicals(self) -> None:
        """
        Compute the logical oprators of the CSS code.
        It considers n data qubits, rx X checks, rz Z checks.
        """
        import numpy as np
        from .utils_linalg import inverse, row_echelon, kernel, rank

        def compute_lz(ker_hx, im_hzT):
            log_stack = np.vstack([im_hzT, ker_hx])
            pivots = row_echelon(log_stack.T)[3]
            log_op_indices = [i for i in range(im_hzT.shape[0], log_stack.shape[0]) if i in pivots]
            log_ops = log_stack[log_op_indices]
            return log_ops

        HX = np.array(self.Hx)
        HZ = np.array(self.Hz)
        hz_perp, _, pivot_hz = kernel(HZ)
        hx_perp, _, pivot_hx = kernel(HX)

        hx_basis = HX[pivot_hx]
        hz_basis = HZ[pivot_hz]

        lx = compute_lz(hz_perp, hx_basis)
        lz = compute_lz(hx_perp, hz_basis)

        k_expected = int(self.num_data_q - rank(HX) - rank(HZ))
        if k_expected < 0:
            raise ValueError(f"Invalid CSS ranks: computed k={k_expected} < 0")
        if lx.shape[0] < k_expected or lz.shape[0] < k_expected:
            raise ValueError(
                "Failed to extract enough logical operators: "
                f"k={k_expected}, lx={lx.shape}, lz={lz.shape}"
            )

        commutation = lx @ lz.T % 2

        if lz.shape[0] > k_expected:
            pivot_cols = row_echelon(commutation)[3]
            if len(pivot_cols) < k_expected:
                raise ValueError(
                    "Commutation matrix does not have enough independent columns: "
                    f"need {k_expected}, have {len(pivot_cols)}"
                )
            keep_cols = pivot_cols[:k_expected]
            lz = lz[keep_cols]
            commutation = commutation[:, keep_cols]
        if lx.shape[0] > k_expected:
            pivot_cols = row_echelon(commutation.T)[3]
            if len(pivot_cols) < k_expected:
                raise ValueError(
                    "Commutation matrix does not have enough independent rows: "
                    f"need {k_expected}, have {len(pivot_cols)}"
                )
            keep_rows = pivot_cols[:k_expected]
            lx = lx[keep_rows]
            commutation = commutation[keep_rows]

        if lx.shape[0] != k_expected or lz.shape[0] != k_expected:
            raise ValueError(
                "Logical operator extraction produced wrong count after trimming: "
                f"k={k_expected}, lx={lx.shape}, lz={lz.shape}"
            )
        if commutation.shape != (k_expected, k_expected):
            raise ValueError(
                "Commutation matrix is not square after trimming: "
                f"shape={commutation.shape}, k={k_expected}"
            )
        if rank(commutation) != k_expected:
            raise ValueError(
                "Commutation matrix is singular; extracted logical bases are not paired. "
                f"rank={rank(commutation)}, k={k_expected}"
            )

        lz = inverse(commutation).T @ lz % 2

        if not np.array_equal(lx @ lz.T % 2, np.eye(k_expected, dtype=int)):
            raise ValueError("Failed to canonicalize logical commutation matrix")

        self.z_logicals = lz.tolist()
        self.x_logicals = lx.tolist()

    def schedule_coloring_separate(self) -> int:
        """
        Create a non-interleaved coloration SE circuit.
        The resulting circuit is minimum depth for a non-interleaved circuit.
        This function is agnostic to the structure (or lack thereof) of the CSS code.
        """
        self.schedule = [-1 for _ in self.edges]

        xgraph_neighbor = [[] for _ in range(self.num_data_q)]
        x_edges = []
        for i, check in enumerate(self.x_checks):
            for q in check:
                xgraph_neighbor[q].append(i + self.num_data_q)
                x_edges.append((q, i + self.num_data_q))
        x_schedule = bipartite_edge_coloring(
            self.num_data_q + self.num_x_check,
            self.num_data_q,
            xgraph_neighbor,
            x_edges,
        )
        x_depth = max(x_schedule) + 1
        for i, time in enumerate(x_schedule):
            q = x_edges[i][0]
            check_idx = x_edges[i][1] - self.num_data_q
            self.schedule[self.edge2id[('X', check_idx, q)]] = time

        zgraph_neighbor = [[] for _ in range(self.num_data_q)]
        z_edges = []
        for i, check in enumerate(self.z_checks):
            for q in check:
                zgraph_neighbor[q].append(i + self.num_data_q)
                z_edges.append((q, i + self.num_data_q))
        z_schedule = bipartite_edge_coloring(
            self.num_data_q + self.num_z_check,
            self.num_data_q,
            zgraph_neighbor,
            z_edges,
        )
        for i, time in enumerate(z_schedule):
            q = z_edges[i][0]
            check_idx = z_edges[i][1] - self.num_data_q
            self.schedule[self.edge2id[('Z', check_idx, q)]] = x_depth + time

        num_edges = len(self.edges)
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                ei = self.edges[i]  # (type, check_idx, qubit)
                ej = self.edges[j]
                same_qubit = ei[2] == ej[2]
                same_check = ei[0] == ej[0] and ei[1] == ej[1]
                if (same_qubit or same_check) and self.schedule[i] == self.schedule[j]:
                    raise ValueError(f'{ei} and {ej} same color')

        return 1 + max(self.schedule)

    def minimize_depth_smt(self) -> int:
        """Minimize depth using SMT solver with consistent ordering constraints between checks that share qubits.
        Will return the minimized depth, and update self.s_x_check, self.s_z_check, and self.depth accordingly."""
        import math
        import z3
        (x_checks, z_checks) = (self.x_checks, self.z_checks)
        D = self.maxdeg_x_graph + self.maxdeg_z_check

        def smt_init(upper_bound_int: int) -> Tuple[Any, Any, Any, Any]:
            """
            Initialize SMT variables and constraints for minimizing depth with consistent ordering for each check&qubit.
                - x_checks, z_checks: the checks of the code
                - upper_bound_int: an upper bound on the stage integers (i.e., depth), used to determine bit vector size
            Will return the SMT solver instance and the variables for stages and residuals.
            """
            bit_vec_length = int(math.log2(upper_bound_int)) + 1
            z3_solver = z3.Solver()
            z3_x = [[
                z3.BitVec(f'stageof2q_xcheck{i}_qubit{j}', bit_vec_length) for j in range(len(check))
            ] for i, check in enumerate(x_checks)]
            z3_z = [[
                z3.BitVec(f'stageof2q_zcheck{i}_qubit{j}', bit_vec_length) for j in range(len(check))
            ] for i, check in enumerate(z_checks)]

            z3_r = [[
                z3.BitVec(f'residual2q_xcheck{i}_zcheck{ii}', bit_vec_length)
                for ii in range(len(z_checks))
            ] for i in range(len(x_checks))]
            
            # Each check's stages must be distinct
            for i, check in enumerate(x_checks):
                for j in range(len(check)):
                    for jj in range(j):
                        z3_solver.add(
                            z3.Or(
                                z3.UGT(z3_x[i][j], z3_x[i][jj]),
                                z3.ULT(z3_x[i][j], z3_x[i][jj])
                            )
                        )
            for i, check in enumerate(z_checks):
                for j in range(len(check)):
                    for jj in range(j):
                        z3_solver.add(
                            z3.Or(
                                z3.UGT(z3_z[i][j], z3_z[i][jj]),
                                z3.ULT(z3_z[i][j], z3_z[i][jj])
                            )
                        )

            # Each qubit's stages must be consistent across all checks it appears in
            for i, x_check in enumerate(x_checks): # X&Z checks that share qubits
                for j in range(len(x_check)):
                    for ii, z_check in enumerate(z_checks):
                        for jj in range(len(z_check)):
                            if x_checks[i][j] == z_checks[ii][jj]:
                                z3_solver.add(
                                    z3.Or(
                                        z3.UGT(z3_x[i][j], z3_z[ii][jj]),
                                        z3.ULT(z3_x[i][j], z3_z[ii][jj])
                                    )
                                )
            
            for i0, x_check in enumerate(x_checks): # X&X checks that share qubits
                for j0 in range(len(x_check)):
                    for i1 in range(i0):
                        for j1 in range(len(x_checks[i1])):
                            if x_checks[i0][j0] == x_checks[i1][j1]:
                                z3_solver.add(
                                    z3.Or(
                                        z3.UGT(z3_x[i0][j0], z3_x[i1][j1]),
                                        z3.ULT(z3_x[i0][j0], z3_x[i1][j1])
                                    )
                                )
            for i0, z_check in enumerate(z_checks): # Z&Z checks that share qubits
                for j0 in range(len(z_check)):
                    for i1 in range(i0):
                        for j1 in range(len(z_checks[i1])):
                            if z_checks[i0][j0] == z_checks[i1][j1]:
                                z3_solver.add(
                                    z3.Or(
                                        z3.UGT(z3_z[i0][j0], z3_z[i1][j1]),
                                        z3.ULT(z3_z[i0][j0], z3_z[i1][j1])
                                    )
                                )

            return z3_solver, z3_x, z3_z, z3_r

        def smt_ordering_xz(
                z3_solver: Any,
                z3_x: Any,
                z3_z: Any,
                z3_r: Any,
                general_formulation: bool = False) -> Any:
            """Add SMT constraints for consistent ordering between X and Z checks that share qubits, using either the general formulation with residuals or the pairwise ordering formulation."""
            for i, x_check in enumerate(x_checks):
                for ii, z_check in enumerate(z_checks):
                    hit_pairs = []
                    for j, q in enumerate(x_check):
                        for jj, qq in enumerate(z_check):
                            if q == qq:
                                hit_pairs.append((j, jj))
                    if hit_pairs:
                        if general_formulation:
                            bvl = z3_r[i][ii].size()
                            z3_solver.add(2 * z3_r[i][ii] == sum([
                                z3.If(z3.UGT(z3_x[i][j], z3_z[ii][jj]),
                                      z3.BitVecVal(1, bvl), z3.BitVecVal(0, bvl))
                                for (j, jj) in hit_pairs
                            ]))
                        else:
                            z3_solver.add(
                                z3.Or(
                                    z3.And(
                                        z3.ULT(z3_x[i][hit_pairs[0][0]], z3_z[ii][hit_pairs[0][1]]),
                                        z3.ULT(z3_x[i][hit_pairs[1][0]], z3_z[ii][hit_pairs[1][1]])
                                    ),
                                    z3.And(
                                        z3.UGT(z3_x[i][hit_pairs[0][0]], z3_z[ii][hit_pairs[0][1]]),
                                        z3.UGT(z3_x[i][hit_pairs[1][0]], z3_z[ii][hit_pairs[1][1]])
                                    )
                                )
                            )
            return z3_solver

        z3_solver, z3_x, z3_z, z3_r = smt_init(D)
        z3_solver = smt_ordering_xz(z3_solver, z3_x, z3_z, z3_r, general_formulation=True)

        while True:
            print(f'trying depth {D}')
            try:
                smt_result = z3_solver.check()
            except KeyboardInterrupt:
                print("Interrupted by user (Ctrl-C). Exiting gracefully.")
                break

            if smt_result == z3.sat:
                values = z3_solver.model()
                s_x_check = [[] for _ in range(len(x_checks))]
                s_z_check = [[] for _ in range(len(z_checks))]
                for i, check in enumerate(x_checks):
                    for j in range(len(check)):
                        s_x_check[i].append(values.evaluate(z3_x[i][j]).as_long())
                for i, check in enumerate(z_checks):
                    for j in range(len(check)):
                        s_z_check[i].append(values.evaluate(z3_z[i][j]).as_long())
                self.set_se_schedule_ir(s_x_check, s_z_check)

                D -= 1
                for i, check in enumerate(x_checks):
                    for j in range(len(check)):
                        z3_solver.add(z3.ULT(z3_x[i][j], D))
                for i, check in enumerate(z_checks):
                    for j in range(len(check)):
                        z3_solver.add(z3.ULT(z3_z[i][j], D))
            else:
                print(f'depth={D} unsat')
                break

        self.depth = D + 1
        return self.depth

    def minimize_depth_ilp(
            self,
            file_name: str,
            save_all_sol: Optional[bool] = True,
            num_threads: Optional[int] = None) -> int:
        """Minimize depth using ILP solver with consistent ordering constraints between checks that share qubits.
        Will return the minimized depth, and update self.s_x_check, self.s_z_check, and self.depth accordingly.
        This function will save the ILP model and solution to file_name, and if save_all_sol is True, it will also save all intermediate solutions found during optimization (not just the final optimal solution). num_threads can be specified to control the number of threads used by the ILP solver."""
        import gurobipy as gp
        from itertools import combinations

        x_checks, z_checks = self.x_checks, self.z_checks
        num_data_q = self.num_data_q
        edges, edge2id = self.edges, self.edge2id

        def ilp_init(var_range: int, ilp_inf: int) -> Tuple[Any, Any]:
            num_edges = len(edges)
            gmodel = gp.Model()
            gvar = []
            for i in range(num_edges):
                gvar.append(gmodel.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=var_range - 1, name=f"edge{edges[i].__repr__().replace(' ', '')}"))
            depth = gmodel.addVar(vtype=gp.GRB.INTEGER, name="depth")

            gmodel.setObjective(depth, gp.GRB.MINIMIZE)

            for i in range(num_edges):
                gmodel.addConstr(gvar[i] + 1 <= depth)

            for i, check in enumerate(x_checks):
                hit_x_check = [edge2id[('X', i, q)] for q in check]
                for a, b in combinations(hit_x_check, 2):
                    delta = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"{a}>{b}_for_X_{i}")
                    gmodel.addConstr(gvar[a] <= gvar[b] - 1 + ilp_inf * delta)
                    gmodel.addConstr(gvar[a] >= gvar[b] + 1 - ilp_inf * (1 - delta))
            for i, check in enumerate(z_checks):
                hit_z_check = [edge2id[('Z', i, q)] for q in check]
                for a, b in combinations(hit_z_check, 2):
                    delta = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"{a}>{b}_for_Z_{i}")
                    gmodel.addConstr(gvar[a] <= gvar[b] - 1 + ilp_inf * delta)
                    gmodel.addConstr(gvar[a] >= gvar[b] + 1 - ilp_inf * (1 - delta))
            for q in range(num_data_q):
                hit_data_q = [i for i, edge in enumerate(edges) if edge[2] == q]
                for a, b in combinations(hit_data_q, 2):
                    delta = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"{a}>{b}_for_data_{q}")
                    gmodel.addConstr(gvar[a] <= gvar[b] - 1 + ilp_inf * delta)
                    gmodel.addConstr(gvar[a] >= gvar[b] + 1 - ilp_inf * (1 - delta))
            return gmodel, gvar

        def ilp_order_xz(gmodel: Any, gvar: Any, ilp_inf: int) -> Any:
            for ix, x_check in enumerate(x_checks):
                for iz, z_check in enumerate(z_checks):
                    x_edges = []
                    z_edges = []
                    for q in x_check:
                        for qq in z_check:
                            if q == qq:
                                x_edges.append(edge2id[('X', ix, q)])
                                z_edges.append(edge2id[('Z', iz, q)])
                    if x_edges:

                        ix_iz_0 = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"X_{ix}<Z_{iz}_0")
                        ix_iz_1 = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"X_{ix}<Z_{iz}_1")

                        iz_ix_0 = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"Z_{iz}<X_{ix}_0")
                        iz_ix_1 = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"Z_{iz}<X_{ix}_1")

                        gmodel.addConstr(gvar[x_edges[0]] - gvar[z_edges[0]] <= -1 + ilp_inf * (1 - ix_iz_0))
                        gmodel.addConstr(gvar[x_edges[1]] - gvar[z_edges[1]] <= -1 + ilp_inf * (1 - ix_iz_1))
                        ix_iz = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"X_{ix}<Z_{iz}")
                        gmodel.addGenConstrAnd(ix_iz, [ix_iz_0, ix_iz_1])

                        gmodel.addConstr(gvar[z_edges[0]] - gvar[x_edges[0]] <= -1 + ilp_inf * (1 - iz_ix_0))
                        gmodel.addConstr(gvar[z_edges[1]] - gvar[x_edges[1]] <= -1 + ilp_inf * (1 - iz_ix_1))
                        iz_ix = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"Z_{iz}<X_{ix}")
                        gmodel.addGenConstrAnd(iz_ix, [iz_ix_0, iz_ix_1])

                        delta = gmodel.addVar(vtype=gp.GRB.BINARY, name=f"X_{ix}_and_Z_{iz}_ordered")
                        gmodel.addGenConstrOr(delta, [ix_iz, iz_ix])
                        gmodel.addConstr(delta == 1)
            return gmodel

        def ilp_special_constraints(gmodel: Any, gvar: Any, ilp_inf: int) -> Any:
            return gmodel

        def load_ilp_sol(file_name_with_ext: str) -> None:
            with open(file_name_with_ext, 'r') as f:
                model_lines = f.readlines()
            s_x_check = [[0 for _ in range(len(check))] for check in x_checks]
            s_z_check = [[0 for _ in range(len(check))] for check in z_checks]

            for line in model_lines:
                if line.startswith('# Objective value'):
                    D = int(line.strip().split(' ')[-1])
                if line.startswith('edge'):
                    stage = int(line.split(' ')[-1].strip())
                    check_basis = line.split("'")[1]
                    check_id = int(line.split(',')[1])
                    data_q = int(line.split(',')[2].split(')')[0])
                    if check_basis == 'X':
                        for j, q in enumerate(x_checks[check_id]):
                            if q == data_q:
                                s_x_check[check_id][j] = stage
                                break
                    if check_basis == 'Z':
                        for j, q in enumerate(z_checks[check_id]):
                            if q == data_q:
                                s_z_check[check_id][j] = stage
                                break
            self.set_se_schedule_ir(s_x_check, s_z_check, depth=D)

        upper_bound = self.maxdeg_x_graph + self.maxdeg_z_check
        ILP_INF = 3 * upper_bound
        gmodel, gvar = ilp_init(upper_bound, ILP_INF)
        gmodel = ilp_order_xz(gmodel, gvar, ILP_INF)
        gmodel = ilp_special_constraints(gmodel, gvar, ILP_INF)

        if save_all_sol:
            gmodel.Params.SolFiles = file_name
        if num_threads is not None:
            gmodel.setParam("Threads", num_threads)
        try:
            gmodel.optimize()
        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl-C). Saving best solution found so far.")
            if gmodel.SolCount > 0:
                gmodel.write(file_name + '_best.sol')
            raise
        if gmodel.SolCount > 0:
            gmodel.write(file_name + '_best.sol')

        load_ilp_sol(file_name + '_best.sol')
        self.depth = int(gmodel.objVal)
        return self.depth
