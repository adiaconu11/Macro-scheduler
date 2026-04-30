"""Unit tests for CSS_Code using the [[4,2,2]] code.

The [[4,2,2]] code encodes 2 logical qubits into 4 physical qubits
with stabilizer generators X0X1X2X3 and Z0Z1Z2Z3.
"""
import pytest
import numpy as np
import os
import tempfile

from sim.qecc.css import CSS_Code

# ---------------------------------------------------------------------------
# [[4,2,2]] code parameters
# ---------------------------------------------------------------------------
HX_422 = [[1, 1, 1, 1]]
HZ_422 = [[1, 1, 1, 1]]
X_CHECKS_422 = [[0, 1, 2, 3]]
Z_CHECKS_422 = [[0, 1, 2, 3]]
NUM_DATA_Q = 4


def make_422(**kwargs):
    """Helper: build a [[4,2,2]] CSS_Code from checks (override with kwargs)."""
    defaults = dict(x_checks=X_CHECKS_422, z_checks=Z_CHECKS_422, num_data_q=NUM_DATA_Q)
    defaults.update(kwargs)
    return CSS_Code(**defaults)


# ===================================================================
# Initialization
# ===================================================================
class TestInit:
    """Test the various construction paths for CSS_Code."""

    def test_from_checks(self):
        code = make_422()
        assert code.num_data_q == 4
        assert code.num_x_check == 1
        assert code.num_z_check == 1
        assert code.num_total_q == 6  # 4 data + 1 X-ancilla + 1 Z-ancilla
        assert code.x_checks == [[0, 1, 2, 3]]
        assert code.z_checks == [[0, 1, 2, 3]]
        assert code.Hx == [[1, 1, 1, 1]]
        assert code.Hz == [[1, 1, 1, 1]]

    def test_from_H(self):
        code = CSS_Code(Hx=HX_422, Hz=HZ_422)
        assert code.num_data_q == 4
        assert code.num_x_check == 1
        assert code.num_z_check == 1
        assert code.x_checks == [[0, 1, 2, 3]]
        assert code.z_checks == [[0, 1, 2, 3]]
        assert code.Hx == [[1, 1, 1, 1]]
        assert code.Hz == [[1, 1, 1, 1]]

    def test_from_checks_and_H_consistent(self):
        """Passing both checks and matrices should succeed when they agree."""
        code = CSS_Code(
            x_checks=X_CHECKS_422, z_checks=Z_CHECKS_422,
            Hx=HX_422, Hz=HZ_422, num_data_q=NUM_DATA_Q,
        )
        assert code.Hx == [[1, 1, 1, 1]]
        assert code.Hz == [[1, 1, 1, 1]]
        assert code.x_checks == [[0, 1, 2, 3]]
        assert code.z_checks == [[0, 1, 2, 3]]

    def test_from_checks_and_H_inconsistent_raises(self):
        """Inconsistent Hx/checks should raise."""
        with pytest.raises(ValueError, match="inconsistent"):
            CSS_Code(
                x_checks=X_CHECKS_422, z_checks=Z_CHECKS_422,
                Hx=[[1, 1, 0, 0]], Hz=HZ_422, num_data_q=NUM_DATA_Q,
            )

    def test_from_code_str(self):
        code_str = "qecc 4 1 1\n0 1 2 3 X\n0 1 2 3 Z\n"
        code = CSS_Code(code_str=code_str)
        assert code.num_data_q == 4
        assert code.num_x_check == 1
        assert code.num_z_check == 1
        assert code.x_checks == [[0, 1, 2, 3]]
        assert code.z_checks == [[0, 1, 2, 3]]
        assert code.Hx == [[1, 1, 1, 1]]
        assert code.Hz == [[1, 1, 1, 1]]

    def test_checks_and_H_agree(self):
        """Codes built from checks vs. from H should produce identical objects."""
        from_checks = make_422()
        from_H = CSS_Code(Hx=HX_422, Hz=HZ_422)
        assert from_checks.Hx == from_H.Hx
        assert from_checks.Hz == from_H.Hz
        assert from_checks.x_checks == from_H.x_checks
        assert from_checks.z_checks == from_H.z_checks
        assert from_checks.num_data_q == from_H.num_data_q

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            CSS_Code()


# ===================================================================
# Tanner-graph degrees
# ===================================================================
class TestDegree:
    """Test compute_degree() for the [[4,2,2]] code."""

    def test_max_degree_is_4(self):
        code = make_422()
        assert code.maxdeg == 4

    def test_check_degrees(self):
        code = make_422()
        assert code.degrees_x_check == [4]
        assert code.degrees_z_check == [4]

    def test_data_qubit_degrees(self):
        code = make_422()
        # Each data qubit appears in exactly 1 X-check and 1 Z-check → degree 2
        assert code.degrees_data_q == [2, 2, 2, 2]
        assert code.degrees_data_q_xpart == [1, 1, 1, 1]
        assert code.degrees_data_q_zpart == [1, 1, 1, 1]
        assert code.maxdeg_data_q == 2

    def test_subgraph_max_degrees(self):
        code = make_422()
        assert code.maxdeg_x_check == 4
        assert code.maxdeg_z_check == 4
        assert code.maxdeg_x_graph == 4  # max(check_deg=4, data_xpart=1)
        assert code.maxdeg_z_graph == 4  # max(check_deg=4, data_zpart=1)


# ===================================================================
# export_str
# ===================================================================
class TestExportStr:
    """Test export_str() and round-trip fidelity."""

    def test_basic_export(self):
        code = make_422()
        s = code.export_str()
        lines = [l.strip() for l in s.strip().split("\n")]
        assert lines[0] == "qecc 4 1 1"
        assert lines[1] == "0 1 2 3 X"
        assert lines[2] == "0 1 2 3 Z"

    def test_roundtrip(self):
        """export → re-import should give an identical code."""
        code = make_422()
        s = code.export_str()
        code2 = CSS_Code(code_str=s)
        assert code2.x_checks == code.x_checks
        assert code2.z_checks == code.z_checks
        assert code2.Hx == code.Hx
        assert code2.Hz == code.Hz
        assert code2.num_data_q == code.num_data_q

    def test_export_with_logicals(self):
        """export_str(print_logical=True) should include LX / LZ lines."""
        code = make_422()
        code.compute_logicals()
        code.num_logical_q = 2
        s = code.export_str(print_logical=True)
        assert "LX" in s
        assert "LZ" in s


# ===================================================================
# SE-schedule validation
# ===================================================================
class TestScheduleValidation:
    """Test validate_se_schedule with correct, incorrect, incomplete, and colliding schedules."""

    # -- valid schedule --------------------------------------------------
    def test_valid_schedule_passes(self):
        """X in order 0,1,2,3 ; Z in order 1,0,3,2 → valid."""
        code = make_422()
        code.set_se_schedule_ir([[0, 1, 2, 3]], [[1, 0, 3, 2]])
        code.validate_se_schedule()  # should not raise

    def test_depth_inference(self):
        code = make_422()
        code.set_se_schedule_ir([[0, 1, 2, 3]], [[1, 0, 3, 2]])
        assert code.depth == 4

    # -- incomplete schedules --------------------------------------------
    def test_incomplete_schedule_raises(self):
        """Partially assigned schedule (-1 entries) should fail."""
        code = make_422()
        code.set_se_schedule_ir([[0, 1, -1, -1]], [[1, 0, -1, -1]])
        with pytest.raises(ValueError, match="not fully assigned"):
            code.validate_se_schedule()

    def test_default_schedule_incomplete(self):
        """Freshly initialised schedule (all -1) should fail."""
        code = make_422()
        with pytest.raises(ValueError, match="not fully assigned"):
            code.validate_se_schedule()

    # -- qubit collision -------------------------------------------------
    def test_collision_raises(self):
        """X and Z both in order 0,1,2,3 → same qubit used in two checks at one stage."""
        code = make_422()
        code.set_se_schedule_ir([[0, 1, 2, 3]], [[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="multiple"):
            code.validate_se_schedule()

    # -- inconsistent ordering -------------------------------------------
    def test_inconsistent_ordering_raises(self):
        """X in order 0,1,2,3 ; Z in order 3,0,1,2 → inconsistent ordering.

        Z order 3,0,1,2 means:
            qubit 0 → stage 1, qubit 1 → stage 2,
            qubit 2 → stage 3, qubit 3 → stage 0
        so s_z_check = [[1,2,3,0]].  The parity product is
            (0-1)*(1-2)*(2-3)*(3-0) = (-1)(-1)(-1)(3) = -3 < 0.
        """
        code = make_422()
        code.set_se_schedule_ir([[0, 1, 2, 3]], [[1, 2, 3, 0]])
        with pytest.raises(ValueError, match="ordering"):
            code.validate_se_schedule()

    # -- shape mismatch in set_se_schedule_ir ----------------------------
    def test_wrong_shape_raises(self):
        code = make_422()
        with pytest.raises(ValueError):
            code.set_se_schedule_ir([[0, 1, 2]], [[0, 1, 2, 3]])  # too few in X


# ===================================================================
# compute_logicals
# ===================================================================
class TestComputeLogicals:
    """Test compute_logicals() for the [[4,2,2]] code (k = 2)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.code = make_422()
        self.code.compute_logicals()
        self.lx = np.array(self.code.x_logicals)
        self.lz = np.array(self.code.z_logicals)
        self.Hx = np.array(self.code.Hx)
        self.Hz = np.array(self.code.Hz)

    def test_correct_count(self):
        assert self.lx.shape[0] == 2
        assert self.lz.shape[0] == 2

    def test_logicals_commute_with_stabilizers(self):
        # X logicals (X-type) must have even overlap with every Z stabilizer
        assert np.all(self.lx @ self.Hz.T % 2 == 0)
        # Z logicals (Z-type) must have even overlap with every X stabilizer
        assert np.all(self.lz @ self.Hx.T % 2 == 0)

    def test_canonical_commutation(self):
        """lx[i] anticommutes with lz[j] iff i == j."""
        commutation = self.lx @ self.lz.T % 2
        np.testing.assert_array_equal(commutation, np.eye(2, dtype=int))

    def test_logicals_not_in_stabilizer_group(self):
        """Each logical must be linearly independent of the stabiliser generators."""
        rank_hx = np.linalg.matrix_rank(self.Hx)
        rank_hz = np.linalg.matrix_rank(self.Hz)
        for i in range(2):
            # Stacking the logical with the stabiliser should increase rank
            assert np.linalg.matrix_rank(np.vstack([self.Hx, self.lx[i:i+1]])) > rank_hx, \
                f"X logical {i} lies in the X stabiliser group"
            assert np.linalg.matrix_rank(np.vstack([self.Hz, self.lz[i:i+1]])) > rank_hz, \
                f"Z logical {i} lies in the Z stabiliser group"

    def test_logicals_are_binary(self):
        assert set(self.lx.flatten()).issubset({0, 1})
        assert set(self.lz.flatten()).issubset({0, 1})


# ===================================================================
# schedule_coloring_separate
# ===================================================================
class TestScheduleColoringSeparate:
    """Test that schedule_coloring_separate() produces a conflict-free schedule."""

    def test_returns_positive_depth(self):
        code = make_422()
        depth = code.schedule_coloring_separate()
        assert depth > 0

    def test_all_edges_assigned(self):
        code = make_422()
        code.schedule_coloring_separate()
        assert all(t >= 0 for t in code.schedule)

    def test_no_qubit_or_check_conflicts(self):
        """Each qubit and each check should have at most one CNOT per time step."""
        code = make_422()
        code.schedule_coloring_separate()
        # Group edges by time step and verify no qubit/check collision
        from collections import defaultdict
        stage_qubits = defaultdict(list)
        stage_checks = defaultdict(list)
        for idx, t in enumerate(code.schedule):
            basis, check_id, qubit = code.edges[idx]
            stage_qubits[t].append(qubit)
            stage_checks[t].append((basis, check_id))
        for t, qubits in stage_qubits.items():
            assert len(qubits) == len(set(qubits)), f"qubit collision at stage {t}"
        for t, checks in stage_checks.items():
            assert len(checks) == len(set(checks)), f"check collision at stage {t}"


# ===================================================================
# minimize_depth_smt  (requires z3)
# ===================================================================
class TestMinimizeDepthSMT:
    """Test that minimize_depth_smt finds depth 4 for the [[4,2,2]] code."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_z3(self):
        pytest.importorskip("z3")

    def test_optimal_depth_is_4(self):
        code = make_422()
        depth = code.minimize_depth_smt()
        assert depth == 4

    def test_produces_valid_schedule(self):
        code = make_422()
        code.minimize_depth_smt()
        code.validate_se_schedule()  # must not raise


# ===================================================================
# minimize_depth_ilp  (requires gurobipy)
# ===================================================================
class TestMinimizeDepthILP:
    """Test that minimize_depth_ilp finds depth 4 for the [[4,2,2]] code."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_gurobi(self):
        pytest.importorskip("gurobipy")

    def test_optimal_depth_is_4(self):
        code = make_422()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_422")
            depth = code.minimize_depth_ilp(file_name=fpath)
        assert depth == 4

    def test_produces_valid_schedule(self):
        code = make_422()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_422")
            code.minimize_depth_ilp(file_name=fpath)
        code.validate_se_schedule()  # must not raise
