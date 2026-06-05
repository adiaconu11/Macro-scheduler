import numpy as np
import pytest

from macroscheduler.qecc.qalp import Block, Perm, Polynomial, Term, compute_basis


class TestPerm:
    def test_identity(self):
        p = Perm.identity(5)
        np.testing.assert_array_equal(p.p, np.array([0, 1, 2, 3, 4], dtype=np.int32))

    def test_right_shift(self):
        p = Perm.right_shift(5)
        np.testing.assert_array_equal(p.p, np.array([4, 0, 1, 2, 3], dtype=np.int32))

    def test_invalid_permutation_raises(self):
        with pytest.raises(ValueError, match="exactly once"):
            Perm([0, 1, 1])

    def test_composition_and_mul(self):
        p = Perm([1, 2, 0])
        q = Perm([2, 0, 1])
        composed = p.comp(q)
        mul_composed = p * q
        np.testing.assert_array_equal(composed.p, p.p[q.p])
        np.testing.assert_array_equal(mul_composed.p, composed.p)

    def test_pow_zero_and_positive(self):
        p = Perm.right_shift(4)
        np.testing.assert_array_equal((p ** 0).p, Perm.identity(4).p)
        np.testing.assert_array_equal((p ** 2).p, (p * p).p)
        np.testing.assert_array_equal((p ** 3).p, (p * p * p).p)

    def test_pow_negative_raises(self):
        p = Perm.right_shift(4)
        with pytest.raises(ValueError, match="non-negative"):
            _ = p ** -1

    def test_kron_and_matmul(self):
        p = Perm([1, 0])
        q = Perm([2, 0, 1])
        expected = np.array([5, 3, 4, 2, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(p.kron(q).p, expected)
        np.testing.assert_array_equal((p @ q).p, expected)

    def test_commutes_with(self):
        p = Perm.right_shift(5)
        assert p.commutes_with(p ** 2)


class TestTerm:
    def test_num_monomials(self):
        t = Term([3, 1, 4])
        assert t.num_monomials() == 3

    def test_get_perm_matches_manual_product(self):
        moduli = [2, 3]
        monomial_bases = compute_basis(moduli)
        term = Term([1, 2])

        got = term.get_perm(moduli, monomial_bases)
        expected = (monomial_bases[0] ** 1) * (monomial_bases[1] ** 2)

        np.testing.assert_array_equal(got.p, expected.p)

    def test_get_perm_computes_bases_when_none(self):
        moduli = [2, 3]
        term = Term([1, 0])

        got = term.get_perm(moduli, None)
        bases = compute_basis(moduli)
        expected = (bases[0] ** 1) * (bases[1] ** 0)

        np.testing.assert_array_equal(got.p, expected.p)


class TestPolynomial:
    def test_validate_nonempty_and_homogeneous(self):
        p = Polynomial([Term([1, 0]), Term([0, 1])])
        p.validate(2)

    def test_validate_empty_raises(self):
        p = Polynomial([])
        with pytest.raises(AssertionError, match="cannot be empty"):
            p.validate(2)

    def test_validate_inconsistent_term_size_raises(self):
        p = Polynomial([Term([1, 0]), Term([1])])
        with pytest.raises(AssertionError, match="Inconsistent"):
            p.validate(2)

    def test_compute_block_returns_block_of_perms(self):
        moduli = [2, 3]
        monomial_bases = compute_basis(moduli)
        poly = Polynomial([Term([1, 0]), Term([0, 2])])

        block = poly.compute_block(moduli, monomial_bases)

        assert isinstance(block, Block)
        assert len(block) == 2
        assert all(isinstance(x, Perm) for x in block)
