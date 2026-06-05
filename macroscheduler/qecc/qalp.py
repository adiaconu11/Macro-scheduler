from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional

from .css import CSS_Code

class Perm:
    """
    Class representing a permutation and implementing some basic logic.
    The permutations here are 0-indexed!
    """
    __slots__ = ('p',)

    def __init__(self, p, validate=True):
        a = np.asarray(p, dtype=np.int32)
        if a.ndim != 1:
            raise ValueError("Permutation must be 1D!")
        if validate:
            self._validate(a)
        
        object.__setattr__(self, "p", a)

    @classmethod
    def right_shift(cls, n: int) -> "Perm":
        """Constructs the right cyclic shift [n-1, 0, 1, ..., n-2]."""
        if n <= 0:
            raise ValueError("n must be a positive integer")
        p = np.concatenate((np.array([n - 1], dtype=np.int32), np.arange(n - 1, dtype=np.int32)))
        return cls(p)

    @classmethod
    def identity(cls, n: int) -> "Perm":
        """Constructs the identity permutation [0, 1, 2, ... n-1]."""
        if n <= 0:
            raise ValueError("n must be a positive integer")
        return cls(np.arange(n, dtype=np.int32))

    @property
    def n(self) -> int:
        """The dimension of the permutation."""
        return self.p.shape[0]

    def _validate(self, a=None):
        """
        Will validate the current permutation. If it is wrong, will raise error.
        """
        p = self.p if a is None else a

        # Require integer type
        n = p.shape[0]

        # Range check (O(n))
        mn = int(p.min())
        mx = int(p.max())
        if mn < 0 or mx >= n:
            raise ValueError(f"Permutation values must be in [0, {n-1}]. Got min={mn}, max={mx}.")

        # Uniqueness check (O(n) average) using counting via bincount
        counts = np.bincount(p, minlength=n)
        if np.any(counts != 1):
            raise ValueError(f"The permutation must contain each value 0...n-1 exactly once")

    def to_matrix(self, dtype=int) -> np.ndarray:
        """Transforms the permutation into its matrix.
        The convention is that if p[i] = j, then mat[j][i] = 1
        """
        n = self.n
        return np.eye(n, dtype=dtype)[:, self.p] # permute the columns

    def commutes_with(self, other: "Perm") -> bool:
        if self.n != other.n:
            return False
        p, q = self.p, other.p
        return bool(np.all(p[q] == q[p]))

    def comp(self, other: "Perm") -> "Perm":
        """Returns composition self ∘ other (apply other, then self)."""
        if self.n != other.n:
            raise ValueError("Permutations must have the same size to compose")
        return Perm(self.p[other.p], validate=False)

    def inverse(self) -> "Perm":
        """Return the inverse permutation."""
        inv = np.empty_like(self.p)
        inv[self.p] = np.arange(self.n, dtype=np.int32)
        return Perm(inv, validate=False)

    def __mul__(self, other: "Perm") -> "Perm":
        return self.comp(other)

    def __pow__(self, exponent: int, modulo=None) -> "Perm":
        """Returns self composed with itself exponent times."""
        if modulo is not None:
            raise TypeError("Perm does not support modular exponentiation")
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        if exponent < 0:
            raise ValueError("Exponent must be non-negative")
        if exponent == 0:
            return Perm.identity(self.n)

        result = Perm.identity(self.n)
        base = self
        power = exponent
        while power > 0:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def kron(self, other: "Perm") -> "Perm":
        """
        Returns the underlying permutation of the kronecker product of the perm matrices.
        This operation is called a "product action".
        """
        n, m = self.n, other.n
        r = [0]*n*m
        p, q = self.p, other.p
        for i in range(n):
            for j in range(m):
                r[i*m + j] = p[i]*m + q[j]
        return Perm(r, validate=False)

    def __matmul__(self, other: "Perm") -> "Perm":
        return self.kron(other)

class Block(list[Perm]):
    """
    A lifted-matrix block: an additive list of permutations.
    Behaves like a regular list[Perm] for iteration/indexing/mutation.
    """
    def as_lifted_mat(self) -> "LiftedMatrix":
        return LiftedMatrix([[self]], self[0].n)



class LiftedMatrix:
    """
     Class that defines a lifted parity check matrix.
    """
    def __init__(self, M_tilde: List[List[Block]], l:int, validate=True):
        self.M_tilde = M_tilde
        self.m = len(M_tilde)
        self.n = len(M_tilde[0])
        self.l = l

        if validate:
            self._validate()
    
    def _validate(self):
        for i, row in enumerate(self.M_tilde):
            assert len(row) == self.n, f"The given lifted matrix isn't homogenous at row {i}"
            for j, entry in enumerate(row):
                for perm in entry:
                    assert self.l == perm.n, f"The entry {i}{j} doesn't have correct lift size!"

    def get_full_matrix(self) -> np.ndarray:
        full_mat = np.zeros((self.m * self.l, self.n * self.l), dtype=int)
        for i in range(self.m):
            for j in range(self.n):
                entry = np.zeros((self.l, self.l), dtype=int)
                for perm in self.M_tilde[i][j]:
                    entry += perm.to_matrix()
                full_mat[i*self.l:(i+1)*self.l, j*self.l:(j+1)*self.l] = entry
        return full_mat
    
    def full_mat_shape(self) -> Tuple[int, int]:
        """Returns the shape of the full matrix as a tuple (num_rows, num_cols)."""
        return (self.m * self.l, self.n * self.l)

    def dagger(self) -> "LiftedMatrix":
        """Return the lifted adjoint: protograph transpose and inverse blocks."""
        blocks = [[Block([]) for _ in range(self.m)] for _ in range(self.n)]
        for i in range(self.m):
            for j in range(self.n):
                blocks[j][i] = Block([perm.inverse() for perm in self.M_tilde[i][j]])
        return LiftedMatrix(blocks, self.l, validate=False)

def _Z(r: int, c: int) -> np.ndarray:
    return np.zeros((r, c), dtype=int)

def lifted_identity(size: int, l: int) -> LiftedMatrix:
    """Return a lifted identity matrix with identity permutations on the diagonal."""
    if size < 0:
        raise ValueError("identity size must be non-negative")
    identity = Perm.identity(l)
    blocks = [[Block([]) for _ in range(size)] for _ in range(size)]
    for i in range(size):
        blocks[i][i] = Block([identity])
    return LiftedMatrix(blocks, l, validate=False)

def _flatten_index(indices: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    idx = 0
    for value, dim in zip(indices, dims):
        idx = idx * dim + value
    return idx

def _multiply_blocks(blocks: Tuple[Block, ...], l: int) -> List[Perm]:
    if any(len(block) == 0 for block in blocks):
        return []

    products = [Perm.identity(l)]
    for block in blocks:
        products = [product * perm for product in products for perm in block]
    return products

def kron_lifted(*factors: LiftedMatrix) -> np.ndarray:
    """
    Evaluate a Kronecker product over the shared lifted permutation algebra.

    The protograph tensor factors are flattened in the same order as
    ``np.kron``. The lift coordinate is kept as the innermost coordinate and is
    expanded only after multiplying the permutation entries in each block.
    """
    if not factors:
        raise ValueError("at least one lifted factor is required")

    l = factors[0].l
    if not all(factor.l == l for factor in factors):
        raise ValueError("all lifted factors must have the same lift size")

    row_dims = tuple(factor.m for factor in factors)
    col_dims = tuple(factor.n for factor in factors)
    num_row_sectors = int(np.prod(row_dims, dtype=int))
    num_col_sectors = int(np.prod(col_dims, dtype=int))
    mat = np.zeros((num_row_sectors * l, num_col_sectors * l), dtype=int)

    for row_indices in np.ndindex(*row_dims):
        row_sector = _flatten_index(row_indices, row_dims)
        row = slice(row_sector * l, (row_sector + 1) * l)
        for col_indices in np.ndindex(*col_dims):
            col_sector = _flatten_index(col_indices, col_dims)
            blocks = tuple(
                factor.M_tilde[row_idx][col_idx]
                for factor, row_idx, col_idx in zip(factors, row_indices, col_indices)
            )
            products = _multiply_blocks(blocks, l)
            if not products:
                continue

            col = slice(col_sector * l, (col_sector + 1) * l)
            for perm in products:
                mat[row, col] += perm.to_matrix()

    return mat

def construct_lifted_product_mats(lifted_check_mats: List[LiftedMatrix]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct QALP check matrices by taking Kronecker products over the lift."""
    dim = len(lifted_check_mats)
    if dim < 2:
        raise ValueError("at least two lifted check matrices are required")

    l = lifted_check_mats[0].l
    if not all(mat.l == l for mat in lifted_check_mats):
        raise ValueError("all lifted check matrices must have the same lift size")

    m_array = [mat.m for mat in lifted_check_mats]
    n_array = [mat.n for mat in lifted_check_mats]

    match dim:
        case 2:
            nA, nB = n_array
            mA, mB = m_array
            A, B = lifted_check_mats
            Hx = np.hstack((
                kron_lifted(A, lifted_identity(nB, l)),
                kron_lifted(lifted_identity(mA, l), B.dagger()),
            ))
            Hz = np.hstack((
                kron_lifted(lifted_identity(nA, l), B),
                kron_lifted(A.dagger(), lifted_identity(mB, l)),
            ))
            return Hx, Hz

        case 3:
            nA, nB, nC = n_array
            mA, mB, mC = m_array
            A, B, C = lifted_check_mats

            Hx = np.hstack((
                kron_lifted(A, lifted_identity(mB, l), lifted_identity(mC, l)),
                kron_lifted(lifted_identity(mA, l), B, lifted_identity(mC, l)),
                kron_lifted(lifted_identity(mA, l), lifted_identity(mB, l), C),
            ))

            z11 = kron_lifted(lifted_identity(nA, l), B.dagger(), lifted_identity(mC, l))
            z12 = kron_lifted(A.dagger(), lifted_identity(nB, l), lifted_identity(mC, l))
            z21 = kron_lifted(lifted_identity(nA, l), lifted_identity(mB, l), C.dagger())
            z23 = kron_lifted(A.dagger(), lifted_identity(mB, l), lifted_identity(nC, l))
            z32 = kron_lifted(lifted_identity(mA, l), lifted_identity(nB, l), C.dagger())
            z33 = kron_lifted(lifted_identity(mA, l), B.dagger(), lifted_identity(nC, l))

            r1, r2, r3 = z11.shape[0], z21.shape[0], z32.shape[0]
            c1, c2, c3 = z11.shape[1], z12.shape[1], z21.shape[1]

            Hz = np.block([
                [z11, z12, _Z(r1, c3)],
                [z21, _Z(r2, c2), z23],
                [_Z(r3, c1), z32, z33],
            ])
            return Hx, Hz

        case _:
            raise NotImplementedError("Only up to 3D codes are currently supported!")

class QALP(CSS_Code):
    """
    Class of general quasi-abelian lifted product codes.
    """
    
    def __init__(self, lifted_check_mats: List[LiftedMatrix], validate=True):
        self.lifted_check_mats = lifted_check_mats

        if len(lifted_check_mats) < 2:
            raise ValueError("At least two lifted check matrices are required to define a QALP code!")            

        if validate:
            self._validate()

        self.check_mats = [x.get_full_matrix() for x in lifted_check_mats]

        self.m_array = [mat.m for mat in self.lifted_check_mats]
        self.n_array = [mat.n for mat in self.lifted_check_mats]
        self.l = self.lifted_check_mats[0].l
        if not all(mat.l == self.l for mat in self.lifted_check_mats):
            raise ValueError("All lifted check matrices must have the same lift size!")

        Hx, Hz = construct_lifted_product_mats(self.lifted_check_mats)

        super().__init__(Hx=Hx, Hz=Hz)

    
    def _validate(self):
        # We need to check that all permutations commute with each other
        for x in range(len(self.lifted_check_mats)):
            for y in range(x+1, len(self.lifted_check_mats)):
                for i in range(self.lifted_check_mats[x].m):
                    for j in range(self.lifted_check_mats[x].n):
                        for k in range(self.lifted_check_mats[y].m):
                            for l in range(self.lifted_check_mats[y].n):
                                for p1 in self.lifted_check_mats[x].M_tilde[i][j]:
                                    for p2 in self.lifted_check_mats[y].M_tilde[k][l]:
                                        if not p1.commutes_with(p2):
                                            raise ValueError(f"Permutations at {x}{i}{j} and {y}{k}{l} don't commute!")

class HGP(QALP):
    """
    This the class of HGP codes.
    These codes have trivial lift sizes, but complicated protographs.
    """
    def __init__(self, classical_check_mats: List[List[List[int]]], validate=True):
        """
        Construct an HGP code from classical parity check matrices.
        Each parity matrix is a binary m by n matrix: m checks, n data bits
        """
        lifted_mats = []
        for check_mat in classical_check_mats:
            m = len(check_mat)
            n = len(check_mat[0])
            for row in check_mat:
                assert len(row) == n, "Invalid classical check matrix!"
            blocks = [[[] for _ in range(n)] for _ in range(m)]
            for i in range(m):
                for j in range(n):
                    if check_mat[i][j] == 1:
                        blocks[i][j].append(Perm.identity(1))
            
            lifted_mats.append(LiftedMatrix(blocks, 1, validate=validate))
        super().__init__(lifted_mats, validate=validate)


class BlockCode(QALP):
    """
    This is the general class of a Block code, a subclass of QALP.
    These codes have trivial protographs of size 1x1. As such M_tilde = M.
    """

    def __init__(self, blocks: List[Block], validate=True):
        super().__init__(
            [b.as_lifted_mat() for b in blocks],
            validate=validate,
        )
    

def compute_basis(moduli: List[int]) -> List[Perm]:
    """
    Compute the base permutations for each monomial dimension.

    Args:
        moduli (List[int]): Lift sizes for each monomial dimension.

    Returns:
        List[Perm]: Base permutation for each monomial dimension.
    """
    num_mods = len(moduli)
    monomial_bases = [0] * num_mods
    for i in range(num_mods):
        before = Perm.identity(np.prod(moduli[:i], dtype=int))
        after = Perm.identity(np.prod(moduli[i+1:], dtype=int))
        base = before @ Perm.right_shift(moduli[i]) @ after
        monomial_bases[i] = base
    
    return monomial_bases

class Term(list[int]):
    """
    A term: each entry is the power corresponding to each monomial.
    """
    def num_monomials(self) -> int:
        return len(self)
    
    def get_perm(self, moduli, monomial_bases: Optional[List[Perm]]) -> Perm:

        if monomial_bases is None:
            monomial_bases = compute_basis(moduli)

        ans = monomial_bases[0] ** int(self[0])
        for i in range(1,len(moduli)):
            ans = ans * (monomial_bases[i] ** int(self[i]))
        
        return ans

    def reversed(self, moduli) -> Term:
        "Find the reversed term given moduli"
        return Term([moduli[i] - self[i] for i in range(len(self))])


class Polynomial(list[Term]):
    """
    A polynomial: each entry is a Term.
    """
    
    def validate(self, num_monomials):
        assert len(self) > 0, "The polynomial cannot be empty!"
        for term in self:
            assert num_monomials == term.num_monomials(), "Inconsistent number of monomials!"
    
    def compute_block(self, moduli, monomial_bases: Optional[List[Perm]]) -> Block:
        return Block([t.get_perm(moduli, monomial_bases) for t in self])
    
    def reversed(self, moduli) -> Polynomial:
        """Find the reversed polynomial given moduli"""
        return Polynomial([term.reversed(moduli) for term in self])
    

class AlgebraicBlockCode(BlockCode):
    """
    This is the overarching class that encompasses the BB and TT codes etc.
    """
    def __init__(self, moduli: List[int], polys: List[Polynomial] | List[List[List[int]]], validate=True):
        self.num_monomials = len(moduli)

        self.monomial_bases = compute_basis(moduli)

        # Convert List[List[List[int]]] to List[Polynomial] if needed
        if polys and isinstance(polys[0][0], (list, tuple)):
            polys = [Polynomial([Term(term) for term in poly]) for poly in polys]
        self.moduli = moduli
        self.polys = polys

        if validate:
            for p in polys:
                p.validate(self.num_monomials)

        blocks = [p.compute_block(moduli, self.monomial_bases) for p in polys]

        super().__init__(blocks, validate=False)

class BB_Code(AlgebraicBlockCode):
    """
    This is the general class of a BB code.
    """
    def __init__(self, l: int, m: int, A_poly: Polynomial | List[List[int]], B_poly: Polynomial | List[List[int]], validate=True):
        moduli = [l, m]
        if isinstance(A_poly, (list, tuple)):
            A_poly = Polynomial([Term(term) for term in A_poly])
        if isinstance(B_poly, (list, tuple)):
            B_poly = Polynomial([Term(term) for term in B_poly])

        # To match ABC definition, we need to have B --> B.T
        B_poly = B_poly.reversed([l, m])

        polys = [A_poly, B_poly]
        super().__init__(moduli, polys, validate)

class TT_Code(AlgebraicBlockCode):
    """
    This is the general class of a TT code.
    """
    def __init__(self, l: int, m: int, n: int, A_poly: Polynomial | List[List[int]],
                  B_poly: Polynomial | List[List[int]], C_poly: Polynomial | List[List[int]], validate=True):
        moduli = [l, m, n]
        polys = [A_poly, B_poly, C_poly]
        super().__init__(moduli, polys, validate)
