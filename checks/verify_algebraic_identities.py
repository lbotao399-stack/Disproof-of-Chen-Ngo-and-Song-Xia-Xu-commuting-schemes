#!/usr/bin/env python3
"""Exact algebraic checks for ``Gram determinants on commuting schemes''.

Reproduces in exact integer / rational arithmetic (stdlib only):
  1. Lemma 3.1: Grassmann Gram determinant det_{i,j<=d}(u_i v_j + u_j v_i)
     = (d + 1)! * prod_{i=1}^d (u_i v_i) for d = 1..5.
  2. Proposition 6.1: Disproof of Song-Xia-Xu (Adv. Math. 426 (2023) 109104,
     arXiv:2207.03147) Lemma 5.2 --- on rational points of h_{2d+2} = 0 with
     h_{2d+3} != 0, rk(A^{SXX}) = 2 and det(A^{SXX}) != 0 at d = 1, whereas
     rk(A^{SXX}) = 3 < 2d and det(A^{SXX}) = 0 for all d = 2, 3, 4.
  3. Remark 6.3: Exact univariate polynomial check in t_{12} over Z at d = 2, 3
     showing deg_{t_{12}} det(A^{SXX}) = 4d - 2 (6 and 10, not <= 4d - 4),
     mismatch of the [t_{12}^{4d-4}] coefficient with the claimed product on
     p. 17 of arXiv:2207.03147v3, and exact divisibility h_{2d+2} | det(A^{SXX}).
"""

from fractions import Fraction
import itertools
import math
import random


# ---------------------------------------------------------------------------
# 1. Lemma 3.1: Grassmann Gram identity
# ---------------------------------------------------------------------------

def grassmann_mul(a, b):
    out = {}
    for ka, ca in a.items():
        for kb, cb in b.items():
            if set(ka) & set(kb):
                continue
            seq = list(ka) + list(kb)
            sgn = 1
            for i in range(len(seq)):
                for j in range(i + 1, len(seq)):
                    if seq[i] > seq[j]:
                        sgn = -sgn
            k = tuple(sorted(seq))
            out[k] = out.get(k, 0) + sgn * ca * cb
    return {k: v for k, v in out.items() if v}


def grassmann_add(a, b, scale=1):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + scale * v
    return {k: v for k, v in out.items() if v}


def check_lemma_3_1():
    print("=== 1. Lemma 3.1: Grassmann Gram determinant (d = 1..5) ===")
    for d in range(1, 6):
        u = lambda i: {(2 * i,): 1}
        v = lambda i: {(2 * i + 1,): 1}
        mat = [
            [grassmann_add(grassmann_mul(u(i), v(j)), grassmann_mul(u(j), v(i))) for j in range(d)]
            for i in range(d)
        ]
        det_val = {}
        for perm in itertools.permutations(range(d)):
            sgn = 1
            for i in range(d):
                for j in range(i + 1, d):
                    if perm[i] > perm[j]:
                        sgn = -sgn
            term = {(): sgn}
            for i in range(d):
                term = grassmann_mul(term, mat[i][perm[i]])
            det_val = grassmann_add(det_val, term)
        top = {(): 1}
        for i in range(d):
            top = grassmann_mul(top, grassmann_mul(u(i), v(i)))
        (top_key, top_sign), = top.items()
        coeff = det_val.get(top_key, 0) // top_sign
        expected = math.factorial(d + 1)
        assert coeff == expected and len(det_val) == 1, (d, coeff, expected, det_val)
        print(f"  d = {d}: det = {coeff} * prod(u_i v_i)  [(d+1)! = {expected}, extra terms = 0] OK")


# ---------------------------------------------------------------------------
# Pfaffian and exact linear algebra over Q
# ---------------------------------------------------------------------------

def pfaffian_exact(mat):
    """Exact Pfaffian of an m x m skew-symmetric matrix over Z or Fraction."""
    m = len(mat)
    if m % 2 == 1:
        return 0
    if m == 0:
        return 1
    memo = {}

    def rec(idxs):
        if not idxs:
            return 1
        if idxs in memo:
            return memo[idxs]
        i0 = idxs[0]
        total = 0
        for pos in range(1, len(idxs)):
            j0 = idxs[pos]
            val = mat[i0][j0]
            if val != 0:
                rem = idxs[1:pos] + idxs[pos + 1:]
                sub = rec(rem)
                if (pos - 1) % 2 == 1:
                    total -= val * sub
                else:
                    total += val * sub
        memo[idxs] = total
        return total

    return rec(tuple(range(m)))


def minor_pfaffian(t_mat, deleted_one_based):
    """Pfaffian h_{i_1,...,i_k} with 1-based indices; 0 if repeated or odd size left."""
    if len(set(deleted_one_based)) != len(deleted_one_based):
        return 0
    n = len(t_mat)
    del_zero = {idx - 1 for idx in deleted_one_based}
    keep = [i for i in range(n) if i not in del_zero]
    if len(keep) % 2 == 1:
        return 0
    sub = [[t_mat[i][j] for j in keep] for i in keep]
    return pfaffian_exact(sub)


def build_sxx_matrix(t_mat, d):
    """Construct the 2d x 2d matrix A^{SXX} from a (2d+3) x (2d+3) skew matrix T."""
    h_single = [minor_pfaffian(t_mat, (i, 2 * d + 1, 2 * d + 2)) for i in range(1, 2 * d + 1)]
    mat_a = [[0] * (2 * d) for _ in range(2 * d)]
    for i in range(1, 2 * d + 1):
        for j in range(1, 2 * d + 1):
            hij = minor_pfaffian(t_mat, (i, j, 2 * d + 2))
            mat_a[i - 1][j - 1] = hij * hij + h_single[i - 1] ** 2 + h_single[j - 1] ** 2
    return mat_a


def rank_and_det_q(mat):
    """Exact rank and determinant over Q using Gaussian elimination."""
    n = len(mat)
    a = [[Fraction(x) for x in row] for row in mat]
    det = Fraction(1)
    rank = 0
    for col in range(n):
        piv = None
        for r in range(rank, n):
            if a[r][col] != 0:
                piv = r
                break
        if piv is None:
            det = Fraction(0)
            continue
        if piv != rank:
            a[rank], a[piv] = a[piv], a[rank]
            det = -det
        pv = a[rank][col]
        det *= pv
        for r in range(rank + 1, n):
            if a[r][col] != 0:
                fac = a[r][col] / pv
                for c in range(col, n):
                    a[r][c] -= fac * a[rank][c]
        rank += 1
    if rank < n:
        det = Fraction(0)
    return rank, det


# ---------------------------------------------------------------------------
# 2. Proposition 6.1: Rank <= 3 on h_{2d+2} = 0 for d = 1..4
# ---------------------------------------------------------------------------

def sample_locus_h_zero(d, rng):
    """Construct an exact rational (2d+3)x(2d+3) skew matrix with h_{2d+2}=0, h_{2d+3}!=0."""
    size_p = 2 * d + 2
    # Build P of rank 2d via sum_{r=1}^d (u_r ^ v_r) over Z
    p_mat = [[Fraction(0)] * size_p for _ in range(size_p)]
    for _ in range(d):
        u = [Fraction(rng.randint(-4, 4)) for _ in range(size_p)]
        v = [Fraction(rng.randint(-4, 4)) for _ in range(size_p)]
        for a in range(size_p):
            for b in range(size_p):
                p_mat[a][b] += u[a] * v[b] - v[a] * u[b]
    # Embed P into (2d+3)x(2d+3) by inserting row/col (2d+2) (0-indexed: 2d+1)
    keep = list(range(2 * d + 1)) + [2 * d + 2]
    n = 2 * d + 3
    t_mat = [[Fraction(0)] * n for _ in range(n)]
    for ia, a in enumerate(keep):
        for ib, b in enumerate(keep):
            t_mat[a][b] = p_mat[ia][ib]
    for a in range(n):
        if a == 2 * d + 1:
            continue
        val = Fraction(rng.randint(-5, 5))
        t_mat[a][2 * d + 1] = val
        t_mat[2 * d + 1][a] = -val
    return t_mat


def check_proposition_6_1():
    print("\n=== 2. Proposition 6.1: Song-Xia-Xu matrix A^{SXX} on h_{2d+2} = 0 ===")
    rng = random.Random(20261111)
    for d in (1, 2, 3, 4):
        while True:
            t_mat = sample_locus_h_zero(d, rng)
            h_zero = minor_pfaffian(t_mat, (2 * d + 2,))
            h_last = minor_pfaffian(t_mat, (2 * d + 3,))
            assert h_zero == 0
            if h_last != 0:
                break
        mat_a = build_sxx_matrix(t_mat, d)
        rk, det_a = rank_and_det_q(mat_a)
        if d == 1:
            assert rk == 2 and det_a != 0
            print(f"  d = 1 (2x2 base case): h_4 = 0, h_5 != 0 -> rk(A^SXX) = {rk}, det(A^SXX) != 0 OK")
        else:
            assert rk <= 3 and det_a == 0, (d, rk, det_a)
            print(f"  d = {d} ({2*d}x{2*d} matrix): h_{2*d+2} = 0, h_{2*d+3} != 0 -> rk(A^SXX) = {rk} < {2*d}, det(A^SXX) = 0 OK")


# ---------------------------------------------------------------------------
# 3. Remark 6.3: Exact univariate polynomial in t_{12} at d = 2 and d = 3
# ---------------------------------------------------------------------------

def lagrange_interpolate_q(xs, ys):
    """Exact polynomial coefficients [c_0, c_1, ..., c_m] in Q[x] from m+1 samples."""
    m = len(xs)
    coeffs = [Fraction(0)] * m
    for i in range(m):
        # basis L_i(x)
        num = [Fraction(1)]
        denom = Fraction(1)
        for j in range(m):
            if i == j:
                continue
            denom *= (xs[i] - xs[j])
            nxt = [Fraction(0)] * (len(num) + 1)
            for k, c in enumerate(num):
                nxt[k] -= c * xs[j]
                nxt[k + 1] += c
            num = nxt
        scale = ys[i] / denom
        for k, c in enumerate(num):
            coeffs[k] += c * scale
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs


def poly_divmod_q(dividend, divisor):
    a = list(dividend)
    b = list(divisor)
    while len(b) > 1 and b[-1] == 0:
        b.pop()
    assert b[-1] != 0
    deg_b = len(b) - 1
    if len(a) - 1 < deg_b:
        return [Fraction(0)], a
    q = [Fraction(0)] * (len(a) - deg_b)
    for k in range(len(a) - 1, deg_b - 1, -1):
        c = a[k] / b[deg_b]
        q[k - deg_b] = c
        for j in range(deg_b + 1):
            a[k - deg_b + j] -= c * b[j]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return q, a


def sxx_claimed_coeff(t_base, d):
    """Compute the claimed [t_{12}^{4d-4}] product on p. 17 of arXiv:2207.03147v3."""
    # In Remark 6.3, the claimed coefficient of t_{12}^{4d-4} is:
    # det([[2 B1, C + B1 + B2], [C + B1 + B2, 2 B2]]) * det(A')
    # where B1, B2, C are the t_{12}-independent parts of h^2_{1,2d+1,2d+2},
    # h^2_{2,2d+1,2d+2}, h^2_{1,2,2d+2} (evaluated at t_{12}=0), and A' is the
    # SXX matrix for T with rows/cols 1, 2 deleted.
    t0 = [row[:] for row in t_base]
    t0[0][1] = Fraction(0)
    t0[1][0] = Fraction(0)
    b1 = minor_pfaffian(t0, (1, 2 * d + 1, 2 * d + 2)) ** 2
    b2 = minor_pfaffian(t0, (2, 2 * d + 1, 2 * d + 2)) ** 2
    c12 = minor_pfaffian(t0, (1, 2, 2 * d + 2)) ** 2
    two_by_two = 4 * b1 * b2 - (c12 + b1 + b2) ** 2
    t_prime = [[t0[i][j] for j in range(2, 2 * d + 3)] for i in range(2, 2 * d + 3)]
    a_prime = build_sxx_matrix(t_prime, d - 1)
    _, det_a_prime = rank_and_det_q(a_prime)
    return two_by_two * det_a_prime


def check_remark_6_3():
    print("\n=== 3. Remark 6.3: Exact t_{12}-degree and divisibility by h_{2d+2} ===")
    rng = random.Random(20261028)
    for d in (2, 3):
        n = 2 * d + 3
        t_base = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                val = Fraction(rng.randint(-4, 4))
                if val == 0:
                    val = Fraction(1)
                t_base[i][j] = val
                t_base[j][i] = -val
        max_deg = 4 * d
        xs = [Fraction(k) for k in range(-(max_deg // 2), max_deg // 2 + 2)]
        ys_det = []
        ys_h = []
        for x in xs:
            t_cur = [row[:] for row in t_base]
            t_cur[0][1] = x
            t_cur[1][0] = -x
            _, det_val = rank_and_det_q(build_sxx_matrix(t_cur, d))
            ys_det.append(det_val)
            ys_h.append(minor_pfaffian(t_cur, (2 * d + 2,)))
        poly_det = lagrange_interpolate_q(xs, ys_det)
        poly_h = lagrange_interpolate_q(xs, ys_h)
        deg_det = len(poly_det) - 1
        actual_4d_4 = poly_det[4 * d - 4] if 4 * d - 4 < len(poly_det) else Fraction(0)
        claimed_4d_4 = sxx_claimed_coeff(t_base, d)
        _, rem = poly_divmod_q(poly_det, poly_h)
        assert deg_det == 4 * d - 2, (d, deg_det)
        assert actual_4d_4 != claimed_4d_4, (d, actual_4d_4, claimed_4d_4)
        assert rem == [Fraction(0)], (d, rem)
        print(
            f"  d = {d}: deg_{{t_12}}(det A^SXX) = {deg_det} = 4d - 2 (SXX claimed <= {4*d-4}); "
            f"[t_12^{4*d-4}] differs from SXX formula ({actual_4d_4 != claimed_4d_4}); "
            f"det(A^SXX) mod h_{2*d+2} = 0 OK"
        )


if __name__ == "__main__":
    check_lemma_3_1()
    check_proposition_6_1()
    check_remark_6_3()
    print("\nAll exact algebraic checks passed.")
