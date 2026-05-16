"""Dump a small triangular-CSR solve plus its analytical gradients for the
ddrs gradcheck test.

Runs DDR's `TriangularSparseSolver` (the reference custom autograd Function from
`~/projects/ddr/src/ddr/routing/utils.py:515`) on a hand-crafted 10-reach DAG,
captures inputs / outputs / gradients, and writes them as plain text CSVs.

The ddrs test (`tests/sparse_gradcheck.rs`) loads these and asserts the BURN
custom backward registered in `src/sparse.rs` produces the same gradients at
f32 precision.

Output (in fixtures/gradcheck/):
    a_values.csv        (nnz,)            non-zero values of A, in CSR order
    crow.csv            (n+1,)            CSR row pointers
    col.csv             (nnz,)            CSR column indices
    b.csv               (n,)              right-hand side
    x.csv               (n,)              forward solution
    grad_output.csv     (n,)              "upstream" gradient ∂L/∂x
    grad_a_values.csv   (nnz,)            ∂L/∂A_values from DDR
    grad_b.csv          (n,)              ∂L/∂b from DDR
"""

import sys
from pathlib import Path

import numpy as np
import torch

DDR = Path.home() / "projects" / "ddr"
sys.path.insert(0, str(DDR / "src"))

from ddr.routing.utils import triangular_sparse_solve  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "gradcheck"
OUT.mkdir(parents=True, exist_ok=True)


def build_test_matrix() -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """A 10-reach lower-triangular DAG with two branches and two confluences.

    Topology (j → i means j drains into i, encoded as A[i,j] != 0):
        0: source                               diag=1.5
        1: source                               diag=1.7
        2: 0 → 2                                diag=2.0, A[2,0]=-0.3
        3: 1 → 3                                diag=1.8, A[3,1]=-0.4
        4: 2,3 → 4   (confluence)               diag=1.6, A[4,2]=-0.5, A[4,3]=-0.6
        5: 4 → 5                                diag=1.9, A[5,4]=-0.7
        6: source                               diag=2.1
        7: 5,6 → 7   (confluence)               diag=1.4, A[7,5]=-0.2, A[7,6]=-0.3
        8: 7 → 8                                diag=2.2, A[8,7]=-0.4
        9: 8 → 9                                diag=1.3, A[9,8]=-0.5
    """
    n = 10
    entries = [
        (0, 0, 1.5),
        (1, 1, 1.7),
        (2, 0, -0.3), (2, 2, 2.0),
        (3, 1, -0.4), (3, 3, 1.8),
        (4, 2, -0.5), (4, 3, -0.6), (4, 4, 1.6),
        (5, 4, -0.7), (5, 5, 1.9),
        (6, 6, 2.1),
        (7, 5, -0.2), (7, 6, -0.3), (7, 7, 1.4),
        (8, 7, -0.4), (8, 8, 2.2),
        (9, 8, -0.5), (9, 9, 1.3),
    ]
    entries.sort(key=lambda e: (e[0], e[1]))  # CSR row-major, col ascending

    col, values = [], []
    row_counts = [0] * n
    for row, c, v in entries:
        col.append(c)
        values.append(v)
        row_counts[row] += 1

    crow = [0] * (n + 1)
    for i in range(n):
        crow[i + 1] = crow[i] + row_counts[i]

    return (
        n,
        np.array(values, dtype=np.float32),
        np.array(crow, dtype=np.int32),
        np.array(col, dtype=np.int32),
    )


def main() -> None:
    n, a_values_np, crow_np, col_np = build_test_matrix()
    rng = np.random.default_rng(42)
    b_np = rng.uniform(0.5, 2.0, size=n).astype(np.float32)
    grad_output_np = rng.standard_normal(n).astype(np.float32)

    a_values_t = torch.tensor(a_values_np, dtype=torch.float32, requires_grad=True)
    b_t = torch.tensor(b_np, dtype=torch.float32, requires_grad=True)
    crow_t = torch.tensor(crow_np, dtype=torch.int32)
    col_t = torch.tensor(col_np, dtype=torch.int32)

    x = triangular_sparse_solve(a_values_t, crow_t, col_t, b_t, True, False, "cpu")
    grad_output_t = torch.tensor(grad_output_np, dtype=torch.float32)
    loss = (x * grad_output_t).sum()
    loss.backward()

    x_np = x.detach().cpu().numpy()
    grad_a_np = a_values_t.grad.detach().cpu().numpy()
    grad_b_np = b_t.grad.detach().cpu().numpy()

    np.savetxt(OUT / "a_values.csv", a_values_np, fmt="%.9e")
    np.savetxt(OUT / "crow.csv", crow_np, fmt="%d")
    np.savetxt(OUT / "col.csv", col_np, fmt="%d")
    np.savetxt(OUT / "b.csv", b_np, fmt="%.9e")
    np.savetxt(OUT / "x.csv", x_np, fmt="%.9e")
    np.savetxt(OUT / "grad_output.csv", grad_output_np, fmt="%.9e")
    np.savetxt(OUT / "grad_a_values.csv", grad_a_np, fmt="%.9e")
    np.savetxt(OUT / "grad_b.csv", grad_b_np, fmt="%.9e")

    print(f"Wrote fixtures to {OUT}")
    print(f"  n={n}, nnz={len(a_values_np)}")
    print(f"  ||x||    = {np.linalg.norm(x_np):.6e}")
    print(f"  ||∂L/∂A|| = {np.linalg.norm(grad_a_np):.6e}")
    print(f"  ||∂L/∂b|| = {np.linalg.norm(grad_b_np):.6e}")


if __name__ == "__main__":
    main()
