# gemm.mojo
# Vectorised GEMM kernel: C = A @ B (row-major, no transpose).
#
# Used for every weight projection in Project Singularity:
#   • Q / K / V projections in AttentionHead
#   • W1 / W2 projections inside each ExpertLayer
#   • Gating projection in MoEBlock
#   • Output-head projection in LMHead (training.mojo)
#
# Layout convention: all matrices are row-major and contiguous.
#   A : [M × K]
#   B : [K × N]
#   C : [M × N]  — zeroed and written by gemm()

from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof
from algorithm import vectorize


fn gemm[dtype: DType](
    a: UnsafePointer[Scalar[dtype]],   # [M × K]
    b: UnsafePointer[Scalar[dtype]],   # [K × N]
    c: UnsafePointer[Scalar[dtype]],   # [M × N]  — output
    M: Int,
    K: Int,
    N: Int,
):
    """Compute C = A @ B (row-major, no transpose).

    Uses a vectorised outer-product accumulation: iterates over M then K and
    uses SIMD to stride across the N dimension.  This keeps a single element
    of A in a register while streaming matching rows of B and C through cache.

    Args:
        a: Pointer to matrix A [M × K].
        b: Pointer to matrix B [K × N].
        c: Pointer to output matrix C [M × N] (zeroed by this function).
        M: Number of rows in A / C.
        K: Inner (contraction) dimension.
        N: Number of columns in B / C.
    """
    alias simd_w = simdwidthof[dtype]()
    memset_zero(c, M * N)

    for i in range(M):
        for k in range(K):
            var a_ik = a[i * K + k]

            @parameter
            fn scale_add[w: Int](j: Int):
                c.store[width=w](
                    i * N + j,
                    c.load[width=w](i * N + j)
                    + a_ik * b.load[width=w](k * N + j),
                )

            vectorize[scale_add, simd_w](N)
