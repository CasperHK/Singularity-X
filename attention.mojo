# attention.mojo
# Core compute kernel for Project Singularity.
# Implements a high-performance Scaled Dot-Product Attention (SDPA) kernel
# using Mojo's SIMD / vectorisation primitives, explicit manual memory
# management via UnsafePointer, and compile-time tile-size selection through
# @parameter + autotune.
#
# Layout convention (all tensors are row-major, contiguous):
#   Q  : [seq_len, head_dim]
#   K  : [seq_len, head_dim]
#   V  : [seq_len, head_dim]
#   Out: [seq_len, head_dim]

from math import sqrt, exp
from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof
from algorithm import vectorize

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------

alias DEFAULT_DTYPE = DType.float32

# SIMD lane width for the chosen dtype – determined at compile time.
alias SIMD_WIDTH = simdwidthof[DEFAULT_DTYPE]()

# ---------------------------------------------------------------------------
# Tile-size candidates exposed to Mojo's autotune mechanism.
# The compiler will benchmark each candidate and select the fastest one.
# ---------------------------------------------------------------------------
@parameter
fn get_tile_size() -> Int:
    """Return the row-tile size used for the attention inner loop.

    Annotate with autotune so Modular's compiler can pick the optimal value
    for the target AI chip at build time.
    """
    return autotune(16, 32, 64, 128)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

@always_inline
fn _dot_product[dtype: DType, simd_w: Int](
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    length: Int,
) -> Scalar[dtype]:
    """Vectorised dot product of two contiguous arrays of length *length*."""
    var acc = SIMD[dtype, simd_w](0)

    @parameter
    fn dot_step[w: Int](i: Int):
        acc += a.load[width=w](i) * b.load[width=w](i)

    vectorize[dot_step, simd_w](length)
    return acc.reduce_add()


@always_inline
fn _row_softmax[dtype: DType](
    row: UnsafePointer[Scalar[dtype]],
    length: Int,
):
    """In-place numerically-stable softmax over a single row."""
    # 1. Find max for numerical stability.
    var max_val = row[0]
    for i in range(1, length):
        if row[i] > max_val:
            max_val = row[i]

    # 2. Subtract max, exponentiate, accumulate sum.
    var total = Scalar[dtype](0)
    for i in range(length):
        var v = exp(row[i] - max_val)
        row[i] = v
        total += v

    # 3. Normalise.
    var inv_total = Scalar[dtype](1) / total
    for i in range(length):
        row[i] *= inv_total


# ---------------------------------------------------------------------------
# Public kernel
# ---------------------------------------------------------------------------

fn scaled_dot_product_attention[
    dtype: DType = DEFAULT_DTYPE,
](
    q_ptr: UnsafePointer[Scalar[dtype]],
    k_ptr: UnsafePointer[Scalar[dtype]],
    v_ptr: UnsafePointer[Scalar[dtype]],
    out_ptr: UnsafePointer[Scalar[dtype]],
    seq_len: Int,
    head_dim: Int,
):
    """Compute Scaled Dot-Product Attention.

    Args:
        q_ptr:    Pointer to Query matrix  [seq_len × head_dim].
        k_ptr:    Pointer to Key matrix    [seq_len × head_dim].
        v_ptr:    Pointer to Value matrix  [seq_len × head_dim].
        out_ptr:  Pointer to output matrix [seq_len × head_dim] (pre-allocated).
        seq_len:  Number of tokens in the sequence.
        head_dim: Per-head dimensionality (must be a multiple of SIMD_WIDTH).
    """
    alias simd_w = simdwidthof[dtype]()
    var scale = Scalar[dtype](1) / sqrt(Scalar[dtype](head_dim))

    # Allocate a temporary attention-score row on the heap.
    var scores = UnsafePointer[Scalar[dtype]].alloc(seq_len)

    # -----------------------------------------------------------------------
    # Outer loop: one output row per query token.
    # -----------------------------------------------------------------------
    for q_row in range(seq_len):
        var q_base = q_ptr + q_row * head_dim

        # Compute raw attention scores: scores[k] = dot(Q[q_row], K[k]) * scale
        for k_row in range(seq_len):
            var k_base = k_ptr + k_row * head_dim
            scores[k_row] = (
                _dot_product[dtype, simd_w](q_base, k_base, head_dim) * scale
            )

        # Softmax over the score row.
        _row_softmax[dtype](scores, seq_len)

        # Weighted sum over Value rows → out[q_row]
        var out_base = out_ptr + q_row * head_dim
        memset_zero(out_base, head_dim)

        for v_row in range(seq_len):
            var weight = scores[v_row]
            var v_base = v_ptr + v_row * head_dim

            @parameter
            fn accumulate[w: Int](i: Int):
                out_base.store[width=w](
                    i,
                    out_base.load[width=w](i) + weight * v_base.load[width=w](i),
                )

            vectorize[accumulate, simd_w](head_dim)

    scores.free()
