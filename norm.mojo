# norm.mojo
# Root-Mean-Square Layer Normalisation (RMSNorm).
#
# RMSNorm is the normalisation layer used in LLaMA, Mistral, and other
# modern transformer variants.  Unlike LayerNorm it omits the mean-
# subtraction step, which reduces compute while preserving training
# stability.
#
# Formula (per vector x of length L):
#   rms(x) = sqrt( mean(x²) + eps )
#   x_norm[i] = gamma[i] * x[i] / rms(x)

from math import sqrt
from memory import UnsafePointer


fn rms_norm[dtype: DType](
    x:      UnsafePointer[Scalar[dtype]],   # [length] — normalised in-place
    gamma:  UnsafePointer[Scalar[dtype]],   # [length] — learnable scale (init 1)
    length: Int,
    eps:    Scalar[dtype],
):
    """Apply RMSNorm to a single vector in-place.

    Args:
        x:      Input vector (overwritten with normalised output).
        gamma:  Per-element learnable scale (typically initialised to 1.0).
        length: Number of elements in x and gamma.
        eps:    Small constant added to the RMS for numerical stability.
    """
    # 1. Compute mean of squares.
    var sum_sq = Scalar[dtype](0)
    for i in range(length):
        sum_sq += x[i] * x[i]

    var rms_inv = Scalar[dtype](1) / sqrt(
        sum_sq / Scalar[dtype](length) + eps
    )
    # Note: on hardware with a fast rsqrt intrinsic, this division + sqrt pair
    # can be replaced by a single rsqrt call for better throughput.

    # 2. Scale each element by gamma[i] / rms.
    for i in range(length):
        x[i] = gamma[i] * x[i] * rms_inv
