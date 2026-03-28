# training.mojo
# Training primitives for Project Singularity.
#
# Provides the building blocks needed to train the SingularityModel as a
# causal language model:
#
#   EmbeddingTable  — learnable token → hidden-vector lookup table.
#   LMHead          — linear projection from hidden dim to vocabulary logits.
#   cross_entropy   — mean cross-entropy loss for next-token prediction.
#   sgd_step        — vectorised in-place SGD parameter update.
#
# These components are intentionally self-contained so they can be tested
# independently of the full model.  Back-propagation gradients are expected
# to be supplied by a future autograd layer; the functions here handle the
# forward pass and the parameter-update arithmetic only.

from math import exp, log
from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof
from algorithm import vectorize

from gemm import gemm

# ---------------------------------------------------------------------------
# Shared hyper-parameters
# These aliases must stay in sync with main.mojo.
# ---------------------------------------------------------------------------

alias TRAIN_DTYPE  = DType.float32
alias VOCAB_SIZE   = 32_768   # BPE vocabulary (matches common tokeniser size)
alias _HIDDEN_DIM  = 512      # must equal main.HIDDEN_DIM
alias _SEQ_LEN     = 128      # must equal main.SEQ_LEN


# ---------------------------------------------------------------------------
# EmbeddingTable — token ID → dense vector lookup
# ---------------------------------------------------------------------------

struct EmbeddingTable:
    """Learned token embedding table.

    Maps each integer token ID to a dense vector of dimension *hidden_dim*.
    Weight layout: row i (of length hidden_dim) is the embedding for token i.
    """
    var weight:     UnsafePointer[Scalar[TRAIN_DTYPE]]
    var vocab_size: Int
    var hidden_dim: Int

    fn __init__(out self, vocab_size: Int, hidden_dim: Int):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weight = UnsafePointer[Scalar[TRAIN_DTYPE]].alloc(
            vocab_size * hidden_dim
        )
        memset_zero(self.weight, vocab_size * hidden_dim)

    fn __del__(owned self):
        self.weight.free()

    fn forward(
        self,
        token_ids: UnsafePointer[Scalar[DType.int32]],       # [seq_len]
        out_ptr:   UnsafePointer[Scalar[TRAIN_DTYPE]],        # [seq_len × hidden_dim]
        seq_len:   Int,
    ):
        """Copy embedding rows into out_ptr — one row per token ID.

        Args:
            token_ids: Integer token indices in [0, vocab_size).
            out_ptr:   Output buffer of shape [seq_len × hidden_dim].
            seq_len:   Number of tokens to embed.
        """
        for s in range(seq_len):
            var tid = Int(token_ids[s])
            var src = self.weight + tid * self.hidden_dim
            var dst = out_ptr   + s   * self.hidden_dim
            for d in range(self.hidden_dim):
                dst[d] = src[d]

    fn embed_grad_update(
        self,
        token_ids: UnsafePointer[Scalar[DType.int32]],        # [seq_len]
        grad_ptr:  UnsafePointer[Scalar[TRAIN_DTYPE]],         # [seq_len × hidden_dim]
        seq_len:   Int,
        lr:        Scalar[TRAIN_DTYPE],
    ):
        """Apply SGD gradient update to embedding rows in-place.

        Each token's embedding row is updated by subtracting lr × gradient.

        Args:
            token_ids: Token indices used in the forward pass.
            grad_ptr:  Upstream gradient of shape [seq_len × hidden_dim].
            seq_len:   Sequence length.
            lr:        Learning rate.
        """
        for s in range(seq_len):
            var tid   = Int(token_ids[s])
            var w_row = self.weight + tid * self.hidden_dim
            var g_row = grad_ptr    + s   * self.hidden_dim
            for d in range(self.hidden_dim):
                w_row[d] -= lr * g_row[d]


# ---------------------------------------------------------------------------
# LMHead — hidden states → vocabulary logits
# ---------------------------------------------------------------------------

struct LMHead:
    """Language-model head: linear projection to vocabulary logits.

    Computes: logits = hidden @ weight + bias
      hidden : [seq_len × hidden_dim]
      weight : [hidden_dim × vocab_size]
      bias   : [vocab_size]
      logits : [seq_len × vocab_size]
    """
    var weight:     UnsafePointer[Scalar[TRAIN_DTYPE]]
    var bias:       UnsafePointer[Scalar[TRAIN_DTYPE]]
    var hidden_dim: Int
    var vocab_size: Int

    fn __init__(out self, hidden_dim: Int, vocab_size: Int):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.weight = UnsafePointer[Scalar[TRAIN_DTYPE]].alloc(
            hidden_dim * vocab_size
        )
        self.bias = UnsafePointer[Scalar[TRAIN_DTYPE]].alloc(vocab_size)
        memset_zero(self.weight, hidden_dim * vocab_size)
        memset_zero(self.bias,   vocab_size)

    fn __del__(owned self):
        self.weight.free()
        self.bias.free()

    fn forward(
        self,
        hidden:  UnsafePointer[Scalar[TRAIN_DTYPE]],   # [seq_len × hidden_dim]
        logits:  UnsafePointer[Scalar[TRAIN_DTYPE]],   # [seq_len × vocab_size]
        seq_len: Int,
    ):
        """Compute logits = hidden @ weight + bias.

        Args:
            hidden:  Input hidden states [seq_len × hidden_dim].
            logits:  Output logits buffer [seq_len × vocab_size] (overwritten).
            seq_len: Number of tokens.
        """
        gemm[TRAIN_DTYPE](
            hidden, self.weight, logits,
            seq_len, self.hidden_dim, self.vocab_size,
        )
        # Add bias broadcast across the sequence dimension using SIMD.
        alias simd_w = simdwidthof[TRAIN_DTYPE]()
        for s in range(seq_len):
            var row = logits + s * self.vocab_size

            @parameter
            fn add_bias[w: Int](v: Int):
                row.store[width=w](
                    v, row.load[width=w](v) + self.bias.load[width=w](v)
                )

            vectorize[add_bias, simd_w](self.vocab_size)


# ---------------------------------------------------------------------------
# Cross-entropy loss — next-token prediction objective
# ---------------------------------------------------------------------------

fn cross_entropy_loss(
    logits:     UnsafePointer[Scalar[TRAIN_DTYPE]],   # [seq_len × vocab_size]
    targets:    UnsafePointer[Scalar[DType.int32]],   # [seq_len]  next-token IDs
    seq_len:    Int,
    vocab_size: Int,
) -> Scalar[TRAIN_DTYPE]:
    """Compute the mean cross-entropy loss for next-token prediction.

    For each position t:
        loss_t = -log( softmax(logits[t])[targets[t]] )
               = log_sum_exp(logits[t]) - logits[t][targets[t]]

    Returns the arithmetic mean of loss_t over all seq_len positions.

    Args:
        logits:     Raw (un-normalised) model output [seq_len × vocab_size].
        targets:    Ground-truth next-token IDs      [seq_len].
        seq_len:    Number of token positions.
        vocab_size: Vocabulary size (must match second dim of logits).
    """
    var total_loss = Scalar[TRAIN_DTYPE](0)

    for s in range(seq_len):
        var row = logits + s * vocab_size

        # Numerically-stable log-sum-exp: subtract max before exponentiating.
        var max_val = row[0]
        for v in range(1, vocab_size):
            if row[v] > max_val:
                max_val = row[v]

        var sum_exp = Scalar[TRAIN_DTYPE](0)
        for v in range(vocab_size):
            sum_exp += exp(row[v] - max_val)
        var log_sum_exp = log(sum_exp) + max_val

        var target_logit = row[Int(targets[s])]
        total_loss += log_sum_exp - target_logit

    return total_loss / Scalar[TRAIN_DTYPE](seq_len)


# ---------------------------------------------------------------------------
# SGD parameter update
# ---------------------------------------------------------------------------

fn sgd_step(
    params: UnsafePointer[Scalar[TRAIN_DTYPE]],   # updated in-place
    grads:  UnsafePointer[Scalar[TRAIN_DTYPE]],   # upstream gradients
    count:  Int,
    lr:     Scalar[TRAIN_DTYPE],
):
    """Apply a vanilla SGD update: params -= lr * grads.

    Uses SIMD vectorisation for throughput on large parameter tensors.

    Args:
        params: Flat parameter buffer (mutated in-place).
        grads:  Flat gradient buffer (same length as params).
        count:  Number of scalar elements to update.
        lr:     Learning rate (step size).
    """
    alias simd_w = simdwidthof[TRAIN_DTYPE]()

    @parameter
    fn update_step[w: Int](i: Int):
        params.store[width=w](
            i,
            params.load[width=w](i) - lr * grads.load[width=w](i),
        )

    vectorize[update_step, simd_w](count)
