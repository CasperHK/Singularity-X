# main.mojo
# Project Singularity — entry point.
#
# Defines:
#   • ExpertLayer      — single FFN expert with W1/GeLU/W2 transform.
#   • MoEBlock         — Mixture-of-Experts with learnable top-k gating.
#   • AttentionHead    — single attention head with Q/K/V/O projections.
#   • TransformerLayer — pre-norm attention + MoE for one transformer layer.
#   • SingularityModel — full model with multi-device sharding metadata.
#   • main()           — minimal inference smoke-test.

from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof, num_logical_cores
from algorithm import parallelize
from math import sqrt, exp

from attention import scaled_dot_product_attention, DEFAULT_DTYPE
from gemm import gemm
from norm import rms_norm

# ---------------------------------------------------------------------------
# Compile-time model hyper-parameters
# (Override with `mojo build -D` flags for different configurations.)
# ---------------------------------------------------------------------------

alias DTYPE        = DEFAULT_DTYPE
alias SEQ_LEN      = 128     # tokens per context window
alias HEAD_DIM     = 64      # dimensionality of each attention head
alias NUM_HEADS    = 8       # number of attention heads per layer
alias NUM_LAYERS   = 12      # transformer layers
alias NUM_EXPERTS  = 8       # total experts in each MoE block
alias TOP_K        = 2       # how many experts are activated per token
alias HIDDEN_DIM   = HEAD_DIM * NUM_HEADS   # 512
alias FFN_DIM      = HIDDEN_DIM * 4         # 2048

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

@always_inline
fn _gelu[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Fast GeLU approximation: x · sigmoid(1.702 · x).

    This is the Hendrycks & Gimpel (2016) GeLU approximated via sigmoid,
    which requires only a single exp() call and is accurate to < 0.1 %.
    """
    var e = exp(-Scalar[dtype](1.702) * x)
    return x * (Scalar[dtype](1) / (Scalar[dtype](1) + e))


# ---------------------------------------------------------------------------
# Device descriptor — lightweight struct tracking which physical device
# (GPU/NPU index) is responsible for a layer shard.
# ---------------------------------------------------------------------------

@value
struct DeviceType(EqualityComparable, Stringable):
    """Type-safe enumeration of supported compute device categories."""
    var _value: Int

    alias GPU = DeviceType(0)
    alias NPU = DeviceType(1)
    alias CPU = DeviceType(2)

    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    fn __str__(self) -> String:
        if self._value == 0:
            return "GPU"
        elif self._value == 1:
            return "NPU"
        else:
            return "CPU"


struct DeviceDescriptor:
    """Identifies a physical compute device."""
    var device_id:   Int
    var device_type: DeviceType

    fn __init__(out self, id: Int, type: DeviceType):
        self.device_id   = id
        self.device_type = type

    fn __copyinit__(out self, other: Self):
        self.device_id   = other.device_id
        self.device_type = other.device_type

    fn describe(self):
        print(String(self.device_type) + ":" + String(self.device_id), end="")


# ---------------------------------------------------------------------------
# ExpertLayer — a single Feed-Forward Network expert.
# Weights live in manually managed heap memory for zero-overhead access.
# ---------------------------------------------------------------------------

struct ExpertLayer:
    """Two-layer FFN: out = W2 · GeLU(W1 · x)."""
    var w1: UnsafePointer[Scalar[DTYPE]]   # [HIDDEN_DIM × FFN_DIM]
    var w2: UnsafePointer[Scalar[DTYPE]]   # [FFN_DIM × HIDDEN_DIM]
    var expert_id: Int

    fn __init__(out self, expert_id: Int):
        self.expert_id = expert_id
        self.w1 = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM * FFN_DIM)
        self.w2 = UnsafePointer[Scalar[DTYPE]].alloc(FFN_DIM * HIDDEN_DIM)
        # Initialise weights to zero (real training would populate these).
        memset_zero(self.w1, HIDDEN_DIM * FFN_DIM)
        memset_zero(self.w2, FFN_DIM * HIDDEN_DIM)

    fn __del__(owned self):
        self.w1.free()
        self.w2.free()

    fn forward(
        self,
        x_ptr:   UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HIDDEN_DIM]
        out_ptr: UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HIDDEN_DIM]
    ):
        """Two-layer FFN forward pass: out = W2 · GeLU(W1 · x).

        Computes the full sequence in one batched GEMM pair:
          tmp     = x @ w1          [SEQ_LEN × FFN_DIM]
          GeLU applied element-wise
          out_ptr = tmp @ w2         [SEQ_LEN × HIDDEN_DIM]
        """
        var tmp = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * FFN_DIM)
        # tmp = x @ w1  (SEQ_LEN × HIDDEN_DIM) @ (HIDDEN_DIM × FFN_DIM)
        gemm[DTYPE](x_ptr, self.w1, tmp, SEQ_LEN, HIDDEN_DIM, FFN_DIM)
        # GeLU activation in-place.
        for i in range(SEQ_LEN * FFN_DIM):
            tmp[i] = _gelu(tmp[i])
        # out = tmp @ w2  (SEQ_LEN × FFN_DIM) @ (FFN_DIM × HIDDEN_DIM)
        gemm[DTYPE](tmp, self.w2, out_ptr, SEQ_LEN, FFN_DIM, HIDDEN_DIM)
        tmp.free()

    fn forward_single(
        self,
        x_row:   UnsafePointer[Scalar[DTYPE]],   # [HIDDEN_DIM] — one token
        out_row: UnsafePointer[Scalar[DTYPE]],   # [HIDDEN_DIM] — one token output
    ):
        """Single-token FFN forward: out = W2 · GeLU(W1 · x).

        Used by MoEBlock to dispatch individual tokens to their chosen experts
        without processing the entire sequence through each expert.
        """
        var tmp = UnsafePointer[Scalar[DTYPE]].alloc(FFN_DIM)
        # tmp = x_row @ w1  (1 × HIDDEN_DIM) @ (HIDDEN_DIM × FFN_DIM)
        gemm[DTYPE](x_row, self.w1, tmp, 1, HIDDEN_DIM, FFN_DIM)
        for i in range(FFN_DIM):
            tmp[i] = _gelu(tmp[i])
        # out_row = tmp @ w2  (1 × FFN_DIM) @ (FFN_DIM × HIDDEN_DIM)
        gemm[DTYPE](tmp, self.w2, out_row, 1, FFN_DIM, HIDDEN_DIM)
        tmp.free()


# ---------------------------------------------------------------------------
# MoEBlock — routes each token to the top-k experts.
# ---------------------------------------------------------------------------

struct MoEBlock:
    """Mixture-of-Experts block: gating network + NUM_EXPERTS ExpertLayer."""
    var gate_w:  UnsafePointer[Scalar[DTYPE]]   # [HIDDEN_DIM × NUM_EXPERTS]
    var experts: UnsafePointer[ExpertLayer]

    fn __init__(out self):
        self.gate_w = UnsafePointer[Scalar[DTYPE]].alloc(
            HIDDEN_DIM * NUM_EXPERTS
        )
        memset_zero(self.gate_w, HIDDEN_DIM * NUM_EXPERTS)
        # Allocate storage for ExpertLayer objects.
        self.experts = UnsafePointer[ExpertLayer].alloc(NUM_EXPERTS)
        for i in range(NUM_EXPERTS):
            (self.experts + i).init_pointee_move(ExpertLayer(i))

    fn __del__(owned self):
        self.gate_w.free()
        for i in range(NUM_EXPERTS):
            (self.experts + i).destroy_pointee()
        self.experts.free()

    fn forward(
        self,
        x_ptr:   UnsafePointer[Scalar[DTYPE]],
        out_ptr: UnsafePointer[Scalar[DTYPE]],
    ):
        """Route each token to the top-k experts and aggregate their outputs.

        Algorithm:
          1. Compute gate logits: (SEQ_LEN × HIDDEN_DIM) @ (HIDDEN_DIM × NUM_EXPERTS)
          2. Softmax over expert dimension per token.
          3. For each token select the top-2 experts by gate probability.
          4. Renormalise selected weights to sum to 1.
          5. Run each selected expert on the single token row (forward_single).
          6. Accumulate weighted expert outputs into out_ptr.
        """
        # --- gate logits --------------------------------------------------- #
        var gate_logits = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * NUM_EXPERTS)
        gemm[DTYPE](x_ptr, self.gate_w, gate_logits, SEQ_LEN, HIDDEN_DIM, NUM_EXPERTS)

        # Softmax over experts for each token (numerically stable).
        for s in range(SEQ_LEN):
            var row = gate_logits + s * NUM_EXPERTS
            var max_val = row[0]
            for e in range(1, NUM_EXPERTS):
                if row[e] > max_val:
                    max_val = row[e]
            var total = Scalar[DTYPE](0)
            for e in range(NUM_EXPERTS):
                var v = exp(row[e] - max_val)
                row[e] = v
                total += v
            var inv_total = Scalar[DTYPE](1) / total
            for e in range(NUM_EXPERTS):
                row[e] *= inv_total

        # --- top-k dispatch ------------------------------------------------ #
        memset_zero(out_ptr, SEQ_LEN * HIDDEN_DIM)
        var expert_out = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM)

        for s in range(SEQ_LEN):
            var logit_row = gate_logits + s * NUM_EXPERTS

            # Find top-1 expert.
            var top1_idx = 0
            var top1_val = logit_row[0]
            for e in range(1, NUM_EXPERTS):
                if logit_row[e] > top1_val:
                    top1_val = logit_row[e]
                    top1_idx = e

            # Find top-2 expert (different from top-1).
            # NUM_EXPERTS >= 2 is guaranteed by the model hyper-parameters.
            var top2_idx = 0 if top1_idx != 0 else 1
            var top2_val = logit_row[top2_idx]
            for e in range(NUM_EXPERTS):
                if e != top1_idx and logit_row[e] > top2_val:
                    top2_val = logit_row[e]
                    top2_idx = e

            # Renormalise top-2 weights so they sum to 1.
            var norm = top1_val + top2_val
            var w1: Scalar[DTYPE]
            var w2: Scalar[DTYPE]
            if norm > Scalar[DTYPE](0):
                w1 = top1_val / norm
                w2 = top2_val / norm
            else:
                w1 = Scalar[DTYPE](0.5)
                w2 = Scalar[DTYPE](0.5)

            var x_row   = x_ptr   + s * HIDDEN_DIM
            var out_row = out_ptr + s * HIDDEN_DIM

            # Expert 1.
            memset_zero(expert_out, HIDDEN_DIM)
            (self.experts + top1_idx)[].forward_single(x_row, expert_out)
            for d in range(HIDDEN_DIM):
                out_row[d] += w1 * expert_out[d]

            # Expert 2.
            memset_zero(expert_out, HIDDEN_DIM)
            (self.experts + top2_idx)[].forward_single(x_row, expert_out)
            for d in range(HIDDEN_DIM):
                out_row[d] += w2 * expert_out[d]

        expert_out.free()
        gate_logits.free()


# ---------------------------------------------------------------------------
# AttentionHead — wraps the SDPA kernel for a single head.
# ---------------------------------------------------------------------------

struct AttentionHead:
    """One multi-head-attention head with explicit QKV weight storage."""
    var wq: UnsafePointer[Scalar[DTYPE]]   # [HIDDEN_DIM × HEAD_DIM]
    var wk: UnsafePointer[Scalar[DTYPE]]
    var wv: UnsafePointer[Scalar[DTYPE]]
    var wo: UnsafePointer[Scalar[DTYPE]]   # [HEAD_DIM  × HIDDEN_DIM]
    var head_id: Int

    fn __init__(out self, head_id: Int):
        self.head_id = head_id
        self.wq = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM * HEAD_DIM)
        self.wk = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM * HEAD_DIM)
        self.wv = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM * HEAD_DIM)
        self.wo = UnsafePointer[Scalar[DTYPE]].alloc(HEAD_DIM  * HIDDEN_DIM)
        memset_zero(self.wq, HIDDEN_DIM * HEAD_DIM)
        memset_zero(self.wk, HIDDEN_DIM * HEAD_DIM)
        memset_zero(self.wv, HIDDEN_DIM * HEAD_DIM)
        memset_zero(self.wo, HEAD_DIM  * HIDDEN_DIM)

    fn __del__(owned self):
        self.wq.free()
        self.wk.free()
        self.wv.free()
        self.wo.free()

    fn forward(
        self,
        x_ptr:   UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HIDDEN_DIM]
        out_ptr: UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HEAD_DIM]
    ):
        """Project input to Q/K/V, run SDPA, write result to out_ptr.

        Q = x @ wq   [SEQ_LEN × HEAD_DIM]
        K = x @ wk   [SEQ_LEN × HEAD_DIM]
        V = x @ wv   [SEQ_LEN × HEAD_DIM]
        out = SDPA(Q, K, V)
        """
        # Allocate projected Q, K, V buffers.
        var q = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)
        var k = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)
        var v = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)

        # Q/K/V projections via GEMM.
        # x: (SEQ_LEN × HIDDEN_DIM) @ wq: (HIDDEN_DIM × HEAD_DIM) → (SEQ_LEN × HEAD_DIM)
        gemm[DTYPE](x_ptr, self.wq, q, SEQ_LEN, HIDDEN_DIM, HEAD_DIM)
        gemm[DTYPE](x_ptr, self.wk, k, SEQ_LEN, HIDDEN_DIM, HEAD_DIM)
        gemm[DTYPE](x_ptr, self.wv, v, SEQ_LEN, HIDDEN_DIM, HEAD_DIM)

        scaled_dot_product_attention[DTYPE](
            q, k, v, out_ptr, SEQ_LEN, HEAD_DIM
        )

        q.free()
        k.free()
        v.free()


# ---------------------------------------------------------------------------
# TransformerLayer — attention + MoE for one Transformer layer.
# ---------------------------------------------------------------------------

struct TransformerLayer:
    """One Transformer layer: pre-norm attention followed by a pre-norm MoE block.

    Architecture (pre-LayerNorm style, as used in LLaMA / Mistral):
        mha_in  = RMSNorm(x, gamma_attn)
        mha_out = MultiHeadAttention(mha_in)   + x          # residual
        moe_in  = RMSNorm(mha_out, gamma_moe)
        out     = MoEBlock(moe_in)             + mha_out    # residual
    """
    var heads:      UnsafePointer[AttentionHead]
    var moe_block:  MoEBlock
    var gamma_attn: UnsafePointer[Scalar[DTYPE]]   # [HIDDEN_DIM] RMSNorm scale
    var gamma_moe:  UnsafePointer[Scalar[DTYPE]]   # [HIDDEN_DIM] RMSNorm scale
    var layer_id:   Int

    fn __init__(out self, layer_id: Int):
        self.layer_id  = layer_id
        self.moe_block = MoEBlock()
        # Allocate attention heads.
        self.heads = UnsafePointer[AttentionHead].alloc(NUM_HEADS)
        for h in range(NUM_HEADS):
            (self.heads + h).init_pointee_move(AttentionHead(h))
        # RMSNorm scale parameters — initialised to 1 (identity transform).
        self.gamma_attn = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM)
        self.gamma_moe  = UnsafePointer[Scalar[DTYPE]].alloc(HIDDEN_DIM)
        for i in range(HIDDEN_DIM):
            self.gamma_attn[i] = Scalar[DTYPE](1.0)
            self.gamma_moe[i]  = Scalar[DTYPE](1.0)

    fn __del__(owned self):
        for h in range(NUM_HEADS):
            (self.heads + h).destroy_pointee()
        self.heads.free()
        self.gamma_attn.free()
        self.gamma_moe.free()

    fn forward(
        self,
        x_ptr:   UnsafePointer[Scalar[DTYPE]],
        out_ptr: UnsafePointer[Scalar[DTYPE]],
    ):
        """Run pre-norm multi-head attention then pre-norm MoE on the input."""
        alias eps = Scalar[DTYPE](1e-6)

        # ------------------------------------------------------------------ #
        # 1. Pre-attention RMSNorm (operate on a copy to preserve x for residual)
        # ------------------------------------------------------------------ #
        var norm_x = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
        for i in range(SEQ_LEN * HIDDEN_DIM):
            norm_x[i] = x_ptr[i]
        for s in range(SEQ_LEN):
            rms_norm[DTYPE](norm_x + s * HIDDEN_DIM, self.gamma_attn, HIDDEN_DIM, eps)

        # ------------------------------------------------------------------ #
        # 2. Multi-head attention on norm_x
        # ------------------------------------------------------------------ #
        var mha_out  = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
        memset_zero(mha_out, SEQ_LEN * HIDDEN_DIM)
        var head_out = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)

        for h in range(NUM_HEADS):
            memset_zero(head_out, SEQ_LEN * HEAD_DIM)
            (self.heads + h)[].forward(norm_x, head_out)
            # Scatter head output into the correct slice of mha_out.
            var offset = h * HEAD_DIM
            for s in range(SEQ_LEN):
                for d in range(HEAD_DIM):
                    mha_out[s * HIDDEN_DIM + offset + d] = (
                        head_out[s * HEAD_DIM + d]
                    )

        head_out.free()
        norm_x.free()

        # Residual connection: mha_out += x
        for i in range(SEQ_LEN * HIDDEN_DIM):
            mha_out[i] += x_ptr[i]

        # ------------------------------------------------------------------ #
        # 3. Pre-MoE RMSNorm (operate on a copy to preserve mha_out for residual)
        # ------------------------------------------------------------------ #
        var norm_mha = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
        for i in range(SEQ_LEN * HIDDEN_DIM):
            norm_mha[i] = mha_out[i]
        for s in range(SEQ_LEN):
            rms_norm[DTYPE](norm_mha + s * HIDDEN_DIM, self.gamma_moe, HIDDEN_DIM, eps)

        # ------------------------------------------------------------------ #
        # 4. MoE block on norm_mha → out_ptr, then residual
        # ------------------------------------------------------------------ #
        self.moe_block.forward(norm_mha, out_ptr)
        norm_mha.free()

        # Second residual: out += mha_out
        for i in range(SEQ_LEN * HIDDEN_DIM):
            out_ptr[i] += mha_out[i]

        mha_out.free()


# ---------------------------------------------------------------------------
# SingularityModel — top-level model struct with multi-device distribution.
# ---------------------------------------------------------------------------

struct SingularityModel:
    """Full Singularity LLM: NUM_LAYERS transformer layers with MoE experts.

    Multi-device sharding: each layer is assigned a DeviceDescriptor so the
    runtime (or a future distributed executor) can schedule work on the
    correct GPU/NPU.  The model itself is device-agnostic; actual data
    migration is handled by the executor backend.
    """
    var layers:  UnsafePointer[TransformerLayer]
    var devices: UnsafePointer[DeviceDescriptor]

    fn __init__(out self, num_devices: Int):
        # Allocate transformer layers.
        self.layers = UnsafePointer[TransformerLayer].alloc(NUM_LAYERS)
        for l in range(NUM_LAYERS):
            (self.layers + l).init_pointee_move(TransformerLayer(l))

        # Assign each layer to a device in a round-robin fashion.
        self.devices = UnsafePointer[DeviceDescriptor].alloc(NUM_LAYERS)
        for l in range(NUM_LAYERS):
            var dev_id = l % num_devices
            (self.devices + l).init_pointee_move(
                DeviceDescriptor(dev_id, DeviceType.GPU)
            )

    fn __del__(owned self):
        for l in range(NUM_LAYERS):
            (self.layers  + l).destroy_pointee()
            (self.devices + l).destroy_pointee()
        self.layers.free()
        self.devices.free()

    fn print_device_map(self):
        """Print the layer → device assignment for debugging."""
        print("=== Singularity device map ===")
        for l in range(NUM_LAYERS):
            print("  layer " + String(l) + " → ", end="")
            (self.devices + l)[].describe()
            print("")

    fn forward(
        self,
        input_ptr:  UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HIDDEN_DIM]
        output_ptr: UnsafePointer[Scalar[DTYPE]],   # [SEQ_LEN × HIDDEN_DIM]
    ):
        """Sequential forward pass through all transformer layers.

        In a real distributed setting each layer dispatch would be
        asynchronous and data would be migrated to the target device before
        the kernel launch.
        """
        # Use double-buffering: alternate between two heap buffers so we
        # never need a third allocation.
        var buf_a = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
        var buf_b = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)

        # Copy input into buf_a.
        for i in range(SEQ_LEN * HIDDEN_DIM):
            buf_a[i] = input_ptr[i]

        for l in range(NUM_LAYERS):
            var src = buf_a if l % 2 == 0 else buf_b
            var dst = buf_b if l % 2 == 0 else buf_a
            memset_zero(dst, SEQ_LEN * HIDDEN_DIM)
            (self.layers + l)[].forward(src, dst)

        # Copy final result to output_ptr.
        var final_buf = buf_a if NUM_LAYERS % 2 == 0 else buf_b
        for i in range(SEQ_LEN * HIDDEN_DIM):
            output_ptr[i] = final_buf[i]

        buf_a.free()
        buf_b.free()


# ---------------------------------------------------------------------------
# main — minimal smoke-test / inference demo.
# ---------------------------------------------------------------------------

fn main():
    print("🔥 Project Singularity — initialising model")
    print(
        "   Layers:", NUM_LAYERS,
        "| Heads:", NUM_HEADS,
        "| Experts:", NUM_EXPERTS,
        "| Top-k:", TOP_K,
    )

    # Detect available logical cores and use that as a proxy for device count.
    var num_devices = max(1, num_logical_cores() // 4)
    print("   Simulating", num_devices, "device(s)")

    var model = SingularityModel(num_devices)
    model.print_device_map()

    # Allocate a zero-filled dummy input sequence.
    var seq_in  = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
    var seq_out = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
    memset_zero(seq_in,  SEQ_LEN * HIDDEN_DIM)
    memset_zero(seq_out, SEQ_LEN * HIDDEN_DIM)

    # Inject a trivial non-zero token embedding at position 0.
    seq_in[0] = Scalar[DTYPE](1.0)

    print("   Running forward pass …")
    model.forward(seq_in, seq_out)

    # Report the first few output values as a sanity check.
    print("   Output[0:8]:", end=" ")
    for i in range(8):
        print(seq_out[i], end=" ")
    print("")
    print("✅ Forward pass complete.")

    seq_in.free()
    seq_out.free()
