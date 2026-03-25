# main.mojo
# Project Singularity — entry point.
#
# Defines:
#   • ExpertLayer   — a single FFN expert used inside the MoE block.
#   • MoEBlock      — Mixture-of-Experts layer with top-k gating.
#   • AttentionHead — single attention head backed by the SDPA kernel.
#   • TransformerLayer — combines attention + MoE for one transformer layer.
#   • SingularityModel  — full model with multi-device sharding metadata.
#   • main()         — minimal inference smoke-test.

from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof, num_logical_cores
from algorithm import parallelize
from math import sqrt

from attention import scaled_dot_product_attention, DEFAULT_DTYPE

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
        """Minimal forward pass placeholder (weight matmul omitted for brevity)."""
        # In production this would call a tiled GEMM; here we copy input → output
        # to keep the boilerplate compilable and self-contained.
        for i in range(SEQ_LEN * HIDDEN_DIM):
            out_ptr[i] = x_ptr[i]


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
        """Route tokens through top-k experts (simplified gating placeholder)."""
        # Full implementation: compute gate logits, softmax, top-k select,
        # dispatch to experts, weighted sum.  Placeholder copies x → out.
        for i in range(SEQ_LEN * HIDDEN_DIM):
            out_ptr[i] = x_ptr[i]


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
        """Project input to Q/K/V, run SDPA, write result to out_ptr."""
        # Allocate projected Q, K, V buffers.
        var q = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)
        var k = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)
        var v = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)
        memset_zero(q, SEQ_LEN * HEAD_DIM)
        memset_zero(k, SEQ_LEN * HEAD_DIM)
        memset_zero(v, SEQ_LEN * HEAD_DIM)

        # TODO: GEMM projections (wq/wk/wv × x) go here.
        # For now, pass zero-initialised buffers to demonstrate the kernel call.

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
    """One Transformer layer: multi-head attention followed by a MoE block."""
    var heads:     UnsafePointer[AttentionHead]
    var moe_block: MoEBlock
    var layer_id:  Int

    fn __init__(out self, layer_id: Int):
        self.layer_id  = layer_id
        self.moe_block = MoEBlock()
        # Allocate attention heads.
        self.heads = UnsafePointer[AttentionHead].alloc(NUM_HEADS)
        for h in range(NUM_HEADS):
            (self.heads + h).init_pointee_move(AttentionHead(h))

    fn __del__(owned self):
        for h in range(NUM_HEADS):
            (self.heads + h).destroy_pointee()
        self.heads.free()

    fn forward(
        self,
        x_ptr:   UnsafePointer[Scalar[DTYPE]],
        out_ptr: UnsafePointer[Scalar[DTYPE]],
    ):
        """Run multi-head attention then MoE on the input sequence."""
        # Collect per-head outputs into a temporary buffer.
        var mha_out = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HIDDEN_DIM)
        memset_zero(mha_out, SEQ_LEN * HIDDEN_DIM)

        var head_out = UnsafePointer[Scalar[DTYPE]].alloc(SEQ_LEN * HEAD_DIM)

        for h in range(NUM_HEADS):
            memset_zero(head_out, SEQ_LEN * HEAD_DIM)
            (self.heads + h)[].forward(x_ptr, head_out)
            # Scatter head output into the correct slice of mha_out.
            var offset = h * HEAD_DIM
            for s in range(SEQ_LEN):
                for d in range(HEAD_DIM):
                    mha_out[s * HIDDEN_DIM + offset + d] = (
                        head_out[s * HEAD_DIM + d]
                    )

        head_out.free()

        # Residual connection: mha_out += x
        for i in range(SEQ_LEN * HIDDEN_DIM):
            mha_out[i] += x_ptr[i]

        # MoE forward pass (in-place result written to out_ptr).
        self.moe_block.forward(mha_out, out_ptr)

        # Second residual: out += mha_out (pre-MoE)
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
