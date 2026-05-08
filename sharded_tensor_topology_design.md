# `sharded_tensor` Topology Metadata Design

## Overview

This document complements [sharded_tensor.md](sharded_tensor.md) by detailing:
1. **Why topology metadata is critical** for correctness and performance in distributed systems
2. **How frameworks (JAX, PyTorch, Megatron, DeepSpeed) handle topology**
3. **Concrete implementation plan for simpler runtime**
4. **Integration with pypto's Linqu hierarchy**

The goal is to make topology a first-class concern in `sharded_tensor` design, grounded in existing pypto infrastructure.

---

## 1. The Topology Problem

A `rank_shape = (4, 8)` tensor describing 32 ranks is equally representable whether:
- All 32 ranks share a single UB domain — within a pod (**L4 = Cluster-level-0**, high bandwidth, tight coupling)
- 4 pods of 8 ranks each, connected by fabric — within a supernode (**L5 = Cluster-level-1**, medium bandwidth)
- 32 nodes distributed across racks — **L6 = Cluster-level-2** (contracted bandwidth, wide-area)

Without topology metadata, the runtime **cannot**:

1. **Choose the right collective algorithm**
   - Ring all-reduce works for same-pod but is O(N) hops across pods
   - Hierarchical tree (reduce within pod, reduce across pods) is significantly faster on multi-pod clusters

2. **Generate correct direct-access code** (§5.3.1 of sharded_tensor.md)
   - TLOAD from same die (L1 = Chip die): UB hop (lowest latency)
   - TLOAD from same pod (L4 = Cluster-level-0): fabric hop (medium latency)
   - TLOAD from different pod / supernode (L5–L6): network hop (highest latency)
   - Compiler cannot emit a locality hint without topology info

3. **Validate partition soundness at compile time**
   - Tensor Parallelism (TP) groups should be within-pod (**L4 = Cluster-level-0**, high BW)
   - Data Parallelism (DP) groups can be cross-pod (**L5 = Cluster-level-1** or **L6 = Cluster-level-2**)
   - Compiler can warn if TP ranks span L4 boundaries

---

## 2. How Frameworks Handle Topology

### JAX: Explicit Named Mesh + Device Placement

```python
import jax
import numpy as np

# Explicit device mesh with named axes
devices = np.array(jax.devices()).reshape(2, 4)  # 2×4 device mesh
mesh = jax.sharding.Mesh(devices, axis_names=('data', 'model'))

# Sharding specs reference axis names; jax.jit respects the mesh context
with mesh:
    # parallelism axes ('data', 'model') are user-named strings
    pass
```

**Strengths:**
- User explicitly names parallelism axes (semantics are clear)
- Mesh maintains device-to-rank mapping and can compute distance
- Composable: multiple meshes for different program phases

**Weakness:**
- Axis names are user-invented strings, not tied to hardware hierarchy
- No built-in notion of "pod" or "supernode" (JAX assumes homogeneous grid); pypto's `pl.Level` addresses this: L4 = pod, L5 = supernode, L6 = cross-rack are first-class named levels

---

### PyTorch Distributed: Implicit + Backend-Specific

```python
import torch
import torch.distributed as dist

# Standard multi-rank launch via torchrun
# Rank and world size come from env vars (no topology specified by user)
dist.init_process_group(backend='nccl')  # or 'gloo' for CPU

# Optional: DeviceMesh (PyTorch 2.0+) for explicit sharding
device_mesh = torch.distributed.device_mesh.init_device_mesh(
    'cuda',
    mesh_shape=(2, 4),
    mesh_dim_names=('replicate', 'shard'),
)

# Topology is handled by the backend:
# - NCCL backend: uses NCCL topology queries (ring, tree) automatically per collective
# - GLOO backend (CPU): auto-detects intra-node vs cross-node, uses shared memory for intra-node
```

**Strengths:**
- Simple: `RANK` and `WORLD_SIZE` env vars, works at any scale
- Backend makes topology decisions automatically (no user annotation)
- Scales to millions of parameters (standard in production)

**Weakness:**
- Topology is implicit and backend-specific; hard to query programmatically
- User cannot easily override backend topology decisions
- Different backends (NCCL vs GLOO) have different topology awareness

---

### Megatron-LM: Explicit Communicator Groups

```python
# Megatron's approach: decompose distributed training into named groups
def initialize_model_parallel_groups(
    tensor_model_parallel_size=2,        # TP: should be intra-pod for high BW
    pipeline_model_parallel_size=2,      # PP: can be cross-pod
    data_parallel_size=4,                # DP: typically cross-pod
    virtual_pipeline_model_parallel_size=None,
):
    # Internally:
    # rank → (tp_rank, pp_rank, dp_rank) via deterministic mapping
    # Each (TP, PP, DP) group gets its own MPI communicator
    # NCCL comm per group for collectives
    
    # User can query: "which ranks are in my TP group?"
    tp_group = get_tp_rank_group()
    print(f"TP group size: {len(tp_group)}")  # Should be ~8-16 for optimal BW
```

**Strengths:**
- Explicit semantics: TP, PP, DP groups are first-class (high-level intent is clear)
- Per-group communicators allow topology hints per group
- Widely used and battle-tested

**Weakness:**
- Requires manual group construction for each parallelism strategy
- Doesn't scale well to new strategies (expert parallelism, sequence parallelism, etc.)
- No direct integration with hardware topology (users must infer from NCCL_DEBUG output)

---

### DeepSpeed / FSDP2: Backend-Driven Optimization

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Minimal user specification
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)

# Under the hood:
# - FSDP2 discovers process topology via PyTorch process group
# - For each all-reduce, queries NCCL backend for topology
# - NCCL automatically selects ring, tree, or other algorithm
# - User never sees topology details (implicit + automatic)
```

**Strengths:**
- Simplest user API: no topology specification needed
- Automatic optimization: backend handles it
- Works everywhere (scales from laptop to 10k-node cluster)

**Weakness:**
- Zero visibility: user cannot query or influence topology decisions
- Difficult to debug performance issues (is it topology? algorithm? bandwidth?)

---

## 3. Recommended Design for pypto `sharded_tensor`

### Philosophy

**Adopt PyTorch Distributed's approach (implicit multi-node init via env vars) + JAX's explicit Mesh concept, grounded in pypto's Linqu hierarchy.**

This balances:
- **Simplicity** (standard env vars, single-node default)
- **Explicitness** (Linqu levels are named, not user-invented)
- **Performance** (compiler can emit topology-aware code)

### Data Model: Add `rank_level` to `sharded_tensor`

In pypto's type system, extend `sharded_tensor` with optional topology metadata:

```python
# Python API sketch (final names TBD)
ST = pl.sharded_tensor(
    shape=(M, N),                                       # Global shape
    shard_shape=(M // R0, N // R1),                     # Per-rank slab
    rank_shape=(R0, R1),                                # Rank grid shape
    rank_level=(pl.Level.CLUSTER_0, pl.Level.CLUSTER_1),# OPTIONAL: Linqu level per axis
                                                        # Dim-0 sharded within pod (L4 = Cluster-level-0)
                                                        # Dim-1 sharded within supernode (L5 = Cluster-level-1)
)
```

**Semantics:**
- `rank_level[i]` ∈ {3, 4, 5, 6, 7} = the Linqu level at which axis `i` is sharded
- Omitting `rank_level` means "all axes at same default level" (backward compatible, single-node default)
- Linqu levels are defined in [`machine_hierarchy_and_function_hierarchy.md`](machine_hierarchy_and_function_hierarchy.md) (§1, §5.3):
  - `pl.Level.HOST` = L3 — Host (single OS instance, one or more chips; runs orchestration)
  - `pl.Level.CLUSTER_0` = L4 — Cluster-level-0 (within pod, high BW, tight coupling)
  - `pl.Level.CLUSTER_1` = L5 — Cluster-level-1 (within supernode, medium BW)
  - `pl.Level.CLUSTER_2` = L6 — Cluster-level-2 (cross-rack, contracted BW)
  - `pl.Level.GLOBAL` = L7 — Global Coordinator (top of hierarchy)

### Runtime Implementation in simpler

#### Layer 1: Multi-Node Initialization (in `comm_hccl.cpp`)

**File:** `src/a2a3/platform/onboard/host/comm_hccl.cpp`

Extend HCCL topology parsing to populate new fields:

```cpp
// In comm_hccl.cpp, after HCCL init:

int rank_id = std::stoi(std::getenv("RANK") ?: "0");
int rank_num = std::stoi(std::getenv("WORLD_SIZE") ?: "1");

// Parse rank_shape from env (e.g., PTO_RANK_SHAPE="2,4")
std::vector<int> rank_shape = parse_rank_shape(std::getenv("PTO_RANK_SHAPE") ?: "");
// Parse rank_level from env (e.g., PTO_RANK_LEVEL="4,5")
std::vector<int> rank_level = parse_rank_level(std::getenv("PTO_RANK_LEVEL") ?: "");

// Set on orchestrator
orch->rank_id = rank_id;
orch->rank_num = rank_num;
orch->rank_shape = rank_shape;
orch->rank_level = rank_level;
orch->hccl_comm = hccl_comm;  // HCCL communicator for this rank
```

#### Layer 2: Orchestrator State (in `pto_orchestrator.h`)

**File:** `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.h`

Add multi-node topology fields:

```cpp
struct PTO2OrchestratorState {
    // === Per-chip topology (existing fields; exact names may differ) ===
    int32_t total_cluster_count{0};
    int32_t total_aiv_count{0};

    // === NEW: Multi-node topology (cross-node) ===
    int32_t rank_id{-1};              // This node's ID in the cluster
    int32_t rank_num{1};              // Total number of participating nodes
    std::vector<int32_t> rank_shape;  // Shape of rank grid (e.g., [2, 4])
    std::vector<int32_t> rank_level;  // Linqu level per axis (e.g., [4, 5])
    
    // For UB direct-access embodiment (sharded_tensor §5.3.1):
    std::vector<void*> ubmem_import_bases;  // Mapped base per remote rank
    
    // HCCL communicator (if multi-node)
    HcclComm hccl_comm{nullptr};
};
```

#### Layer 3: Python Setup (in `platform_info.py`)

**File:** `simpler_setup/platform_info.py`

Add topology discovery function:

```python
def discover_multi_node_topology() -> dict:
    """Query environment and HCCL for multi-node topology.
    
    Returns:
        {
            'rank_id': int,
            'rank_num': int,
            'rank_shape': tuple,
            'rank_level': tuple or None,
            'hccl_available': bool,
        }
    """
    rank_id = int(os.getenv('RANK', '0'))
    rank_num = int(os.getenv('WORLD_SIZE', '1'))
    rank_shape = parse_shape_from_env('PTO_RANK_SHAPE')
    rank_level = parse_shape_from_env('PTO_RANK_LEVEL')
    
    hccl_available = (rank_num > 1)  # Multi-node implies HCCL
    
    return {
        'rank_id': rank_id,
        'rank_num': rank_num,
        'rank_shape': rank_shape,
        'rank_level': rank_level,
        'hccl_available': hccl_available,
    }
```

### Compiler Usage: Lowering `sharded_tensor` Operations

When the pypto compiler lowers a `sharded_tensor.all_reduce()` or direct-access `TLOAD`:

```cpp
// Pseudocode: lowering of ST.all_reduce(op=SUM) in pypto compiler

void lower_sharded_tensor_all_reduce(ShardedTensorOp *op, PassContext ctx) {
    const auto* handler = ctx->GetBackendHandler();
    const auto& rank_level = op->sharded_tensor()->rank_level();
    
    // Strategy 1: UB direct-access (preferred if available)
    if (handler->SupportsUBDirectAccess() && rank_level[0] <= 5) {
        // Use ring algorithm on UB, works well intra/cross-pod
        emit_ub_ring_all_reduce(op);
    }
    // Strategy 2: HCCL collective (fallback)
    else {
        // Let HCCL backend pick algorithm (ring or tree)
        emit_hccl_all_reduce(op, rank_level);
    }
}
```

### Usage Example (Python DSL)

```python
import pypto as pl

@pl.program
class DistributedMatMul:
    @pl.function
    def main(self):
        # Global shape: (4096, 4096)
        # 2×4 rank grid: 2 pods × 4 supernodes per pod
        # TP across supernodes (L5), DP across pods (L6)
        weight = pl.sharded_tensor(
            shape=(4096, 4096),
            shard_shape=(2048, 1024),
            rank_shape=(2, 4),
            rank_level=(pl.Level.CLUSTER_1, pl.Level.CLUSTER_2),
            # Axis 0: within supernode (L5 = Cluster-level-1)
            # Axis 1: within cross-rack domain (L6 = Cluster-level-2)
        )
        
        # Gradient all-reduce (DP group)
        grad_ST = ...  # Computed gradient for this rank's slab
        grad_ST.all_reduce(op=pl.ReduceOp.SUM)
        # Compiler generates: hierarchical reduce within pod (tree), then across pods
        
        # Direct access to remote slab (for TP communication)
        remote_slab = weight.view(rank_index=(1, 2))  # Explicit remote rank
        # Compiler generates: UB load or network DMA depending on rank_level
```

---

## 4. Integration with Existing pypto Infrastructure

### Alignment with `machine_hierarchy_and_function_hierarchy.md`

The Linqu levels (L0–L7) are defined in pypto's hierarchy guide (§1, §5.3). This design reuses those levels as topology metadata, giving them operational meaning in `sharded_tensor`.

| Linqu Level | `pl.Level` constant | Bandwidth Character | Use in `rank_level` | Typical Collective Algorithm |
|---|---|---|---|---|
| L3 | `pl.Level.HOST` | Intra-host (shared memory) | Intra-node sharding across chips | Shared-memory copy |
| L4 | `pl.Level.CLUSTER_0` | High BW, tight coupling (pod) | Intra-pod TP groups | Ring, UB direct access |
| L5 | `pl.Level.CLUSTER_1` | Medium BW (within supernode) | Intra-supernode DP groups | Tree reduce, UB fabric |
| L6 | `pl.Level.CLUSTER_2` | Contracted BW (cross-rack) | Cross-rack groups | Multi-stage tree, network |
| L7 | `pl.Level.GLOBAL` | Global coordinator | Rarely used for sharding | Platform-specific |

### Backward Compatibility

- Single-node programs: omit `rank_level`, defaults to `rank_num=1` (no collective overhead)
- Multi-node without topology: omit `rank_level`, runtime uses safe defaults (HCCL picks algorithm)
- Multi-node with topology: provide `rank_level`, runtime optimizes

---

## 5. Open Design Decisions

1. **Environment Variable Naming**
   - Proposed: `RANK`, `WORLD_SIZE`, `PTO_RANK_SHAPE`, `PTO_RANK_LEVEL`
   - Alternative: Use HCCL's existing `HCCL_RANK`, `HCCL_WORLD_SIZE`

2. **Default Topology When `rank_level` Omitted**
   - Proposed: All axes at L6 (conservative, works across racks)
   - Alternative: All axes at L4 (optimistic, works within pod)
   - Alternative: Query HCCL for inferred topology

3. **UB vs HCCL Transport Decision**
   - Proposed: Compiler queries `rank_level[i]` to decide (UB for L≤5, HCCL for L≥6)
   - Alternative: Always use HCCL (simpler, but misses UB optimization)
   - Alternative: Runtime decides based on performance model

4. **Topology Validation at Compile Time**
   - Should the compiler warn if TP ranks span L6 boundary?
   - Should the compiler reject non-Linqu-aligned `rank_level` values?

---

## 6. Testing and Validation

### Unit Test Checklist

- [ ] Single-rank `sharded_tensor` (rank_num=1) overhead is zero
- [ ] Multi-rank with `rank_level` set: collective algorithm matches expected topology
- [ ] Multi-rank without `rank_level`: falls back to safe default
- [ ] Direct-access TLOAD respects `rank_level`: UB for L≤5, network for L≥6
- [ ] Environment variable parsing handles missing/malformed input

### Integration Test Checklist

- [ ] All-reduce on 8-rank pod (L4): uses ring, achieves high BW
- [ ] All-reduce across 2 pods (L6): uses tree, achieves acceptable BW
- [ ] Mixed TP (L4) + DP (L6): correct group semantics

---

## 7. Future Extensions

1. **Topology-aware scheduling:** Assign `ShardedTensor` collectives to least-congested network paths
2. **Fault-aware groups:** Add `rank_level_failover` for handling rank failures within a group
3. **Custom topologies:** Allow user-defined topology functions instead of fixed Linqu levels
4. **Topology-aware profiling:** Report actual BW per `rank_level` axis for debugging

