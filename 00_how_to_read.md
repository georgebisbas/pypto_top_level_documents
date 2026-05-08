# How to Read This Repository

Grouping of the documents under `pypto_top_level_documents/` and a structured reading path.

## Grouping by topic

### A. Foundations — the conceptual model (read first, everything else assumes these)

- [machine_hierarchy_and_function_hierarchy.md](machine_hierarchy_and_function_hierarchy.md) — Defines the **Linqu L0–L7 hierarchy** (Core → Chip → Host → Cluster tiers → Global Coordinator) and the `pl.function` / `pl.at` scope grammar. This is the vocabulary every other doc uses.
- [multi_level_runtime_ring_and_pypto_free_api.md](multi_level_runtime_ring_and_pypto_free_api.md) — The **ring-buffer per scope depth + `fanout_count`/`ref_count`** lifetime model, and the `pl.free` API. Foundation for resource management throughout.

### B. Runtime architecture — how the system actually executes

- [linqu_runtime_design.md](linqu_runtime_design.md) — Synthesis doc: **full distributed runtime** across L0–L6 (orchestrator/scheduler/worker, IP-to-coordinate discovery, two-phase code/data residency, multi-tenancy).
- [simpler_distributed_runtime_design.md](simpler_distributed_runtime_design.md) — Concrete L3 `HostWorker`/`DistWorker` implementation in the `simpler` repo (the layer that builds on top of the L2 ChipWorker).
- [pypto-runtime-arch-docs/](pypto-runtime-arch-docs/) — Formal **4+1 view** restructuring of the runtime architecture (logical / development / process / physical / scenario / cross-cutting / ADRs). Use [white-paper.md](pypto-runtime-arch-docs/white-paper.md) as the on-ramp, then [00-index.md](pypto-runtime-arch-docs/00-index.md).
- [runtime_async.md](runtime_async.md) — Extension to the run-to-completion model for **asynchronous hardware engines** (decoupling "function returned" from "task complete").

### C. Tensor / type system

- [tensor_layout.md](tensor_layout.md) — `tile_shape` for tile-contiguous **physical GM layout** (TLOAD/TSTORE bandwidth).
- [tensor_valid_shape.md](tensor_valid_shape.md) — `valid_shape`: separating **storage layout** from **logical data extent** to satisfy 512-byte alignment.
- [sharded_tensor.md](sharded_tensor.md) — New value type for **rank-partitioned, symmetric shared-memory** tensors (cross-node programming paradigm).
- [sharded_tensor_topology_design.md](sharded_tensor_topology_design.md) — **Topology metadata** that makes `sharded_tensor` performance-aware (collective algorithm choice).

### D. ISA / compiler features (intra-cluster communication)

- [tpush_tpop_isa_design_v3.md](tpush_tpop_isa_design_v3.md) — Latest (v3) TPUSH/TPOP/TFREE ISA design for 1×Cube + 2×Vector clusters.
- [HL_ptoisa_newfeature20260306_TPUSH_TPOP.md](HL_ptoisa_newfeature20260306_TPUSH_TPOP.md) — Earlier dated draft of the same feature; cross-reference.
- [HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md](HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md) — `ExpandMixedKernel` compiler pass that **lowers mixed InCore functions into separate AIC/AIV kernels** wired by TPUSH/TPOP.

### E. Distributed data services

- [linqu_data_system.md](linqu_data_system.md) — Four base services over the UB network: `lingqu_shmem`, block, dfs, db.

### F. Serving / inference engine (LLM application layer, mostly Chinese)

- [UBL128_serving.md](UBL128_serving.md) — UBL128 HBD hardware topology + prefill/decode-disaggregated serving design with prefix caching.
- [pypto_serving_design goal.md](pypto_serving_design%20goal.md) — Goals for pypto-serving (vLLM/SGLang-class engine on Linqu).
- [pypto_serving_reference_sglang_vllm.md](pypto_serving_reference_sglang_vllm.md) — vLLM / SGLang reference notes.
- [pypto_serving_implementation_plan.md](pypto_serving_implementation_plan.md) — Implementation plan that maps L2–L7 to inference responsibilities.

### G. Auxiliary (skip for a structured pass)

- [Gemini_conversation.md](Gemini_conversation.md) — Informal Q&A log; useful only as background.
- `gen_mixed_kernel_local_usage.py` — Helper script, not a design doc.

## Recommended reading path

A linear path that builds dependencies:

1. **Foundations** → A1, A2.
2. **Runtime, top-down** → B1 for the big picture, then B3 ([white-paper.md](pypto-runtime-arch-docs/white-paper.md) → [00-index.md](pypto-runtime-arch-docs/00-index.md) → 01–07) for the formal view, then B2 for the concrete L3 implementation, then B4 for the async extension.
3. **Tensors** → C1 → C2 → C3 → C4 (each adds a property: physical layout → valid extent → sharding → topology).
4. **ISA + compiler pass** → D1 (TPUSH/TPOP v3), then D3 (ExpandMixedKernel). Use D2 only if you need the historical context.
5. **Data services** → E1.
6. **Serving** (only if relevant) → F1 → F2 → F3 → F4.

**One-week minimum spine:** A1, A2, B1, B3 (white-paper only), C1, D1 — covers the machine model, runtime, tensor layout, and cluster-comm ISA without the application layer.
