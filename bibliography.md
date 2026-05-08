# Bibliography

Related work and reference papers for pypto design and research.

Entries are grouped by topic area and sorted by year within each group. Short inline keys (e.g., `[Shoeybi19]`) are used to cross-reference from design documents.

**Verification status.** Entries marked ✅ have been verified against arXiv / publisher metadata (title, authors, year, ID). Entries marked ⚠️ have been included based on widely-known references but the exact citation details (full author list, venue, page numbers) have not been re-verified against a primary source — re-check before formal citation in a paper or external document.

---

## Distributed Training Frameworks

✅ **[Shoeybi19]** M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, B. Catanzaro.
*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.*
arXiv:1909.08053, 2019.
<https://arxiv.org/abs/1909.08053>

> Introduces tensor parallelism (TP) for transformer attention and MLP blocks. Defines the TP/DP/PP group decomposition that Megatron uses. Directly relevant to how `sharded_tensor` should model TP groups as same-pod, high-bandwidth rank axes (`rank_level=4`).

---

✅ **[Narayanan21]** D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. A. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, M. Zaharia.
*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.*
SC'21 (Supercomputing), 2021. arXiv:2104.04473.
<https://arxiv.org/abs/2104.04473>

> Extends Megatron-LM with interleaved pipeline parallelism. Covers schedule, memory footprint, and throughput analysis across TP/PP/DP combinations. Relevant to the interaction between pipeline stages and `sharded_tensor` lifetime barriers (§6).

---

✅ **[Rajbhandari20]** S. Rajbhandari, J. Rasley, O. Ruwase, Y. He.
*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.*
SC'20 (Supercomputing), 2020. arXiv:1910.02054.
<https://arxiv.org/abs/1910.02054>

> Introduces the ZeRO optimizer partitioning model (parameter sharding, gradient sharding, optimizer-state sharding). The mental model of ZeRO Stage 3 (full parameter sharding) is the closest existing precedent to `sharded_tensor` for optimizer states. Relevant to the all-gather/reduce-scatter pattern over DP groups.

---

✅ **[Zhao23]** Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, et al.
*PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.*
arXiv:2304.11277, 2023. (A version was published in VLDB; verify which version you cite before formal use.)
<https://arxiv.org/abs/2304.11277>

> Details the PyTorch FSDP design: mixed-precision sharding, prefetching strategies, and NCCL backend interaction. Most directly comparable to the `sharded_tensor` memory management model (§6): FSDP has the same construction barrier (all_gather before forward) and retirement pattern (reduce_scatter after backward).

---

✅ **[Lepikhin21]** D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, Z. Chen.
*GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.*
ICLR 2021. arXiv:2006.16668.
<https://arxiv.org/abs/2006.16668>

> Introduces sharding annotations (how to partition a tensor) and collective insertion for MoE transformers, implemented as an XLA extension. Closely related to the design philosophy of `sharded_tensor`: partitioning metadata lives in the type annotation, not scattered across the code.

---

## Compiler Frameworks for Distributed Computation

⚠️ **[Frostig18]** R. Frostig, M. J. Johnson, C. Leary.
*Compiling machine learning programs via high-level tracing.*
SysML 2018.
<https://mlsys.org/Conferences/doc/2018/146.pdf>

> Original JAX paper. Introduces the XLA-tracing model. The JAX `Mesh` abstraction (named axes + device placement) referenced in `sharded_tensor_topology_design.md §2.1` is a descendant of this work.

---

⚠️ **[JAX]** JAX project (Google Research and contributors).
*JAX: composable transformations of Python+NumPy programs.*
Project repository.
<https://github.com/jax-ml/jax>

> The `jax.sharding.NamedSharding` and `jax.sharding.Mesh` APIs implement the named-axis topology model surveyed in `sharded_tensor_topology_design.md §2.1`. (Cite the repository or current documentation rather than a single paper for API references.)

---

✅ **[Xu21]** Y. Xu, H. Lee, D. Chen, B. Hechtman, Y. Huang, R. Joshi, M. Krikun, D. Lepikhin, et al.
*GSPMD: General and Scalable Parallelization for ML Computation Graphs.*
arXiv:2105.04663, 2021.
<https://arxiv.org/abs/2105.04663>

> The GSPMD compiler (ancestor of today's JAX/XLA sharding) automatically propagates partition annotations through computation graphs and inserts collectives. The sharding annotation model is the closest existing analogue to pypto's `shard_shape` + `rank_shape` design philosophy.

---

## Collective Communication

⚠️ **[Rabenseifner04]** R. Rabenseifner.
*Optimization of Collective Reduction Operations.*
ICCS 2004 (International Conference on Computational Science), Springer LNCS 3036.

> Foundational paper on recursive halving / recursive doubling algorithms for all-reduce and reduce-scatter. The ring algorithm referenced in `sharded_tensor_topology_design.md §3` is a special case. Foundational reference for collective algorithm selection based on topology.

---

⚠️ **[Thakur05]** R. Thakur, R. Rabenseifner, W. Gropp.
*Optimization of Collective Communication Operations in MPICH.*
International Journal of High Performance Computing Applications, 19(1):49–66, 2005.

> Defines topology-aware collective selection rules still used by MPI implementations today. Directly relevant to the compiler integration sketch in `sharded_tensor_topology_design.md §5.1`.

---

✅ **[Li20]** S. Li, Y. Zhao, R. Varma, O. Salpekar, P. Noordhuis, T. Li, A. Paszke, J. Smith, B. Vaughan, P. Damania, S. Chintala.
*PyTorch Distributed: Experiences on Accelerating Data Parallel Training.*
VLDB 2020. arXiv:2006.15704.
<https://arxiv.org/abs/2006.15704>

> Describes the PyTorch `DistributedDataParallel` design and the `RANK` / `WORLD_SIZE` initialization model surveyed in `sharded_tensor_topology_design.md §2.2`. The gradient bucketing and communication overlap design is relevant to `sharded_tensor` retirement and collective ordering.

---

## Symmetric Shared Memory and PGAS Models

✅ **[Yelick07]** K. Yelick, D. Bonachea, W.-Y. Chen, P. Colella, K. Datta, J. Duell.
*Productivity and Performance Using Partitioned Global Address Space Languages.*
PASCO'07 (Workshop on Parallel Symbolic Computation), pages 24–32, 2007.
<https://doi.org/10.1145/1278177.1278183>

> Surveys PGAS languages (UPC, Titanium) and their performance / productivity trade-offs. The `open_share_memory` model that `sharded_tensor` inherits from is conceptually related to PGAS one-sided communication.

---

⚠️ **[OpenSHMEM]** OpenSHMEM Specification Committee.
*OpenSHMEM Application Programming Interface.*
Cite the specific version you reference (e.g., 1.4 or 1.5).
<http://www.openshmem.org/site/Specification>

> Defines the symmetric heap, put/get primitives, and collective completion model (`shmem_quiet`, `shmem_fence`, barriers). The epoch-based retirement model adopted in `sharded_tensor §6.3` mirrors the OpenSHMEM `shmem_quiet()` epoch model. Relevant to the memory consistency discussion for UB direct-access (audit finding: Critical Issue 1).

---

## Parallelism Search and Cost Models

✅ **[Jia18]** Z. Jia, M. Zaharia, A. Aiken.
*Beyond Data and Model Parallelism for Deep Neural Networks.*
ICML 2018 (PMLR 80). arXiv:1807.05358.
<https://arxiv.org/abs/1807.05358>

> Defines the SOAP space for parallelism strategies (Sample / Operator / Attribute / Parameter) and the FlexFlow framework. Provides theoretical grounding for why `rank_shape` can have multiple axes with different semantics (TP, DP, PP, SP). (Note: an extended version appeared at MLSys 2019 — verify which version you cite.)

---

## Fault Tolerance

⚠️ **[Moody10]** A. Moody, G. Bronevetsky, K. Mohror, B. R. de Supinski.
*Design, Modeling, and Evaluation of a Scalable Multi-Level Checkpointing System.*
SC'10 (Supercomputing), 2010.

> Multi-level checkpointing for HPC (the SCR library). Relevant background for Open Question 9 (fault tolerance) in `sharded_tensor.md` — the scope-epoch retirement model of §6.3 has a natural checkpoint boundary at scope exit.

---

## Notes on This Document

- **Add new entries** in the most relevant group; add a new group if none fits.
- **Citation key format:** `[FirstAuthorYY]` — first author's last name + two-digit year. Add a letter suffix (`[Smith23a]`, `[Smith23b]`) if two entries share the same key.
- **Verification marks:**
  - ✅ — title, authors, year, and arXiv/DOI ID directly verified against the primary source.
  - ⚠️ — citation included based on common knowledge but not re-verified end-to-end. Re-check before formal use.
- **Abstract field** (the indented `>` block under each entry): one or two sentences on why this reference is relevant to pypto, not just what the paper does.
- **Links:** use stable URLs (arXiv abstract page, DOI, or official spec). Avoid personal homepages.
- **When in doubt about a long author list,** truncate to the first few verified authors + `et al.` rather than guessing tail authors.
