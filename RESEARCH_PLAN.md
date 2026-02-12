# MD-PM Further Research Plan

## Executive Summary

MD-PM (Memoization-based Molecular Dynamics with Persistent Memory) trades compute for memory by precomputing interatomic force/energy values into lookup tables, replacing expensive per-timestep analytic evaluations with O(1) table lookups. This research plan identifies concrete directions to deepen the memoization strategy, exploit modern memory hardware, broaden the scope of tabulated physics, and rigorously characterize when and why the approach wins.

---

## 1. Persistent Memory and Tiered Memory Integration

### 1.1 Intel Optane / CXL-Attached Memory Support

**Motivation.** MD-PM's core premise — large precomputed tables — maps directly onto byte-addressable persistent memory (PMEM) and CXL-attached DRAM tiers, where capacity is abundant but latency is higher than local DRAM. Currently, all table allocation goes through LAMMPS's `memory->create()` (malloc-based), with no awareness of memory tiers.

**Research tasks.**
- Integrate `libmemkind` or `libpmem2` into `memory.cpp` so that table arrays can be explicitly placed on PMEM/CXL tiers via an allocator policy flag.
- Add a `pair_style table` keyword (e.g., `pmem yes`) that routes the `compute_table()` allocations to persistent memory.
- Persist precomputed tables across simulation restarts: serialize the `Table` struct to a memory-mapped file on PMEM so that a restarted simulation skips the spline-fitting / table-construction phase entirely.
- Benchmark table lookup latency and throughput on DRAM vs. Optane vs. CXL memory, varying table sizes from 2^10 to 2^20 entries.

**Expected outcome.** Quantified speedup from eliminating table reconstruction on restart; characterization of the memory-bandwidth/latency tradeoff curve for each tier; practical guidelines for table sizing on tiered-memory nodes.

### 1.2 NUMA-Aware Table Replication

**Motivation.** Tables are currently MPI-broadcast and identically replicated on every rank. On multi-socket NUMA nodes, remote-socket table accesses can double latency.

**Research tasks.**
- Add per-socket table replication using `numa_alloc_onnode()` or `hwloc`-guided placement.
- Measure NUMA-local vs. NUMA-remote table lookup throughput in the pair-force inner loop.
- Evaluate selective replication: replicate only the hot type-pair tables, and share cold ones.

---

## 2. Expanded Memoization Scope

### 2.1 Tabulated Many-Body Potentials

**Motivation.** Only pairwise potentials are fully tabulated today. Many-body potentials (EAM, Tersoff, MEAM) dominate metallic and semiconductor simulations and spend significant time evaluating embedding functions or angular terms. `pair_vashishta_table.cpp` and `pair_tersoff_table.cpp` already exist as partial examples.

**Research tasks.**
- Generalize the memoization framework to multi-dimensional tables: 2D tables for EAM embedding functions F(rho) where rho is a summed density, and 2D/3D tables for angular-dependent potentials g(cos(theta)).
- Implement a `pair_style eam/table` that precomputes the electron-density function rho(r), pairwise energy phi(r), and embedding function F(rho) on grids, replacing the per-step spline evaluation in `pair_eam.cpp`.
- Extend to SNAP/MLIAP descriptors: precompute bispectrum components on a grid of neighbor-environment descriptors.

**Expected outcome.** Table-accelerated EAM with measured speedup vs. stock `pair_eam`; feasibility assessment for angular and many-body tabulation.

### 2.2 Memoization of Long-Range Electrostatics

**Motivation.** PPPM is the second-largest cost center after pair forces (20% in the rhodopsin benchmark). The real-space Coulomb sum already uses interpolation tables (`pair.h` lines 84-93: `rtable`, `ftable`, `etable`), but the reciprocal-space FFT grid charges and force interpolation are computed fresh each step.

**Research tasks.**
- Cache and reuse the reciprocal-space mesh charges across timesteps when atomic displacements are below a threshold, updating only the changed grid points (delta-PPPM).
- Tabulate the charge-spreading (interpolation) weights as a function of fractional grid position, replacing per-atom polynomial evaluation.
- Evaluate accuracy degradation as a function of the reuse interval and displacement threshold.

### 2.3 Tabulated Fix Operations

**Motivation.** Several compute-intensive fixes (e.g., `fix wall/*`, `fix cmap`) evaluate analytic wall potentials or cross-map correction surfaces repeatedly.

**Research tasks.**
- Tabulate wall potentials (LJ 9-3, LJ 10-4-3, harmonic, Morse) as a function of wall-distance.
- Precompute CMAP correction energy and force grids at initialization (already partially done in `fix_cmap.cpp`; benchmark and optimize the grid resolution/interpolation tradeoff).

---

## 3. Table Construction and Accuracy

### 3.1 Adaptive Table Resolution

**Motivation.** The current implementation uses uniform spacing in r^2, which wastes entries in flat potential regions and under-resolves steep repulsive walls. The bitmap style uses logarithmic spacing (via IEEE 754 bit patterns) but only with power-of-two granularity.

**Research tasks.**
- Implement adaptive (non-uniform) table spacing that concentrates entries near the repulsive wall and near cutoff features, using an error-driven refinement criterion.
- Compare accuracy-per-byte against uniform and bitmap spacing for LJ, Morse, Buckingham, and Coulomb potentials.
- Develop an automatic table-size selector: given a target force error tolerance, compute the minimum N for each interpolation style.

### 3.2 Higher-Order Interpolation

**Motivation.** LINEAR gives O(h^2) error, SPLINE gives O(h^4) but costs 6 multiplies. Hermite interpolation using stored force values as derivatives could give O(h^4) accuracy at O(h^2) cost.

**Research tasks.**
- Implement a HERMITE table style that stores (e, f) pairs and uses f = -dE/dr as the Hermite derivative, achieving cubic accuracy with only 2 stored values per point.
- Benchmark accuracy and speed vs. LINEAR and SPLINE at matched table sizes and matched error tolerances.

### 3.3 Error Analysis Framework

**Research tasks.**
- Build an automated test harness that compares table-lookup forces against analytic reference forces for all supported pair styles, measuring maximum and RMS force error as a function of N and interpolation style.
- Quantify energy drift in NVE simulations as a function of table parameters; establish recommended minimum table sizes for energy-conserving dynamics.
- Test sensitivity to round-off: compare float vs. double table storage and the impact on the bitmap union-cast technique.

---

## 4. Hardware-Specific Acceleration

### 4.1 GPU Table Optimization

**Motivation.** `pair_table_gpu.cpp` offloads to GPU but uses the same data layout as the CPU version. GPU texture memory and L1/L2 cache have very different access patterns.

**Research tasks.**
- Store tables in CUDA texture memory (1D textures with hardware linear interpolation) and benchmark against global-memory tables with software interpolation.
- Evaluate read-only `__ldg()` intrinsics for table loads on modern NVIDIA architectures (Ampere, Hopper).
- Optimize the Kokkos table implementation (`pair_table_kokkos.cpp`) for AMD GPUs (HIP backend) and Intel GPUs (SYCL backend), comparing memory-access strategies.
- Benchmark bandwidth-limited vs. compute-limited regimes: at what table size does GPU memory bandwidth saturate?

### 4.2 SIMD Vectorization of Table Lookups

**Motivation.** Table lookups (gather operations) do not auto-vectorize well. AVX-512 provides `_mm512_i32gather_pd` for vectorized gathers.

**Research tasks.**
- Implement an explicit AVX-512 gather path for LINEAR and BITMAP table lookups, processing 8 pairs simultaneously.
- Compare with the USER-INTEL package's existing vectorization strategy for analytic pair styles.
- Measure speedup on Xeon Scalable (Ice Lake, Sapphire Rapids) and evaluate whether the gather bottleneck or the index-computation bottleneck dominates.

### 4.3 Prefetching Strategies

**Research tasks.**
- Insert software prefetch hints (`__builtin_prefetch`) for the next pair's table entry while processing the current pair, exploiting the neighbor-list traversal order.
- Evaluate neighbor-list sorting (by type pair, by distance bin) to improve table access locality.
- Measure L1/L2/L3 cache miss rates for table lookups using hardware performance counters, and quantify the benefit of prefetching at each level.

---

## 5. Scalability and Performance Characterization

### 5.1 Compute-vs-Memory Crossover Analysis

**Motivation.** The memoization tradeoff (table lookup vs. analytic computation) depends on the cost of the analytic formula, table size, memory bandwidth, and cache behavior. No systematic crossover analysis exists.

**Research tasks.**
- For each pair style that has both analytic and table variants (LJ, Morse, Buckingham, Coulomb, EAM), measure time-per-pair as a function of table size (N = 2^8 to 2^20) and compare against the analytic baseline.
- Identify the crossover table size where table lookup becomes slower than analytic computation due to cache misses.
- Characterize how the crossover shifts with: (a) number of atom types (more tables compete for cache), (b) system size (neighbor list length), (c) hardware generation.

### 5.2 Strong and Weak Scaling Studies

**Research tasks.**
- Run the 5 standard benchmarks (LJ, chain, EAM, chute, rhodo) with table-based pair styles at varying core counts (1 to 1024+) and measure parallel efficiency.
- Compare scaling of table-based vs. analytic pair styles, focusing on whether table replication memory cost becomes limiting at high core counts.
- Evaluate MPI+OpenMP hybrid configurations: does sharing a single table copy across threads on a socket reduce memory pressure while maintaining performance?

### 5.3 Large-System Benchmarks

**Research tasks.**
- Design a benchmark suite targeting the big-memory regime: 10M to 1B atoms with large table sizes (2^16 to 2^20 entries per type pair).
- Measure memory footprint vs. atom count and table size; identify the memory-limited maximum system size on a given node.
- Benchmark on actual big-memory hardware (e.g., 1-4 TB DRAM nodes, Optane-equipped systems) and compare against commodity 256 GB nodes.

---

## 6. Algorithmic Extensions

### 6.1 On-the-Fly Table Refinement

**Motivation.** For reactive simulations or systems with large temperature ranges, the relevant distance range changes during the simulation. Static tables may waste entries on unvisited regions.

**Research tasks.**
- Implement a dynamic table that monitors the accessed distance range per type pair and refines (or coarsens) the table at periodic intervals.
- Track access histograms in the inner loop (using lightweight counters, not per-access logging) and trigger refinement when the distribution shifts.
- Evaluate the overhead of refinement checks and table reconstruction vs. the accuracy/memory benefit.

### 6.2 Machine-Learned Potential Tabulation

**Motivation.** Neural network potentials (NNPs) and Gaussian approximation potentials (GAPs) are expensive to evaluate per atom. If the potential energy surface can be tabulated in a descriptor space, table lookups could dramatically accelerate ML/MD.

**Research tasks.**
- For low-dimensional descriptors (e.g., 2-body or 3-body symmetry functions), precompute the NNP output on a grid and use interpolation during MD.
- Assess the curse of dimensionality: determine the maximum descriptor dimensionality for which tabulation remains practical (likely 3-5D).
- Integrate with the MLIAP package: add a `mliap model table` variant that stores precomputed model outputs.

### 6.3 Memoization with Spatial Hashing

**Motivation.** Instead of tabulating in distance space, cache (memoize) recently computed force values keyed by quantized pair coordinates. If atoms move slowly, many pairs will hit the cache.

**Research tasks.**
- Implement a spatial hash table that stores (type_i, type_j, quantized_r) -> (force, energy) pairs.
- Evaluate hit rates as a function of quantization granularity, temperature (atomic velocity), and timestep size.
- Compare amortized cost (hash lookup + occasional miss + analytic computation) against always-tabulate and always-compute baselines.

---

## 7. Software Engineering and Testing

### 7.1 Unified Table Framework

**Motivation.** Table construction logic is duplicated across `pair_table.cpp`, `bond_table.cpp`, `angle_table.cpp`, `dihedral_table.cpp`, and the kspace Coulomb tables. Each has its own file parsing, splining, and interpolation code.

**Research tasks.**
- Extract a shared `TableInterpolator` class that encapsulates: file I/O, spline fitting, table construction, and interpolation for all 4 styles (LOOKUP, LINEAR, SPLINE, BITMAP).
- Refactor existing table-based styles to use the shared class, reducing code duplication and ensuring consistent error handling.
- Add unit tests for the interpolator itself (accuracy, edge cases, memory management).

### 7.2 Comprehensive Regression Test Suite

**Research tasks.**
- Extend `unittest/force-styles/` with tests that compare table-based forces against analytic reference values at high precision, for every pair style that supports tabulation.
- Add energy-conservation tests: run NVE for 10,000+ steps and verify total energy drift stays below a tolerance as a function of table size.
- Add performance regression tests that flag unexpected slowdowns when table parameters change.

---

## 8. Research Prioritization

| Priority | Direction | Expected Impact | Effort |
|----------|-----------|----------------|--------|
| **High** | 5.1 Crossover analysis | Foundational — guides all other work | Medium |
| **High** | 2.1 Many-body tabulation (EAM) | Large user base, significant speedup potential | High |
| **High** | 3.1 Adaptive table resolution | Better accuracy per byte, fewer user-tuning knobs | Medium |
| **Medium** | 1.1 PMEM/CXL integration | Enables much larger tables; hardware-dependent | High |
| **Medium** | 4.1 GPU texture tables | Direct speedup for GPU users | Medium |
| **Medium** | 3.2 Hermite interpolation | Better accuracy/speed tradeoff | Low |
| **Medium** | 6.2 ML potential tabulation | High-impact niche, growing user base | High |
| **Lower** | 6.3 Spatial hashing | Speculative; unclear win vs. tables | Medium |
| **Lower** | 2.2 Delta-PPPM caching | Narrow applicability (large electrostatic systems) | High |
| **Lower** | 1.2 NUMA-aware replication | Marginal gains on most systems | Low |

---

## Timeline Sketch

**Phase 1 — Foundations (characterization and framework):**
- 5.1 Crossover analysis for all existing table/analytic pairs
- 3.3 Error analysis framework
- 7.1 Unified table interpolator class
- 3.1 Adaptive table resolution prototype

**Phase 2 — Expanded memoization (new physics):**
- 2.1 EAM tabulation
- 3.2 Hermite interpolation style
- 2.3 Tabulated fix operations

**Phase 3 — Hardware exploitation:**
- 1.1 PMEM/CXL integration
- 4.1 GPU texture memory tables
- 4.2 AVX-512 gather vectorization
- 1.2 NUMA-aware replication

**Phase 4 — Advanced algorithms:**
- 6.2 ML potential tabulation
- 6.1 Dynamic table refinement
- 6.3 Spatial hash memoization
- 2.2 Delta-PPPM caching
