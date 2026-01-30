# Vector Dot Product - CUDA Kernel Implementations

Parallel GPU implementations of vector dot product with progressive optimization strategies.

**Problem**: Compute dot product of two float vectors: 
$$
result =  \sum_{i=0}^{n-1} a_{i} \cdot b_{i} 
$$

## Implementations

| Kernel | Strategy | Synchronization | Expected Performance | Status |
| -------- | -------- | -------- | -------- | -------- |
| Stage1Navive | Single-stage atomic reduction | Global atomic (high contention) | Baseline (worst) | Implementated |
| Stage2Naive | Two-stage with per-block atomics | Atomic per block | Better | Implemented |
| SharedMem | Two-stage with shared memory reduction | Shared memory + tree reduction | Best | Implemented |
| WarpShuffle | Warp-level primitives | Warp shuffle instructions | Future optimization | Implemented |

## Kernel Descriptions

### Stage1Navive

**Strategy**:

- grid-stride loop for load balancing across any data size
- each thread compute partial sum
- all threads atomically add their result to a single global `result` variable

**Code**:

```cuda

for (int i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
}
atomicAdd(result, sum); // high contention

```

**Performance**:

- very slow, becase all the threads update the same location, the GPU must serialize these operations.

## Optimization Progression

### Problem

The dot-product is a **reduction problem**: many inputs -> single output. Naive approache faces:

1. **Contention**: All threads competing for the same output location
2. **Serialization**: Atomic operations force sequential execution

### Solution strategy: Hierarchical Reduction

**Key insight**: Match GPU memory hierarchy levels (global -> shared -> registers)

![alt text](kernel.png)

## Code Pattern

### Grid-stride loop

All kernels use the grid-stride loop for robustness:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = idx; i < N; i += stride) {
    // process element i ...
}

```

**Benefits**:

- handles any n size, regardless of grid dimensions
- load balancing: distributes work evenly when n is not divisible by grid size
- allows limiting grid size while processing billions of elements

### Two-stage reduction

Stage 1:

- Input: vectors a[n], b[n]
- Output: partialSum[numBlocks] // one value per block

Stage 2:

- Input: partialSum[numBlocks]
- Output: result // single scalar

**Why two stages?**:

- avoid requiring all blocks to fit in GPU simultaneously
- allow processing datasets larger than GPU memory
- second stage has minimal cost (only 1024 elements max)

## Building and Running

### Build

```bash
    # From project root
    mkdir -p build && cd build
    cmake ..
    cmake --build . --target dot-product
```

### Run benchmark

```bash
    ./bin/dot-product
```

**Output**: CSV file dot_product_benchmark.csv with timing statistics

### Run tests

```bash
    ctest -R dot-product
```

### Visualize Results

```bash
python scripts/analyze_benchmark.py dot_product_benchmark.csv
```

**Generates plots**:

- bandwidth_vs_size.png: Bandwidth scaling across data sizes
- kernel_comparison.png: Execution time comparison with error bars
- speedup_comparison.png: Relative speedup vs baseline
- variance_analysis.png: Timing consistency (cofficient of variation)

## Performance

**Test Configuration**:

- GPU: NVIDIA RTX 3090 (Ampere, compute capability 8.6)
- Block size: 256 threads
- Max grid blocks: 1024
- Warmup iterations: 10
- Trial runs: 5
- Data type: `float` (4 bytes)

**Bandwidth**:
![alt text](bandwidth.png)

**Precision Considerations**:

- CPU reference uses `double` accumulation for higher precision
- GPU kernels use `float` with epsilon tolerance: 1e-5
- **Large datasets**: Floating-point accumulation order can cause slight differences
  - Atomic operations have non-deterministic ordering
  - Relative error remains acceptable (<1e-5) for most use case
