# Kernel 3: Direction Classification - Analysis & Implementation

## Overview

Kernel 3 performs direction classification for edge detection in the Canny edge detection pipeline. Given gradient components (g_x, g_y) from Sobel convolutions, it classifies the gradient direction into one of 4 bins (0°, 45°, 90°, 135°).

## Algorithm

For each pixel with gradients (g_x, g_y):
1. Classify direction based on angle of gradient vector
2. Assign to one of 4 direction bins:
   - Direction 0: |g_x| > |g_y| (horizontal, ~0°/180°)
   - Direction 1: |g_x| > |g_y| && sign(g_x)*sign(g_y) > 0 (diagonal +45°)
   - Direction 2: |g_x| ≤ |g_y| (vertical, ~90°)
   - Direction 3: |g_x| ≤ |g_y| && sign(g_x)*sign(g_y) < 0 (diagonal -45°)

## Kernel Specifications

- **Input**: Two float arrays (g_x, g_y), row-major layout, 32-byte aligned
- **Output**: Single float array (direction), values in {0.0, 1.0, 2.0, 3.0}
- **Kernel Size**: 16×1 (process 16 pixels per iteration for data reuse)
- **SIMD Width**: 8 floats per AVX2 vector (2 vectors = 16 pixels)
- **Data Type**: float32

## Architecture Bottleneck Analysis

### Broadwell (Intel Xeon E5-2680 v4)

**Original Approach - VBLENDVPS bottleneck**:
- VBLENDVPS: 2.0 CPI reciprocal throughput
- Issue: Small kernel sizes (1 pixel) lead to poor data reuse

**Optimized Approach - MOVEMASK + VBLEND**:
- MOVEMASK (extract sign bits): 1 cycle latency, high throughput
- VBLEND (regular): 1.0 CPI reciprocal throughput (better than VBLENDVPS)
- Larger kernel: 16 pixels per iteration improves constant reuse

**Theoretical Peak Performance**:
- At 2.4 GHz (base): ~9.6 GFLOP/s
- At 3.3 GHz (turbo): ~13.2 GFLOP/s

## Implementation Variants

### Kernel 3.1: Naive Scalar Implementation
**Strategy**: Process pixels one at a time with scalar operations
- Simple conditional logic
- No SIMD optimization
- Baseline for comparison

**Key Instructions**:
- Load g_x, g_y
- Compare absolute values
- Compare signs
- Store direction
- **Operations per pixel**: ~15 scalar operations

### Kernel 3.2: Movemask + Blend (16 pixels)
**Strategy**: Process 16 pixels (2 AVX2 vectors) using movemask for sign extraction
- MOVEMASK to extract sign bits from floating-point representations
- Integer VBLEND (1.0 CPI) instead of VBLENDVPS (2.0 CPI)
- 16-pixel kernel for better constant reuse

**Key Optimizations**:
- MOVEMASK: Fast extraction of sign information
- Regular VBLEND: 1.0 CPI (vs 2.0 CPI for VBLENDVPS)
- Larger kernel: amortizes load/store and constant setup costs
- Process 2 vectors in parallel to hide latency

**Operations per 16 pixels**: ~18 instructions (avg ~1.1 per pixel)

### Kernel 3.3: Integer Bit Operations with Parallelism
**Strategy**: Use integer operations for all bitwise work, process 16 pixels
- _mm256_and_si256() for magnitude masking
- _mm256_xor_si256() for sign detection (no multiplication)
- Parallel computation of direction bits
- Better throughput on integer execution units

**Key Optimizations**:
- All bitwise ops on integer units (better utilization)
- XOR for sign checks (avoids 3-cycle multiplication)
- 16-pixel processing with 2 accumulators in flight
- Reduced dependency chains

**Operations per 16 pixels**: ~16 instructions (avg ~1.0 per pixel)
**Expected Speedup**: ~1.5× vs Kernel 3.2

### Kernel 3.4: Highly Optimized with Instruction Pipelining
**Strategy**: Maximum parallelism with 3-4 vector streams
- Process 3-4 independent vector pairs (24-32 pixels) 
- Minimize stall cycles by keeping pipeline full
- Use MOVEMASK results immediately without intermediate stores
- Overlap sign checking with magnitude comparisons

**Key Optimizations**:
- Multiple independent instruction streams
- No dependency between vector lanes
- Instruction-level parallelism across 32 pixels
- Pipelined compute with prefetch

**Operations per 16 pixels**: ~14 instructions (avg ~0.875 per pixel)
**Expected Speedup**: ~2.0× vs Kernel 3.2, ~1.3× vs Kernel 3.3

## Performance Metrics

All kernels measure:
- **FLOPS/cycle**: Total floating-point operations divided by cycle count
- **Throughput**: Pixels processed per second
- **Utilization**: Ratio to theoretical peak

## Memory Access Pattern

**Layout**: Row-major contiguous arrays
- g_x[i], g_y[i] are adjacent in memory
- Sequential access enables cache prefetching
- L1 cache (32 KB) easily holds working set for 1024×1024 images

**Cache Efficiency**:
- 3 × 4 MB = 12 MB data for 1024×1024 image
- L3 cache: 35 MB shared (sufficient for full image)
- No data reorganization needed

## Testing Strategy

1. Generate synthetic gradient data
2. Warm cache with initial runs
3. Measure total cycles and operations
4. Compute FLOPS/cycle
5. Verify numerical correctness
6. Compare relative performance

## Expected Results

| Kernel | FLOPS/cycle | Speedup | Notes |
|--------|-------------|---------|-------|
| 3.1 Naive | 0.5-0.8 | 1.0× | Scalar baseline |
| 3.2 Basic AVX2 | 3.0-4.0 | 4-6× | Vectorization gain |
| 3.3 Integer Opt | 4.5-5.5 | 5.5-7× | Sign XOR optimization |
| 3.4 Reduction Tree | 5.5-6.5 | 7-8× | Parallelism optimization |

**Target**: 5.5-6.5 FLOPS/cycle for optimized kernels (87-100% efficiency vs theoretical 7.5 FLOPS/cycle peak at 2.4 GHz base frequency)
