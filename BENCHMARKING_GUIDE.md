# Gaussian Blur Benchmarking Program

This document provides an overview of the Gaussian blur benchmarking program for other Claude instances to understand the codebase structure and implementation.

## Overview

This program benchmarks different implementations of a separable Gaussian blur filter (3-tap kernel: [0.25, 0.5, 0.25]) on 2D float arrays. The blur is implemented as two 1D convolutions: horizontal then vertical (separable convolution).

## Image Configuration

- **Image Size**: 512×600 pixels (WIDTH × HEIGHT)
- **Data Type**: `float` arrays
- **Benchmark Runs**: 100 iterations (configurable via `RUNS` in `config.h`)

## Program Structure

### Core Files

1. **config.h** - Configuration constants (WIDTH, HEIGHT, RUNS)
2. **blur.h/blur.c** - High-level blur implementations
3. **kernels.h** - Kernel function declarations
4. **kernels_vert.c** - Vertical convolution kernels (5×16 variants)
5. **kernels_vert_10x8.c** - Vertical convolution kernels (10×8 block size)
6. **kernels_vert_4x80.c** - Vertical convolution kernels (4×80 block size, unrolled)
7. **kernels_horiz.c** - Horizontal convolution kernels
8. **main.c** - Benchmarking harness
9. **utils.h/utils.c** - Utility functions (image comparison, random fill, timing)

## Blur Implementation Strategy

All optimized implementations use a **two-pass separable convolution**:

1. **Horizontal Pass**: Apply 1D convolution horizontally, store in temporary buffer
2. **Vertical Pass**: Apply 1D convolution vertically on the temporary buffer

This reduces the 2D 3×3 convolution from 9 operations per pixel to 6 operations (3 horizontal + 3 vertical).

### Edge Handling

The implementations use **boundary replication** (clamp to edge):
- Pixels beyond image boundaries are treated as if the edge pixel value is replicated
- Special kernel variants handle this:
  - **upper** variants: For the first block (top edge)
  - **regular** variants: For middle blocks
  - **lower** variants: For the last block (bottom edge)
  - **left/right** variants: For horizontal pass edge handling

## Implemented Versions

### 1. `gaussian_blur` (Naive Reference)
- Simple nested loops with boundary checks
- No optimizations
- Used for correctness validation

### 2. `gaussian_blur_reference` (Scalar Reference)
- Clean scalar implementation with boundary checks
- **Auto-vectorization explicitly disabled** using compiler pragmas and volatile pointers
- Serves as true scalar baseline for performance comparison
- Used for correctness validation
- Uses both `#pragma GCC optimize("no-tree-vectorize")` and `volatile` pointers to prevent compiler optimization

### 3. `gaussian_blur` (Naive Reference)
- Simple nested loops with runtime boundary checks
- May be auto-vectorized by compiler
- Simplest implementation for understanding

### 4. `gaussian_blur_5x16`
- Block size: 5 rows × 16 columns
- Uses AVX2 instructions (`__m256` for 8 floats)
- Vertical pass: processes 5 rows, 16 columns (2 accumulators × 8 floats each)
- Horizontal pass: processes 5 rows, 16 columns

### 5. `gaussian_blur_5x16_lowload`
- Same block size as 5×16
- Optimized memory access pattern to reduce redundant loads
- Groups operations to reuse loaded data across multiple accumulators

### 6. `gaussian_blur_10x8`
- Block size: 10 rows × 8 columns
- Vertical pass: processes 10 rows, 8 columns (10 accumulators × 8 floats each)
- Trades wider blocks for taller blocks
- Horizontal pass: still uses 5×16 kernels

### 7. `gaussian_blur_4x80` **(NEW - Unrolled)**
- Block size: 4 rows × 80 columns
- **Fully unrolled vertical pass**: Each of the 4 output rows is computed explicitly (no loops in the kernel)
- Uses 10 accumulators × 8 floats = 80 columns (processes more columns at once)
- For 512-wide images: only needs WIDTH/80 = 6-7 kernel calls per row block
- Vertical blocking: HEIGHT/4 = 150 blocks for 600-pixel height
- Horizontal pass: still uses existing 5×16 kernels

## Kernel Design: 4×80 Unrolled

### Rationale
The 4×80 kernel **unrolls the row loop**, writing out all operations for each of the 4 rows explicitly:

```c
// Pseudo-code structure
kernel_conv3_vert_4x80() {
    // Output Row 0: load 3 input rows, compute, store
    ymm0-ymm9 = 0;
    load row[-1] (10 loads × 8 floats), multiply by 0.25, accumulate
    load row[0]  (10 loads × 8 floats), multiply by 0.5,  accumulate
    load row[1]  (10 loads × 8 floats), multiply by 0.25, accumulate
    store 10 results to output row 0

    // Output Row 1: load 3 input rows, compute, store
    ymm0-ymm9 = 0;
    load row[0] × 0.25
    load row[1] × 0.5
    load row[2] × 0.25
    store to output row 1

    // Output Row 2 and 3: similar explicit unrolling
}
```

### Benefits of Unrolling
- **Eliminates loop overhead** for the 4 iterations
- **Better instruction scheduling**: Compiler/CPU can see all operations at once
- **Reduced branch mispredictions**: No loop conditions to evaluate
- **Allows more aggressive optimization**: All data dependencies are explicit

### Block Configuration
- **80 columns** = 10 × 8 floats matches full AVX2 register usage
- **4 rows** is a sweet spot: not too large (code size), not too small (still amortizes overhead)

## Benchmarking

The benchmark measures:
1. **Single run cycles**: Timing for a single blur operation
2. **Average cycles over 100 runs**: More stable measurement
3. **Throughput**: Operations per cycle (formula: `12.0 * HEIGHT * WIDTH / cycles`)
   - Factor of 12 accounts for: 6 FLOPs per pixel (3 horizontal + 3 vertical) × 2 memory ops

### Timing Method
Uses `rdtsc()` (Read Time-Stamp Counter) for cycle-accurate measurements.

### Output Format
```
Gaussian blur [variant] took [X] cycles
 [throughput in ops/cycle]
Gaussian blur [variant] average took [Y] cycles
 [average throughput]
```

## How to Use This Benchmark

### Compilation
```bash
# Typical compilation (requires AVX2 support)
gcc -O3 -mavx2 -mfma -o blur main.c blur.c kernels_*.c utils.c
```

### Running
```bash
./blur
```

### Adding New Implementations

1. **Create kernel functions** in a new `kernels_*.c` file
   - Follow naming convention: `kernel_conv3_[direction]_[size]`
   - Implement regular, upper, and lower variants

2. **Declare in `kernels.h`**
   ```c
   void kernel_conv3_vert_NxM(const float *restrict src, int src_stride,
                               float *restrict dst, int dst_stride);
   ```

3. **Create blur function in `blur.c`**
   - Follow the two-pass pattern (vertical then horizontal)
   - Use appropriate block sizes for `num_blocks`, `block_h`, `block_w`
   - Call upper variant for first block, regular for middle, lower for last

4. **Declare in `blur.h`**
   ```c
   void gaussian_blur_NxM(float input[HEIGHT][WIDTH],
                           float output[HEIGHT][WIDTH]);
   ```

5. **Add benchmark in `main.c`**
   ```c
   t0 = rdtsc();
   gaussian_blur_NxM(input, output);
   t1 = rdtsc();
   // ... print timing results
   ```

## Performance Considerations

### What Affects Performance

1. **Block Size**:
   - Larger blocks: Better cache reuse, more work per kernel call
   - Smaller blocks: Better register usage, simpler code

2. **Memory Access Patterns**:
   - Stride-1 access (horizontal) is fastest
   - Strided access (vertical) benefits from prefetching

3. **Register Pressure**:
   - 16 AVX2 registers (`ymm0`-`ymm15`) available
   - Use 10 for accumulators, 2 for constants, 1 for temporary loads

4. **Loop Unrolling**:
   - Reduces branch overhead
   - Exposes more instruction-level parallelism
   - Trade-off: larger code size

### Typical Performance Ranking (Best to Worst)
1. Unrolled variants (e.g., 4×80) - Explicit computation, minimal overhead
2. Low-load variants - Optimized memory access
3. Standard SIMD variants - Good vectorization
4. Naive (may be auto-vectorized at -O3)
5. Scalar reference - **Deliberately not vectorized**, slowest but true scalar baseline

## Correctness Validation

All implementations are compared against `gaussian_blur_reference()`:
```c
float diff = compare_images(reference, output);
printf("Average difference from reference: %.6f\n", diff);
```

The difference should be very small (< 0.001 typically) due to floating-point rounding.

## Key Insights for Claude Instances

- **Separable convolution** is the core optimization (2D → two 1D passes)
- **Block processing** amortizes kernel call overhead
- **SIMD** (AVX2 `__m256`) processes 8 floats in parallel
- **Boundary variants** eliminate runtime conditionals for edge cases
- **Unrolling** trades code size for performance by eliminating loop overhead
- The program is designed to benchmark **different space-time trade-offs** in convolution implementation

## Preventing Auto-Vectorization

The `gaussian_blur_reference` function uses multiple techniques to prevent the compiler from auto-vectorizing at -O3:

1. **GCC-specific pragma**: `#pragma GCC optimize("no-tree-vectorize")`
2. **Clang-specific pragmas**:
   - `#pragma clang optimize off`
   - `#pragma clang loop vectorize(disable)`
3. **Volatile pointers**: `volatile float *dst` forces the compiler to treat each access independently
4. **Pragma push/pop**: Ensures only this function is affected, not the entire file

This multi-layered approach ensures the reference implementation remains truly scalar across different compilers and optimization levels, providing an accurate baseline for measuring SIMD speedup.

**Why both pragmas AND volatile?**
- Pragmas alone might be ignored by some compilers or optimization levels
- Volatile alone creates memory barriers that slow down even scalar code unnaturally
- Together, they provide defense-in-depth: pragmas prevent vectorization, volatile ensures scalar semantics are preserved
- The combination works reliably across GCC, Clang, and different optimization levels

## Common Questions

**Q: Why 4×80 instead of 5×16 or 10×8?**
A: 4×80 unrolls the vertical dimension completely (4 rows explicit) and uses the full width (80 = 10×8 floats) to minimize the number of kernel invocations per image.

**Q: Why use different block sizes?**
A: Different block sizes explore the design space between register pressure, cache behavior, and loop overhead. No single size is optimal for all architectures.

**Q: Why separate upper/lower variants instead of runtime checks?**
A: Eliminates branch mispredictions and allows more aggressive compiler optimization. The overhead of function calls is lower than per-pixel conditionals.

**Q: Can WIDTH and HEIGHT be changed?**
A: Yes, in `config.h`. Note that some implementations assume WIDTH is divisible by the block width and HEIGHT by block height. For general sizes, you'd need to handle remainders.

---

**Last Updated**: 2025-11-13
**Implementation**: 4×80 unrolled vertical convolution kernel
