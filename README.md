## Overview

This repository contains:

- Highly optimized Gaussian blur (`gaussian_blur_4x96`) and Sobel (`sobel_4x96`) kernels.
- A benchmark harness (`main.c`, `run_benchmarks.sh`) to measure FLOPS/cycle.
- A real-image pipeline (`main_image.c` + Python helpers) to run the blur+Sobel pipeline on JPEG images and visualize gradients.

All C benchmarks and pipelines are now built with OpenMP enabled.

---

## 1. Manual benchmark (single build / run)

Use the existing `Makefile` to build and run the default benchmark executable (`blur_test` from `main.c` and `kernel1/*`).

Before running, set OpenMP environment variables (choose a thread count suitable for your machine; 8 is a good starting point on an 8-core CPU):

```bash
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_SCHEDULE=static
```

Then build and run from the repo root:

```bash
make clean
make          # uses -fopenmp from the Makefile
make run      # runs ./blur_test
```

This will:

- Compile with `-O3 -march=native -Wall -Wextra -fopenmp -Ikernel1`.
- Run the benchmark defined in `main.c`, which:
  - Fills a random `HEIGHT x WIDTH` image.
  - Benchmarks reference vs. 4x96 Gaussian blur and Sobel.
  - Prints FLOPS/cycle and correctness checks.

You can override dimensions at compile time, for example:

```bash
make clean
make CFLAGS_ADD="-DWIDTH=1024 -DHEIGHT=1024"
```

where `CFLAGS_ADD` is an optional extra flags variable you can add to your local build command (or just call `gcc` manually as in the automated script below).

---

## 2. Automated benchmark (`run_benchmarks.sh`)

The automated script compiles and benchmarks several image sizes and writes average FLOPS/cycle results into CSV files under `benchmark_results/`.

### 2.1 Make the script executable

From the repo root:

```bash
chmod +x run_benchmarks.sh
```

### 2.2 Run the full suite with OpenMP

Set your OpenMP environment variables (you can change the thread count as needed):

```bash
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_SCHEDULE=static
```

Then run the script:

```bash
./run_benchmarks.sh
```

What it does:

- For each configured size (by default `128x128`, `256x256`, `512x512`, `1024x1024`):
  - Compiles `blur_benchmark` with:

    ```bash
    gcc -O3 -march=native -Wall -Wextra -fopenmp -Ikernel1 \
        -DHEIGHT=<H> -DWIDTH=<W> -DRUNS=100 \
        -o blur_benchmark \
        main.c \
        kernel1/blur_main.c kernel1/utils.c \
        kernel1/kernels_vert_4x96_macro.c kernel1/kernels_horiz_4x96_macro.c \
        kernel1/kernels_sobel_4x96_macro.c
    ```

  - Runs `./blur_benchmark --csv` multiple times (`OUTER_RUNS`, default 10).
  - Averages FLOPS/cycle for:
    - `gaussian_ref`, `sobel_ref`, `pipeline_ref`
    - `gaussian_4x96`, `sobel_4x96`, `pipeline_4x96`
  - Writes a CSV like:

    ```csv
    Benchmark,FLOPS_per_Cycle
    gaussian_ref,2.017462
    sobel_ref,2.022600
    pipeline_ref,1.988885
    gaussian_4x96,3.356434
    sobel_4x96,3.691219
    pipeline_4x96,3.872222
    ```

You can then visualize these results using `visualize_benchmarks.py`:

```bash
python3 visualize_benchmarks.py
```

This generates:

- `benchmark_results/gaussian_benchmark.png`
- `benchmark_results/sobel_benchmark.png`
- `benchmark_results/pipeline_benchmark.png`

Each shows FLOPS/cycle vs. image size for reference and 4x96, with a dashed horizontal line at 32 FLOPS/cycle (a theoretical peak).

---

## 3. Real-image pipeline (`main_image.c` + Python helpers)

This pipeline applies:

1. `gaussian_blur_4x96` to a real grayscale image.
2. `sobel_4x96` to the blurred image to compute `grad_x` and `grad_y`.
3. Python tools to prepare inputs and visualize outputs.

All C code is built with OpenMP; Python is used for I/O and visualization.

### 3.1 Prepare real images (Python)

`prepare_real_images.py` converts JPEGs to float32 binaries compatible with the C pipeline.

Install Python dependencies (on your local machine):

```bash
pip install numpy pillow matplotlib
```

Run the preparation script (on your local machine):

```bash
python3 prepare_real_images.py
```

Defaults:

- Reads `.jpg` images from: `archive/images/test`
- For each image and each size in `[128, 256, 512, 1024]`:
  - Converts to grayscale and resizes to `size x size`.
  - Writes:
    - Binary: `archive/images/test_prepared/<size>x<size>/<name>_<size>x<size>.bin`
    - PNG preview: same directory with `.png` extension.

Example outputs for `some_image.jpg` at 512×512:

- `archive/images/test_prepared/512x512/some_image_512x512.bin`
- `archive/images/test_prepared/512x512/some_image_512x512.png`

Copy the `.bin` file you want to use to the server where you run the C code, or access it directly if you share the filesystem.

### 3.2 Run the C pipeline on the server

`main_image.c`:

- Reads a single-channel float32 binary image into `input[HEIGHT][WIDTH]`.
- Runs:
  - `gaussian_blur_4x96(input, blurred);`
  - `sobel_4x96(blurred, grad_x, grad_y);`
- Writes:
  - `<output_prefix>_grad_x.bin`
  - `<output_prefix>_grad_y.bin`
  as float32 binaries (row-major, `HEIGHT x WIDTH`).

Because the kernels are compiled with OpenMP, you should:

```bash
export OMP_NUM_THREADS=8     # or another suitable number
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_SCHEDULE=static
```

Compile for a specific size (e.g., 512×512):

```bash
gcc -O3 -march=native -Wall -Wextra -fopenmp -Ikernel1 \
    -DWIDTH=512 -DHEIGHT=512 \
    main_image.c kernel1/*.c -o run_real_512
```

Run on one of the prepared 512×512 binaries, for example:

```bash
./run_real_512 \
    archive/images/test_prepared/512x512/some_image_512x512.bin \
    out_some_image_512x512
```

This produces:

- `out_some_image_512x512_grad_x.bin`
- `out_some_image_512x512_grad_y.bin`

You can repeat this pattern for other sizes by changing `-DWIDTH`, `-DHEIGHT`, the executable name, and the input/output filenames.

### 3.3 Visualize gradients (Python)

`visualize_gradients.py` loads the original input and the gradient binaries and renders them as images.

On your local machine (where Python and the prepared `.bin` files are available), run:

```bash
python3 visualize_gradients.py \
    --input-bin archive/images/test_prepared/512x512/some_image_512x512.bin \
    --grad-x-bin out_some_image_512x512_grad_x.bin \
    --grad-y-bin out_some_image_512x512_grad_y.bin \
    --width 512 \
    --height 512
```

This:

- Loads all three binaries (`float32`, row-major, 512×512).
- Computes gradient magnitude `sqrt(gx^2 + gy^2)`.
- Normalizes and displays four images:
  - Input
  - Grad X
  - Grad Y
  - Gradient magnitude

You can also save the visualization to a PNG:

```bash
python3 visualize_gradients.py \
    --input-bin archive/images/test_prepared/512x512/some_image_512x512.bin \
    --grad-x-bin out_some_image_512x512_grad_x.bin \
    --grad-y-bin out_some_image_512x512_grad_y.bin \
    --width 512 \
    --height 512 \
    --output vis_some_image_512x512.png
```

Adjust `--width` / `--height` and all filenames to match the size and names you used when compiling and running `run_real_*`.

