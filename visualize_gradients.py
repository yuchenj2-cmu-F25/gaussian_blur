import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize input image and Sobel gradients from C pipeline.",
    )
    parser.add_argument(
        "--input-bin",
        type=Path,
        required=True,
        help="Path to input float32 binary (WIDTH*HEIGHT values).",
    )
    parser.add_argument(
        "--grad-x-bin",
        type=Path,
        required=True,
        help="Path to grad_x float32 binary.",
    )
    parser.add_argument(
        "--grad-y-bin",
        type=Path,
        required=True,
        help="Path to grad_y float32 binary.",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Image width used when running the C pipeline.",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Image height used when running the C pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the visualization as a PNG.",
    )
    return parser.parse_args()


def load_image(path: Path, width: int, height: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    expected = width * height
    if arr.size != expected:
        raise ValueError(
            f"{path} has {arr.size} floats, expected {expected} "
            f"for shape {height}x{width}"
        )
    return arr.reshape((height, width))


def normalize_to_uint8(arr: np.ndarray, use_abs: bool = False) -> np.ndarray:
    if use_abs:
        arr = np.abs(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max <= arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    arr_norm = (arr - arr_min) / (arr_max - arr_min)
    return (arr_norm * 255.0).astype(np.uint8)


def main() -> None:
    args = parse_args()

    w, h = args.width, args.height

    img = load_image(args.input_bin, w, h)
    gx = load_image(args.grad_x_bin, w, h)
    gy = load_image(args.grad_y_bin, w, h)

    mag = np.sqrt(gx * gx + gy * gy)

    img_u8 = normalize_to_uint8(img, use_abs=False)
    gx_u8 = normalize_to_uint8(gx, use_abs=True)
    gy_u8 = normalize_to_uint8(gy, use_abs=True)
    mag_u8 = normalize_to_uint8(mag, use_abs=False)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(img_u8, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(gx_u8, cmap="gray")
    axes[1].set_title("Grad X")
    axes[2].imshow(gy_u8, cmap="gray")
    axes[2].set_title("Grad Y")
    axes[3].imshow(mag_u8, cmap="gray")
    axes[3].set_title("Gradient magnitude")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Saved visualization to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

