import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real images as float32 binaries for the C pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("archive/images/test"),
        help="Directory containing input .jpg images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("archive/images/test_prepared"),
        help="Directory to store prepared .bin files.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="List of target square sizes (e.g. 128 256 512 1024).",
    )
    return parser.parse_args()


def prepare_images(input_dir: Path, output_dir: Path, sizes: list[int]) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jpgs = sorted(input_dir.glob("*.jpg"))
    if not jpgs:
        print(f"No .jpg images found in {input_dir}")
        return

    print(f"Found {len(jpgs)} images in {input_dir}")

    for img_path in jpgs:
        base = img_path.stem
        print(f"Processing {img_path.name}...")

        img_original = Image.open(img_path).convert("L")

        for size in sizes:
            resized = img_original.resize((size, size), Image.BILINEAR)
            arr = np.array(resized, dtype=np.float32)

            size_dir = output_dir / f"{size}x{size}"
            size_dir.mkdir(parents=True, exist_ok=True)

            bin_path = size_dir / f"{base}_{size}x{size}.bin"
            png_path = size_dir / f"{base}_{size}x{size}.png"

            arr.tofile(bin_path)
            resized.save(png_path)

            print(f"  -> {bin_path} (float32, row-major, {size}x{size})")
            print(f"  -> {png_path} (preview)")


def main() -> None:
    args = parse_args()
    prepare_images(args.input_dir, args.output_dir, args.sizes)


if __name__ == "__main__":
    main()

