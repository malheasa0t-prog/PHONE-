"""Find or remove near-duplicate images using perceptual hashing."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import imagehash
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def iter_images(image_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect near-duplicate images.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete duplicate files. Omit this flag for a dry run.",
    )
    args = parser.parse_args()

    hashes: dict[imagehash.ImageHash, Path] = {}
    duplicates: list[tuple[Path, Path]] = []

    image_paths = iter_images(args.image_dir)
    for image_path in tqdm(image_paths, desc="Scanning images"):
        try:
            with Image.open(image_path) as image:
                image_hash = imagehash.phash(image)
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")
            continue

        match = next(
            (
                existing_path
                for existing_hash, existing_path in hashes.items()
                if abs(image_hash - existing_hash) < args.threshold
            ),
            None,
        )

        if match is not None:
            duplicates.append((image_path, match))
            if args.delete:
                image_path.unlink()
        else:
            hashes[image_hash] = image_path

    print(f"Scanned images: {len(image_paths)}")
    print(f"Near-duplicates found: {len(duplicates)}")

    for duplicate, original in duplicates:
        action = "deleted" if args.delete else "would delete"
        print(f"{action}: {duplicate} (similar to {original})")


if __name__ == "__main__":
    main()
