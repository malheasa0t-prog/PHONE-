"""Create augmented copies of image files for training data expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def adjust_brightness_contrast(image: np.ndarray) -> np.ndarray:
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.randint(-30, 30)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def rotate_image(image: np.ndarray) -> np.ndarray:
    angle = np.random.uniform(-15, 15)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
    )


def augment_image(image: np.ndarray) -> np.ndarray:
    augmented = image.copy()

    if np.random.rand() < 0.5:
        augmented = cv2.flip(augmented, 1)
    if np.random.rand() < 0.3:
        augmented = cv2.flip(augmented, 0)

    augmented = adjust_brightness_contrast(augmented)
    return rotate_image(augmented)


def iter_images(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment image datasets.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--copies", type=int, default=3)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_dir)
    for image_path in tqdm(image_paths, desc="Augmenting images"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        for index in range(args.copies):
            augmented = augment_image(image)
            output_name = f"{image_path.stem}_aug_{index}.jpg"
            cv2.imwrite(str(output_dir / output_name), augmented)

    print(f"Saved augmented images to: {output_dir}")


if __name__ == "__main__":
    main()
