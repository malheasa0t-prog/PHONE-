# Phone Damage Detection

YOLO-based computer vision project for detecting and classifying visible phone
damage from images. The repository includes reusable utilities for image
augmentation, duplicate-image cleanup, YOLO training, inference, and saved
training artifacts from earlier experiments.

## Project Goals

- Detect common phone-condition classes from images.
- Support repeatable YOLO model training and inference.
- Keep preprocessing and evaluation scripts reusable outside the original
  notebook environment.

## Classes

The current inference script is configured for these labels:

- Crack
- oil
- Dislodged Screen
- Scratches
- phone good
- Dents

Update the class list in `scripts/run_inference.py` if the trained model uses a
different label order.

## Repository Structure

```text
TEST.ipynb        Original experimentation notebook
scripts/          Reusable Python utilities
model/            Trained model artifacts and YOLO training outputs
result.txt        Training/evaluation notes
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Augment Images

```bash
python scripts/augment_images.py --input-dir path/to/images --copies 3
```

### Remove Duplicate Images

Run a dry check first:

```bash
python scripts/remove_duplicate_images.py --image-dir path/to/images
```

Delete near-duplicates after reviewing the output:

```bash
python scripts/remove_duplicate_images.py --image-dir path/to/images --delete
```

### Train YOLO

```bash
python scripts/train_yolo.py --data data.yaml --model yolo11s.pt --epochs 60 --imgsz 640
```

### Run Inference

```bash
python scripts/run_inference.py --model path/to/best.pt --source path/to/test.jpg
```

## Notes

Large trained weights are currently kept in the repository for reproducibility.
For a cleaner production portfolio, future model binaries should be moved to
GitHub Releases or an external model registry while the repository keeps code,
configs, and evaluation summaries.
