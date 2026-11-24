"""
Synthetic data generator.

Creates fake drone images by pasting programmatically generated drone
silhouettes onto real backgrounds from COCO128. Annotations are produced
automatically — no human labeling needed.
"""

from pathlib import Path

import cv2
import numpy as np

from dataset import DATASET_DIR

OUTPUT_DIR = Path("synthetic_output")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
NUM_IMAGES = 30
DRONE_CLASS_ID = 0  # our custom dataset has one class: drone


# ---------------------------------------------------------------------------
# Drone silhouette generator
# ---------------------------------------------------------------------------

def generate_drone(size: int, angle: float = 0.0) -> np.ndarray:
    """
    Draw a quadcopter silhouette and return an RGBA image (alpha = transparency).
    size controls wingspan roughly. angle rotates the whole drone.
    """
    canvas = size * 3
    img = np.zeros((canvas, canvas, 4), dtype=np.uint8)
    cx, cy = canvas // 2, canvas // 2
    color = (50, 50, 50, 255)  # dark gray, opaque

    # Central body
    body_r = size // 5
    cv2.ellipse(img, (cx, cy), (body_r, body_r // 2), angle, 0, 360, color, -1)

    # 4 arms + rotors at 45° / 135° / 225° / 315°
    arm_len = size // 2
    rotor_rx = size // 5
    rotor_ry = size // 10
    arm_thickness = max(2, size // 12)

    for base_angle in [45, 135, 225, 315]:
        rad = np.radians(base_angle + angle)
        tip_x = int(cx + np.cos(rad) * arm_len)
        tip_y = int(cy + np.sin(rad) * arm_len)
        cv2.line(img, (cx, cy), (tip_x, tip_y), color, arm_thickness)
        cv2.ellipse(img, (tip_x, tip_y), (rotor_rx, rotor_ry),
                    base_angle + angle, 0, 360, color, -1)

    # Crop away empty rows/cols to get a tight bounding box
    alpha = img[:, :, 3]
    rows_with_content = np.any(alpha > 0, axis=1)
    cols_with_content = np.any(alpha > 0, axis=0)
    r0, r1 = np.where(rows_with_content)[0][[0, -1]]
    c0, c1 = np.where(cols_with_content)[0][[0, -1]]

    return img[r0:r1 + 1, c0:c1 + 1]


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def paste_onto(background: np.ndarray, drone_rgba: np.ndarray,
               x: int, y: int) -> np.ndarray:
    """
    Alpha-blend drone_rgba onto background at top-left corner (x, y).
    Returns a new BGR image (background is not modified).
    """
    result = background.copy()
    dh, dw = drone_rgba.shape[:2]
    bh, bw = background.shape[:2]

    # Clip drone to stay within image bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + dw, bw), min(y + dh, bh)
    dx1, dy1 = x1 - x, y1 - y  # offsets into the drone image

    drone_crop = drone_rgba[dy1:dy1 + (y2 - y1), dx1:dx1 + (x2 - x1)]
    if drone_crop.size == 0:
        return result

    alpha = drone_crop[:, :, 3:4] / 255.0  # normalize to 0-1
    drone_bgr = drone_crop[:, :, :3]
    region = result[y1:y2, x1:x2]

    # Standard alpha blending: out = alpha * foreground + (1 - alpha) * background
    result[y1:y2, x1:x2] = (alpha * drone_bgr + (1 - alpha) * region).astype(np.uint8)
    return result


def to_yolo_annotation(x: int, y: int, dw: int, dh: int,
                        bw: int, bh: int) -> str:
    """Convert pixel position to normalized YOLO annotation line."""
    x_center = (x + dw / 2) / bw
    y_center = (y + dh / 2) / bh
    width = dw / bw
    height = dh / bh
    return f"{DRONE_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(n: int) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    backgrounds = sorted((DATASET_DIR / "images" / "train2017").glob("*.jpg"))
    rng = np.random.default_rng(seed=42)

    for i in range(n):
        # Pick a random background
        bg_path = backgrounds[rng.integers(len(backgrounds))]
        bg = cv2.imread(str(bg_path))
        bh, bw = bg.shape[:2]

        # Random drone: size between 3% and 12% of image width, random rotation
        size = int(rng.uniform(0.03, 0.12) * bw)
        angle = float(rng.uniform(0, 360))
        drone = generate_drone(size, angle)
        dh, dw = drone.shape[:2]

        # Random position (allow partial overlap with edges)
        x = int(rng.uniform(-dw * 0.2, bw - dw * 0.8))
        y = int(rng.uniform(-dh * 0.2, bh - dh * 0.8))

        composite = paste_onto(bg, drone, x, y)

        # Clamp box to image bounds for the annotation
        x_clamped = max(x, 0)
        y_clamped = max(y, 0)
        w_clamped = min(x + dw, bw) - x_clamped
        h_clamped = min(y + dh, bh) - y_clamped

        name = f"synthetic_{i:04d}"
        cv2.imwrite(str(IMAGES_DIR / f"{name}.jpg"), composite)
        (LABELS_DIR / f"{name}.txt").write_text(
            to_yolo_annotation(x_clamped, y_clamped, w_clamped, h_clamped, bw, bh)
        )

    print(f"Generated {n} synthetic images → {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_dataset(NUM_IMAGES)
