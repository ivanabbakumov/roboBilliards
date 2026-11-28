"""RoboBilliards Vision — camera and ball-detection test.

Builds on the existing macOS OpenCV camera test:
    cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

Keys:
    s — save the current frame and detection overlay
    q — quit

This is an initial calibration/diagnostic tool, not yet the scoring system.
Color thresholds and ball-size limits must be calibrated using the real balls
under the final competition lighting.
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera settings — kept consistent with the user's working camera script.
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
CAMERA_INIT_DELAY_SEC = 0.5

# Set to 0 to use the camera's default resolution. The camera may ignore these
# requests if it does not support the selected mode.
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080

SAVE_FOLDER = "captured_photos"

# Approximate starting thresholds for the lighting in the supplied photo.
# OpenCV HSV uses H: 0..179, S: 0..255, V: 0..255.
# These values are deliberately exposed as trackbars for live adjustment.
DEFAULTS = {
    "white_max_s": 105,
    "white_min_v": 135,
    "orange_h_min": 3,
    "orange_h_max": 28,
    "orange_min_s": 95,
    "orange_min_v": 75,
    "black_max_v": 65,
    "black_max_s": 255,
    "min_area": 250,
    "max_area": 3600,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def nothing(_value: int) -> None:
    """Trackbar callback placeholder."""


def make_mask(hsv: np.ndarray, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> np.ndarray:
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def find_ball_candidates(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> list[tuple[tuple[int, int], int, float]]:
    """Return plausible circular objects as (center, radius, circularity).

    The area and radius filters help reject the much larger 100 mm pockets.
    They do not yet distinguish a real black ball from every dark object in
    the scene; this is why the output is for visual testing only.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[tuple[int, int], int, float]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)

        # At the supplied camera distance, a 40 mm ball should be roughly
        # 30–45 pixels wide. Keep a wider initial range for perspective.
        if radius < 8 or radius > 28:
            continue
        if circularity < 0.48:
            continue

        results.append(((int(round(x)), int(round(y))), int(round(radius)), circularity))

    return results


def create_trackbars() -> None:
    cv2.namedWindow("Detection settings", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection settings", 440, 520)

    trackbars = [
        ("White max saturation", "white_max_s", 255),
        ("White min brightness", "white_min_v", 255),
        ("Orange hue min", "orange_h_min", 179),
        ("Orange hue max", "orange_h_max", 179),
        ("Orange min saturation", "orange_min_s", 255),
        ("Orange min brightness", "orange_min_v", 255),
        ("Black max brightness", "black_max_v", 255),
        ("Black max saturation", "black_max_s", 255),
        ("Minimum contour area", "min_area", 5000),
        ("Maximum contour area", "max_area", 10000),
    ]

    for label, key, maximum in trackbars:
        cv2.createTrackbar(label, "Detection settings", DEFAULTS[key], maximum, nothing)


def get_settings() -> dict[str, int]:
    values: dict[str, int] = {}
    for key, label in [
        ("white_max_s", "White max saturation"),
        ("white_min_v", "White min brightness"),
        ("orange_h_min", "Orange hue min"),
        ("orange_h_max", "Orange hue max"),
        ("orange_min_s", "Orange min saturation"),
        ("orange_min_v", "Orange min brightness"),
        ("black_max_v", "Black max brightness"),
        ("black_max_s", "Black max saturation"),
        ("min_area", "Minimum contour area"),
        ("max_area", "Maximum contour area"),
    ]:
        values[key] = cv2.getTrackbarPos(label, "Detection settings")
    return values


def main() -> None:
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Use the same backend and capture flow as the already-working script.
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    time.sleep(CAMERA_INIT_DELAY_SEC)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        print("Попробуйте CAMERA_INDEX = 1, если камера имеет другой индекс.")
        return

    if REQUESTED_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    if REQUESTED_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Камера подключена.")
    print(f"Разрешение: {actual_width} x {actual_height}")
    print(f"FPS, сообщённый камерой: {actual_fps:.1f}")
    print("Нажмите 's', чтобы сохранить кадры, или 'q', чтобы выйти.")
    print("Настройки HSV и фильтры можно менять в окне Detection settings.")

    create_trackbars()
    cv2.namedWindow("RoboBilliards Vision", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masks", cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_timer = time.time()
    measured_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Предупреждение: не удалось получить кадр с камеры.")
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = now

            settings = get_settings()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # White balls: low saturation and relatively high brightness.
            white_mask = make_mask(
                hsv,
                (0, 0, settings["white_min_v"]),
                (179, settings["white_max_s"], 255),
            )

            # Orange balls: hue interval is adjustable for the real lighting.
            h_min = min(settings["orange_h_min"], settings["orange_h_max"])
            h_max = max(settings["orange_h_min"], settings["orange_h_max"])
            orange_mask = make_mask(
                hsv,
                (h_min, settings["orange_min_s"], settings["orange_min_v"]),
                (h_max, 255, 255),
            )

            # Black ball: dark pixels. The 100 mm pockets are expected to be
            # rejected mainly by the ball-size filter at this camera distance.
            black_mask = make_mask(
                hsv,
                (0, 0, 0),
                (179, settings["black_max_s"], settings["black_max_v"]),
            )

            min_area = settings["min_area"]
            max_area = max(settings["max_area"], min_area + 1)

            detections = [
                ("WHITE", (255, 255, 255), white_mask),
                ("ORANGE", (0, 140, 255), orange_mask),
                ("BLACK", (30, 30, 30), black_mask),
            ]

            output = frame.copy()
            detection_counts: dict[str, int] = {}

            for label, color, mask in detections:
                candidates = find_ball_candidates(mask, min_area, max_area)
                detection_counts[label] = len(candidates)

                for (x, y), radius, circularity in candidates:
                    cv2.circle(output, (x, y), radius, color, 2)
                    cv2.circle(output, (x, y), 2, color, -1)
                    cv2.putText(
                        output,
                        f"{label} {circularity:.2f}",
                        (x - radius, y - radius - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            status = (
                f"FPS: {measured_fps:.1f} | "
                f"White: {detection_counts['WHITE']} | "
                f"Orange: {detection_counts['ORANGE']} | "
                f"Black: {detection_counts['BLACK']} | s: save | q: quit"
            )
            cv2.putText(
                output,
                status,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RoboBilliards Vision", output)

            # Side-by-side masks: white, orange, black.
            mask_panel = np.hstack((white_mask, orange_mask, black_mask))
            mask_panel = cv2.resize(mask_panel, (960, 180))
            cv2.putText(mask_panel, "WHITE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.putText(mask_panel, "ORANGE", (335, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.putText(mask_panel, "BLACK", (655, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            cv2.imshow("Masks", mask_panel)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join(SAVE_FOLDER, f"photo_{timestamp}.jpg")
                overlay_path = os.path.join(SAVE_FOLDER, f"detection_{timestamp}.jpg")
                masks_path = os.path.join(SAVE_FOLDER, f"masks_{timestamp}.jpg")

                cv2.imwrite(photo_path, frame)
                cv2.imwrite(overlay_path, output)
                cv2.imwrite(masks_path, mask_panel)

                print(f"Исходное фото сохранено: {photo_path}")
                print(f"Распознавание сохранено: {overlay_path}")
                print(f"Маски цветов сохранены: {masks_path}")

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Камера отключена.")


if __name__ == "__main__":
    main()
