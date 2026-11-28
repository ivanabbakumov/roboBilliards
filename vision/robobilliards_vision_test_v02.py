"""RoboBilliards Vision v0.2 — white and orange ball detection and field calibration.

Based on the user's working macOS camera capture code:
    cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

This version is a diagnostic prototype only. It detects white and orange balls; it does
not yet award points or decide whether a ball has entered a pocket.

Controls (focus the main video window first):
    C — calibrate the playing-area polygon (click 4 corners clockwise:
        top-left, top-right, bottom-right, bottom-left)
    S — save original frame, detection overlay and both color masks
    V — start/stop recording raw camera frames
    R — reset/remove saved field calibration
    Q — quit

The field polygon is saved in vision_config.json and reused next time.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Camera settings — same backend and capture approach as the user's script.
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_BACKEND = cv2.CAP_AVFOUNDATION
CAMERA_INIT_DELAY_SEC = 0.5
REQUESTED_WIDTH = 1920
REQUESTED_HEIGHT = 1080

SAVE_FOLDER = "captured_photos"
CONFIG_FILE = "vision_config.json"
WINDOW_NAME = "RoboBilliards Vision v0.2"
MASK_WINDOW_NAME = "White ball mask"
SETTINGS_WINDOW_NAME = "Detection settings"

# The current calibration test uses 9 white balls. Change this to 10 when
# testing with the full standard set; it is only a visual reference.
EXPECTED_WHITE_BALLS = 9

# Initial values calibrated on the green competition field at 1920 x 1080.
DEFAULTS = {
    "white_max_s": 45,
    "white_min_v": 165,
    "hough_param2": 25,
    "min_radius": 15,
    "max_radius": 24,
    "min_mask_percent": 60,
}

# Display is scaled down so a 1920x1080 frame fits on a MacBook screen.
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720


# ---------------------------------------------------------------------------
# Calibration state
# ---------------------------------------------------------------------------
field_points: list[tuple[int, int]] = []
calibration_clicks: list[tuple[int, int]] = []
calibration_mode = False
display_scale = 1.0
current_frame_width = 0
current_frame_height = 0


def noop(_value: int) -> None:
    """OpenCV trackbar callback placeholder."""


def load_config() -> None:
    """Load the previously selected playable-area polygon, if it exists."""
    global field_points
    if not os.path.exists(CONFIG_FILE):
        field_points = []
        return

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        points = data.get("field_polygon", [])
        if len(points) == 4:
            field_points = [(int(point[0]), int(point[1])) for point in points]
            print(f"Калибровка игрового поля загружена из {CONFIG_FILE}.")
        else:
            field_points = []
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Не удалось прочитать {CONFIG_FILE}: {error}")
        field_points = []


def save_config(frame_width: int, frame_height: int) -> None:
    """Save the selected field polygon and the frame size used for it."""
    data = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "field_polygon": [[x, y] for x, y in field_points],
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(f"Калибровка сохранена в {CONFIG_FILE}.")


def make_field_mask(frame_shape: tuple[int, ...]) -> np.ndarray:
    """Create a binary mask containing only the calibrated playing area."""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(field_points) == 4:
        polygon = np.array(field_points, dtype=np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
    else:
        mask[:] = 255
    return mask


def mouse_callback(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
    """Collect four field corners in displayed-image coordinates."""
    global calibration_clicks, calibration_mode, field_points

    if event != cv2.EVENT_LBUTTONDOWN or not calibration_mode:
        return

    # Convert coordinates from the resized preview back to camera-frame pixels.
    frame_x = int(round(x / display_scale))
    frame_y = int(round(y / display_scale))
    calibration_clicks.append((frame_x, frame_y))
    print(f"Точка {len(calibration_clicks)}/4: ({frame_x}, {frame_y})")

    if len(calibration_clicks) == 4:
        field_points = calibration_clicks.copy()
        calibration_mode = False
        if current_frame_width > 0 and current_frame_height > 0:
            save_config(current_frame_width, current_frame_height)
        print("Контур игрового поля задан и сохранён.")


def create_trackbars() -> None:
    cv2.namedWindow(SETTINGS_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SETTINGS_WINDOW_NAME, 440, 360)

    trackbars = [
        ("White max saturation", "white_max_s", 255),
        ("White min brightness", "white_min_v", 255),
        ("Hough sensitivity (lower = more)", "hough_param2", 60),
        ("Minimum ball radius", "min_radius", 35),
        ("Maximum ball radius", "max_radius", 60),
        ("Minimum mask coverage %", "min_mask_percent", 100),
    ]
    for label, key, maximum in trackbars:
        cv2.createTrackbar(label, SETTINGS_WINDOW_NAME, DEFAULTS[key], maximum, noop)


def get_settings() -> dict[str, int]:
    labels = {
        "white_max_s": "White max saturation",
        "white_min_v": "White min brightness",
        "hough_param2": "Hough sensitivity (lower = more)",
        "min_radius": "Minimum ball radius",
        "max_radius": "Maximum ball radius",
        "min_mask_percent": "Minimum mask coverage %",
    }
    return {key: cv2.getTrackbarPos(label, SETTINGS_WINDOW_NAME) for key, label in labels.items()}


def circle_mask_coverage(mask: np.ndarray, x: int, y: int, radius: int) -> float:
    """Fraction of white-mask pixels inside the inner part of a candidate."""
    height, width = mask.shape
    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return 0.0

    local_mask = mask[y1:y2, x1:x2]
    circle = np.zeros(local_mask.shape, dtype=np.uint8)
    cv2.circle(circle, (x - x1, y - y1), max(2, int(radius * 0.68)), 255, -1)
    circle_area = cv2.countNonZero(circle)
    if circle_area == 0:
        return 0.0
    white_area = cv2.countNonZero(cv2.bitwise_and(local_mask, circle))
    return white_area / float(circle_area)


def add_candidate(
    candidates: list[dict[str, float | int | str]],
    x: int,
    y: int,
    radius: int,
    source: str,
    score: float,
    min_mask_percent: int,
    white_mask: np.ndarray,
    field_mask: np.ndarray,
) -> None:
    """Validate and append a candidate, rejecting duplicate/outside objects."""
    height, width = white_mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return
    if field_mask[y, x] == 0:
        return

    coverage = circle_mask_coverage(white_mask, x, y, radius)
    if coverage < min_mask_percent / 100.0:
        return

    # Merge detections from contour and Hough stages when they describe the
    # same ball. The distance threshold is intentionally smaller than a ball
    # diameter so that two nearby balls are not merged into one.
    for existing in candidates:
        dx = x - int(existing["x"])
        dy = y - int(existing["y"])
        distance = float(np.hypot(dx, dy))
        duplicate_limit = max(7.0, min(radius, int(existing["radius"])) * 0.55)
        if distance < duplicate_limit:
            if score > float(existing["score"]):
                existing.update({"x": x, "y": y, "radius": radius, "source": source, "score": score, "coverage": coverage})
            return

    candidates.append({
        "x": x,
        "y": y,
        "radius": radius,
        "source": source,
        "score": score,
        "coverage": coverage,
    })


def detect_white_balls(
    frame: np.ndarray,
    field_mask: np.ndarray,
    settings: dict[str, int],
) -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    balls, white_mask, _, _ = detect_balls(frame, field_mask, settings)
    return balls, white_mask


def detect_balls(frame, field_mask, settings):
    """Share circle proposals across the two color masks."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, settings["white_min_v"]), (179, settings["white_max_s"], 255))
    orange = cv2.inRange(hsv, (10, 50, 120), (30, 255, 255))
    # Reserve orange pixels when the white saturation slider is raised.
    white = cv2.bitwise_and(white, cv2.bitwise_not(orange))
    min_radius = max(5, settings["min_radius"])
    max_radius = max(min_radius + 1, settings["max_radius"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    hough = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(min_radius * 1.65)),
        param1=100,
        param2=max(10, settings["hough_param2"]),
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    def detect(mask):
        return detect_mask(cv2.bitwise_and(mask, field_mask), field_mask, settings,
                           min_radius, max_radius, hough)

    white_balls, white_mask = detect(white)
    orange_balls, orange_mask = detect(orange)
    return white_balls, white_mask, orange_balls, orange_mask


def detect_mask(mask, field_mask, settings, min_radius, max_radius, hough):
    # Remove isolated speckles while preserving a ping-pong ball's round shape.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    min_mask_percent = settings["min_mask_percent"]
    candidates: list[dict[str, float | int | str]] = []

    # Method 1: connected components/contours. Best when the balls are separated.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < np.pi * min_radius * min_radius * 0.25:
            continue
        if area > np.pi * max_radius * max_radius * 2.0:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (center_x, center_y), radius_value = cv2.minEnclosingCircle(contour)
        radius = int(round(radius_value))
        if radius < min_radius or radius > max_radius:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        aspect = box_w / float(max(1, box_h))
        if not 0.60 <= aspect <= 1.55:
            continue
        if circularity < 0.42:
            continue

        score = min(1.0, 0.55 * circularity + 0.45 * min(1.0, area / (np.pi * radius * radius)))
        add_candidate(
            candidates, int(round(center_x)), int(round(center_y)), radius,
            "contour", score, min_mask_percent, mask, field_mask,
        )

    # Method 2: Hough circles can split adjacent bright regions that became
    # connected in the threshold mask. Mask coverage rejects most wood texture.
    if hough is not None:
        for circle in np.round(hough[0]).astype(int):
            x, y, radius = map(int, circle)
            coverage = circle_mask_coverage(mask, x, y, radius)
            score = min(1.0, 0.35 + coverage)
            add_candidate(
                candidates, x, y, radius, "hough", score,
                min_mask_percent, mask, field_mask,
            )

    # Sort left-to-right purely for predictable display; no ball identity is
    # assigned yet. Tracking IDs will be introduced in the next stage.
    candidates.sort(key=lambda item: (int(item["y"]), int(item["x"])))
    return candidates, mask


def find_robot_card(frame, detector):
    """Locate ID 0 inside the calibrated field and include the printed white margin."""
    if len(field_points) != 4:
        return None
    polygon = np.asarray(field_points, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(polygon)
    x0, y0 = max(0, x - 10), max(0, y - 10)
    crop = frame[y0:y + height + 10, x0:x + width + 10]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for scale in (1, 2):
        image = gray if scale == 1 else cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(image)
        if ids is None:
            continue
        for marker_id, quad in zip(ids.ravel(), corners):
            if marker_id != 0:
                continue
            points = quad.reshape(4, 2) / scale + np.float32([x0, y0])
            center = points.mean(axis=0)
            if cv2.pointPolygonTest(polygon, tuple(map(float, center)), False) >= 0:
                # Both printed cards have a 4:3 ratio of card width to marker width.
                return center + (points - center) * (4 / 3)
    return None


def outside_card(balls, card):
    if card is None:
        return balls
    return [ball for ball in balls if cv2.pointPolygonTest(
        card, (float(ball["x"]), float(ball["y"])), False) < 0]


def draw_overlay(
    frame: np.ndarray,
    candidates: list[dict[str, float | int | str]],
    expected_count: int,
    fps: float,
    orange_balls=(),
    card=None,
) -> np.ndarray:
    output = frame.copy()

    if len(field_points) == 4:
        polygon = np.array(field_points, dtype=np.int32)
        cv2.polylines(output, [polygon], True, (255, 80, 0), 3)

    for balls, prefix, color in ((candidates, "W", (0, 255, 0)), (orange_balls, "O", (0, 140, 255))):
        for index, candidate in enumerate(balls, start=1):
            x = int(candidate["x"])
            y = int(candidate["y"])
            radius = int(candidate["radius"])
            score = float(candidate["score"])
            cv2.circle(output, (x, y), radius, color, 2, cv2.LINE_AA)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)
            label_y = max(20, y - radius - 7)
            cv2.putText(
                output,
                f"{prefix}{index} {score:.2f}",
                (x - radius, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

    if card is not None:
        outline = np.rint(card).astype(np.int32)
        cv2.polylines(output, [outline], True, (255, 255, 0), 2, cv2.LINE_AA)
        anchor = (int(outline[:, 0].min()), max(62, int(outline[:, 1].min()) - 6))
        cv2.putText(output, "ID 0", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    count = len(candidates)
    count_color = (0, 220, 0) if count == expected_count else (0, 180, 255)
    status = f"FPS: {fps:.1f} | WHITE: {count}/{expected_count} | ORANGE: {len(orange_balls)} | C: field | S: save | V: video | R: reset | Q: quit"
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, count_color, 2, cv2.LINE_AA)

    if calibration_mode:
        cv2.putText(output, "CLICK 4 FIELD CORNERS: TL, TR, BR, BL", (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for point in calibration_clicks:
            cv2.circle(output, point, 9, (0, 255, 255), 3)
    elif len(field_points) != 4:
        cv2.putText(output, "FIELD NOT CALIBRATED - press C and click 4 corners", (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return output


def main() -> None:
    global calibration_mode, calibration_clicks, display_scale, field_points
    global current_frame_width, current_frame_height

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    load_config()

    # Keep the exact backend used by the working read_camera.py script.
    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    time.sleep(CAMERA_INIT_DELAY_SEC)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        print("Проверьте разрешение macOS для Terminal/Python и CAMERA_INDEX.")
        return

    if REQUESTED_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    if REQUESTED_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Камера подключена.")
    print(f"Разрешение: {actual_width} x {actual_height}")
    print("Распознаются белые и оранжевые мячи.")
    print(f"Порог белого цвета: насыщенность <= {DEFAULTS['white_max_s']}, яркость >= {DEFAULTS['white_min_v']}.")
    print("Нажмите C и щёлкните 4 угла игровой поверхности по часовой стрелке:")
    print("верхний левый -> верхний правый -> нижний правый -> нижний левый.")
    print("S — снимок, V — начать/остановить видео, R — сбросить калибровку, Q — выход.")

    create_trackbars()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MASK_WINDOW_NAME, 960, 540)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    writer = None
    frame_counter = 0
    fps_timer = time.time()
    measured_fps = 0.0
    current_frame: np.ndarray | None = None
    current_overlay: np.ndarray | None = None
    current_mask: np.ndarray | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Предупреждение: не удалось получить кадр с камеры.")
                time.sleep(0.01)
                continue

            frame_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                measured_fps = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

            field_mask = make_field_mask(frame.shape)
            settings = get_settings()
            candidates, white_mask, orange_balls, orange_mask = detect_balls(frame, field_mask, settings)
            card = find_robot_card(frame, detector)
            candidates = outside_card(candidates, card)
            orange_balls = outside_card(orange_balls, card)
            overlay = draw_overlay(frame, candidates, EXPECTED_WHITE_BALLS, measured_fps, orange_balls, card)

            height, width = frame.shape[:2]
            current_frame_width = width
            current_frame_height = height
            if writer is not None:
                writer.write(frame)
                cv2.putText(overlay, "REC", (width - 85, 29), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
            display_scale = min(
                DISPLAY_MAX_WIDTH / float(width),
                DISPLAY_MAX_HEIGHT / float(height),
                1.0,
            )
            display_width = int(round(width * display_scale))
            display_height = int(round(height * display_scale))
            preview = cv2.resize(overlay, (display_width, display_height), interpolation=cv2.INTER_AREA)
            mask_preview = cv2.resize(white_mask, (display_width, display_height), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(WINDOW_NAME, preview)
            cv2.imshow(MASK_WINDOW_NAME, mask_preview)

            current_frame = frame
            current_overlay = overlay
            current_mask = white_mask

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("c"), ord("C")):
                calibration_clicks = []
                calibration_mode = True
                print("Калибровка: щёлкните 4 угла игровой поверхности по часовой стрелке.")
            elif key in (ord("r"), ord("R")):
                field_points = []
                calibration_clicks = []
                calibration_mode = False
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                print("Калибровка сброшена. Нажмите C, чтобы настроить поле заново.")
            elif key in (ord("s"), ord("S")):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                photo_path = os.path.join(SAVE_FOLDER, f"photo_{timestamp}.jpg")
                overlay_path = os.path.join(SAVE_FOLDER, f"detection_{timestamp}.jpg")
                mask_path = os.path.join(SAVE_FOLDER, f"white_mask_{timestamp}.jpg")
                cv2.imwrite(photo_path, current_frame)
                cv2.imwrite(overlay_path, current_overlay)
                cv2.imwrite(mask_path, current_mask)
                orange_path = os.path.join(SAVE_FOLDER, f"orange_mask_{timestamp}.jpg")
                cv2.imwrite(orange_path, orange_mask)
                if len(field_points) == 4:
                    save_config(width, height)
                print(f"Исходное фото: {photo_path}")
                print(f"Результат распознавания: {overlay_path}")
                print(f"Маска белых объектов: {mask_path}")
                print(f"Маска оранжевых объектов: {orange_path}")
            elif key in (ord("v"), ord("V")):
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"Видео сохранено: {video_path}")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    video_path = os.path.abspath(os.path.join("captured_videos", f"field_{timestamp}.avi"))
                    try:
                        os.makedirs(os.path.dirname(video_path), exist_ok=True)
                        fps = measured_fps or cap.get(cv2.CAP_PROP_FPS)
                        if not np.isfinite(fps) or fps <= 0:
                            fps = 15.0
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                                                 fps, (width, height))
                        if not writer.isOpened():
                            raise OSError("не удалось открыть видеофайл")
                        writer.write(frame)
                        print(f"Запись началась: {video_path}")
                    except (OSError, cv2.error) as error:
                        if writer is not None:
                            writer.release()
                            writer = None
                        print(f"Ошибка записи: {error}")
            elif key in (ord("q"), ord("Q")):
                break

    finally:
        if writer is not None:
            writer.release()
            print(f"Видео сохранено: {video_path}")
        cap.release()
        cv2.destroyAllWindows()
        print("Камера отключена.")


if __name__ == "__main__":
    main()
