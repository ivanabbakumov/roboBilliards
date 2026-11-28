"""Run the scoreboard's vision and scoring pipeline against saved recordings."""
import argparse
import json
import math
from pathlib import Path

import cv2

from pocket_scoring import PocketCounter, find_pockets
import robobilliards_vision_test_v02 as vision


DEFAULT_CONFIG = Path(__file__).resolve().parent / 'tests/auditorium_2026-11-28/vision_config.json'


def replay(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Не удалось открыть видео или прочитать FPS: {path}')

        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        counter = PocketCounter([])
        scores = dict(white=0, orange=0)
        goals = []
        frame_count = card_frames = 0
        pockets_frame = None
        next_search = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = frame_count
            frame_count += 1
            now = frame_index / fps
            if frame_index == 0:
                field = vision.make_field_mask(frame.shape)
            if not counter.pockets and now >= next_search:
                counter = PocketCounter(find_pockets(frame, vision.field_points))
                if counter.pockets:
                    pockets_frame = frame_index
                next_search = now + 1

            white, _, orange, _ = vision.detect_balls(frame, field, vision.DEFAULTS)
            card = vision.find_robot_card(frame, detector)
            card_frames += card is not None
            detections = {
                team: [(float(ball['x']), float(ball['y']))
                       for ball in vision.outside_card(balls, card)]
                for team, balls in [('white', white), ('orange', orange)]
            }
            for team, pocket in counter.update(detections, now, card):
                scores[team] += 1
                goals.append(dict(frame=frame_index, seconds=round(now, 3),
                                  team=team, pocket=pocket + 1))

        if not frame_count:
            raise ValueError(f'Не удалось декодировать ни одного кадра: {path}')
        return dict(video=path.name, fps=fps, frames=frame_count, aruco_frames=card_frames,
                    pockets_frame=pockets_frame, pockets=counter.pockets, score=scores, goals=goals)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('videos', nargs='+', type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        points = config['field_polygon']
        if len(points) != 4:
            raise ValueError('Калибровка должна содержать четыре угла поля')
        vision.field_points = [(int(x), int(y)) for x, y in points]
        cv2.setNumThreads(3)
        results = [replay(path) for path in args.videos]
    except (OSError, ValueError, KeyError, TypeError, cv2.error) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
