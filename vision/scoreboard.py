"""Local RoboBilliard referee console and LAN projector display."""
import argparse
import json
import math
import sys
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from pocket_scoring import PocketCounter, find_pockets
import robobilliards_vision_test_v02 as vision


class Match:
    def __init__(self):
        self.white = self.orange = 0
        self.remaining = 60
        self.end = None
        self.last_goal = None

    def snapshot(self):
        if self.end is not None:
            self.remaining = max(0, self.end - time.monotonic())
            if not self.remaining:
                self.end = None
        return dict(white=self.white, orange=self.orange, seconds=math.ceil(self.remaining),
                    running=self.end is not None, last_goal=self.last_goal)

    def change(self, action):
        self.snapshot()
        if action == 'start' and self.end is None and self.remaining:
            self.end = time.monotonic() + self.remaining
        elif action == 'pause':
            self.end = None
        elif action == 'reset':
            self.__init__()
        elif action == 'extra':
            self.remaining, self.end = 30, None
        elif action in ('white+', 'white-', 'orange+', 'orange-'):
            team = action[:-1]
            delta = 1 if action.endswith('+') else -1
            setattr(self, team, max(0, getattr(self, team) + delta))


def capture(source):
    global latest, frame_time, pockets_ready
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    while True:
        cap = cv2.VideoCapture(source) if source else cv2.VideoCapture(args.camera, backend)
        counter = PocketCounter([])
        next_search = 0
        tracked_epoch = None
        previous_time = None
        with lock:
            pockets_ready = False
        try:
            if not source:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, vision.REQUESTED_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vision.REQUESTED_HEIGHT)
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.monotonic()
                with lock:
                    active = match.snapshot()['running']
                    epoch = round_epoch
                if not counter.pockets and now >= next_search:
                    counter = PocketCounter(find_pockets(frame, vision.field_points))
                    next_search = now + 1
                field = vision.make_field_mask(frame.shape)
                white, _, orange, _ = vision.detect_balls(frame, field, vision.DEFAULTS)
                card = vision.find_robot_card(frame, detector)
                white = vision.outside_card(white, card)
                orange = vision.outside_card(orange, card)
                if tracked_epoch != epoch or not active or (previous_time is not None and now-previous_time > .5):
                    counter = PocketCounter(counter.pockets)
                previous_time, tracked_epoch = now, epoch
                detections = {team: [(float(b['x']), float(b['y'])) for b in balls]
                              for team, balls in [('white', white), ('orange', orange)]}
                goals = counter.update(detections, now, card) if active else []
                with lock:
                    pockets_ready = bool(counter.pockets)
                    if epoch == round_epoch and match.snapshot()['running']:
                        for team, pocket in goals:
                            match.change(team + '+')
                            match.last_goal = {'team': team, 'pocket': pocket + 1}
                for number, (center, radius) in enumerate(counter.pockets, 1):
                    xy = tuple(round(value) for value in center)
                    cv2.circle(frame, xy, round(radius), (200, 195, 60), 1)
                    cv2.putText(frame, str(number), xy, cv2.FONT_HERSHEY_SIMPLEX, .5, (200, 195, 60), 1)
                for balls, color in ((white, (60, 220, 90)), (orange, (0, 145, 255))):
                    for ball in balls:
                        cv2.circle(frame, (ball['x'], ball['y']), ball['radius'], color, 2)
                if card is not None:
                    outline = card.round().astype('int32')
                    cv2.polylines(frame, [outline], True, (255, 255, 0), 2)
                    cv2.putText(frame, 'ID 0', tuple(outline[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)
                width = min(1280, frame.shape[1])
                frame = cv2.resize(frame, (width, round(frame.shape[0]*width/frame.shape[1])), interpolation=cv2.INTER_AREA)
                ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with lock:
                        latest, frame_time = data.tobytes(), time.monotonic()
                if source:
                    time.sleep(1 / max(1, cap.get(cv2.CAP_PROP_FPS)))
        finally:
            cap.release()
        time.sleep(.1 if source else 2)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, mime, body):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/judge'):
            if path == '/judge' and self.client_address[0] != '127.0.0.1':
                return self.send_error(403)
            return self.reply(200, 'text/html; charset=utf-8', page)
        if path == '/state':
            with lock:
                state = match.snapshot()
                state['pockets_ready'] = pockets_ready
                state['capture_started'] = worker is not None and worker.is_alive()
                state['camera'] = time.monotonic() - frame_time < 3
            return self.reply(200, 'application/json', json.dumps(state).encode())
        if path == '/frame.jpg':
            with lock:
                frame = latest if time.monotonic() - frame_time < 3 else None
            return self.reply(200 if frame else 503, 'image/jpeg', frame or b'')
        self.send_error(404)

    def do_POST(self):
        global worker, round_epoch
        if self.path != '/action' or self.client_address[0] != '127.0.0.1' or self.headers.get('X-Referee') != '1' or self.headers.get('Host') not in local_hosts:
            return self.send_error(403)
        length = self.headers.get('Content-Length', '')
        if not length.isdigit() or int(length) > 32:
            return self.send_error(400)
        action = self.rfile.read(int(length)).decode('ascii', errors='replace')
        if action not in {'start','pause','extra','reset','white+','white-','orange+','orange-'}:
            return self.send_error(400)
        with lock:
            if action == 'start' and (worker is None or not worker.is_alive()):
                worker = threading.Thread(target=capture, args=(args.video,), daemon=True)
                worker.start()
            if action in ('reset', 'extra', 'pause') or (action == 'start' and not match.snapshot()['running']):
                round_epoch += 1
            match.change(action)
        self.reply(200, 'text/plain', b'ok')

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--video', help='Play a test recording in a loop')
    parser.add_argument('--camera', type=int, default=vision.CAMERA_INDEX)
    root = Path(__file__).resolve().parent
    parser.add_argument('--config', type=Path, default=root / 'tests/auditorium_2026-11-28/vision_config.json')
    args = parser.parse_args()
    vision.CONFIG_FILE = str(args.config)
    vision.load_config()
    round_epoch = 0
    pockets_ready = False
    worker = None
    if args.video and not Path(args.video).is_file():
        parser.error('Видео не найдено')
    page = Path(__file__).with_name('scoreboard.html').read_bytes()
    match, lock = Match(), threading.Lock()
    latest, frame_time = None, float('-inf')
    local_hosts = {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}
    server = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'Судья: http://127.0.0.1:{args.port}/judge', flush=True)
    address = socket.gethostname()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(('192.0.2.1', 9))
            address = probe.getsockname()[0]
    except OSError:
        pass
    print(f'Проектор: http://{address}:{args.port}/', flush=True)
    server.serve_forever()
