"""Detect the four pocket rims and confirm ball entries across consecutive frames."""
from dataclasses import dataclass
from math import dist

import cv2
import numpy as np


def find_pockets(frame, field):
    if len(field) != 4:
        return []
    points = np.asarray(field, dtype=float)
    side = np.mean(np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1))
    center = points.mean(axis=0)
    pockets = []
    for corner in points:
        target = corner + .20 * (center - corner)
        x, y = np.maximum(0, (target - side * .105).astype(int))
        right, bottom = (target + side * .105).astype(int)
        area = frame[y:bottom, x:right]
        if not area.size:
            return []
        gray = cv2.medianBlur(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY), 5)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, side * .05,
                                   param1=70, param2=23, minRadius=round(side*.033),
                                   maxRadius=round(side*.055))
        if circles is None:
            return []
        options = []
        for cx, cy, radius in circles[0]:
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            interior = gray[(xx-cx)**2 + (yy-cy)**2 < (.65*radius)**2]
            if interior.size and np.median(interior) < 105:
                pos = (float(x+cx), float(y+cy))
                options.append((dist(pos, target), pos, float(radius)))
        if not options:
            return []
        _, pos, radius = min(options)
        pockets.append((pos, radius))
    return pockets


@dataclass
class Track:
    pos: tuple
    last_seen: float
    outside: bool
    pocket: int = -1
    misses: int = 0


class PocketCounter:
    def __init__(self, pockets):
        self.pockets = pockets
        self.tracks = {'white': [], 'orange': []}

    def update(self, detections, now, card=None):
        goals = []
        if not self.pockets:
            return goals
        radius = max(r for _, r in self.pockets)
        all_points = [p for points in detections.values() for p in points]
        for team, points in detections.items():
            tracks = self.tracks[team]
            pairs = sorted((dist(t.pos, p), i, j) for i, t in enumerate(tracks)
                           for j, p in enumerate(points))
            matched, used = set(), set()
            for distance, i, j in pairs:
                track = tracks[i]
                if i in matched or j in used or distance > radius * 1.6:
                    continue
                matched.add(i)
                used.add(j)
                track.pos, track.last_seen, track.misses = points[j], now, 0
            survivors = []
            for i, track in enumerate(tracks):
                if i not in matched:
                    track.misses += 1
                    if track.pocket >= 0:
                        center, r = self.pockets[track.pocket]
                        occupied = any(dist(p, track.pos) < .65*r for p in all_points)
                        covered = card is not None and cv2.pointPolygonTest(card, track.pos, True) >= -r
                        if occupied or covered:
                            track.pocket = -1
                    if now-track.last_seen >= .35 and track.misses >= 3:
                        if track.pocket >= 0:
                            goals.append((team, track.pocket))
                        continue
                survivors.append(track)
            for j, point in enumerate(points):
                if j not in used:
                    survivors.append(Track(point, now, False))
            for track in survivors:
                if track.misses:
                    continue
                idx = min(range(4), key=lambda k: dist(track.pos, self.pockets[k][0]))
                center, r = self.pockets[idx]
                distance = dist(track.pos, center)
                if distance > 1.10*r:
                    track.outside = True
                track.pocket = idx if track.outside and distance < .88*r else -1
            self.tracks[team] = survivors
        return goals
