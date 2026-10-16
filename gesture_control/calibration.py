"""Calibrazione guidata: tieni il palmo (posa indice) fermo nei 4 angoli comodi per te."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2

from . import hand as H

STEPS = [
    ("in ALTO a SINISTRA", (0, 0)),
    ("in ALTO a DESTRA", (1, 0)),
    ("in BASSO a DESTRA", (1, 1)),
    ("in BASSO a SINISTRA", (0, 1)),
]
HOLD_TIME = 1.0
STILL_TOLERANCE = 0.025


class Calibrator:
    def __init__(self, controller, cfg):
        self.ctrl = controller
        self.cfg = cfg
        self.camera_index = 0
        self.previous_mode = None
        self.points: List[Tuple[float, float]] = []
        self.hold_start: Optional[float] = None
        self.hold_origin = None
        self.progress = 0.0
        self.anchor = None

    @property
    def active(self) -> bool:
        return self.ctrl.mode == "calibrazione"

    def start(self, camera_index: int) -> None:
        self.camera_index = camera_index
        if not self.active:
            self.previous_mode = self.ctrl.mode
        self.ctrl.mode = "calibrazione"
        self.ctrl.mouse.release_all()
        self.points = []
        self.hold_start = None

    def cancel(self) -> None:
        self.ctrl.mode = self.previous_mode or "mouse"

    def update(self, hs, t: float) -> None:
        self.anchor = hs.anchor
        if hs.pose != H.POINT:
            self.hold_start, self.progress = None, 0.0
            return
        if self.hold_start is None or \
                abs(hs.anchor[0] - self.hold_origin[0]) + abs(hs.anchor[1] - self.hold_origin[1]) > STILL_TOLERANCE:
            self.hold_start, self.hold_origin = t, hs.anchor
        self.progress = min((t - self.hold_start) / HOLD_TIME, 1.0)
        if self.progress >= 1.0:
            self.points.append(hs.anchor)
            self.hold_start, self.progress = None, 0.0
            if len(self.points) == len(STEPS):
                self._finish()

    def _finish(self) -> None:
        tl, tr, br, bl = self.points
        region = [(tl[0] + bl[0]) / 2, (tl[1] + tr[1]) / 2, (tr[0] + br[0]) / 2, (bl[1] + br[1]) / 2]
        if region[2] - region[0] < 0.08 or region[3] - region[1] < 0.08:
            print("[calibrazione] zona troppo piccola o angoli invertiti, ripeto")
            self.points = []
            return
        self.ctrl.set_region(region)
        self.cfg.set_region_for_camera(self.camera_index, region)
        self.ctrl.mode = self.previous_mode or "mouse"
        self.ctrl.last_action = ("calibrazione salvata", self.ctrl.last_action[1])

    def draw(self, img) -> None:
        h, w = img.shape[:2]
        step = len(self.points)
        if step >= len(STEPS):
            return
        label, (cx, cy) = STEPS[step]
        lines = [
            f"CALIBRAZIONE {step + 1}/{len(STEPS)}",
            f"Alza solo l'INDICE e porta la mano {label}",
            "dove ti e' comodo, poi tienila ferma 1 secondo.  (ESC = annulla)",
        ]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (20, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (20, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        corner = (int(30 + cx * (w - 60)), int(30 + cy * (h - 60)))
        cv2.circle(img, corner, 22, (0, 255, 255), 3)
        for p in self.points:
            cv2.circle(img, (int(p[0] * w), int(p[1] * h)), 8, (0, 200, 0), -1)
        if self.anchor:
            pt = (int(self.anchor[0] * w), int(self.anchor[1] * h))
            cv2.circle(img, pt, 14, (255, 255, 255), 2)
            if self.progress > 0:
                cv2.ellipse(img, pt, (22, 22), -90, 0, 360 * self.progress, (0, 255, 0), 4)
