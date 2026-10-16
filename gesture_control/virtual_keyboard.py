"""Tastiera virtuale disegnata sull'anteprima: punti con la pinza pollice-indice, pinch = tasto."""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
from pynput.keyboard import Key

from .filters import OneEuro2D

# (etichetta, testo da scrivere oppure tasto speciale, larghezza in unità)
ROWS = [
    [(c, c, 1) for c in "1234567890"] + [("<-", Key.backspace, 1.5)],
    [(c, c, 1) for c in "qwertyuiop"] + [("e'", "è", 1)],
    [(c, c, 1) for c in "asdfghjkl"] + [("o'", "ò", 1), ("a'", "à", 1), ("u'", "ù", 1)],
    [("SHIFT", "shift", 1.5)] + [(c, c, 1) for c in "zxcvbnm,.?"] + [("i'", "ì", 1)],
    [("ESC", Key.esc, 1.5), ("@", "@", 1), ("SPAZIO", " ", 5), ("TAB", Key.tab, 1.5), ("INVIO", Key.enter, 2.5)],
]
AREA = (0.06, 0.30, 0.94, 0.88)  # x0, y0, x1, y1 normalizzati sull'immagine


class VirtualKeyboard:
    def __init__(self, keyboard, cfg):
        self.kb = keyboard
        self.cfg = cfg
        self.filter = OneEuro2D(1.5, 2.0)
        self.pointer: Optional[Tuple[float, float]] = None
        self.pinched = False
        self.shift = False
        self.flash: Tuple[Optional[str], float] = (None, 0.0)
        self.keys = self._layout()

    def _layout(self) -> List[Tuple[str, object, Tuple[float, float, float, float]]]:
        x0, y0, x1, y1 = AREA
        row_h = (y1 - y0) / len(ROWS)
        unit = (x1 - x0) / max(sum(w for _, _, w in row) for row in ROWS)
        keys = []
        for r, row in enumerate(ROWS):
            total = sum(w for _, _, w in row) * unit
            x = x0 + ((x1 - x0) - total) / 2
            y = y0 + r * row_h
            for label, value, w in row:
                keys.append((label, value, (x, y, x + w * unit, y + row_h)))
                x += w * unit
        return keys

    def key_at(self, p) -> Optional[tuple]:
        if p is None:
            return None
        for key in self.keys:
            kx0, ky0, kx1, ky1 = key[2]
            if kx0 <= p[0] < kx1 and ky0 <= p[1] < ky1:
                return key
        return None

    def update(self, hs, t: float) -> None:
        lm = hs.landmarks
        # punto medio tra pollice e indice: resta fermo mentre si chiude la pinza
        mx = (lm[4].x + lm[8].x) / 2
        my = (lm[4].y + lm[8].y) / 2
        self.pointer = self.filter(mx, my, t)

        if self.pinched:
            if hs.pinch_index > self.cfg.pinch_off:
                self.pinched = False
            return
        if hs.pinch_index < self.cfg.pinch_on:
            self.pinched = True
            key = self.key_at(self.pointer)
            if key:
                self._press(key)
                self.flash = (key[0], t)

    def _press(self, key) -> None:
        label, value, _ = key
        if value == "shift":
            self.shift = not self.shift
            return
        if isinstance(value, str):
            self.kb.type_text(value.upper() if self.shift else value)
            self.shift = False
        else:
            self.kb.tap(value)

    def draw(self, img, t: float) -> None:
        h, w = img.shape[:2]
        hover = self.key_at(self.pointer)
        overlay = img.copy()
        for key in self.keys:
            label, value, (kx0, ky0, kx1, ky1) = key
            p0 = (int(kx0 * w) + 3, int(ky0 * h) + 3)
            p1 = (int(kx1 * w) - 3, int(ky1 * h) - 3)
            color = (60, 60, 60)
            if key is hover:
                color = (200, 140, 40)
            if self.flash[0] == label and t - self.flash[1] < 0.25:
                color = (60, 200, 60)
            if value == "shift" and self.shift:
                color = (40, 120, 220)
            cv2.rectangle(overlay, p0, p1, color, -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        for label, value, (kx0, ky0, kx1, ky1) in self.keys:
            text = label.upper() if (self.shift and len(label) == 1) else label
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cx = int((kx0 + kx1) / 2 * w - size[0] / 2)
            cy = int((ky0 + ky1) / 2 * h + size[1] / 2)
            cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if self.pointer:
            pt = (int(self.pointer[0] * w), int(self.pointer[1] * h))
            cv2.circle(img, pt, 10, (0, 255, 255) if not self.pinched else (0, 255, 0), 3)
