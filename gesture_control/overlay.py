"""Disegno dell'anteprima: scheletro mano, zona attiva, stato, guida gesti."""
from __future__ import annotations

import cv2

from .controller import MODE_CALIBRATE, MODE_KEYBOARD, MODE_MOUSE

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

HELP = [
    "GESTI",
    "indice alzato ......... muovi cursore",
    "pugno ................. frizione (riposiziona)",
    "pinch pollice-indice .. click / tieni = trascina",
    "  2 pinch veloci ...... doppio click",
    "pinch pollice-medio ... click destro",
    "indice+medio (V) ...... scroll su/giu",
    "mano aperta + swipe ... desktop / panoramica",
    "T con due mani 0.5s ... play/pausa",
    "pugni che si scontrano .. armatura Iron Man",
    "pollice su / giu ...... volume",
    "pinch pollice-mignolo + ruota polso .. volume",
    "rock (indice+mignolo) 1s .. pausa/attiva",
    "telefono (pollice+mignolo) 1s .. tastiera",
    "",
    "TASTI: q esci  p pausa  c calibra  n camera",
    "       k tastiera  i iron man  g gira guanto  d debug  h aiuto",
]


def _text(img, text, org, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _pinch_bars(img, cfg, hs) -> None:
    """Distanza pollice-dito live: la barra deve scendere sotto la linea verde per cliccare."""
    h, w = img.shape[:2]
    full = 1.0
    bx, bw = w - 230, 200
    rows = [("indice", hs.pinch_index), ("medio", hs.pinch_middle), ("mignolo", hs.pinch_pinky)]
    for i, (label, value) in enumerate(rows):
        y = 30 + i * 34
        on = value < cfg.pinch_on
        cv2.rectangle(img, (bx, y), (bx + bw, y + 14), (60, 60, 60), -1)
        cv2.rectangle(img, (bx, y), (bx + int(bw * min(value / full, 1.0)), y + 14),
                      (0, 220, 0) if on else (0, 160, 255), -1)
        for thr, col in ((cfg.pinch_on, (0, 255, 0)), (cfg.pinch_off, (0, 0, 255))):
            x = bx + int(bw * thr / full)
            cv2.line(img, (x, y - 3), (x, y + 17), col, 2)
        _text(img, f"{label} {value:.2f}", (bx - 125, y + 13), 0.5)
    y = 30 + len(rows) * 34 + 10
    for label, tip, pip, reach, need_ratio, need_reach in (
            ("ind", hs.pinch_index, hs.index_pip_gap, hs.index_reach, cfg.left_pip_ratio, 0.0),
            ("med", hs.pinch_middle, hs.middle_pip_gap, hs.middle_reach, cfg.pinch_pip_ratio, cfg.pinch_min_reach)):
        ratio = pip / max(tip, 1e-3)
        ok = ratio >= need_ratio and reach >= need_reach
        _text(img, f"{label}: nocca/punta {min(ratio, 99):.1f}  portata {reach:.2f}",
              (bx - 125, y), 0.45, (0, 220, 0) if ok else (0, 120, 255), 1)
        y += 20


def draw(img, ctrl, hs, fps: float, t: float, show_help: bool, camera_index: int) -> None:
    h, w = img.shape[:2]

    if ctrl.mode == MODE_MOUSE:
        x0, y0, x1, y1 = ctrl.region
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 200, 0), 1)

    for other in ([] if ctrl.ironman is not None and ctrl.ironman.active else ctrl.other_hands):
        pts = [(int(p.x * w), int(p.y * h)) for p in other.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (150, 150, 150), 1, cv2.LINE_AA)

    armor = ctrl.ironman is not None and ctrl.ironman.active
    if hs is not None and not armor:
        pts = [(int(p.x * w), int(p.y * h)) for p in hs.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (230, 230, 230), 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(img, p, 4 if i not in (4, 8, 12, 20) else 6, (0, 140, 255), -1, cv2.LINE_AA)
        ax, ay = hs.anchor
        cv2.circle(img, (int(ax * w), int(ay * h)), 9, (255, 200, 0), 2, cv2.LINE_AA)

    if ctrl.mode == MODE_KEYBOARD and ctrl.virtual_keyboard is not None:
        ctrl.virtual_keyboard.draw(img, t)
    if ctrl.mode == MODE_CALIBRATE and ctrl.calibrator is not None:
        ctrl.calibrator.draw(img)
        return

    state = "ATTIVO" if ctrl.enabled else "PAUSA"
    color = (0, 220, 0) if ctrl.enabled else (0, 0, 255)
    _text(img, f"{state} | {ctrl.mode} | cam {camera_index} | {fps:.0f} fps", (12, h - 44), 0.6, color)
    pose = hs.pose if hs is not None else "-"
    _text(img, f"posa: {pose}   {ctrl.message}", (12, h - 16), 0.6)

    label, at = ctrl.last_action
    if label and t - at < 1.5:
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        _text(img, label, ((w - size[0]) // 2, 60), 1.0, (0, 255, 255))

    if ctrl.hold_progress > 0:
        bx, by, bw = w // 2 - 120, 90, 240
        cv2.rectangle(img, (bx, by), (bx + bw, by + 14), (80, 80, 80), -1)
        cv2.rectangle(img, (bx, by), (bx + int(bw * ctrl.hold_progress), by + 14), (0, 220, 255), -1)
        _text(img, ctrl.hold_label, (bx, by - 8), 0.55)

    if ctrl.cfg.show_debug and hs is not None:
        _pinch_bars(img, ctrl.cfg, hs)
    if ctrl.cfg.show_debug and ctrl.bump_distance is not None:
        ok = ctrl.bump_distance <= ctrl.cfg.bump_contact
        _text(img, f"pugni: distanza {ctrl.bump_distance:.2f} (contatto < {ctrl.cfg.bump_contact})",
              (w - 330, h - 70), 0.5, (0, 255, 0) if ok else (200, 200, 200), 1)

    if show_help and ctrl.mode == MODE_MOUSE:
        for i, line in enumerate(HELP):
            _text(img, line, (12, 26 + i * 22), 0.5, (255, 255, 255), 1)
