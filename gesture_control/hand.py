"""Analisi dei 21 landmark MediaPipe: dita alzate, pinch, posa, ancora del palmo."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Pose riconosciute
FIST = "pugno"
POINT = "indice"
TWO = "due dita"
OPEN = "mano aperta"
ROCK = "rock"
CALL = "telefono"
THUMB_UP = "pollice su"
THUMB_DOWN = "pollice giu"
OTHER = "-"


def _d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _d3(a, b) -> float:
    # z di MediaPipe ha circa la stessa scala di x: il 3D regge la mano inclinata verso la camera
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


@dataclass
class HandState:
    landmarks: list
    handedness: str
    size: float
    fingers: Tuple[bool, bool, bool, bool, bool]  # pollice, indice, medio, anulare, mignolo
    pose: str
    anchor: Tuple[float, float]       # centro palmo, normalizzato 0..1
    index_tip: Tuple[float, float]
    pinch_index: float                # distanza pollice-indice / dimensione mano
    pinch_middle: float
    pinch_pinky: float
    index_open: float                 # quanto è aperto l'indice (anti falsi pinch col pugno)
    middle_open: float
    # pollice -> nocca centrale (PIP) del dito: grande nel pinch vero, piccolo se il
    # pollice è solo appoggiato sul dito chiuso
    index_pip_gap: float
    middle_pip_gap: float
    # polso -> punta del dito: piccolo se il dito è ripiegato nel palmo
    index_reach: float
    middle_reach: float
    roll: float                       # angolo polso->medio in gradi


def _finger_extended(lm, tip: int, pip: int) -> bool:
    w = lm[WRIST]
    return _d(w, lm[tip]) > _d(w, lm[pip]) * 1.12


def _classify(lm, fingers, size) -> str:
    thumb, index, middle, ring, pinky = fingers
    four = (index, middle, ring, pinky)
    if four == (True, True, True, True):
        return OPEN
    if four == (True, False, False, False):
        return POINT
    if four == (True, True, False, False):
        return TWO
    if four == (True, False, False, True):
        return ROCK
    if four == (False, False, False, True) and thumb:
        return CALL
    if four == (False, False, False, False):
        if thumb:
            dy = (lm[THUMB_TIP].y - lm[THUMB_MCP].y) / size
            if dy < -0.45:
                return THUMB_UP
            if dy > 0.45:
                return THUMB_DOWN
        return FIST
    return OTHER


def analyze(lm, handedness: str) -> HandState:
    size = _d(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
    thumb = _d(lm[THUMB_TIP], lm[INDEX_MCP]) / size > 0.6 and \
        _d(lm[THUMB_TIP], lm[PINKY_MCP]) > _d(lm[THUMB_IP], lm[PINKY_MCP])
    fingers = (
        thumb,
        _finger_extended(lm, INDEX_TIP, INDEX_PIP),
        _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP),
        _finger_extended(lm, RING_TIP, RING_PIP),
        _finger_extended(lm, PINKY_TIP, PINKY_PIP),
    )
    palm = [lm[i] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)]
    anchor = (sum(p.x for p in palm) / 5, sum(p.y for p in palm) / 5)
    w, m = lm[WRIST], lm[MIDDLE_MCP]
    size3 = _d3(w, m) or 1e-6
    return HandState(
        landmarks=lm,
        handedness=handedness,
        size=size,
        fingers=fingers,
        pose=_classify(lm, fingers, size),
        anchor=anchor,
        index_tip=(lm[INDEX_TIP].x, lm[INDEX_TIP].y),
        pinch_index=_d3(lm[THUMB_TIP], lm[INDEX_TIP]) / size3,
        pinch_middle=_d3(lm[THUMB_TIP], lm[MIDDLE_TIP]) / size3,
        pinch_pinky=_d3(lm[THUMB_TIP], lm[PINKY_TIP]) / size3,
        index_open=_d3(lm[INDEX_TIP], lm[INDEX_MCP]) / size3,
        middle_open=_d3(lm[MIDDLE_TIP], lm[MIDDLE_MCP]) / size3,
        index_pip_gap=_d3(lm[THUMB_TIP], lm[INDEX_PIP]) / size3,
        middle_pip_gap=_d3(lm[THUMB_TIP], lm[MIDDLE_PIP]) / size3,
        index_reach=_d3(lm[WRIST], lm[INDEX_TIP]) / size3,
        middle_reach=_d3(lm[WRIST], lm[MIDDLE_TIP]) / size3,
        roll=math.degrees(math.atan2(m.y - w.y, m.x - w.x)),
    )


def detect_hands(result) -> List[HandState]:
    return [analyze(lm, handed[0].category_name.lower())
            for lm, handed in zip(result.hand_landmarks, result.handedness)]


class HandPicker:
    """Sceglie la mano che controlla il PC; con "any" resta fedele alla prima vista,
    così la seconda mano (usata per la T) non ruba il cursore."""

    def __init__(self, wanted: str):
        self.wanted = wanted
        self.locked: Optional[str] = None

    def split(self, hands: List[HandState]):
        if not hands:
            self.locked = None
            return None, []
        target = self.wanted if self.wanted != "any" else self.locked
        main = next((h for h in hands if h.handedness == target), None)
        if main is None:
            if self.wanted != "any":
                return None, hands
            main = hands[0]
            self.locked = main.handedness
        return main, [h for h in hands if h is not main]


def _axis(h: HandState, aspect: float):
    lm = h.landmarks
    return (lm[MIDDLE_MCP].x - lm[WRIST].x) * aspect, lm[MIDDLE_MCP].y - lm[WRIST].y


def is_timeout_t(a: HandState, b: HandState, aspect: float) -> bool:
    """Gesto "T" del timeout: una mano orizzontale sopra, l'altra verticale con la
    punta delle dita sotto il palmo della prima. aspect = larghezza/altezza frame."""
    for top, stem in ((a, b), (b, a)):
        tdx, tdy = _axis(top, aspect)
        sdx, sdy = _axis(stem, aspect)
        if abs(tdx) < 1.3 * abs(tdy):           # sopra: orizzontale
            continue
        if sdy >= 0 or abs(sdy) < 1.3 * abs(sdx):  # gambo: verticale, dita in su
            continue
        if sum(top.fingers[1:]) < 2 or sum(stem.fingers[1:]) < 2:
            continue
        tip = max((stem.landmarks[i] for i in (INDEX_TIP, MIDDLE_TIP, RING_TIP)), key=lambda p: -p.y)
        size = (math.hypot(tdx, tdy) + math.hypot(sdx, sdy)) / 2
        dx = (tip.x - top.anchor[0]) * aspect
        dy = tip.y - top.anchor[1]
        if math.hypot(dx, dy) < 0.9 * size and dy > -0.5 * size:
            return True
    return False
