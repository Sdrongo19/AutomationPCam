"""Macchina a stati: trasforma le pose della mano in azioni di mouse/tastiera."""
from __future__ import annotations

import math
import time
from collections import deque
from typing import List, Optional

from . import hand as H
from .filters import OneEuro2D
from .platform_io import Keyboard, Mouse, ScreenSpace

MODE_MOUSE = "mouse"
MODE_KEYBOARD = "tastiera"
MODE_CALIBRATE = "calibrazione"

POSE_STABLE_FRAMES = 3


class Hold:
    """Gesto da tenere fermo per un tempo: scatta una volta, poi aspetta il rilascio."""

    def __init__(self):
        self.start: Optional[float] = None
        self.fired = False

    def update(self, active: bool, t: float, duration: float) -> bool:
        if not active:
            self.start, self.fired = None, False
            return False
        if self.start is None:
            self.start = t
        if not self.fired and t - self.start >= duration:
            self.fired = True
            return True
        return False

    def progress(self, t: float, duration: float) -> float:
        if self.start is None or self.fired:
            return 0.0
        return min((t - self.start) / duration, 1.0)


class Controller:
    def __init__(self, cfg, screen: ScreenSpace, mouse: Mouse, keyboard: Keyboard):
        self.cfg = cfg
        self.screen = screen
        self.mouse = mouse
        self.kb = keyboard
        self.region = list(cfg.active_region)

        self.enabled = True
        self.mode = MODE_MOUSE
        self.message = ""
        self.last_action = ("", 0.0)
        self.hold_label, self.hold_progress = "", 0.0

        self.filter = OneEuro2D(cfg.smoothing_min_cutoff, cfg.smoothing_beta)
        self.prev_anchor = None
        self.cursor = self._initial_cursor()

        self._raw_pose, self._pose_count, self.pose = H.OTHER, 0, H.OTHER
        self.left_down = False
        self.right_latched = False
        self._pinch_pending, self._pinch_frames = None, 0
        self.dragging = False
        self.pinch_start = 0.0
        self.freeze_until = 0.0
        self.now = 0.0

        self.swipe_history = deque()
        self.swipe_until = 0.0
        self.palm_hold = Hold()
        self.palm_hold_origin = None
        self.toggle_hold = Hold()
        self.keyboard_hold = Hold()
        self.timeout_hold = Hold()
        self.bump_history = deque()  # (istante, distanza tra le nocche in "mani")
        self.bump_until = 0.0
        self.bump_distance = None    # per il debug
        self.face = None       # ultimo viso rilevato (ironman.Face)
        self.ironman = None    # agganciato da app.py
        self.picker = H.HandPicker(cfg.hand)
        self.aspect = 16 / 9
        self.main_hand: Optional[H.HandState] = None
        self.other_hands: List[H.HandState] = []
        self.volume_next = None
        self.knob_angle = None

        # agganciati da app.py
        self.virtual_keyboard = None
        self.calibrator = None

    # ------------------------------------------------------------ utilità

    def _initial_cursor(self):
        try:
            from pynput.mouse import Controller as _M
            x, y = _M().position
            s = self.screen
            return [(x - s.x0) / (s.x1 - s.x0), (y - s.y0) / (s.y1 - s.y0)]
        except Exception:
            return [0.5, 0.5]

    def set_region(self, region) -> None:
        self.region = list(region)

    def _action(self, name: str, label: Optional[str] = None) -> None:
        self.kb.run_action(name)
        self.last_action = (label or name, time.monotonic())

    def toggle_enabled(self) -> None:
        self.enabled = not self.enabled
        self.mouse.release_all()
        self.left_down = False
        self.last_action = ("ATTIVO" if self.enabled else "IN PAUSA", time.monotonic())

    def toggle_keyboard(self) -> None:
        self.mouse.release_all()
        self.left_down = False
        self.mode = MODE_MOUSE if self.mode == MODE_KEYBOARD else MODE_KEYBOARD
        self.last_action = (f"modalita {self.mode}", time.monotonic())

    def _stable_pose(self, raw: str) -> str:
        if raw == self._raw_pose:
            self._pose_count += 1
        else:
            self._raw_pose, self._pose_count = raw, 1
        if self._pose_count >= POSE_STABLE_FRAMES:
            self.pose = raw
        return self.pose

    def _lost_hand(self) -> None:
        self.mouse.release_all()
        self.left_down = False
        self.filter.reset()
        self.prev_anchor = None
        self.swipe_history.clear()
        self.knob_angle = None
        self.volume_next = None
        for h in (self.palm_hold, self.toggle_hold, self.keyboard_hold):
            h.update(False, 0, 1)

    # ------------------------------------------------------------ ciclo

    def update(self, hands: List[H.HandState], t: float) -> None:
        self.hold_label, self.hold_progress = "", 0.0
        self.now = t
        hs, others = self.picker.split(hands)
        self.main_hand, self.other_hands = hs, others

        # pugni che si scontrano -> armatura Iron Man on/off
        self._fist_bump(hands, t)

        # gesto T a due mani (timeout) -> play/pausa. Mentre lo fai il mouse resta fermo.
        t_pose = self.enabled and self.mode == MODE_MOUSE and hs is not None and \
            any(H.is_timeout_t(hs, o, self.aspect) for o in others)
        if self.timeout_hold.update(t_pose, t, self.cfg.timeout_hold_time):
            self._action(self.cfg.actions.get("timeout_t", "none"), "T: play/pausa")
        if t_pose:
            self._show_hold("T: " + self.cfg.actions.get("timeout_t", "none"), self.timeout_hold, t,
                            self.cfg.timeout_hold_time)
            self.message = "timeout (T)"
            self.prev_anchor = None
            self.mouse.release_all()
            self.left_down = False
            return

        if hs is None:
            self._lost_hand()
            self._pose_count = 0
            self.message = "nessuna mano"
            return

        pose = self._stable_pose(hs.pose)

        if self.mode == MODE_CALIBRATE and self.calibrator is not None:
            self.calibrator.update(hs, t)
            return

        # 🤘 tenuto: attiva/pausa (funziona anche in pausa)
        if self.toggle_hold.update(pose == H.ROCK, t, self.cfg.hold_toggle_time):
            self.toggle_enabled()
        if pose == H.ROCK:
            self._show_hold("pausa/attiva", self.toggle_hold, t, self.cfg.hold_toggle_time)

        if not self.enabled:
            self.message = "IN PAUSA - rock 1s per riattivare"
            self.prev_anchor = None
            return

        # 🤙 tenuto: tastiera virtuale on/off
        if self.keyboard_hold.update(pose == H.CALL, t, self.cfg.hold_toggle_time):
            self.toggle_keyboard()
        if pose == H.CALL:
            self._show_hold("tastiera", self.keyboard_hold, t, self.cfg.hold_toggle_time)

        if self.mode == MODE_KEYBOARD and self.virtual_keyboard is not None:
            self.virtual_keyboard.update(hs, t)
            self.message = "tastiera virtuale - telefono 1s per uscire"
            return

        self._mouse_mode(hs, pose, t)

    def _fist_bump(self, hands, t: float) -> None:
        """Due pugni che si avvicinano rapidamente e si toccano con le nocche.

        Conta quanto si avvicinano (non da dove partono), ignora singoli frame imperfetti e
        accetta come contatto anche la mano che MediaPipe perde proprio mentre i pugni si
        sovrappongono."""
        cfg = self.cfg
        hist = self.bump_history
        while hist and t - hist[0][0] > cfg.bump_window:
            hist.popleft()
        self.bump_distance = None
        if self.ironman is None or self.mode == MODE_CALIBRATE or t < self.bump_until:
            return

        fists = len(hands) >= 2 and all(h.fingers[1:] == (False, False, False, False) for h in hands[:2])
        if not fists:
            # una mano sparita subito dopo un avvicinamento rapido: i pugni si sono scontrati
            if len(hands) < 2 and len(hist) >= 3 and t - hist[-1][0] < 0.15:
                last_d, point = hist[-1][1], hist[-1][2]
                if last_d < cfg.bump_contact * 1.35 and max(e[1] for e in hist) - last_d >= cfg.bump_approach:
                    self._bump(point, t)
            return

        def knuckles(h):
            lm = h.landmarks
            return (sum(lm[i].x for i in (5, 9, 13, 17)) / 4 * self.aspect,
                    sum(lm[i].y for i in (5, 9, 13, 17)) / 4)

        def hand_size(h):
            lm = h.landmarks
            return math.hypot((lm[9].x - lm[0].x) * self.aspect, lm[9].y - lm[0].y) or 1e-3

        a, b = hands[0], hands[1]
        ka, kb = knuckles(a), knuckles(b)
        d = math.hypot(ka[0] - kb[0], ka[1] - kb[1]) / ((hand_size(a) + hand_size(b)) / 2)
        point = ((ka[0] + kb[0]) / 2 / self.aspect, (ka[1] + kb[1]) / 2)
        hist.append((t, d, point))
        self.bump_distance = d
        if d <= cfg.bump_contact and max(e[1] for e in hist) - d >= cfg.bump_approach:
            self._bump(point, t)

    def _bump(self, point, t: float) -> None:
        self.bump_until = t + self.cfg.bump_cooldown
        self.bump_history.clear()
        self.ironman.toggle(t)
        self.ironman.impact(point, t)
        self.last_action = ("ARMATURA " + ("ON" if self.ironman.armored else "OFF"), t)

    def _show_hold(self, label: str, hold: Hold, t: float, duration: float) -> None:
        p = hold.progress(t, duration)
        if p > 0:
            self.hold_label, self.hold_progress = label, p

    def _mouse_mode(self, hs: H.HandState, pose: str, t: float) -> None:
        cfg = self.cfg
        x0, y0, x1, y1 = self.region
        fx, fy = self.filter(hs.anchor[0], hs.anchor[1], t)
        delta = (0.0, 0.0)
        if self.prev_anchor is not None:
            delta = (fx - self.prev_anchor[0], fy - self.prev_anchor[1])
        self.prev_anchor = (fx, fy)
        # spostare la mano per tutta la zona attiva = attraversare tutto lo schermo
        gain = (1.0 / max(x1 - x0, 0.05), 1.0 / max(y1 - y0, 0.05))

        # ---- click sinistro / trascinamento (pinch pollice-indice)
        # Pinch rapido = click nel punto mirato. Pinch tenuto drag_hold_time = tasto premuto,
        # poi la mano trascina. Così muoversi durante un click non seleziona né sposta nulla.
        if self.left_down:
            if hs.pinch_index > cfg.pinch_off:
                if self.dragging:
                    self.mouse.up("left")
                else:
                    self.mouse.click("left")
                    self.last_action = ("click", t)
                self.left_down = self.dragging = False
                self.freeze_until = t + cfg.click_freeze_time
                return
            if not self.dragging and t - self.pinch_start >= cfg.drag_hold_time:
                self.mouse.down("left")
                self.dragging = True
                self.last_action = ("TRASCINA", t)
            if self.dragging:
                # mentre le dita si riaprono il cursore resta fermo: si rilascia dove si mirava
                if hs.pinch_index < cfg.pinch_on:
                    self._move_cursor(delta, gain)
                self.message = "trascino"
            else:
                self.hold_label = "tieni per trascinare"
                self.hold_progress = (t - self.pinch_start) / cfg.drag_hold_time
                self.message = "click"
            return
        else:
            finger, raw = self._pinch_candidate(hs, pose)
            if finger == "left":
                self.left_down, self.dragging, self.pinch_start = True, False, t
                self.message = "click"
                return
            if finger == "right" and not self.right_latched:
                self.mouse.click("right")
                self.right_latched = True
                self.last_action = ("click destro", t)
            # il blocco dura solo finché il pinch destro è ancora in corso
            if raw != "right":
                self.right_latched = False
            if self.right_latched:
                self.message = "click destro"
                return

        # ---- manopola volume (pinch pollice-mignolo + rotazione polso)
        knob = cfg.volume_mode in ("knob", "both") and hs.pinch_pinky < cfg.pinch_on \
            and hs.pinch_index > cfg.pinch_off
        if knob:
            self._volume_knob(hs.roll)
            self.message = "manopola volume"
            return
        self.knob_angle = None

        # ---- posa corrente
        if pose == H.POINT and hs.pinch_index < cfg.aim_zone and hs.pinch_index <= hs.pinch_middle:
            # pollice che si avvicina all'indice: cursore fermo per cliccare dove si mira
            self.message = "mira"
        elif pose == H.POINT:
            self._move_cursor(delta, gain)
            self.message = "muovo"
        elif pose == H.TWO:
            px = delta[1] * cfg.scroll_speed
            self.mouse.scroll(px if cfg.scroll_natural else -px)
            self.message = "scroll"
        elif pose == H.FIST:
            self.message = "frizione (riposiziona la mano)"
        elif pose in (H.THUMB_UP, H.THUMB_DOWN) and cfg.volume_mode in ("thumbs", "both"):
            self._volume_thumbs(pose == H.THUMB_UP, t)
            self.message = "volume"
        else:
            self.message = pose
        if pose not in (H.THUMB_UP, H.THUMB_DOWN):
            self.volume_next = None

        # ---- mano aperta: swipe e "tieni fermo"
        self._open_palm(hs, pose, t)

    def _pinch_candidate(self, hs: H.HandState, pose: str) -> Optional[str]:
        """Quale dito tocca il pollice: vince il più vicino, con margine, confermato su più frame.

        Toccando il medio il pollice passa vicino all'indice (e viceversa): senza il
        confronto tra i due un click destro diventava sinistro.
        """
        cfg = self.cfg
        candidate = None
        if pose not in (H.THUMB_UP, H.THUMB_DOWN):
            idx, mid = hs.pinch_index, hs.pinch_middle
            # sinistro: basta che l'indice sia il dito più vicino al pollice. Nel pinch il
            # medio piegato sta accanto all'indice, quindi niente margine qui.
            if idx < cfg.pinch_on and idx <= mid and hs.index_pip_gap >= idx * cfg.left_pip_ratio:
                candidate = "left"
            # destro: controlli severi, il pollice appoggiato sul medio chiuso dava falsi click
            elif mid < cfg.pinch_on and mid * cfg.pinch_margin <= idx and hs.middle_open > 0.45 \
                    and hs.middle_pip_gap >= mid * cfg.pinch_pip_ratio \
                    and hs.middle_reach >= cfg.pinch_min_reach:
                candidate = "right"
        if candidate != self._pinch_pending:
            self._pinch_pending, self._pinch_frames = candidate, 0
        if candidate is None:
            return None, None
        self._pinch_frames += 1
        confirmed = candidate if self._pinch_frames >= cfg.pinch_confirm_frames else None
        return confirmed, candidate

    def _move_cursor(self, delta, gain) -> None:
        if self.now < self.freeze_until:
            return  # i movimenti scartati non si accumulano: niente salto allo sblocco
        c = self.cursor
        c[0] = min(max(c[0] + delta[0] * gain[0], 0.0), 1.0)
        c[1] = min(max(c[1] + delta[1] * gain[1], 0.0), 1.0)
        x, y = self.screen.to_screen(c[0], c[1])
        self.mouse.move(x, y)

    def _volume_thumbs(self, up: bool, t: float) -> None:
        cfg = self.cfg
        if self.volume_next is None:
            self.volume_next = t + cfg.volume_repeat_delay
        elif t >= self.volume_next:
            self.volume_next = t + cfg.volume_repeat_interval
        else:
            return
        self._action("volume_up" if up else "volume_down", "volume +" if up else "volume -")

    def _volume_knob(self, roll: float) -> None:
        if self.knob_angle is None:
            self.knob_angle = roll
            return
        diff = (roll - self.knob_angle + 180) % 360 - 180
        step = self.cfg.knob_degrees_per_step
        while abs(diff) >= step:
            up = diff > 0
            self._action("volume_up" if up else "volume_down", "volume +" if up else "volume -")
            self.knob_angle += step if up else -step
            diff += -step if up else step

    def _open_palm(self, hs: H.HandState, pose: str, t: float) -> None:
        cfg = self.cfg
        if pose != H.OPEN:
            self.swipe_history.clear()
            self.palm_hold.update(False, t, 1)
            return

        ax, ay = hs.anchor
        self.swipe_history.append((t, ax, ay))
        while self.swipe_history and t - self.swipe_history[0][0] > cfg.swipe_window:
            self.swipe_history.popleft()

        if t >= self.swipe_until and len(self.swipe_history) >= 3:
            _, sx, sy = self.swipe_history[0]
            dx, dy = ax - sx, ay - sy
            name = None
            if abs(dx) > cfg.swipe_min_distance and abs(dx) > 2 * abs(dy):
                name = "swipe_right" if dx > 0 else "swipe_left"
            elif abs(dy) > cfg.swipe_min_distance * 0.8 and abs(dy) > 2 * abs(dx):
                name = "swipe_down" if dy > 0 else "swipe_up"
            if name:
                self._action(cfg.actions.get(name, "none"), f"{name} -> {cfg.actions.get(name)}")
                self.swipe_until = t + cfg.swipe_cooldown
                self.swipe_history.clear()
                self.palm_hold.update(False, t, 1)
                return

        # mano aperta ferma -> play/pausa
        if self.palm_hold_origin is None or self.palm_hold.start is None:
            self.palm_hold_origin = (ax, ay)
        moved = abs(ax - self.palm_hold_origin[0]) + abs(ay - self.palm_hold_origin[1])
        if moved > 0.05:
            self.palm_hold.update(False, t, 1)
            self.palm_hold_origin = (ax, ay)
        action = cfg.actions.get("palm_hold", "none")
        if self.palm_hold.update(t >= self.swipe_until, t, cfg.hold_play_pause_time):
            self._action(action)
        self._show_hold(action, self.palm_hold, t, cfg.hold_play_pause_time)
