"""Input di sistema (mouse, tastiera, media) e geometria monitor per macOS e Windows."""
from __future__ import annotations

import math
import sys
import time
from typing import List, Tuple

from pynput import keyboard as pkb
from pynput import mouse as pms

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

if IS_WIN:
    import ctypes
    from ctypes import wintypes

    # Coordinate in pixel fisici, coerenti tra monitor con scaling diverso
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

if IS_MAC:
    import Quartz

Rect = Tuple[float, float, float, float]  # x, y, larghezza, altezza


# ---------------------------------------------------------------- monitor

def list_displays() -> Tuple[List[Rect], int]:
    """Ritorna (rettangoli dei monitor in coordinate globali, indice del principale)."""
    if IS_MAC:
        err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
        main = Quartz.CGMainDisplayID()
        rects, primary = [], 0
        for i, did in enumerate(ids[:count]):
            b = Quartz.CGDisplayBounds(did)
            rects.append((b.origin.x, b.origin.y, b.size.width, b.size.height))
            if did == main:
                primary = i
        return rects, primary
    if IS_WIN:
        rects: List[Rect] = []
        proc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
        )

        def _cb(_hmon, _hdc, r, _lp):
            rc = r.contents
            rects.append((rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top))
            return 1

        ctypes.windll.user32.EnumDisplayMonitors(None, None, proc_type(_cb), 0)
        primary = next((i for i, r in enumerate(rects) if r[0] == 0 and r[1] == 0), 0)
        return rects or [(0, 0, 1920, 1080)], primary
    return [(0, 0, 1920, 1080)], 0


class ScreenSpace:
    """Mappa un punto normalizzato 0..1 sull'area dei monitor scelti."""

    def __init__(self, monitor_setting):
        rects, primary = list_displays()
        if monitor_setting == "primary":
            rects = [rects[primary]]
        elif isinstance(monitor_setting, int) and 0 <= monitor_setting < len(rects):
            rects = [rects[monitor_setting]]
        self.rects = rects
        self.x0 = min(r[0] for r in rects)
        self.y0 = min(r[1] for r in rects)
        self.x1 = max(r[0] + r[2] for r in rects)
        self.y1 = max(r[1] + r[3] for r in rects)

    def to_screen(self, nx: float, ny: float) -> Tuple[float, float]:
        x = self.x0 + nx * (self.x1 - self.x0)
        y = self.y0 + ny * (self.y1 - self.y0)
        return self._clamp(x, y)

    def _clamp(self, x: float, y: float) -> Tuple[float, float]:
        # Monitor di dimensioni diverse lasciano "buchi" nel rettangolo totale:
        # porta il punto sul monitor più vicino.
        best, best_d = (x, y), math.inf
        for rx, ry, rw, rh in self.rects:
            cx = min(max(x, rx), rx + rw - 1)
            cy = min(max(y, ry), ry + rh - 1)
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best, best_d = (cx, cy), d
                if d == 0:
                    break
        return best


# ---------------------------------------------------------------- mouse

class Mouse:
    LEFT, RIGHT = "left", "right"

    def __init__(self, double_click_time: float = 0.4):
        self.double_click_time = double_click_time
        self.pos = (0.0, 0.0)
        self.down_buttons = set()
        self._scroll_rest = 0.0
        self._last_down = (0.0, None, (0.0, 0.0))
        self._click_count = 1
        if not IS_MAC:
            self._pm = pms.Controller()

    # Su Windows il doppio click lo riconosce il sistema dai tempi; su macOS
    # serve impostare esplicitamente il contatore di click sull'evento.
    def _next_click_count(self, button: str) -> int:
        t, last_button, last_pos = self._last_down
        now = time.monotonic()
        near = math.dist(last_pos, self.pos) < 12
        self._click_count = self._click_count + 1 if (
            last_button == button and now - t < self.double_click_time and near
        ) else 1
        self._last_down = (now, button, self.pos)
        return self._click_count

    if IS_MAC:
        _TYPES = {
            "left": (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp,
                     Quartz.kCGEventLeftMouseDragged, Quartz.kCGMouseButtonLeft),
            "right": (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp,
                      Quartz.kCGEventRightMouseDragged, Quartz.kCGMouseButtonRight),
        }

        def _post(self, etype, button, clicks=None):
            ev = Quartz.CGEventCreateMouseEvent(None, etype, self.pos, button)
            if clicks is not None:
                Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, clicks)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)

        def move(self, x: float, y: float) -> None:
            self.pos = (x, y)
            if "left" in self.down_buttons:
                self._post(self._TYPES["left"][2], Quartz.kCGMouseButtonLeft)
            elif "right" in self.down_buttons:
                self._post(self._TYPES["right"][2], Quartz.kCGMouseButtonRight)
            else:
                self._post(Quartz.kCGEventMouseMoved, Quartz.kCGMouseButtonLeft)

        def down(self, button: str) -> None:
            d, _, _, b = self._TYPES[button]
            self._post(d, b, self._next_click_count(button))
            self.down_buttons.add(button)

        def up(self, button: str) -> None:
            _, u, _, b = self._TYPES[button]
            self._post(u, b, self._click_count)
            self.down_buttons.discard(button)

        def scroll(self, pixels: float) -> None:
            self._scroll_rest += pixels
            step = int(self._scroll_rest)
            if step:
                self._scroll_rest -= step
                ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 1, step)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
    else:
        _PIXELS_PER_NOTCH = 60.0

        def move(self, x: float, y: float) -> None:
            self.pos = (x, y)
            self._pm.position = (int(x), int(y))

        def down(self, button: str) -> None:
            self._next_click_count(button)
            self._pm.press(getattr(pms.Button, button))
            self.down_buttons.add(button)

        def up(self, button: str) -> None:
            self._pm.release(getattr(pms.Button, button))
            self.down_buttons.discard(button)

        def scroll(self, pixels: float) -> None:
            self._scroll_rest += pixels / self._PIXELS_PER_NOTCH
            step = int(self._scroll_rest)
            if step:
                self._scroll_rest -= step
                self._pm.scroll(0, step)

    def click(self, button: str) -> None:
        self.down(button)
        self.up(button)

    def release_all(self) -> None:
        for b in list(self.down_buttons):
            self.up(b)


# ---------------------------------------------------------------- tastiera

_MAC_KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "enter": 36, "l": 37,
    "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46,
    ".": 47, "tab": 48, "space": 49, "`": 50, "backspace": 51, "esc": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111, "home": 115, "pageup": 116,
    "delete": 117, "end": 119, "pagedown": 121, "left": 123, "right": 124, "down": 125, "up": 126,
}
_MAC_FN_KEYS = {"left", "right", "up", "down", "home", "end", "pageup", "pagedown", "delete"}

_ALIASES = {"option": "alt", "win": "cmd", "super": "cmd", "command": "cmd", "control": "ctrl",
            "return": "enter", "escape": "esc"}
_MODIFIERS = ("cmd", "ctrl", "alt", "shift")

if IS_MAC:
    _MAC_FLAGS = {
        "cmd": Quartz.kCGEventFlagMaskCommand,
        "ctrl": Quartz.kCGEventFlagMaskControl,
        "alt": Quartz.kCGEventFlagMaskAlternate,
        "shift": Quartz.kCGEventFlagMaskShift,
    }

SYSTEM_ACTIONS = {
    "darwin": {
        "desktop_left": "ctrl+left",
        "desktop_right": "ctrl+right",
        "overview": "ctrl+up",
        "app_windows": "ctrl+down",
        "show_desktop": "f11",
    },
    "win32": {
        "desktop_left": "ctrl+cmd+left",
        "desktop_right": "ctrl+cmd+right",
        "overview": "cmd+tab",
        "app_windows": "alt+tab",
        "show_desktop": "cmd+d",
    },
}
MEDIA_ACTIONS = {
    "play_pause": pkb.Key.media_play_pause,
    "next_track": pkb.Key.media_next,
    "prev_track": pkb.Key.media_previous,
    "volume_up": pkb.Key.media_volume_up,
    "volume_down": pkb.Key.media_volume_down,
    "mute": pkb.Key.media_volume_mute,
}


def _pynput_key(name: str):
    if hasattr(pkb.Key, name):
        return getattr(pkb.Key, name)
    if name == "pageup":
        return pkb.Key.page_up
    if name == "pagedown":
        return pkb.Key.page_down
    return pkb.KeyCode.from_char(name)


class Keyboard:
    def __init__(self):
        self._kb = pkb.Controller()

    def hotkey(self, combo: str) -> None:
        keys = [_ALIASES.get(k.strip().lower(), k.strip().lower()) for k in combo.split("+") if k.strip()]
        mods = [k for k in keys if k in _MODIFIERS]
        main = [k for k in keys if k not in _MODIFIERS]
        if IS_MAC and len(main) == 1 and main[0] in _MAC_KEYCODES:
            # Evento Quartz con flag: le scorciatoie di sistema (Spaces, Mission Control)
            # richiedono anche il flag Fn sulle frecce.
            flags = 0
            for m in mods:
                flags |= _MAC_FLAGS[m]
            if main[0] in _MAC_FN_KEYS:
                flags |= Quartz.kCGEventFlagMaskSecondaryFn | Quartz.kCGEventFlagMaskNumericPad
            code = _MAC_KEYCODES[main[0]]
            for is_down in (True, False):
                ev = Quartz.CGEventCreateKeyboardEvent(None, code, is_down)
                Quartz.CGEventSetFlags(ev, flags)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return
        pressed = [_pynput_key(k) for k in mods + main]
        for k in pressed:
            self._kb.press(k)
        for k in reversed(pressed):
            self._kb.release(k)

    def tap(self, key) -> None:
        self._kb.press(key)
        self._kb.release(key)

    def type_text(self, text: str) -> None:
        self._kb.type(text)

    def run_action(self, action: str) -> None:
        if not action or action == "none":
            return
        if action.startswith("hotkey:"):
            self.hotkey(action[len("hotkey:"):])
        elif action in MEDIA_ACTIONS:
            self.tap(MEDIA_ACTIONS[action])
        elif action in SYSTEM_ACTIONS.get(sys.platform, {}):
            self.hotkey(SYSTEM_ACTIONS[sys.platform][action])
        else:
            print(f"[azione] sconosciuta: {action}")


# ---------------------------------------------------------------- permessi

def check_permissions() -> None:
    """Su macOS chiede il permesso Accessibilità (serve per muovere mouse e tastiera)."""
    if not IS_MAC:
        return
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
        if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
            print("\n[!] Concedi il permesso 'Accessibilità' all'app/terminale in\n"
                  "    Impostazioni di Sistema > Privacy e sicurezza > Accessibilità,\n"
                  "    poi riavvia il programma.\n")
    except Exception as e:
        print(f"[permessi] verifica non riuscita: {e}")
