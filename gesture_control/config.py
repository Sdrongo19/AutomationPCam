"""Configurazione: valori di default + override da config.json utente."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

APP_NAME = "GestureControl"

DEFAULTS = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "show_preview": True,
    "preview_always_on_top": True,
    # "right", "left" oppure "any" (la prima vista): quale mano controlla il PC.
    # L'altra mano serve solo per il gesto T.
    "hand": "any",
    # "all" = tutti i monitor, "primary" = solo principale, oppure indice (0, 1, ...)
    "monitor": "all",
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.5,

    # Zona attiva della webcam (coordinate normalizzate 0..1) mappata sullo schermo.
    # Sovrascritta dalla calibrazione, salvata per ogni camera in "calibration".
    "active_region": [0.20, 0.15, 0.80, 0.65],
    "calibration": {},

    # Filtro One Euro: min_cutoff basso = più stabile, beta alto = più reattivo
    "smoothing_min_cutoff": 1.0,
    "smoothing_beta": 15.0,

    # Pinch (distanza punte / dimensione mano), con isteresi
    "pinch_on": 0.30,
    "pinch_off": 0.45,
    # il dito scelto deve essere almeno così più vicino al pollice dell'altro
    "pinch_margin": 1.35,
    "pinch_confirm_frames": 2,
    # click destro vero: pollice-nocca del medio >= ratio x pollice-punta; punta lontana dal polso
    "pinch_pip_ratio": 1.8,
    "pinch_min_reach": 0.95,
    # click sinistro: controllo leggero anti-pugno (pollice appoggiato sull'indice chiuso)
    "left_pip_ratio": 1.3,
    "show_debug": True,
    # pinch tenuto più di così = trascinamento; più breve = click
    "drag_hold_time": 0.45,
    # dopo un click il cursore resta fermo (riaprire le dita non lo sposta)
    "click_freeze_time": 0.2,
    # pollice così vicino all'indice = sto per cliccare, cursore fermo
    "aim_zone": 0.5,
    "double_click_time": 0.40,

    "scroll_speed": 2500,
    "scroll_natural": True,

    "swipe_min_distance": 0.22,
    "swipe_window": 0.30,
    "swipe_cooldown": 0.9,

    "hold_toggle_time": 1.0,
    "hold_play_pause_time": 1.0,

    # "thumbs", "knob" oppure "both"
    "volume_mode": "both",
    "volume_repeat_delay": 0.45,
    "volume_repeat_interval": 0.15,
    "knob_degrees_per_step": 12,

    "global_hotkey": "<ctrl>+<alt>+g",

    # Gesto -> azione. Azioni: desktop_left, desktop_right, overview, show_desktop,
    # app_windows, play_pause, next_track, prev_track, volume_up, volume_down, mute,
    # none, oppure "hotkey:cmd+shift+4" (tasti separati da +).
    "actions": {
        "swipe_left": "desktop_right",
        "swipe_right": "desktop_left",
        "swipe_up": "overview",
        "swipe_down": "show_desktop",
        "palm_hold": "none",
        "timeout_t": "play_pause",
    },
    "timeout_hold_time": 0.5,
    # scontro tra pugni (armatura Iron Man): distanze tra le nocche in "grandezze di mano"
    "bump_contact": 1.2,     # pugni a contatto sotto questa distanza
    "bump_approach": 0.6,    # avvicinamento minimo negli ultimi bump_window secondi
    "bump_window": 0.7,
    "bump_cooldown": 1.0,
    # guanto 3D: lato mostrato (palmo/dorso). Se è invertito premi g nell'anteprima
    "gauntlet_flip_side": True,
}


def user_config_dir() -> Path:
    """Cartella scrivibile per config.json (accanto al progetto se non impacchettato)."""
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parent.parent
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".config"
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(relative: str) -> Path:
    """Percorso di un file incluso (assets), funziona anche dentro PyInstaller."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / relative


class Config(dict):
    def __init__(self, path: Path):
        super().__init__(copy.deepcopy(DEFAULTS))
        self.path = path
        if path.exists():
            try:
                user = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                print(f"[config] {path} non valido ({e}), uso default")
                user = {}
            for key, value in user.items():
                if isinstance(value, dict) and isinstance(self.get(key), dict):
                    self[key].update(value)
                else:
                    self[key] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def save(self) -> None:
        # Solo i valori diversi dal default: i default nuovi del codice restano attivi
        diff = {}
        for key, value in self.items():
            default = DEFAULTS.get(key)
            if isinstance(value, dict) and isinstance(default, dict):
                sub = {k: v for k, v in value.items() if default.get(k) != v}
                if sub:
                    diff[key] = sub
            elif value != default:
                diff[key] = value
        self.path.write_text(json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")

    def region_for_camera(self, index: int) -> list:
        return self["calibration"].get(str(index), self["active_region"])

    def set_region_for_camera(self, index: int, region: list) -> None:
        self["calibration"][str(index)] = [round(v, 4) for v in region]
        self.save()


def load_config() -> Config:
    return Config(user_config_dir() / "config.json")
