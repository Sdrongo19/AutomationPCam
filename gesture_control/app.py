"""Ciclo principale: webcam -> MediaPipe -> controller -> anteprima."""
from __future__ import annotations

import argparse
import sys
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from . import overlay
from .calibration import Calibrator
from .config import load_config, resource_path
from .controller import MODE_CALIBRATE, Controller
from .hand import detect_hands
from .ironman import Face, IronMan
from .platform_io import IS_MAC, IS_WIN, Keyboard, Mouse, ScreenSpace, check_permissions
from .virtual_keyboard import VirtualKeyboard

WINDOW = "Gesture Control"


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_DSHOW if IS_WIN else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10):  # alcune webcam restituiscono frame vuoti all'apertura
        ok, _ = cap.read()
        if ok:
            return cap
        time.sleep(0.05)
    cap.release()
    return None


def make_landmarker(cfg):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resource_path("assets/hand_landmarker.task"))),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=cfg.min_detection_confidence,
        min_hand_presence_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def make_face_detector():
    path = resource_path("assets/blaze_face_short_range.tflite")
    if not path.exists():
        print("[viso] modello assente: gesto occhiali disattivato")
        return None
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def start_hotkeys(cfg, events: dict) -> None:
    from pynput import keyboard

    bindings = {}
    if cfg.global_hotkey:
        bindings[cfg.global_hotkey] = events["toggle"].set
    bindings["<ctrl>+<alt>+q"] = events["quit"].set
    try:
        listener = keyboard.GlobalHotKeys(bindings)
        listener.daemon = True
        listener.start()
    except Exception as e:
        print(f"[hotkey] scorciatoie globali non disponibili: {e}")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Controlla il PC con i gesti della mano.")
    p.add_argument("--camera", type=int, help="indice webcam (0, 1, ...)")
    p.add_argument("--no-preview", action="store_true", help="nessuna finestra anteprima")
    p.add_argument("--calibrate", action="store_true", help="avvia la calibrazione guidata")
    p.add_argument("--hand", choices=["any", "right", "left"])
    p.add_argument("--monitor", help="all, primary oppure indice monitor")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config()
    if args.camera is not None:
        cfg["camera_index"] = args.camera
    if args.no_preview:
        cfg["show_preview"] = False
    if args.hand:
        cfg["hand"] = args.hand
    if args.monitor:
        cfg["monitor"] = int(args.monitor) if args.monitor.isdigit() else args.monitor
    print(f"[config] {cfg.path}")

    check_permissions()
    screen = ScreenSpace(cfg.monitor)
    print(f"[schermo] area {screen.x0:.0f},{screen.y0:.0f} -> {screen.x1:.0f},{screen.y1:.0f} "
          f"({len(screen.rects)} monitor)")
    keyboard = Keyboard()
    ctrl = Controller(cfg, screen, Mouse(cfg.double_click_time), keyboard)
    ctrl.virtual_keyboard = VirtualKeyboard(keyboard, cfg)
    ctrl.calibrator = calibrator = Calibrator(ctrl, cfg)

    cam_index = cfg.camera_index
    cap = open_camera(cam_index, cfg.camera_width, cfg.camera_height)
    if cap is None:
        print(f"[camera] impossibile aprire la webcam {cam_index}. "
              "Controlla il permesso Fotocamera o prova --camera 1")
        return 1
    ctrl.set_region(cfg.region_for_camera(cam_index))

    preview = cfg.show_preview
    if preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 720, 405)
        if cfg.preview_always_on_top:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
        if args.calibrate or str(cam_index) not in cfg.calibration:
            calibrator.start(cam_index)
    elif args.calibrate:
        print("[calibrazione] richiede l'anteprima, ignoro --calibrate")

    events = {"toggle": threading.Event(), "quit": threading.Event()}
    start_hotkeys(cfg, events)

    landmarker = make_landmarker(cfg)
    face_detector = make_face_detector()
    ironman = IronMan(resource_path("assets/gauntlet.npz"), cfg.gauntlet_flip_side)
    ironman.load()  # subito, non al primo scontro di pugni (OpenGL + 9 MB di modello)
    ctrl.ironman = ironman
    frame_no = 0
    face_seen = 0.0
    t0 = time.monotonic()
    last_ts = -1
    fps, last_frame_t = 0.0, time.monotonic()
    show_help = True
    failures = 0

    print("[avvio] pronto. 'q' nell'anteprima o Ctrl+Alt+Q per uscire.")
    try:
        while not events["quit"].is_set():
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    print("[camera] nessun frame dalla webcam, esco")
                    break
                time.sleep(0.02)
                continue
            failures = 0
            frame = cv2.flip(frame, 1)  # effetto specchio: destra è destra

            now = time.monotonic()
            ts = max(int((now - t0) * 1000), last_ts + 1)
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, ts)
            frame_no += 1
            if face_detector is not None and frame_no % 2 == 0:  # il viso si muove poco: 1 frame su 2
                faces = face_detector.detect_for_video(image, ts).detections
                if faces:
                    ctrl.face = Face.from_detection(faces[0], frame.shape[1], frame.shape[0])
                    face_seen = now
                elif now - face_seen > 1.0:  # la mano sugli occhi può nascondere il viso per un attimo
                    ctrl.face = None
            hands = detect_hands(result)
            ctrl.aspect = frame.shape[1] / frame.shape[0]

            if events["toggle"].is_set():
                events["toggle"].clear()
                ctrl.toggle_enabled()
            ctrl.update(hands, now)

            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt

            if not preview:
                continue
            ironman.draw(frame, hands, ctrl.main_hand, ctrl.face, now)
            overlay.draw(frame, ctrl, ctrl.main_hand, fps, now, show_help, cam_index)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == 27 and ctrl.mode == MODE_CALIBRATE:
                calibrator.cancel()
            elif key == ord("p"):
                ctrl.toggle_enabled()
            elif key == ord("i"):
                ironman.toggle(now)
            elif key == ord("g") and ironman.gpu is not None:
                # palmo e dorso invertiti: gira il guanto e ricorda la scelta
                ironman.gpu.flip_side = not ironman.gpu.flip_side
                cfg["gauntlet_flip_side"] = ironman.gpu.flip_side
                cfg.save()
            elif key == ord("d"):
                cfg["show_debug"] = not cfg.show_debug
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("c"):
                calibrator.start(cam_index)
            elif key == ord("k"):
                ctrl.toggle_keyboard()
            elif key == ord("n"):
                cap, cam_index = switch_camera(cap, cam_index, cfg)
                ctrl.set_region(cfg.region_for_camera(cam_index))
                if str(cam_index) not in cfg.calibration:
                    calibrator.start(cam_index)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.mouse.release_all()
        cap.release()
        landmarker.close()
        if face_detector is not None:
            face_detector.close()
        cv2.destroyAllWindows()
    return 0


def switch_camera(cap, index: int, cfg):
    cap.release()
    for candidate in list(range(index + 1, index + 5)) + list(range(0, index + 1)):
        new = open_camera(candidate, cfg.camera_width, cfg.camera_height)
        if new is not None:
            print(f"[camera] uso webcam {candidate}")
            cfg["camera_index"] = candidate
            cfg.save()
            return new, candidate
    raise RuntimeError("nessuna webcam disponibile")
