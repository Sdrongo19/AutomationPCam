# PyInstaller: genera GestureControl.app (macOS) o GestureControl.exe (Windows).
# Uso:  pyinstaller gesture_control.spec
import sys
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("mediapipe")
datas += [("assets/hand_landmarker.task", "assets"), ("assets/blaze_face_short_range.tflite", "assets"),
          ("assets/gauntlet.npz", "assets"), ("assets/gauntlet_left.npz", "assets")]
hiddenimports += ["moderngl", "glcontext"]
hiddenimports += ["pynput.keyboard._darwin", "pynput.mouse._darwin"] if sys.platform == "darwin" \
    else ["pynput.keyboard._win32", "pynput.mouse._win32"]

a = Analysis(["main.py"], datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["tkinter", "IPython", "pandas", "scipy"])
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="GestureControl", console=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="GestureControl")
    app = BUNDLE(
        coll,
        name="GestureControl.app",
        bundle_identifier="dev.managify.gesturecontrol",
        info_plist={
            "NSCameraUsageDescription": "La webcam serve a riconoscere i gesti della mano.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, name="GestureControl", console=False)
