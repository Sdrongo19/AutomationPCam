#!/bin/bash
# Crea dist/GestureControl.app
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
.venv/bin/pip install -q pyinstaller && .venv/bin/pyinstaller --noconfirm gesture_control.spec
