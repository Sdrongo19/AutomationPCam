#!/bin/bash
# Avvio su macOS: crea l'ambiente al primo avvio, poi lancia il programma.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt || exit 1
fi
exec .venv/bin/python main.py "$@"
