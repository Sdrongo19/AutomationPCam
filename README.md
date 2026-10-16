# Gesture Control

Controlla il PC (macOS e Windows) con i gesti della mano davanti alla webcam.
MediaPipe riconosce 21 punti della mano, il programma li trasforma in mouse, scroll,
scorciatoie di sistema, volume e una tastiera virtuale.

## Avvio rapido

| | macOS | Windows |
|---|---|---|
| Da sorgente | `./run_mac.sh` | `run_windows.bat` |
| Crea app | `./build_mac.sh` → `dist/GestureControl.app` | `build_windows.bat` → `dist\GestureControl.exe` |

Richiede Python 3.9–3.12. Il primo avvio crea `.venv` e installa le dipendenze.
L'app per Windows va compilata **su Windows** (PyInstaller non fa cross-compilazione).

Opzioni: `--camera 1`, `--no-preview`, `--calibrate`, `--hand right|left|any`, `--monitor all|primary|0|1`.

### Permessi macOS
Impostazioni di Sistema → Privacy e sicurezza:
- **Fotocamera**: Terminale (o GestureControl.app)
- **Accessibilità**: necessaria per muovere mouse e premere tasti
- **Monitoraggio input**: solo per le scorciatoie globali (Ctrl+Alt+G / Ctrl+Alt+Q)

Dopo aver concesso un permesso, riavvia il programma.

## Gesti

| Gesto | Azione |
|---|---|
| ☝️ Solo indice alzato, muovi la mano | Muovi il cursore |
| ✊ Pugno | Frizione: il cursore resta fermo, riposiziona la mano |
| 👌 Pinch pollice-indice rapido | Click sinistro (due pinch veloci = doppio click). Avvicinando il pollice il cursore si ferma per mirare |
| 👌 Pinch tenuto 0.45 s, poi muovi | **Trascina** (selezione testo, spostare finestre/file). Barra "tieni per trascinare" |
| Pinch pollice-medio | Click destro |
| ✌️ Indice + medio, muovi su/giù | Scroll |
| 🖐 Mano aperta, swipe ← → | Cambia desktop |
| 🖐 Mano aperta, swipe ↑ / ↓ | Mission Control / Task View · Mostra desktop |
| **T con due mani** (una orizzontale sopra, l'altra verticale sotto il palmo) 0.5 s | Play/pausa musica/video |
| 👍 / 👎 tenuto | Volume su / giù (ripete) |
| Pinch pollice-mignolo + ruota il polso | Volume a manopola (orario = su) |
| 🤜🤛 **Scontra i pugni** (da lontani a nocche a contatto) | **Guanto 3D di Iron Man Mark 85** on/off con onda d'urto: le 253 placche del modello arrivano in volo e si montano dal polso alle dita (o si smontano). Metallo PBR che riflette la stanza, dita che seguono le tue falangi, repulsore acceso quando il palmo guarda la camera (mano aperta = carico). Tasto `i` |
| 🤘 Indice + mignolo tenuto 1 s | Pausa / riattiva il controllo |
| 🤙 Pollice + mignolo tenuto 1 s | Apri / chiudi tastiera virtuale |

Il cursore funziona come un mouse: spostare la mano lungo tutta la zona attiva (rettangolo
azzurro nell'anteprima) attraversa tutti gli schermi; usa il pugno per "sollevare il mouse".

**Tastiera virtuale**: appare sull'anteprima. Punta con la pinza pollice-indice (il cerchio
giallo) e chiudila per premere il tasto. Il testo va nell'app che ha il focus: non cliccare
sulla finestra dell'anteprima.

### Tasti nell'anteprima
`q` esci · `i` Iron Man · `g` gira guanto (palmo/dorso invertiti) · `p` pausa · `c` calibra · `n` cambia webcam · `k` tastiera · `h` aiuto · `Esc` annulla calibrazione

## Calibrazione
Parte da sola al primo uso di ogni webcam (o con `c`). Alza solo l'indice, porta la mano nei
4 angoli che ti sono comodi e tienila ferma 1 secondo ciascuno. Il risultato viene salvato per
quella webcam, quindi integrata ed esterna hanno ognuna la propria zona.

## Configurazione
`config.json` (accanto a `main.py`; per l'app: `~/Library/Application Support/GestureControl/`
su Mac, `%APPDATA%\GestureControl\` su Windows). Scrivi solo le chiavi che vuoi cambiare:

```json
{
  "volume_mode": "both",
  "monitor": "all",
  "scroll_speed": 2500,
  "scroll_natural": true,
  "smoothing_min_cutoff": 1.0,
  "smoothing_beta": 15.0,
  "actions": {
    "swipe_left": "desktop_right",
    "swipe_right": "desktop_left",
    "swipe_up": "overview",
    "swipe_down": "hotkey:cmd+shift+4",
    "timeout_t": "play_pause",
    "palm_hold": "none"
  }
}
```

- `timeout_t`: azione del gesto T; `palm_hold`: mano aperta ferma 1 s (spento di default)
- `hand`: `any` = controlla la prima mano vista, l'altra serve solo per la T
- `volume_mode`: `thumbs` (👍/👎), `knob` (manopola), `both`
- Azioni: `desktop_left`, `desktop_right`, `overview`, `app_windows`, `show_desktop`,
  `play_pause`, `next_track`, `prev_track`, `volume_up`, `volume_down`, `mute`, `none`,
  oppure `hotkey:tasto+tasto` (`cmd` = ⌘ su Mac, tasto Windows su PC)
- Cursore che trema → abbassa `smoothing_min_cutoff`; cursore in ritardo → alza `smoothing_beta`
- Scontro pugni non riconosciuto → alza `bump_contact` o abbassa `bump_approach` (la distanza tra i pugni si vede in debug, tasto `d`); scatta da solo → il contrario
- Trascinamenti per sbaglio → alza `drag_hold_time`; cursore che si blocca troppo presto
  avvicinando il pollice → abbassa `aim_zone`
- Click che scattano da soli → abbassa `pinch_on`; click difficili → alzalo. Le barre in alto
  a destra nell'anteprima (tasto `d`) mostrano la distanza pollice-dito live: deve scendere
  sotto la linea verde (`pinch_on`) e risalire oltre la rossa (`pinch_off`) per rilasciare
- Click destro che diventa sinistro → alza `pinch_margin` (quanto il dito scelto deve essere
  più vicino al pollice rispetto all'altro)
- Click falsi con la mano chiusa (pollice appoggiato sul medio) → alza `pinch_pip_ratio` o
  `pinch_min_reach`; pinch veri ignorati → abbassali. Nel debug, le righe "nocca/punta" e
  "portata" sono verdi quando il pinch è considerato vero
- Tutte le chiavi e i default: `gesture_control/config.py`

## Struttura
```
main.py                      avvio
gesture_control/app.py       webcam, MediaPipe, finestra, tasti
gesture_control/hand.py      landmark -> dita, pose, pinch
gesture_control/controller.py  macchina a stati gesti -> azioni
gesture_control/platform_io.py mouse/tastiera/monitor per macOS (Quartz) e Windows
gesture_control/virtual_keyboard.py, calibration.py, overlay.py, filters.py
assets/                     modelli MediaPipe (mano, viso)
gesture_control/ironman.py   effetto armatura + scontro pugni
```

## Guanto 3D

`assets/gauntlet.npz` è generato da `tools/build_gauntlet.py` a partire dal modello GLB:
ritaglia mano e avambraccio, trova le 21 articolazioni facendo guardare il modello a MediaPipe
(le immagini di controllo finiscono in `build/`) e calcola lo skinning. Per rigenerarlo:

```
.venv/bin/pip install -r tools/requirements-build.txt
.venv/bin/python tools/build_gauntlet.py percorso/iron_man_mark_85.glb
```

Se OpenGL o il file mancano, viene usata l'armatura semplificata disegnata in 2D.

### Crediti
Modello 3D: ["Iron Man Mark 85"](https://sketchfab.com/3d-models/iron-man-mark-85-8da781aa74024366844b36444c650d69)
di LLIypuk, licenza [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Ritagliato e
riadattato (solo mano e avambraccio, scheletro aggiunto). Iron Man è un personaggio Marvel:
progetto personale, non commerciale.
