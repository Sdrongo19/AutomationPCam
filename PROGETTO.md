# Gesture Control — stato del progetto

> Documento di passaggio di consegne per riprendere il lavoro in una nuova chat AI.
> **Va aggiornato ogni volta che si sviluppa qualcosa** (sezioni "Stato", "Problemi aperti",
> "Registro modifiche"). Ultimo aggiornamento: 2026-10-16.

## 1. Obiettivo

Controllare il PC (macOS e Windows) con i gesti della mano davanti alla webcam: mouse completo,
scroll, scorciatoie di sistema, media/volume, tastiera virtuale. In più un effetto "Iron Man":
scontrando i pugni compare sulle mani il guanto 3D del Mark 85, disegnato in tempo reale.

Utente: sviluppa su Mac (Apple M4, macOS, 2 monitor 1920×1080 affiancati), parla italiano,
vuole risultati realistici e gesti che non si attivino per sbaglio. Preferisce che si proceda
senza troppe domande e che i problemi vengano verificati, non dati per risolti.

## 2. Come si avvia

| | macOS | Windows |
|---|---|---|
| Da sorgente | `./run_mac.sh` | `run_windows.bat` |
| App impacchettata | `./build_mac.sh` → `dist/GestureControl.app` | `build_windows.bat` → `dist\GestureControl.exe` (va compilato su Windows) |

- Python di sistema 3.9.6 (`/usr/bin/python3`), ambiente in `.venv/`. Il codice deve restare
  compatibile con Python 3.9 (niente `match`, niente `X | Y` nei tipi a runtime).
- Dipendenze: `requirements.txt` (opencv-python, mediapipe 1.1.1, pynput, moderngl).
  Solo per rigenerare il guanto: `tools/requirements-build.txt` (trimesh, pygltflib, scipy, networkx).
- Permessi macOS: Fotocamera, Accessibilità (mouse/tastiera), Monitoraggio input (hotkey globali).
- Opzioni: `--camera N`, `--no-preview`, `--calibrate`, `--hand right|left|any`, `--monitor all|primary|N`.
- Tasti nell'anteprima: `q` esci · `p` pausa · `c` calibra · `n` cambia webcam · `k` tastiera ·
  `i` Iron Man on/off · `g` gira guanto (palmo/dorso) · `d` debug · `h` aiuto.
- Hotkey globali: Ctrl+Alt+G pausa, Ctrl+Alt+Q esci.

## 3. Architettura

```
main.py                         avvio -> gesture_control.app.main()
gesture_control/
  app.py            ciclo principale: webcam (specchiata) -> MediaPipe mani (2) + viso -> controller -> disegno
  hand.py           21 landmark -> dita alzate, pose, distanze pinch (3D), HandPicker, gesto T
  controller.py     macchina a stati gesti -> azioni (mouse, scroll, swipe, volume, T, pugni, tastiera)
  platform_io.py    mouse/tastiera/monitor: macOS via Quartz, Windows via pynput + ctypes
  filters.py        filtro One Euro (cursore)
  calibration.py    calibrazione guidata 4 angoli, salvata per webcam
  virtual_keyboard.py  tastiera disegnata sull'anteprima
  overlay.py        anteprima: scheletro, zona attiva, stato, aiuto, barre debug
  ironman.py        effetto Iron Man: stati, onda d'urto, HUD JARVIS, bagliori; fallback 2D procedurale
  gauntlet.py       guanto 3D in OpenGL (moderngl): skinning, PBR, montaggio a placche
  config.py         default + config.json (salva solo i valori diversi dal default)
tools/build_gauntlet.py   prepara assets/gauntlet*.npz dal modello GLB (una tantum)
assets/   hand_landmarker.task, blaze_face_short_range.tflite, gauntlet.npz (destro, gemme),
          gauntlet_left.npz (sinistro)
gesture_control.spec      PyInstaller (include assets, moderngl, glcontext)
```

- MediaPipe 1.1.1 **non ha più `mp.solutions`**: si usa solo la Tasks API
  (`vision.HandLandmarker`, `vision.FaceDetector`, modalità VIDEO con timestamp crescenti).
- Il frame viene specchiato (`cv2.flip`) prima del rilevamento: l'etichetta destra/sinistra di
  MediaPipe presume un'immagine specchiata.
- `config.json` sta accanto a `main.py` da sorgente, in `~/Library/Application Support/GestureControl/`
  (Mac) o `%APPDATA%\GestureControl\` (Windows) per l'app impacchettata.

## 4. Gesti attuali

| Gesto | Azione |
|---|---|
| Solo indice alzato | Muove il cursore (relativo, con frizione) |
| Pugno | Frizione: cursore fermo, si riposiziona la mano |
| Pollice che si avvicina all'indice | Cursore bloccato per mirare |
| Pinch pollice-indice rapido (< 0.45 s) | Click sinistro (due rapidi = doppio click) |
| Pinch pollice-indice tenuto 0.45 s | Tasto premuto → trascinamento |
| Pinch pollice-medio | Click destro |
| Indice + medio su/giù | Scroll |
| Mano aperta, swipe ← → ↑ ↓ | Desktop sinistro/destro, Mission Control/Task View, mostra desktop |
| Pollice su / giù tenuto | Volume +/− |
| Pinch pollice-mignolo + rotazione polso | Volume a manopola |
| Rock (indice+mignolo) 1 s | Pausa/riattiva tutto il tracciamento |
| Telefono (pollice+mignolo) 1 s | Tastiera virtuale |
| T con due mani 0.5 s | Play/pausa media |
| Pugni che si scontrano | Guanto Iron Man on/off |

La mano di controllo è la prima vista (`hand: "any"`); la seconda serve solo per T e pugni.

## 5. Decisioni importanti (e perché)

**Mouse**
- Cursore *relativo* con guadagno = 1/larghezza zona attiva: spostare la mano per tutta la zona
  calibrata attraversa tutti gli schermi. Il filtro One Euro lavora in coordinate normalizzate
  (`smoothing_beta` 15: con valori bassi il cursore restava indietro di ~100 px).
- Ancora del cursore = centro del palmo (polso + 4 nocche), non la punta dell'indice: così il pinch
  non sposta il cursore.
- Click e trascinamento sono **gesti separati**: prima il pinch premeva subito il tasto e il
  tremolio diventava un trascinamento (selezioni e finestre spostate per sbaglio).
- Pinch: distanze **3D** normalizzate (reggono la mano inclinata). Sinistro = l'indice è il dito
  più vicino al pollice, senza margine (nel pinch il medio piegato sta accanto). Destro = controlli
  severi (`pinch_margin`, `pinch_pip_ratio`, `pinch_min_reach`): con l'indice puntato a mano chiusa
  il pollice appoggiato sul medio dava falsi click destri.
- Il blocco del click destro dura solo finché il pinch destro è in corso (prima restava attivo per
  sempre e bloccava click sinistro e cursore).
- macOS: mouse tramite Quartz con `kCGMouseEventClickState` (necessario per il doppio click) ed
  eventi "dragged" durante il trascinamento. Le scorciatoie con le frecce hanno i flag Fn/NumPad
  (servono a Spaces/Mission Control).

**Config**
- `config.save()` scrive solo i valori diversi dal default: in una prima versione salvava tutto e
  i default nuovi del codice venivano ignorati.

**Iron Man / guanto 3D**
- Modello: "Iron Man Mark 85" di LLIypuk (Sketchfab, CC BY 4.0, GLB da 100 MB, 2.2 M triangoli,
  senza scheletro, mesh divise per materiale). Il GLB originale non è nel repo: serve solo per
  rigenerare gli `.npz`.
- `tools/build_gauntlet.py`: ritaglia mano+avambraccio (destro con le 6 gemme, sinistro senza; il
  sinistro viene salvato specchiato come una "destra"). Le articolazioni le trova **MediaPipe
  guardando il render del guanto** (vista dal palmo, mediana di 20 varianti ruotate/con margine); la
  profondità viene dallo z-buffer (vista palmo + dorso). Il punto 5 (nocca indice) e il polso vengono
  riportati sul piano del palmo perché il pollice piegato davanti falsa la profondità.
- Il nome del materiale va letto **prima** di `submesh`, altrimenti le gemme (mesh senza UV) perdono
  il materiale e diventano metallo scuro.
- Runtime (`gauntlet.py`): il palmo si allinea con Umeyama (similitudine) sui punti del palmo; la
  scala si ricalcola sulle sole distanze 2D (la z di MediaPipe è inaffidabile); ogni falange ruota
  verso il dito osservato e si allunga/accorcia (0.7–1.45). Skinning a 2 ossa nello shader.
- PBR GGX metallico, riflessi presi dall'immagine della webcam (mipmap), tonemapping ACES, MSAA 4×,
  output premoltiplicato, ombra sulla pelle, avambraccio tagliato a 10 cm e sfumato.
- **Lato del guanto (palmo/dorso)**: si usa l'etichetta di MediaPipe. Con la webcam live risultava
  invertita rispetto alle foto di prova → `gauntlet_flip_side: true` di default; il tasto `g` lo
  inverte e lo salva. Provato anche un criterio geometrico (adattamento normale vs specchiato): sbagliava
  di più, scartato.
- Mano destra dell'utente → guanto con le gemme (specchiato); mano sinistra → guanto liscio.
  Il repulsore è il materiale `material_3`; gli altri materiali luminosi sono gemme.
- Montaggio: ogni componente connessa del modello è una placca (253 destro, 162 sinistro) che arriva
  in volo da 7–13 cm, in ordine avambraccio → palmo → falangi; smontaggio al contrario.
- Pugni: lo scontro conta l'**avvicinamento** (≥ `bump_approach` in `bump_window` s) fino al contatto
  (`bump_contact`), ignora singoli frame imperfetti e accetta la mano persa al contatto. La prima
  versione richiedeva una distanza di partenza assoluta e non riconosceva il secondo scontro.
- numpy 2.0 su macOS dà falsi `RuntimeWarning` nel prodotto matrice-vettore: in `gauntlet.py` si usa
  una somma esplicita.
- Prima del guanto 3D c'era un'armatura procedurale 2D (capsule + shading): resta come fallback se
  OpenGL o gli `.npz` mancano. L'utente l'ha giudicata poco realistica.

## 6. Come si è verificato finora

- Nessun test automatico nel repo. Le verifiche sono state fatte con script temporanei (fuori dal
  progetto) che simulano mani finte e un mouse/tastiera finti sul `Controller`, e con render del
  guanto su foto di esempio di MediaPipe (`woman_hands.jpg`, `victory.jpg`, `thumb_up.jpg`).
- **Mai testato su Windows.**
- Test reali li fa l'utente con la webcam e riferisce cosa non va.

## 7. Stato e problemi aperti

- [ ] Verificare dal vivo che il secondo scontro di pugni spenga l'armatura (logica corretta il
      2026-10-16, provata solo in simulazione).
- [ ] Verificare dal vivo le gemme sulla mano destra e il guanto liscio sulla sinistra.
- [ ] Il guanto può risultare più grosso della mano in alcune pose (es. pollice in su).
- [ ] Le scorciatoie di sistema su macOS (cambio desktop, Mission Control) sono inviate come eventi
      sintetici: non confermato che macOS le accetti.
- [ ] Soglie di pinch tarate su mani simulate e sui commenti dell'utente: la modalità debug (`d`)
      mostra i valori live per tararle.
- [ ] Windows: tutto da provare (mouse via pynput, DPI, monitor, `CAP_DSHOW`).
- [ ] App impacchettata da ricompilare dopo le ultime modifiche (`./build_mac.sh`, ~300 MB).

Idee proposte e non fatte: colpo di repulsore (mano aperta spinta verso la camera), suono
all'accensione dell'armatura, effetto in finestra a schermo intero o come webcam virtuale per le
videochiamate, profili di gesti per singola app, gesti registrabili dall'utente.

## 8. Registro modifiche

- **2026-10-16** — Prima versione: mouse, click, scroll, swipe, volume, pausa rock, tastiera
  virtuale, multi-monitor, calibrazione, config JSON, packaging PyInstaller.
- **2026-10-16** — Click: distanze 3D, scelta del dito più vicino, conferma su 2 frame, controlli
  anti-pollice-appoggiato; corretto blocco permanente dopo click destro; config salvata solo come
  differenze.
- **2026-10-16** — Seconda mano tracciata; gesto T = play/pausa (la mano aperta ferma 1 s non fa
  più nulla); rock resta pausa del tracciamento.
- **2026-10-16** — Click e trascinamento separati (pinch tenuto 0.45 s), cursore fermo mentre si
  mira e dopo il click.
- **2026-10-16** — Effetto Iron Man: prima col gesto "occhiali" (riconoscimento viso), poi
  sostituito dallo scontro dei pugni; armatura 2D procedurale (poi solo fallback).
- **2026-10-16** — Guanto 3D Mark 85 in OpenGL con skinning, PBR, montaggio a placche; lato del
  palmo invertito per la webcam live + tasto `g`.
- **2026-10-16** — Due guanti come nel modello (gemme sulla destra); scontro pugni più robusto;
  distanza dei pugni nel debug.
