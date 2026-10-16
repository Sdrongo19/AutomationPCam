"""Armatura di Iron Man sulla mano (solo anteprima), messa e tolta scontrando i pugni.

Ogni pezzo è un volume (capsula, sfera o placca bombata) illuminato per pixel con luce
diffusa e riflesso speculare metallico, con z-buffer sulla profondità di MediaPipe:
le dita si coprono a vicenda in modo corretto. L'armatura proietta un'ombra sulla pelle e
si adatta alla luminosità della scena.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

HUD = (255, 220, 90)
CYAN = (255, 235, 170)

# Materiali (BGR 0..1)
RED = dict(base=np.array([0.06, 0.08, 0.60], np.float32), spec=np.array([0.85, 0.88, 1.0], np.float32),
           ks=0.85, shin=28.0, amb=0.16, refl=0.30)
RED_DEEP = dict(base=np.array([0.04, 0.05, 0.42], np.float32), spec=np.array([0.8, 0.8, 1.0], np.float32),
                ks=0.7, shin=24.0, amb=0.14, refl=0.25)
GOLD = dict(base=np.array([0.20, 0.58, 0.86], np.float32), spec=np.array([0.65, 0.9, 1.0], np.float32),
            ks=1.0, shin=22.0, amb=0.18, refl=0.45)
STEEL = dict(base=np.array([0.20, 0.19, 0.18], np.float32), spec=np.array([0.9, 0.9, 0.9], np.float32),
             ks=0.9, shin=40.0, amb=0.2, refl=0.5)

_L = np.array([-0.35, -0.55, 0.76], np.float32)
_L /= np.linalg.norm(_L)
_H = _L + np.array([0, 0, 1], np.float32)
_H /= np.linalg.norm(_H)

_GRAIN = np.random.default_rng(3).normal(0, 0.012, (512, 512, 3)).astype(np.float32)

FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PART_FLIGHT = 0.35      # secondi di volo di ogni pezzo
STAGGER = 0.07          # ritardo tra un "anello" di pezzi e il successivo
DISASSEMBLE_SPEED = 1.4  # lo smontaggio è un po' più rapido


# ---------------------------------------------------------------- viso / gesto

@dataclass
class Face:
    """Punti chiave di MediaPipe FaceDetector, normalizzati."""
    right_eye: tuple
    left_eye: tuple
    right_ear: tuple
    left_ear: tuple
    box: tuple  # x, y, w, h normalizzati

    @classmethod
    def from_detection(cls, det, width: int, height: int) -> "Face":
        k = [(p.x, p.y) for p in det.keypoints]
        b = det.bounding_box
        return cls(k[0], k[1], k[4], k[5],
                   (b.origin_x / width, b.origin_y / height, b.width / width, b.height / height))

    def eye_distance(self, aspect: float) -> float:
        return math.hypot((self.right_eye[0] - self.left_eye[0]) * aspect,
                          self.right_eye[1] - self.left_eye[1]) or 1e-3


# ---------------------------------------------------------------- rendering

def _ease_out(x: float) -> float:
    return 1 - (1 - x) ** 3


class _Layer:
    """Buffer colore/alpha/profondità sul ritaglio dell'immagine attorno a una mano."""

    def __init__(self, x0: int, y0: int, x1: int, y1: int):
        self.x0, self.y0 = x0, y0
        h, w = y1 - y0, x1 - x0
        self.color = np.zeros((h, w, 3), np.float32)
        self.alpha = np.zeros((h, w), np.float32)
        self.depth = np.full((h, w), -1e9, np.float32)
        self.normal = np.zeros((h, w, 2), np.float32)
        self.refl = np.zeros((h, w), np.float32)
        self.tint = np.zeros((h, w, 3), np.float32)

    def _grid(self, bx0, by0, bx1, by1):
        h, w = self.alpha.shape
        x0, y0 = max(int(bx0) - self.x0, 0), max(int(by0) - self.y0, 0)
        x1, y1 = min(int(bx1) - self.x0 + 1, w), min(int(by1) - self.y0 + 1, h)
        if x1 <= x0 or y1 <= y0:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        return (slice(y0, y1), slice(x0, x1)), xs + self.x0 + 0.5, ys + self.y0 + 0.5

    def _write(self, sl, cov, nx, ny, nz, z, mat, opacity):
        diff = np.clip(nx * _L[0] + ny * _L[1] + nz * _L[2], 0, 1)
        spec = np.clip(nx * _H[0] + ny * _H[1] + nz * _H[2], 0, 1) ** mat["shin"]
        rim = (1 - nz) ** 2
        # riflesso d'ambiente: le superfici rivolte verso l'alto prendono un po' di "cielo"
        env = np.clip(-ny, 0, 1) * 0.12
        col = (mat["base"] * (mat["amb"] + 0.95 * diff - 0.35 * rim + env)[..., None]
               + mat["spec"] * (mat["ks"] * spec)[..., None])
        win = (cov > 0) & (z > self.depth[sl])
        self.depth[sl][win] = z[win]
        self.color[sl][win] = col[win]
        self.alpha[sl][win] = np.maximum(self.alpha[sl][win], cov[win] * opacity)
        self.normal[sl][win] = np.stack([nx, ny], -1)[win]
        # fresnel: ai bordi il metallo riflette di più
        self.refl[sl][win] = (mat["refl"] + 0.45 * (1 - nz) ** 3)[win]
        self.tint[sl][win] = mat["base"] / mat["base"].max()

    def capsule(self, a, b, ra, rb, za, zb, mat, opacity=1.0):
        r_max = max(ra, rb)
        g = self._grid(min(a[0], b[0]) - r_max - 1, min(a[1], b[1]) - r_max - 1,
                       max(a[0], b[0]) + r_max + 1, max(a[1], b[1]) + r_max + 1)
        if g is None:
            return
        sl, xs, ys = g
        abx, aby = b[0] - a[0], b[1] - a[1]
        l2 = abx * abx + aby * aby or 1e-6
        t = np.clip(((xs - a[0]) * abx + (ys - a[1]) * aby) / l2, 0, 1)
        dx, dy = xs - (a[0] + abx * t), ys - (a[1] + aby * t)
        r = ra + (rb - ra) * t
        d = np.sqrt(dx * dx + dy * dy)
        cov = np.clip(r - d + 0.5, 0, 1)
        nx, ny = dx / r, dy / r
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        self._write(sl, cov, nx, ny, nz, za + (zb - za) * t + nz * r, mat, opacity)

    def plate(self, poly, z, mat, bulge=0.55, opacity=1.0, across=None):
        """Placca bombata; con `across` (versore) è curva come un cilindro (polsino)."""
        poly = np.asarray(poly, np.float32)
        bx0, by0 = poly.min(0)
        bx1, by1 = poly.max(0)
        g = self._grid(bx0 - 1, by0 - 1, bx1 + 1, by1 + 1)
        if g is None:
            return
        sl, xs, ys = g
        mask = np.zeros(xs.shape, np.uint8)
        off = np.array([sl[1].start + self.x0, sl[0].start + self.y0], np.float32)
        cv2.fillPoly(mask, [((poly - off) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
        cov = mask.astype(np.float32) / 255
        c = poly.mean(0)
        if across is not None:
            rad = float(np.abs((poly - c) @ across).max()) or 1.0
            u = ((xs - c[0]) * across[0] + (ys - c[1]) * across[1]) / rad * bulge
            nx, ny = u * across[0], u * across[1]
        else:
            rad = float(np.linalg.norm(poly - c, axis=1).max()) or 1.0
            nx, ny = (xs - c[0]) / rad * bulge, (ys - c[1]) / rad * bulge
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        self._write(sl, cov, nx, ny, nz, z + nz * rad * 0.25, mat, opacity)


def _pillow(layer: "_Layer", pts, radius: float, z: float, mat, opacity: float = 1.0) -> None:
    """Placca di armatura: forma = inviluppo dei punti arrotondato di `radius`,
    piatta al centro e curva ai bordi (profilo da campo di distanza), con giunzioni dei pannelli."""
    pts = np.asarray(pts, np.float32)
    pad = radius + 2
    bx0, by0 = pts.min(0) - pad
    bx1, by1 = pts.max(0) + pad
    g = layer._grid(bx0, by0, bx1, by1)
    if g is None:
        return
    sl, xs, ys = g
    off = np.array([sl[1].start + layer.x0, sl[0].start + layer.y0], np.float32)
    hull = cv2.convexHull(((pts - off) * 16).astype(np.int32))
    mask = np.zeros(xs.shape, np.uint8)
    cv2.fillPoly(mask, [hull], 255, cv2.LINE_AA, shift=4)
    cv2.polylines(mask, [hull], True, 255, max(int(radius * 2), 1), cv2.LINE_AA, shift=4)
    dist = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 5)
    edge = max(radius * 1.2, 2.0)
    u = np.clip(dist / edge, 0, 1)
    height = np.sqrt(1 - (1 - u) ** 2) * edge          # quarto di cerchio ai bordi, poi piatto
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8
    nz = 1 / np.sqrt(1 + gx * gx + gy * gy)
    nx, ny = -gx * nz, -gy * nz
    cov = mask.astype(np.float32) / 255
    layer._write(sl, cov, nx, ny, nz, z + height, mat, opacity)
    # giunzioni dei pannelli: pannello interno + linea centrale
    c = pts.mean(0) - off
    inner = ((hull.reshape(-1, 2) / 16.0 - c) * 0.62 + c)
    seam = np.zeros(xs.shape, np.uint8)
    cv2.polylines(seam, [(inner * 16).astype(np.int32)], True, 255, max(int(radius * 0.12), 1), cv2.LINE_AA,
                  shift=4)
    sm = (seam > 60) & (cov > 0.9)
    layer.color[sl][sm] *= 0.45


@dataclass
class _Part:
    kind: str           # "capsule" | "plate"
    ring: int           # ordine di montaggio: 0 polso ... 7 punte
    seed: int
    args: tuple
    anchor: np.ndarray  # punto di riferimento per scintille


def palm_facing_camera(hand) -> bool:
    """Palmo verso la camera? (immagine specchiata, etichetta mano di MediaPipe)."""
    lm = hand.landmarks
    ax, ay = lm[5].x - lm[0].x, lm[5].y - lm[0].y
    bx, by = lm[17].x - lm[0].x, lm[17].y - lm[0].y
    cross = ax * by - ay * bx
    return cross > 0 if hand.handedness == "right" else cross < 0


def _hand_parts(hand, w: int, h: int):
    lm = hand.landmarks
    P = np.array([(p.x * w, p.y * h) for p in lm], np.float32)
    Z = np.array([-p.z * w for p in lm], np.float32)  # più grande = più vicino alla camera
    size = float(np.linalg.norm(P[9] - P[0])) or 1.0
    palm_w = float(np.linalg.norm(P[5] - P[17])) or size * 0.8
    rf = max(palm_w * 0.19, 3.0)          # raggio dito: copre tutta la larghezza del dito
    axis = (P[9] - P[0]) / size
    perp = np.array([-axis[1], axis[0]], np.float32)
    zp = float(Z[[0, 5, 9, 13, 17]].mean())
    parts: List[_Part] = []

    # polsino: placca cilindrica lungo l'avambraccio con fascia dorata
    def band(d0, d1, half0, half1):
        a, b = P[0] - axis * size * d0, P[0] - axis * size * d1
        return np.array([a + perp * half0, a - perp * half0, b - perp * half1, b + perp * half1])
    wrist_half = palm_w * 0.55
    parts.append(_Part("plate", 0, 1, (band(-0.08, 0.55, wrist_half, wrist_half * 0.95), Z[0] - 2, RED_DEEP,
                                       dict(bulge=0.92, across=perp)), P[0]))
    parts.append(_Part("plate", 0, 2, (band(0.12, 0.2, wrist_half * 1.02, wrist_half * 1.02), Z[0] + 1, GOLD,
                                       dict(bulge=0.92, across=perp)), P[0] - axis * size * 0.16))

    # palmo/dorso: volumi arrotondati fusi tra loro (niente spigoli)
    parts.append(_Part("pillow", 1, 3, (P[[0, 1, 2, 5, 9, 13, 17]], palm_w * 0.24, zp, RED), P[[0, 9]].mean(0)))

    # barra paranocche
    parts.append(_Part("capsule", 2, 15, (P[5], P[17], rf * 0.95, rf * 0.85, Z[5] + rf * 0.4, Z[17] + rf * 0.4,
                                          GOLD), (P[5] + P[17]) / 2))

    for f, chain in enumerate(FINGERS):
        thumb = f == 0
        r0 = rf * (1.12 if thumb else 1.0) * (0.88 if f == 4 else 1.0)
        radii = [r0, r0 * 0.93, r0 * 0.86, r0 * 0.8]
        mats = (RED, GOLD, RED)
        for s in range(3):
            a, b = chain[s], chain[s + 1]
            d = P[b] - P[a]
            gap = 0.1 if s < 2 else 0.0   # spazio per il giunto, la punta resta piena
            pa, pb = P[a] + d * (0.1 if s > 0 else 0.0), P[b] - d * gap
            parts.append(_Part("capsule", 3 + 2 * s, 20 + f * 3 + s,
                               (pa, pb, radii[s], radii[s + 1], Z[a], Z[b], mats[s]), (P[a] + P[b]) / 2))
            if s < 2:  # giunto meccanico in acciaio
                j = chain[s + 1]
                parts.append(_Part("capsule", 4 + 2 * s, 40 + f * 2 + s,
                                   (P[j], P[j], radii[s + 1] * 0.92, radii[s + 1] * 0.92, Z[j], Z[j], STEEL),
                                   P[j]))
    return parts, P, Z, size, palm_w


class IronMan:
    def __init__(self, gauntlet_path=None, flip_side: bool = True):
        self.state = "off"   # off | assembling | on | disassembling
        self.since = 0.0
        self.gauntlet_path = gauntlet_path
        self.gpu = None          # guanto 3D (gesture_control/gauntlet.py), caricato al primo uso
        self._gpu_tried = False
        self.flip_side = flip_side
        self.rng_dirs = {}
        self.impact_at: Optional[Tuple[Tuple[float, float], float]] = None

    @property
    def armored(self) -> bool:
        return self.state in ("assembling", "on")

    # compatibilità con il codice precedente
    @property
    def active(self) -> bool:
        return self.state != "off"

    def activate(self, t: float) -> None:
        if not self.armored:
            self.state, self.since = "assembling", t

    def deactivate(self, t: float) -> None:
        if self.armored:
            self.state, self.since = "disassembling", t

    def impact(self, point, t: float) -> None:
        """Onda d'urto nel punto dello scontro (coordinate normalizzate)."""
        self.impact_at = (point, t)

    def toggle(self, t: float) -> None:
        self.deactivate(t) if self.armored else self.activate(t)

    def _direction(self, seed: int) -> np.ndarray:
        if seed not in self.rng_dirs:
            r = np.random.default_rng(seed * 7919 + 17)
            a = r.uniform(0, 2 * math.pi)
            self.rng_dirs[seed] = np.array([math.cos(a), math.sin(a)], np.float32)
        return self.rng_dirs[seed]

    def _progress(self, ring: int, f_seed: int, el: float) -> float:
        """0 = pezzo lontano/invisibile, 1 = agganciato."""
        start = ring * STAGGER + (f_seed % 5) * 0.012
        if self.state == "assembling":
            return float(np.clip((el - start) / PART_FLIGHT, 0, 1))
        if self.state == "disassembling":
            start = (8 - ring) * STAGGER / DISASSEMBLE_SPEED
            return 1 - float(np.clip((el - start) / (PART_FLIGHT / DISASSEMBLE_SPEED), 0, 1))
        return 1.0

    # ------------------------------------------------------------ disegno

    def load(self) -> None:
        self._gpu_tried = True
        if self.gauntlet_path is not None:
            from .gauntlet import load_gauntlet
            self.gpu = load_gauntlet(self.gauntlet_path)
            if self.gpu is not None:
                self.gpu.flip_side = self.flip_side
                print("[iron man] guanto 3D Mark 85 caricato")

    def _gpu_progress(self, delay: float, el: float) -> float:
        from .gauntlet import FLIGHT
        if self.state == "assembling":
            return float(np.clip((el - delay) / FLIGHT, 0, 1))
        if self.state == "disassembling":
            # si smonta al contrario: prima le punte, per ultimo l'avambraccio
            back = (self.gpu.total_time - FLIGHT - delay) / DISASSEMBLE_SPEED
            return 1 - float(np.clip((el - back) / (FLIGHT / DISASSEMBLE_SPEED), 0, 1))
        return 1.0

    def draw(self, img, hands: List, main_hand, face: Optional[Face], t: float) -> None:
        self._shockwave(img, t)
        if self.state == "off":
            return
        if not self._gpu_tried:
            self.load()
        el = t - self.since
        total = self.gpu.total_time if self.gpu is not None else 8 * STAGGER + PART_FLIGHT
        if self.state == "assembling" and el > total + 0.15:
            self.state = "on"
        if self.state == "disassembling" and el > total / DISASSEMBLE_SPEED + 0.05:
            self.state = "off"
            return
        if self.gpu is not None:
            info = self.gpu.render(img, hands, lambda d: self._gpu_progress(d, el), t)
            if self.state == "on":
                for inf, hand in zip(info, hands):
                    if inf["facing"]:
                        self._glow(img, inf["repulsor"], inf["size"] * 0.22, hand.pose == "mano aperta", t)
                    else:  # le gemme stanno sul dorso
                        for xy, col in inf["gems"]:
                            self._glow(img, xy, inf["size"] * 0.05, False, t, col)
        else:
            for hand in hands:
                self._armor(img, hand, el, t)
        if self.state == "on" or (self.state == "assembling" and el > total * 0.8):
            self._hud(img, main_hand, face, t)

    def _armor(self, img, hand, el: float, t: float) -> None:
        h, w = img.shape[:2]
        lm = hand.landmarks
        full_size = math.hypot((lm[9].x - lm[0].x) * w, (lm[9].y - lm[0].y) * h) or 1.0
        # mani grandi: si disegna a risoluzione ridotta e si ingrandisce (stessa resa, molto più veloce)
        ds = float(np.clip(95.0 / full_size, 0.3, 1.0))
        sw, sh = w * ds, h * ds
        parts, P, Z, size, palm_w = _hand_parts(hand, sw, sh)
        margin = size * (1.6 if self.state != "on" else 0.6)
        x0, y0 = int(max(P[:, 0].min() - margin, 0)), int(max(P[:, 1].min() - margin, 0))
        x1, y1 = int(min(P[:, 0].max() + margin, sw - 1)), int(min(P[:, 1].max() + margin, sh - 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        layer = _Layer(x0, y0, x1, y1)
        sparks = []

        for part in parts:
            p = self._progress(part.ring, part.seed, el)
            if p <= 0:
                continue
            e = _ease_out(p)
            # il pezzo arriva da poco fuori la mano e si avvicina rallentando
            offset = self._direction(part.seed) * size * 1.1 * (1 - e) ** 2
            opacity = min(1.0, p * 2.5)
            scale = 0.75 + 0.25 * e
            if part.kind == "capsule":
                a, b, ra, rb, za, zb, mat = part.args
                layer.capsule(a + offset, b + offset, ra * scale, rb * scale, za, zb, mat, opacity)
            elif part.kind == "pillow":
                pts, radius, z, mat = part.args
                c = pts.mean(0)
                _pillow(layer, (pts - c) * scale + c + offset, radius * scale, z, mat, opacity)
            else:
                poly, z, mat, opts = part.args
                c = poly.mean(0)
                layer.plate((poly - c) * scale + c + offset, z, mat, opacity=opacity, **opts)
            if self.state == "assembling" and 0.93 < p < 1.0:
                sparks.append(part.anchor / ds)

        self._composite(img, layer, size, ds)

        for sp in sparks:
            self._spark(img, sp, full_size)
        if self._progress(1, 3, el) >= 1.0 and palm_facing_camera(hand):
            self._reactor(img, P / ds, Z / ds, full_size, palm_w / ds, hand.pose == "mano aperta", t)

    @staticmethod
    def _composite(img, layer: _Layer, size: float, ds: float) -> None:
        a = layer.alpha
        if not a.any():
            return
        hh, ww = a.shape
        H, W = img.shape[:2]
        X0, Y0 = int(layer.x0 / ds), int(layer.y0 / ds)
        X1, Y1 = min(int((layer.x0 + ww) / ds), W), min(int((layer.y0 + hh) / ds), H)
        roi = img[Y0:Y1, X0:X1]
        full = roi.astype(np.float32) / 255
        base = cv2.resize(full, (ww, hh), interpolation=cv2.INTER_AREA)
        # adatta l'armatura alla luce della stanza
        color = layer.color * float(np.clip(0.55 + 1.1 * float(base.mean()), 0.65, 1.25))
        # riflessi: il metallo specchia la scena (sfocata), deformata secondo la normale
        env = cv2.GaussianBlur(base, (0, 0), max(size * 0.12, 2))
        gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        k = size * 0.9
        refl = cv2.remap(env, gx - layer.normal[..., 0] * k, gy - layer.normal[..., 1] * k,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        refl = refl * (0.35 + 0.65 * layer.tint) * 1.6
        r = layer.refl[..., None]
        color = np.clip(color * (1 - r * 0.6) + refl * r, 0, 1)
        # ombra portata sulla pelle
        shadow = cv2.GaussianBlur(a, (0, 0), max(size * 0.06, 1.5))
        shift = np.float32([[1, 0, size * 0.05], [0, 1, size * 0.08]])
        shadow = cv2.warpAffine(shadow, shift, (ww, hh))
        # morbidezza e grana da webcam (a bassa risoluzione: costa poco)
        color = cv2.GaussianBlur(color, (0, 0), 0.7 if ds >= 1.0 else 0.4)
        gh, gw = color.shape[:2]
        color = color + (_GRAIN[:gh, :gw] if gh <= 512 and gw <= 512 else np.resize(_GRAIN, color.shape))
        # ritorno alla risoluzione piena
        size_full = (X1 - X0, Y1 - Y0)
        if ds < 1.0:
            color = cv2.resize(color, size_full, interpolation=cv2.INTER_LINEAR)
            a = cv2.resize(a, size_full, interpolation=cv2.INTER_LINEAR)
            shadow = cv2.resize(shadow, size_full, interpolation=cv2.INTER_LINEAR)
        else:
            color, a, shadow = color[:size_full[1], :size_full[0]], a[:size_full[1], :size_full[0]], \
                shadow[:size_full[1], :size_full[0]]
        full *= (1 - 0.5 * shadow)[..., None]
        out = full * (1 - a[..., None]) + np.clip(color, 0, 1) * a[..., None]
        roi[:] = (out * 255).astype(np.uint8)

    def _shockwave(self, img, t: float) -> None:
        if self.impact_at is None:
            return
        (px, py), t0 = self.impact_at
        k = (t - t0) / 0.45
        if k > 1:
            self.impact_at = None
            return
        h, w = img.shape[:2]
        c = (int(px * w), int(py * h))
        fade = 1 - k
        flash = img.copy()
        cv2.circle(flash, c, int(h * 0.05 * (1 + 2 * k)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(flash, 0.6 * fade ** 2, img, 1 - 0.6 * fade ** 2, 0, img)
        for i, col in enumerate(((255, 240, 200), CYAN)):
            r = int(h * (0.04 + 0.35 * _ease_out(k)) * (1 - 0.18 * i))
            cv2.circle(img, c, r, tuple(int(v * fade) for v in col), max(int(6 * fade), 1), cv2.LINE_AA)

    @staticmethod
    def _spark(img, p, size: float) -> None:
        x, y = int(p[0]), int(p[1])
        L = int(size * 0.18)
        for dx, dy in ((1, 0), (0, 1), (0.7, 0.7), (0.7, -0.7)):
            cv2.line(img, (int(x - dx * L), int(y - dy * L)), (int(x + dx * L), int(y + dy * L)),
                     (255, 255, 230), 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), max(int(size * 0.04), 2), (255, 255, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _glow(img, center, r: float, charged: bool, t: float, color=CYAN) -> None:
        """Bagliore del repulsore (la luce vera la emette il materiale del modello 3D)."""
        h, w = img.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        r = max(int(r * (1.25 if charged else 1.0)), 4)
        pulse = 0.8 + 0.2 * math.sin(t * 5) + (0.25 if charged else 0)
        g = int(r * (5 if charged else 3.5))
        x0, y0, x1, y1 = max(cx - g, 0), max(cy - g, 0), min(cx + g, w), min(cy + g, h)
        if x1 <= x0 or y1 <= y0:
            return
        glow = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        cv2.circle(glow, (cx - x0, cy - y0), int(r * 1.2), tuple(int(min(c * pulse, 255)) for c in color), -1,
                   cv2.LINE_AA)
        k = max(int(r * (3.5 if charged else 2.5)) | 1, 3)
        glow = cv2.GaussianBlur(glow, (k, k), 0)
        roi = img[y0:y1, x0:x1]
        cv2.add(roi, glow, dst=roi)

    def _reactor(self, img, P, Z, size: float, palm_w: float, charged: bool, t: float) -> None:
        h, w = img.shape[:2]
        center = P[[0, 5, 9, 13, 17]].mean(0) * 0.5 + (P[0] + P[9]) / 2 * 0.5
        cx, cy = int(center[0]), int(center[1])
        r = max(int(palm_w * (0.27 if not charged else 0.31)), 6)
        pulse = 0.82 + 0.18 * math.sin(t * 5)
        if charged:
            pulse = min(1.0, pulse + 0.25)

        g = int(r * (4.5 if charged else 3.2))
        x0, y0, x1, y1 = max(cx - g, 0), max(cy - g, 0), min(cx + g, w), min(cy + g, h)
        if x1 > x0 and y1 > y0:
            glow = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            cv2.circle(glow, (cx - x0, cy - y0), int(r * 1.3), tuple(int(c * pulse) for c in CYAN), -1,
                       cv2.LINE_AA)
            k = max(int(r * (3 if charged else 2)) | 1, 3)
            glow = cv2.GaussianBlur(glow, (k, k), 0)
            roi = img[y0:y1, x0:x1]
            cv2.add(roi, glow, dst=roi)
            cv2.add(roi, glow, dst=roi)

        # alloggiamento metallico
        cv2.circle(img, (cx, cy), r + 4, (40, 40, 40), 4, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r + 2, (170, 170, 175), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (110, 110, 115), 2, cv2.LINE_AA)
        spin = (t * (220 if charged else 90)) % 360
        col = tuple(int(c * pulse) for c in CYAN)
        for i in range(10):
            a0 = spin + i * 36
            cv2.ellipse(img, (cx, cy), (int(r * 0.78), int(r * 0.78)), 0, a0, a0 + 22, col, max(r // 5, 2),
                        cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(r * 0.45), col, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(r * 0.25), (255, 255, 255), -1, cv2.LINE_AA)

    def _hud(self, img, main_hand, face: Optional[Face], t: float) -> None:
        h, w = img.shape[:2]
        if face is not None:
            fx, fy, fw, fh = face.box
            c = (int((fx + fw / 2) * w), int((fy + fh / 2) * h))
            r = int(max(fw * w, fh * h) * 0.75)
            spin = t * 40
            for i in range(3):
                a0 = spin + i * 120
                cv2.ellipse(img, c, (r, r), 0, a0, a0 + 70, HUD, 2, cv2.LINE_AA)
            cv2.ellipse(img, c, (r + 10, r + 10), 0, -spin, -spin + 200, HUD, 1, cv2.LINE_AA)
        if main_hand is not None:
            tip = main_hand.landmarks[8]
            x, y = int(tip.x * w), int(tip.y * h)
            cv2.circle(img, (x, y), 14, HUD, 1, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(img, (x + dx * 8, y + dy * 8), (x + dx * 20, y + dy * 20), HUD, 1, cv2.LINE_AA)
        label = "J.A.R.V.I.S. ONLINE"
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        org = (w - tw - 20, 30)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD, 2, cv2.LINE_AA)
