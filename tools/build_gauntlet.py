"""Prepara il guanto 3D per il programma (si esegue una volta).

    .venv/bin/python tools/build_gauntlet.py iron_man_mark_85.glb

1. ritaglia mano + avambraccio destro dal modello completo
2. trova i 21 punti della mano nel modello: lo disegna e lo fa guardare a MediaPipe
3. calcola i pesi di skinning (quanto ogni vertice segue ogni osso)
4. salva tutto in assets/gauntlet.npz, letto da gesture_control/gauntlet.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import moderngl
import numpy as np
import trimesh
from mediapipe.tasks.python import BaseOptions, vision
from pygltflib import GLTF2

ROOT = Path(__file__).resolve().parent.parent

# Zone delle mani nel modello di LLIypuk (metri, personaggio alto 1.79). La destra ha le gemme.
CROPS = {
    "right": (np.array([-0.5, 0.74, -0.2]), np.array([-0.225, 1.10, 0.3])),
    "left": (np.array([0.225, 0.74, -0.2]), np.array([0.5, 1.10, 0.3])),
}
OUTPUTS = {"right": "assets/gauntlet.npz", "left": "assets/gauntlet_left.npz"}

# Ossa: (punto di partenza, punto di arrivo) sui 21 punti di MediaPipe. Osso 0 = palmo.
BONES = [(0, 9)] + [(a, b) for chain in ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16),
                                         (17, 18, 19, 20)) for a, b in zip(chain, chain[1:])]


def load_materials(path):
    g = GLTF2().load(str(path))
    mats = []
    for m in g.materials:
        p = m.pbrMetallicRoughness
        mats.append(dict(name=m.name, base=list(p.baseColorFactor[:3]), metal=p.metallicFactor,
                         rough=p.roughnessFactor, emissive=list(m.emissiveFactor or [0, 0, 0])))
    return mats


def extract(path, side):
    """Mesh di una mano. La sinistra viene specchiata in una "destra" così tutto il resto
    (viste, punti, ossa) è identico; il programma la rispecchia quando la disegna."""
    crop_lo, crop_hi = CROPS[side]
    mats = load_materials(path)
    names = [m["name"] for m in mats]
    scene = trimesh.load(str(path), force="scene")
    pos, nor, mat_id, piece = [], [], [], []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        inside = np.all((m.vertices >= crop_lo) & (m.vertices <= crop_hi), axis=1)
        faces = np.nonzero(inside[m.faces].all(1))[0]
        if len(faces) == 0:
            continue
        # il nome del materiale va letto prima del ritaglio: submesh perde il materiale delle
        # mesh senza coordinate UV (le gemme)
        mat = getattr(getattr(m.visual, "material", None), "name", None)
        sub = m.submesh([faces], append=True)
        mid = names.index(mat) if mat in names else names.index("material")
        # pezzi staccati che appartengono al corpo (coscia) e non alla mano
        for comp in sub.split(only_watertight=False):
            if abs(comp.vertices[:, 0].mean()) < 0.25:
                continue
            tri = comp.vertices[comp.faces].reshape(-1, 3)
            pos.append(tri)
            nor.append(np.repeat(comp.face_normals, 3, axis=0))
            mat_id.append(np.full(len(tri), mid))
            piece.append(np.full(len(tri), len(piece)))   # ogni componente connessa = una placca
    pos = np.vstack(pos).astype(np.float32)
    nor = np.nan_to_num(np.vstack(nor)).astype(np.float32)
    if side == "left":
        pos[:, 0] *= -1
        nor[:, 0] *= -1
    return pos, nor, np.concatenate(mat_id).astype(np.int32), np.concatenate(piece).astype(np.int32), mats


def smooth_normals(pos, nor):
    """Normali per vertice mediate sui vertici coincidenti (superfici lisce, spigoli veri restano)."""
    key = np.round(pos * 2e4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    acc = np.zeros((inv.max() + 1, 3), np.float64)
    np.add.at(acc, inv.ravel(), nor)
    sm = acc[inv.ravel()]
    sm /= np.linalg.norm(sm, axis=1, keepdims=True) + 1e-9
    # niente media sugli spigoli vivi (> 50°)
    sharp = (sm * nor).sum(1) < np.cos(np.radians(50))
    sm[sharp] = nor[sharp]
    return sm.astype(np.float32)


class Renderer:
    """Render ortografico semplice per far vedere il guanto a MediaPipe."""

    VS = """#version 410
    in vec3 pos; in vec3 nor; uniform mat4 mvp; uniform mat3 rot; out vec3 n;
    void main(){ n = rot * nor; gl_Position = mvp * vec4(pos, 1.0); }"""
    FS = """#version 410
    in vec3 n; out vec4 color;
    void main(){ vec3 N = normalize(n); if (N.z < 0.0) N = -N;  // verso la camera, anche per mesh specchiate
      float d = max(dot(N, normalize(vec3(0.3, 0.5, 0.8))), 0.0);
      color = vec4(vec3(0.42, 0.55, 0.78) * (0.35 + 0.75 * d), 1.0); }"""   # BGR "pelle"

    def __init__(self, pos, nor, size=768):
        self.ctx = moderngl.create_standalone_context(require=410)
        self.size = size
        self.prog = self.ctx.program(vertex_shader=self.VS, fragment_shader=self.FS)
        vbo = self.ctx.buffer(np.hstack([pos, nor]).astype("f4").tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "pos", "nor")])
        self.fbo = self.ctx.simple_framebuffer((size, size))
        self.ctx.enable(moderngl.DEPTH_TEST)

    def render(self, rot, center, half):
        """rot: matrice camera (righe = asse x schermo, asse y schermo, verso la camera)."""
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = rot / half
        m[:3, 3] = -(rot @ center) / half
        m[2, :] *= -1  # profondità OpenGL
        self.prog["mvp"].write(m.T.astype("f4").tobytes())
        self.prog["rot"].write(rot.T.astype("f4").tobytes())
        self.fbo.use()
        self.fbo.clear(0.12, 0.12, 0.12, 1.0)
        self.vao.render()
        img = np.frombuffer(self.fbo.read(components=3), np.uint8).reshape(self.size, self.size, 3)[::-1].copy()
        depth = np.frombuffer(self.fbo.read(components=1, attachment=-1, dtype="f4"), np.float32).reshape(self.size, self.size)[::-1]
        return img, depth


def _robust_detect(landmarker, img):
    """MediaPipe su varianti dell'immagine (margine, rotazione, sfondo): mediana dei risultati.
    Il guanto non è una mano vera, quindi una singola rilevazione è poco affidabile."""
    size = img.shape[0]
    found = []
    for pad in (0, 200):
        for ang in (-30, -15, 0, 15, 30):
            for bg in (30, 200):
                im = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                im[(im == 30).all(2)] = bg
                c = (im.shape[1] / 2, im.shape[0] / 2)
                M = cv2.getRotationMatrix2D(c, ang, 1)
                im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), borderValue=(bg, bg, bg))
                res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                if not res.hand_landmarks or res.handedness[0][0].score < 0.5:
                    continue
                p = np.array([(q.x * im.shape[1], q.y * im.shape[0]) for q in res.hand_landmarks[0]])
                Minv = cv2.invertAffineTransform(M)
                p = p @ Minv[:, :2].T + Minv[:, 2]
                found.append((p - pad) / size)
    print(f"[landmark] rilevazioni valide: {len(found)}")
    if len(found) < 3:
        return None
    return np.median(np.array(found), axis=0).astype(np.float32)


def find_landmarks(pos, nor, debug_dir, tag):
    """Punti della mano nel modello. Vista dal dorso (-x) e dal palmo (+x), dita in alto."""
    center = (pos.min(0) + pos.max(0)) / 2
    half = float((pos.max(0) - pos.min(0)).max()) * 0.6
    r = Renderer(pos, nor)
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "assets/hand_landmarker.task")),
        num_hands=1, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2))
    views = {
        # schermo x, schermo y (in alto), verso la camera
        "dorso": np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], np.float32),
        "palmo": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], np.float32),
    }
    # La vista dal palmo è quella affidabile (dal dorso il pollice è nascosto e MediaPipe
    # scambia le dita): i punti 2D vengono da lì, il dorso serve solo per la profondità.
    img, front = r.render(views["palmo"], center, half)
    _, back = r.render(views["dorso"], center, half)
    pts = _robust_detect(landmarker, img)
    dbg = img.copy()
    if pts is None:
        cv2.imwrite(str(Path(debug_dir) / f"landmarks_{tag}.png"), dbg)
        sys.exit("MediaPipe non ha riconosciuto la mano nel modello")
    for i, (x, y) in enumerate(pts * r.size):
        cv2.circle(dbg, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(dbg, str(i), (int(x) + 4, int(y)), 0, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(Path(debug_dir) / f"landmarks_{tag}.png"), dbg)

    rot = views["palmo"]
    finger_radius = 0.0095  # metri: raggio di un dito dell'armatura

    def surface(depth, px, py):
        win = depth[max(py - 3, 0):py + 4, max(px - 3, 0):px + 4]
        return float(win[win < 1].min()) if (win < 1).any() else None

    out = []
    for i, (x, y) in enumerate(pts):
        px, py = int(np.clip(x * r.size, 0, r.size - 1)), int(np.clip(y * r.size, 0, r.size - 1))
        d_front = surface(front, px, py)
        d_back = surface(back, r.size - 1 - px, py)   # la vista dal dorso è specchiata in orizzontale
        z_front = -(d_front * 2 - 1) * half if d_front is not None else 0.0   # verso la camera = positivo
        if i in (0, 1, 5, 9, 13, 17) and d_back is not None:
            z_back = (d_back * 2 - 1) * half      # superficie posteriore, nello stesso sistema
            z = (z_front + z_back) / 2
        else:
            z = z_front - finger_radius
        ndc = np.array([x * 2 - 1, 1 - y * 2, 0], np.float32) * half
        ndc[2] = z
        out.append(center + rot.T @ ndc)
    lm = np.array(out)
    # il pollice piegato davanti al palmo può falsare la profondità delle nocche: i punti del
    # palmo lontani dal piano comune vengono riportati sul piano
    palm = [0, 5, 9, 13, 17]
    view_axis = rot[2]
    depths = lm[palm] @ view_axis
    med = float(np.median(depths))
    for i, dep in zip(palm, depths):
        if abs(dep - med) > 0.012:
            lm[i] += view_axis * (med - dep)
            print(f"[landmark] punto {i}: profondità corretta di {(med - dep) * 100:.1f} cm")
    return lm.astype(np.float32)


def skin_weights(pos, lm):
    """Ogni vertice segue i 2 ossi più vicini, con passaggio morbido alle giunture."""
    nb = len(BONES)
    d = np.empty((len(pos), nb), np.float32)
    for i, (a, b) in enumerate(BONES):
        A, B = lm[a], lm[b]
        if i == 0:  # palmo: segmento polso -> nocca medio, ma esteso fino all'avambraccio
            A = lm[0] + (lm[0] - lm[9]) * 2.5
        ab = B - A
        t = np.clip(((pos - A) @ ab) / (ab @ ab), 0, 1)
        d[:, i] = np.linalg.norm(pos - (A + t[:, None] * ab), axis=1)
    # il palmo è largo: distanza dalla superficie del palmo invece che dall'asse
    palm_pts = lm[[0, 1, 5, 9, 13, 17]]
    c = palm_pts.mean(0)
    _, _, vt = np.linalg.svd(palm_pts - c)
    normal = vt[2]
    inside = np.abs((pos - c) @ normal)
    proj = pos - np.outer((pos - c) @ normal, normal)
    # distanza in-piano dal poligono del palmo (approssimata col raggio)
    rad = np.linalg.norm(palm_pts - c, axis=1).max()
    d[:, 0] = np.minimum(d[:, 0], np.maximum(np.linalg.norm(proj - c, axis=1) - rad * 0.8, 0) + inside * 0.5)
    order = np.argsort(d, axis=1)[:, :2]
    d2 = np.take_along_axis(d, order, 1)
    w = 1.0 / (d2 + 1e-4) ** 4
    w /= w.sum(1, keepdims=True)
    return order.astype(np.float32), w.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--debug", default=str(ROOT / "build"))
    args = ap.parse_args()
    Path(args.debug).mkdir(parents=True, exist_ok=True)

    for side in ("right", "left"):
        print(f"--- guanto {'destro (gemme)' if side == 'right' else 'sinistro'}")
        pos, nor, mat_id, piece, mats = extract(args.glb, side)
        print(f"[mesh] {len(pos) // 3} triangoli, {piece.max() + 1} pezzi separati")
        nor = smooth_normals(pos, nor)
        lm = find_landmarks(pos, nor, args.debug, side)
        bones, weights = skin_weights(pos, lm)
        out = ROOT / OUTPUTS[side]
        np.savez_compressed(
            out, pos=pos, nor=nor, mat=mat_id, piece=piece, bones=bones, weights=weights, landmarks=lm,
            mat_base=np.array([m["base"] for m in mats], np.float32),
            mat_metal=np.array([m["metal"] for m in mats], np.float32),
            mat_rough=np.array([m["rough"] for m in mats], np.float32),
            mat_emissive=np.array([m["emissive"] for m in mats], np.float32),
            mat_names=np.array([m["name"] for m in mats]),
        )
        print(f"[ok] {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
