"""Guanto 3D di Iron Man (assets/gauntlet.npz) disegnato in OpenGL sopra la mano.

Il modello viene mosso dai 21 punti di MediaPipe: il palmo si allinea al palmo osservato
(rotazione + scala), ogni falange ruota verso la direzione osservata del dito mantenendo le
proporzioni del modello. Materiali metallici PBR (GGX) con riflessi presi dall'immagine
della webcam, così il guanto riflette la stanza in cui ci si trova.
Preparazione dei dati: tools/build_gauntlet.py.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

PALM_POINTS = [0, 1, 5, 9, 13, 17]
CHAINS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
N_BONES = 16
FLIGHT = 0.45          # secondi di volo di ogni placca
RING_DELAY = 0.12      # ritardo tra "anelli" (avambraccio, palmo, falangi...)
FOREARM_END = 0.10     # metri di avambraccio dopo il polso
FOREARM_FADE = 0.035   # sfumatura finale dell'avambraccio

# il repulsore nel palmo (c'è in entrambi i guanti); gli altri materiali luminosi sono le gemme
REPULSOR_MATERIAL = "material_3"

# ritocchi ai materiali del modello per la resa in tempo reale (rugosità senza texture)
MATERIAL_TWEAKS = {"material": dict(rough=0.42), "Mat.1": dict(rough=0.32), "Mat.2": dict(rough=0.3)}

VERTEX_SHADER = """
#version 410
in vec3 in_pos; in vec3 in_nor; in float in_mat; in vec2 in_bones; in vec2 in_w;
in float in_piece; in vec3 in_dir; in float in_fade;
uniform mat4 bones[16];
uniform mat4 proj;
uniform sampler2D piece_tex;
out vec3 v_nor; out vec3 v_pos; flat out int v_mat; out float v_vis; out float v_fade;
void main() {
    v_fade = in_fade;
    float p = texelFetch(piece_tex, ivec2(int(in_piece), 0), 0).r;
    float e = 1.0 - pow(1.0 - p, 3.0);
    vec3 local = in_pos + in_dir * (1.0 - e) * (1.0 - e);
    mat4 B = bones[int(in_bones.x)] * in_w.x + bones[int(in_bones.y)] * in_w.y;
    vec4 wp = B * vec4(local, 1.0);
    v_nor = mat3(B) * in_nor;
    v_pos = wp.xyz;
    v_mat = int(in_mat + 0.5);
    v_vis = p;
    gl_Position = proj * wp;
}
"""

FRAGMENT_SHADER = """
#version 410
in vec3 v_nor; in vec3 v_pos; flat in int v_mat; in float v_vis; in float v_fade;
uniform vec3 base[16]; uniform float metal[16]; uniform float rough[16]; uniform vec3 emis[16];
uniform sampler2D env; uniform float exposure; uniform float glow;
out vec4 color;
const float PI = 3.14159265;

vec3 env_lookup(vec3 dir, float lod) {
    // la scena della webcam come ambiente: X a destra, Y in basso, Z verso la scena
    vec2 uv = vec2(0.5 + dir.x * 0.48, 0.5 + dir.y * 0.48);
    return pow(textureLod(env, uv, lod).rgb, vec3(2.2));
}
float ggx(float NdH, float a) { float a2 = a * a; float d = NdH * NdH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d); }
float smith(float NdV, float NdL, float a) {
    float k = (a + 1.0) * (a + 1.0) / 8.0;
    return NdV / (NdV * (1.0 - k) + k) * NdL / (NdL * (1.0 - k) + k);
}
void main() {
    if (v_vis <= 0.001) discard;
    vec3 N = normalize(v_nor);
    vec3 V = vec3(0.0, 0.0, -1.0);                  // verso la camera
    if (dot(N, V) < 0.0) N = -N;
    vec3 albedo = base[v_mat];
    float m = metal[v_mat];
    float r = clamp(rough[v_mat], 0.08, 1.0);
    float a = r * r;
    vec3 F0 = mix(vec3(0.04), albedo, m);
    float NdV = max(dot(N, V), 1e-3);

    vec3 Lo = vec3(0.0);
    vec3 dirs[3] = vec3[3](normalize(vec3(-0.45, -0.75, -0.5)), normalize(vec3(0.7, -0.1, -0.6)),
                           normalize(vec3(0.2, 0.8, 0.4)));
    vec3 cols[3] = vec3[3](vec3(3.2, 3.05, 2.9), vec3(0.9, 1.0, 1.15), vec3(1.4, 1.4, 1.5));
    for (int i = 0; i < 3; i++) {
        vec3 L = dirs[i];
        vec3 H = normalize(L + V);
        float NdL = max(dot(N, L), 0.0);
        if (NdL <= 0.0) continue;
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
        vec3 spec = ggx(max(dot(N, H), 0.0), a) * smith(NdV, NdL, a) * F / (4.0 * NdV * NdL + 1e-4);
        vec3 kd = (1.0 - F) * (1.0 - m);
        Lo += (kd * albedo / PI + spec) * cols[i] * NdL;
    }
    // ambiente: riflesso speculare (sfocato secondo la rugosità) + luce diffusa
    vec3 R = reflect(-V, N);
    vec3 Fa = F0 + (max(vec3(1.0 - r), F0) - F0) * pow(1.0 - NdV, 5.0);
    vec3 spec_env = env_lookup(R, r * 7.0) * Fa * 1.6;
    vec3 diff_env = env_lookup(N, 8.0) * albedo * (1.0 - m) * 0.9;
    vec3 c = Lo + spec_env + diff_env + emis[v_mat] * glow;
    c *= exposure;
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);   // tonemapping ACES
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    float alpha = smoothstep(0.0, 0.25, v_vis) * v_fade;
    color = vec4(c * alpha, alpha);   // premoltiplicato: i bordi antialiasing restano puliti
}
"""


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similitudine (scala, rotazione, traslazione) che porta src su dst ai minimi quadrati."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
    return s, R, mu_d - s * R @ mu_s


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotazione minima che porta la direzione a sulla direzione b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.9999:
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _affine(linear: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = linear
    m[:3, 3] = t
    return m


class _Smoother:
    """Filtro One Euro sui 21 punti (in pixel): mano ferma = guanto fermo, movimenti rapidi = pronto."""

    def __init__(self, min_cutoff=1.8, beta=0.015):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.x = self.dx = None
        self.t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None or self.t is None or t - self.t > 0.3:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = 1 / (1 + 1 / (2 * math.pi * 1.0 * dt))
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        a = 1 / (1 + 1 / (2 * math.pi * cutoff * dt))
        self.x = a * x + (1 - a) * self.x
        return self.x


class _Model:
    """Un guanto: geometria, scheletro a riposo, placche, materiali, gemme."""

    def __init__(self, ctx, prog, path: Path):
        import moderngl

        z = np.load(path)
        self.rest = z["landmarks"].astype(np.float64)
        pos, mat, piece = z["pos"], z["mat"], z["piece"]
        bones = z["bones"]
        self.n_pieces = int(piece.max()) + 1

        # placche: centro, osso principale, ordine di montaggio, direzione di arrivo
        counts = np.bincount(piece, minlength=self.n_pieces).astype(np.float64)
        centers = np.stack([np.bincount(piece, pos[:, i], self.n_pieces) for i in range(3)], 1) / counts[:, None]
        main_bone = np.argmax(np.stack([np.bincount(piece, (bones[:, 0] == b).astype(float), self.n_pieces)
                                        for b in range(N_BONES)], 1), 1)
        axis = self.rest[9] - self.rest[0]
        along = ((centers - self.rest[0]) * axis).sum(1) / (axis @ axis)
        ring = np.where(main_bone == 0, np.where(along < 0, 0, 1), 2 + (main_bone - 1) % 3)
        rng = np.random.default_rng(7)
        self.piece_delay = ring * RING_DELAY + rng.uniform(0, 0.08, self.n_pieces)
        palm_c = self.rest[PALM_POINTS].mean(0)
        out = centers - palm_c
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        rnd = rng.normal(0, 1, (self.n_pieces, 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        fly_dir = out * 0.6 + rnd * 0.8
        fly_dir /= np.linalg.norm(fly_dir, axis=1, keepdims=True)
        fly = (fly_dir * rng.uniform(0.07, 0.13, (self.n_pieces, 1))).astype(np.float32)  # 7-13 cm

        # parti luminose: il repulsore è quella più vicina al centro del palmo, le altre sono gemme
        emissive = z["mat_emissive"]
        names = [str(n) for n in z["mat_names"]]
        self.gems = []           # (centro a riposo, osso principale, colore BGR)
        self.repulsor = palm_c
        glowing = [m for m in np.nonzero(emissive.max(1) > 0.5)[0] if (mat == m).any()]
        if glowing:
            cents = {m: pos[mat == m].astype(np.float64).mean(0) for m in glowing}
            rep = names.index(REPULSOR_MATERIAL) if REPULSOR_MATERIAL in names and \
                names.index(REPULSOR_MATERIAL) in glowing else min(glowing, key=lambda m: np.linalg.norm(cents[m] - palm_c))
            self.repulsor = cents[rep]
            for m in glowing:
                if m == rep:
                    continue
                bone = int(np.bincount(bones[mat == m, 0].astype(int), minlength=N_BONES).argmax())
                rgb = (emissive[m] + z["mat_base"][m]) / 2
                rgb = rgb / max(float(rgb.max()), 1e-3)
                self.gems.append((cents[m], bone, tuple(int(c * 255) for c in rgb[::-1])))
        n = np.cross(self.rest[5] - self.rest[0], self.rest[17] - self.rest[0])
        n /= np.linalg.norm(n)
        if ((self.repulsor - palm_c) * n).sum() < 0:
            n = -n
        self.palm_normal = n

        # avambraccio: il modello è tagliato a metà braccio, sfumiamo il bordo invece di mostrarlo
        wrist_axis = axis / np.linalg.norm(axis)
        # somma esplicita: il prodotto matrice-vettore di numpy 2.0 su macOS dà falsi avvisi
        beyond = -((pos - self.rest[0]) * wrist_axis).sum(1)   # metri oltre il polso
        fade = np.clip((FOREARM_END - beyond) / FOREARM_FADE, 0, 1).astype(np.float32)
        keep = beyond < FOREARM_END + 0.01
        data = np.hstack([pos, z["nor"], mat[:, None].astype(np.float32), bones.astype(np.float32),
                          z["weights"], piece[:, None].astype(np.float32), fly[piece], fade[:, None]])
        tri_keep = keep.reshape(-1, 3).all(1).repeat(3)       # triangoli interi oltre il taglio: via
        data = data[tri_keep].astype("f4")
        self.vbo = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(prog, [(self.vbo, "3f 3f 1f 2f 2f 1f 3f 1f", "in_pos", "in_nor", "in_mat",
                                            "in_bones", "in_w", "in_piece", "in_dir", "in_fade")])
        self.piece_tex = ctx.texture((self.n_pieces, 1), 1, dtype="f4")
        self.piece_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.mats = dict(base=z["mat_base"].copy(), metal=z["mat_metal"].copy(), rough=z["mat_rough"].copy(),
                         emis=z["mat_emissive"].copy())
        for i, name in enumerate(names):
            for k, v in MATERIAL_TWEAKS.get(name, {}).items():
                self.mats[k][i] = v

    @property
    def total_time(self) -> float:
        return float(self.piece_delay.max()) + FLIGHT

    def pose(self, pts: np.ndarray, mirror: bool):
        """Matrici degli ossi (modello -> pixel) per i 21 punti osservati (x, y, z in pixel)."""
        Rm = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        rest = self.rest @ Rm.T
        s, R, t = _umeyama(rest[PALM_POINTS], pts[PALM_POINTS])

        # scala dalle sole distanze 2D: la profondità stimata da MediaPipe è poco affidabile
        model2d = (s * rest @ R.T)[:, :2]
        pairs = [(0, 5), (0, 9), (0, 17), (5, 17), (5, 9), (9, 13), (13, 17)]
        obs = sum(np.linalg.norm(pts[a, :2] - pts[b, :2]) for a, b in pairs)
        mod = sum(np.linalg.norm(model2d[a] - model2d[b]) for a, b in pairs)
        if mod > 1e-6:
            k = float(np.clip(obs / mod, 0.6, 1.6))
            s *= k
            t = pts[PALM_POINTS].mean(0) - s * R @ rest[PALM_POINTS].mean(0)
        SR = s * R
        mats = [_affine(SR @ Rm, t)]
        for chain in CHAINS:
            joint = SR @ rest[chain[0]] + t
            for a, b in zip(chain, chain[1:]):
                d_rest = SR @ (rest[b] - rest[a])
                d_obs = pts[b] - pts[a]
                Q = _rot_between(d_rest, d_obs)
                # falange allungata/accorciata sulla lunghezza vera del dito
                u = Q @ d_rest
                L = np.linalg.norm(u) + 1e-9
                stretch = float(np.clip(np.linalg.norm(d_obs[:2]) / (np.linalg.norm(u[:2]) + 1e-6), 0.7, 1.45))
                u /= L
                S_u = np.eye(3) + (stretch - 1) * np.outer(u, u)
                lin = S_u @ Q @ SR @ Rm
                mats.append(_affine(lin, joint - lin @ self.rest[a]))
                joint = joint + u * L * stretch
        return mats, s, R, t, Rm


class Gauntlet:
    """Due guanti come nel modello: il destro con le gemme, il sinistro senza."""

    def __init__(self, right_path: Path, left_path: Optional[Path] = None):
        import moderngl  # importato qui: senza OpenGL il resto del programma funziona lo stesso

        self.ctx = moderngl.create_standalone_context(require=410)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.right = _Model(self.ctx, self.prog, right_path)
        self.left = _Model(self.ctx, self.prog, left_path) if left_path is not None and left_path.exists() \
            else self.right
        self.env = None
        self.fbo_size = None
        self.smoothers = {}
        # quale lato mostrare in base all'etichetta destra/sinistra di MediaPipe (tasto g)
        self.flip_side = True

    @property
    def total_time(self) -> float:
        return max(self.right.total_time, self.left.total_time)

    def _use_materials(self, model: _Model) -> None:
        self._set_array("base", model.mats["base"], 3)
        self._set_array("metal", model.mats["metal"], 1)
        self._set_array("rough", model.mats["rough"], 1)
        self._set_array("emis", model.mats["emis"], 3)

    def _set_array(self, name, values, comps):
        arr = np.zeros((16, comps), np.float32)
        arr[:len(values)] = np.asarray(values, np.float32).reshape(len(values), comps)
        self.prog[name].write(arr.tobytes())

    def _ensure_fbo(self, w: int, h: int) -> None:
        if self.fbo_size == (w, h):
            return
        ctx = self.ctx
        self.ms_fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4, samples=4)],
                                      depth_attachment=ctx.depth_renderbuffer((w, h), samples=4))
        self.fbo = ctx.framebuffer(color_attachments=[ctx.renderbuffer((w, h), 4)])
        self.fbo_size = (w, h)
        proj = np.array([[2 / w, 0, 0, -1], [0, -2 / h, 0, 1], [0, 0, 0.5 / w, 0], [0, 0, 0, 1]], np.float32)
        self.prog["proj"].write(proj.T.tobytes())

    # ------------------------------------------------------------ disegno

    def render(self, img, hands: List, progress, t: float, glow: float = 1.0) -> list:
        """Disegna i guanti su img. progress(delay) -> 0..1 per placca. Ritorna info per i bagliori."""
        h, w = img.shape[:2]
        self._ensure_fbo(w, h)
        self._update_env(img)
        visible = False
        for model in {id(self.right): self.right, id(self.left): self.left}.values():
            arr = np.array([progress(d) for d in model.piece_delay], np.float32)
            model.piece_tex.write(arr.tobytes())
            visible |= bool(arr.any())
        if not visible:
            return []
        lum = float(cv2.resize(img, (32, 18)).mean()) / 255
        self.prog["exposure"].value = float(np.clip(0.75 + 0.9 * lum, 0.8, 1.4))
        self.prog["glow"].value = glow

        self.ms_fbo.use()
        self.ms_fbo.clear(0, 0, 0, 0)
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.env.use(1)
        self.prog["piece_tex"].value = 0
        self.prog["env"].value = 1
        boxes, info = [], []
        for i, hand in enumerate(hands):
            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in hand.landmarks])
            pts = self.smoothers.setdefault(i, _Smoother())(pts, t)
            # forma da mano sinistra nell'immagine = tua mano destra (immagine specchiata):
            # guanto destro con le gemme, specchiato. Altrimenti il sinistro (salvato già specchiato).
            mirror = (hand.handedness == "right") != self.flip_side
            model = self.right if mirror else self.left
            mats, s, R, tr, Rm = model.pose(pts, mirror)
            arr = np.zeros((N_BONES, 4, 4), np.float32)
            arr[:len(mats)] = np.array([m.T for m in mats], np.float32)
            self.prog["bones"].write(arr.tobytes())
            self._use_materials(model)
            model.piece_tex.use(0)
            model.vao.render()
            size = float(np.linalg.norm(pts[9] - pts[0]))
            x0, y0 = pts[:, :2].min(0) - size * 1.6
            x1, y1 = pts[:, :2].max(0) + size * 1.6
            boxes.append((x0, y0, x1, y1))
            normal = R @ Rm @ model.palm_normal
            rep = s * R @ Rm @ model.repulsor + tr
            gems = [((mats[b][:3, :3] @ c + mats[b][:3, 3])[:2], col) for c, b, col in model.gems]
            info.append(dict(repulsor=rep[:2], facing=normal[2] < -0.25, size=size, gems=gems))
        self.ctx.copy_framebuffer(self.fbo, self.ms_fbo)
        for box in boxes:
            self._composite(img, box)
        return info

    def _update_env(self, img) -> None:
        small = cv2.resize(img, (256, 144), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.env is None:
            self.env = self.ctx.texture((256, 144), 3, rgb.tobytes())
            self.env.repeat_x = self.env.repeat_y = False
        else:
            self.env.write(rgb.tobytes())
        self.env.build_mipmaps()

    def _composite(self, img, box) -> None:
        h, w = img.shape[:2]
        x0, y0 = int(max(box[0], 0)), int(max(box[1], 0))
        x1, y1 = int(min(box[2], w)), int(min(box[3], h))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        raw = self.fbo.read(viewport=(x0, h - y1, x1 - x0, y1 - y0), components=4)
        layer = np.frombuffer(raw, np.uint8).reshape(y1 - y0, x1 - x0, 4)[::-1].astype(np.float32) / 255
        a = layer[..., 3:4]
        if not a.any():
            return
        rgb = layer[..., 2::-1]  # RGB -> BGR
        roi = img[y0:y1, x0:x1].astype(np.float32) / 255
        size = max((x1 - x0) / 4, 8)
        shadow = cv2.GaussianBlur(a[..., 0], (0, 0), size * 0.05)
        shadow = cv2.warpAffine(shadow, np.float32([[1, 0, size * 0.04], [0, 1, size * 0.07]]),
                                (x1 - x0, y1 - y0))
        roi *= (1 - 0.45 * shadow)[..., None]
        out = roi * (1 - a) + rgb  # rgb è già premoltiplicato per alpha
        img[y0:y1, x0:x1] = (np.clip(out, 0, 1) * 255).astype(np.uint8)


def load_gauntlet(path: Path) -> Optional[Gauntlet]:
    """path = guanto destro (gemme); il sinistro è accanto, gauntlet_left.npz."""
    if not path.exists():
        return None
    try:
        return Gauntlet(path, path.with_name("gauntlet_left.npz"))
    except Exception as e:  # OpenGL non disponibile o file non valido
        print(f"[guanto 3D] non disponibile ({e}), uso l'armatura semplificata")
        return None
