"""The numpy side of the Blender add-on (no bpy): frame conversion into Blender, SH rotation, and b2ctrain's posing
rule (cage.cu pose_bound) as a reference for the Geometry Nodes graph that does it in Blender.

Frames. Every b2c array is in the b2crunner frame (SPEC 2). glTF world is W · p, and Blender's glTF importer turns
glTF's +Y up into Blender's +Z up with `C` (x, y, z) -> (x, -z, y). So a b2c point lands at `C W p` in Blender and a
b2c direction at `(C W_3x3) d`; `W` must be a rotation times a uniform scale.
"""
from __future__ import annotations

import numpy as np

C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)   # glTF Y-up -> Blender Z-up (the importer's)
SH_BANDS = {1: (0, 3), 2: (3, 8), 3: (8, 15)}                  # band -> radiance:sh_k index range (DC excluded)


class Frame:
    """The b2crunner frame -> Blender: p -> R p * s + t (R a rotation, s a uniform scale)."""

    def __init__(self, W: np.ndarray):
        M = C @ np.asarray(W, np.float64)[:3, :3]
        s = np.cbrt(np.linalg.det(M))
        R = M / s
        if not (s > 0 and np.allclose(R @ R.T, np.eye(3), atol=1e-5)):
            raise ValueError("the subject's W is not a rotation with a uniform scale")
        self.R, self.s, self.t = R, float(s), C @ np.asarray(W, np.float64)[:3, 3]
        self.M = M                                             # the linear part, R * s

    def points(self, p: np.ndarray) -> np.ndarray:
        return p @ self.M.T + self.t

    def vectors(self, v: np.ndarray) -> np.ndarray:
        return v @ self.M.T

    @property
    def matrix(self) -> np.ndarray:
        out = np.eye(4); out[:3, :3] = self.M; out[:3, 3] = self.t
        return out


# ---------------------------------------------------------------- quaternions (w x y z, Blender's order)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] -> w x y z with w >= 0."""
    R = np.asarray(R, np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    q = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    k = np.argmax(q, -1)
    out = np.empty(R.shape[:-2] + (4,))
    s = np.sqrt(np.maximum(np.take_along_axis(q, k[..., None], -1)[..., 0], 1e-30)) * 2
    r = lambda i, j: R[..., i, j]
    cand = [np.stack([s / 4, (r(2, 1) - r(1, 2)) / s, (r(0, 2) - r(2, 0)) / s, (r(1, 0) - r(0, 1)) / s], -1),
            np.stack([(r(2, 1) - r(1, 2)) / s, s / 4, (r(0, 1) + r(1, 0)) / s, (r(0, 2) + r(2, 0)) / s], -1),
            np.stack([(r(0, 2) - r(2, 0)) / s, (r(0, 1) + r(1, 0)) / s, s / 4, (r(1, 2) + r(2, 1)) / s], -1),
            np.stack([(r(1, 0) - r(0, 1)) / s, (r(0, 2) + r(2, 0)) / s, (r(1, 2) + r(2, 1)) / s, s / 4], -1)]
    for i in range(4):
        out[k == i] = cand[i][k == i]
    out *= np.where(out[..., :1] < 0, -1, 1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:], q[..., :3]], -1)


# ---------------------------------------------------------------- spherical harmonics (the 3DGS real basis)

def sh_basis(l: int, d: np.ndarray) -> np.ndarray:
    """Band l of the 3DGS SH basis at unit directions d [..., 3] -> [..., 2l + 1]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    if l == 1:
        c = 0.4886025119029199
        return np.stack([-c * y, c * z, -c * x], -1)
    if l == 2:
        c = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
        return np.stack([c[0] * x * y, c[1] * y * z, c[2] * (2 * z * z - x * x - y * y), c[3] * x * z, c[4] * (x * x - y * y)], -1)
    c = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
         1.445305721320277, -0.5900435899266435]
    xx, yy, zz = x * x, y * y, z * z
    return np.stack([c[0] * y * (3 * xx - yy), c[1] * x * y * z, c[2] * y * (4 * zz - xx - yy),
                     c[3] * z * (2 * zz - 3 * xx - 3 * yy), c[4] * x * (4 * zz - xx - yy), c[5] * z * (xx - yy),
                     c[6] * x * (xx - 3 * yy)], -1)


def sample_dirs(l: int) -> tuple[np.ndarray, np.ndarray]:
    """2l + 1 fixed unit directions whose band-l basis matrix is well conditioned, and that matrix's inverse. A band's
    coefficients are recovered exactly from its values there, which is how SH get rotated (here and in the node graph)."""
    rng = np.random.default_rng(l)
    best = None
    for _ in range(200):
        D = rng.normal(size=(2 * l + 1, 3)); D /= np.linalg.norm(D, axis=1, keepdims=True)
        Y = sh_basis(l, D); cnd = np.linalg.cond(Y)
        if best is None or cnd < best[0]:
            best = (cnd, D, Y)
    return best[1], np.linalg.inv(best[2])


def rotate_sh(sh: np.ndarray, R: np.ndarray) -> np.ndarray:
    """SH coefficients without DC [n, K, 3] (K = 3, 8 or 15), rotations [n, 3, 3] or [3, 3] -> the coefficients of
    g(d) = f(R^T d), i.e. the radiance turned with the splat."""
    out = np.array(sh, np.float64, copy=True)
    for l, (a, b) in SH_BANDS.items():
        if b > sh.shape[1]:
            break
        D, Yinv = sample_dirs(l)
        Dr = np.einsum("...ji,kj->...ki", R, D)                                # R^T d_k
        Y = np.broadcast_to(sh_basis(l, Dr), (len(sh), 2 * l + 1, 2 * l + 1))
        out[:, a:b] = np.einsum("mk,nkc->nmc", Yinv, np.einsum("nkm,nmc->nkc", Y, sh[:, a:b]))
    return out


# ---------------------------------------------------------------- the splat in Blender's frame

def splat_to_blender(s: dict, frame: Frame) -> dict:
    """read.splat()'s display-form arrays -> Blender's GAUSSIAN_SPLAT attributes (positions, w x y z rotations,
    linear scales, radiance:base = (SH DC, opacity), radiance:sh_k)."""
    sh = np.asarray(s["sh"], np.float64)
    rot = frame.R @ quat_to_mat(xyzw_to_wxyz(np.asarray(s["rot"], np.float64)))
    return {"position": frame.points(np.asarray(s["pos"], np.float64)), "rotation": mat_to_quat(rot),
            "scale": np.asarray(s["scale"], np.float64) * frame.s,
            "base": np.c_[sh[:, 0], s["opacity"]], "sh": rotate_sh(sh[:, 1:], frame.R)}


# ---------------------------------------------------------------- b2ctrain's posing rule (the reference)

def tri_frames(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cage.cu tri_frame: columns (e1 along the first edge, n x e1, n) and the size k = sqrt(|(b - a) x (c - a)|)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    u, w = b - a, c - a
    nn = np.cross(u, w); nl = np.linalg.norm(nn, axis=1); ul = np.linalg.norm(u, axis=1)
    e1 = u / np.maximum(ul, 1e-12)[:, None]; n = nn / np.maximum(nl, 1e-20)[:, None]
    return np.stack([e1, np.cross(n, e1), n], -1), np.sqrt(nl)


def pose_reference(V0, P, F, face, bary, off, splat: dict, max_growth: float) -> dict:
    """b2ctrain's cage.cu pose_bound for every splat (stretch fade, fill and gates off), in whatever frame V0 / P /
    splat share: canonical cage V0, posed cage P, binding (face, barycentrics, offset) -> posed position, rotation,
    scale and SH."""
    R0, k0 = tri_frames(V0, F); R, k = tri_frames(P, F)
    if max_growth > 0:
        k = np.clip(k, k0 / max_growth, k0 * max_growth)
    bnd = face >= 0; f = face[bnd]; t = F[f]; b = bary[bnd]
    out = {"position": splat["position"].copy(), "rotation": splat["rotation"].copy(), "scale": splat["scale"].copy(),
           "sh": splat["sh"].copy()}
    foot = P[t[:, 0]] * (1 - b[:, :1] - b[:, 1:]) + P[t[:, 1]] * b[:, :1] + P[t[:, 2]] * b[:, 1:]
    out["position"][bnd] = foot + np.einsum("nij,nj->ni", R[f], off[bnd]) * k[f, None]
    dq = R[f] @ np.transpose(R0[f], (0, 2, 1))
    out["rotation"][bnd] = mat_to_quat(dq @ quat_to_mat(splat["rotation"][bnd]))
    out["scale"][bnd] = splat["scale"][bnd] * np.where(k0[f] > 0, k[f] / np.maximum(k0[f], 1e-30), 1)[:, None]
    out["sh"][bnd] = rotate_sh(splat["sh"][bnd], dq)
    return out
