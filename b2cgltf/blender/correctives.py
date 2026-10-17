"""MHR's pose correctives without torch: the network as numpy arrays, its input from joint rotations, and the
skinning that carries its output into the posed frame. Used by the Blender add-on (no bpy here).

MHR (the TorchScript `mhr_model.pt`) adds a corrective to its unposed mesh before skinning:
  input   the first two columns of prerot_j^-1 L_j, minus identity, for joints 2.. (L_j the joint's local rotation);
          a b2c subject's joint nodes are MHR's joint frames, so L_j is the glTF node rotation (checked to 1e-7);
  layer 1 sparse 750 -> 3000, ReLU;  layer 2 sparse 3000 -> 3 x 18439 (MHR units, cm).
It is not the face: expressions are MHR's separate linear blendshapes.

The weights come from the model, which b2c files reference but never embed (SPEC 4.3). Export them once, where torch
is installed:
    python -m b2cgltf.blender.correctives [MODEL.pt] OUT.npz
MODEL.pt defaults to SPEC 7.3's lookup ($B2C_MHR_MODEL_DIR, then the Hugging Face cache).
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

import numpy as np

from .posing import quat_to_mat, xyzw_to_wxyz

MODEL_REPO, MODEL_FILE = "facebook/sam-3d-body-dinov3", "assets/mhr_model.pt"


class Net:
    """The corrective network, from export()'s arrays."""

    def __init__(self, z):
        self.l1_rows, self.l1_cols = z["l1_rows"], z["l1_cols"]
        self.l1_vals = np.asarray(z["l1_vals"], np.float64)
        self.n_hidden, self.n_out = int(z["n_hidden"]), int(z["n_out"])
        self.rows, self.vals = z["rows"], z["vals"]
        self.unit = np.repeat(np.arange(self.n_hidden, dtype=np.int32), np.diff(z["col_ptr"]))
        self.preT = np.swapaxes(quat_to_mat(xyzw_to_wxyz(np.asarray(z["prerot"], np.float64))), -1, -2)
        self.first = int(z["first_joint"])
        self.Libm = quat_to_mat(xyzw_to_wxyz(np.asarray(z["ibm_q"], np.float64))) * np.asarray(z["ibm_s"])[:, None, None]
        self.model_sha256 = str(z["model_sha256"]) if "model_sha256" in z else ""

    @property
    def joints(self) -> int:
        return len(self.preT)

    def features(self, L: np.ndarray) -> np.ndarray:
        """Local joint rotations [J, 3, 3] -> the network input."""
        R = self.preT @ L
        f = np.concatenate([R[:, :, 0], R[:, :, 1]], 1)[self.first:].copy()
        f[:, 0] -= 1; f[:, 4] -= 1
        return f.ravel()

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Network input -> the unposed corrective [n_out / 3, 3], MHR units (cm)."""
        h = np.maximum(np.bincount(self.l1_rows, self.l1_vals * f[self.l1_cols], self.n_hidden), 0).astype(np.float32)
        return np.bincount(self.rows, self.vals * h[self.unit], self.n_out).reshape(-1, 3)


def local_rotations(G: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Joint rotations relative to the skeleton root [..., J, 3, 3] -> relative to the parent joint."""
    L = G.copy()
    has = parents >= 0
    L[..., has, :, :] = np.swapaxes(G[..., parents[has], :, :], -1, -2) @ G[..., has, :, :]
    return L


def skin(G: np.ndarray, Libm: np.ndarray, idx: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unposed vectors d [n, 3] -> posed, as MHR skins them: sum_k w_k (G_j Libm_j) d with j = idx[:, k]. G: joint
    rotations [J, 3, 3] in the frame the result is wanted in."""
    M = G @ Libm
    out = np.zeros_like(d, dtype=np.float64)
    for k in range(idx.shape[1]):
        out += w[:, k, None] * np.einsum("nab,nb->na", M[idx[:, k]], d)
    return out


# ---------------------------------------------------------------- export (torch, outside Blender)

def find_model() -> str | None:
    """SPEC 7.3: $B2C_MHR_MODEL_DIR/<file name>, then the Hugging Face cache (local files only)."""
    d = os.environ.get("B2C_MHR_MODEL_DIR")
    if d and os.path.isfile(os.path.join(d, os.path.basename(MODEL_FILE))):
        return os.path.join(d, os.path.basename(MODEL_FILE))
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(os.path.join(hub, "models--" + MODEL_REPO.replace("/", "--"), "snapshots", "*", MODEL_FILE)))
    return hits[-1] if hits else None


def export(model_path: str, out: str) -> dict:
    """mhr_model.pt -> OUT.npz (the network, the joint pre-rotations, the inverse bind pose), checked against the
    TorchScript on random poses. Returns a report."""
    import torch
    sha = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    m = torch.jit.load(model_path, map_location="cpu")
    pc = m.pose_correctives_model
    l1, l2 = pc.pose_dirs_predictor._modules["0"], pc.pose_dirs_predictor._modules["2"]
    idx = l1.sparse_indices.numpy(); w1 = l1.sparse_weight.numpy()
    cols = l2.weight.detach().numpy().T                                          # [3000, n_out]
    nz = [np.nonzero(c)[0] for c in cols]
    col_ptr = np.concatenate([[0], np.cumsum([len(r) for r in nz])]).astype(np.int64)
    sk = m.character_torch.skeleton
    ibm = m.character_torch.linear_blend_skinning.inverse_bind_pose.numpy().astype(np.float64)   # t(3) q(x y z w) s
    arrays = dict(l1_rows=idx[0].astype(np.int32), l1_cols=idx[1].astype(np.int32), l1_vals=w1.astype(np.float32),
                  n_hidden=cols.shape[0], n_out=cols.shape[1], col_ptr=col_ptr,
                  rows=np.concatenate(nz).astype(np.int32), vals=np.concatenate([c[r] for c, r in zip(cols, nz)]).astype(np.float32),
                  prerot=sk.joint_prerotations.numpy().astype(np.float64), first_joint=2, ibm_q=ibm[:, 3:7],
                  ibm_s=ibm[:, 7], model_sha256=sha)
    net = Net(arrays)
    jp = torch.zeros(4, net.joints * 7)
    jp.view(4, net.joints, 7)[:, :, 3:6] = torch.from_numpy(np.random.default_rng(0).normal(0, 0.4, (4, net.joints, 3))).float()
    ref = pc(jp).reshape(4, -1, 3).detach().numpy()
    feats = pc._pose_features_from_joint_params(jp).numpy()
    err = max(np.abs(net(f) - r).max() for f, r in zip(feats, ref))
    if err > 1e-3:
        raise RuntimeError(f"the numpy network differs from the TorchScript by {err}")
    np.savez_compressed(out, **arrays)
    return {"out": out, "model_sha256": sha, "entries": int(col_ptr[-1]), "max_err": float(err)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or len(argv) > 2:
        print(__doc__); return 2
    model, out = (argv[0], argv[1]) if len(argv) == 2 else (find_model(), argv[0])
    if not model:
        print(f"no {MODEL_FILE} found ($B2C_MHR_MODEL_DIR or the Hugging Face cache of {MODEL_REPO}); pass it"); return 1
    print(export(model, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
