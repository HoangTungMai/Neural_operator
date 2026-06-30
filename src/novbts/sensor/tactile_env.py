#!/usr/bin/env python3
"""Phase 6a -- differentiable tactile ENV wrapper (the framework core).

This packages the three pieces built across the project -- the frozen forward FNO
surrogate (Phase 3), the differentiable marker-dot sensor (Phase 5), and the
contextual control policy (Phase 4) -- behind ONE small env API. That is what turns
the pile of scripts into a usable *differentiable VBTS simulation framework* (Track A).

Honest scope: the FNO is a STATIC one-step input->field map, so this is a single-step
CONTEXTUAL (goal-conditioned) env, not multi-step dynamics. `reset()` samples a context
(a sphere indentor pressed at some object mu,E,R with a TARGET tactile imprint to match);
`step(action)` applies a contact action (sx,sy), pushes it through frozen-FNO -> disp
field -> marker-dot sensor -> a camera image, and rewards how close that sensor image is
to the target. The whole transition is torch-differentiable: `differentiable_step` returns
tensors whose grad flows back to the action, so a policy can be trained by backprop.

Geometry is kept to the simplest case (sphere indentor, geom=0) on purpose.

  python -m novbts.sensor.tactile_env --demo
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from novbts.operator.field2field import DEV
from novbts.operator.diff_policy import (
    Setup, context_tensors, context_features, fno_field, PolicyMLP,
)
from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, render_dots, sample_field_to_markers,
    sensor_marker_grid_pixel_even, marker_half_extent,
)
from novbts.sensor.realism import add_camera_noise
from novbts.paths import FEM, RUNS, ensure


# ---------------------------------------------------------------------------
# the env
# ---------------------------------------------------------------------------

class TactileEnv:
    """Single-step contextual differentiable VBTS env.

    observation  = rendered marker-dot camera image [B,1,px,px] (the real sensor output)
    action       = contact shear (sx,sy) in metres, [B,2]
    reward       = -||sensor(action) - target||^2   (image-space servo, default)

    The Phase-4 policy consumes the COMPACT context feature (object scalars + a summary
    of the target field), exposed as `self.feat`; the image is the raw sensor view. Both
    describe the same context -- the feature is what a small MLP policy was trained on,
    the image is what a vision policy / real sensor would see.
    """

    def __init__(self, S, *, sensor_side=11, px=64, sigma=1.35, working_dist=0.05,
                 background=0.72, contrast=0.58, reward_mode="image",
                 noise_read=0.0, photons=300.0, seed=0):
        self.S = S
        self.coords = S.coords
        self.dense_t = torch.tensor(self.coords, device=DEV)
        self.px, self.sigma, self.reward_mode = px, sigma, reward_mode
        self.noise_read, self.photons = noise_read, photons

        self.cam = PinholeCamera.from_gel(marker_half_extent(self.coords), px=px,
                                          working_dist=working_dist)
        sc = sensor_marker_grid_pixel_even(self.cam, sensor_side, pixel_fill=0.75)
        self.sensor_t = torch.tensor(sc, device=DEV)
        self.m = sc.shape[0]
        self.render_kw = dict(background=background, contrast=contrast,
                              polarity="dark", saturate=True)
        self.pix_rest = self.cam.project(
            deformed_marker_xyz(self.sensor_t, torch.zeros(1, self.m, 3, device=DEV)))  # [1,m,2]

        # sphere-only context pools (geom==0) -> simplest indentor
        geom = S.P[:, 8]
        tr = S.tr.cpu().numpy(); te = S.te.cpu().numpy()
        self.pool_tr = tr[geom[tr] < 0.5]
        self.pool_te = te[geom[te] < 0.5]
        if len(self.pool_tr) == 0 or len(self.pool_te) == 0:
            raise SystemExit("no sphere (geom=0) frames in train/test split")

        # action box + policy-feature normalisation, fixed from the train split
        self.a_scale = torch.tensor([float(np.abs(S.P[:, 4:6]).max())] * 2,
                                    dtype=torch.float32, device=DEV)
        ctx_tr = context_tensors(S, torch.tensor(self.pool_tr, device=DEV))
        feat_tr = context_features(ctx_tr)
        self.feat_m = feat_tr.mean(0, keepdim=True)
        self.feat_s = feat_tr.std(0, keepdim=True).clamp_min(1e-6)
        self.feat_dim = feat_tr.shape[1]

        self.rng = np.random.default_rng(seed)
        self.idx = self.ctx = self.feat = None
        self.target_img = self.target_flow = self.true_action = None

    # -- rendering (differentiable) ----------------------------------------

    def _field_to_image_flow(self, field):
        """field [B,3,H,W] raw disp -> (image [B,1,px,px], flow [B,m,2]). Differentiable."""
        mk = sample_field_to_markers(field, self.dense_t, self.sensor_t)        # [B,m,3]
        pix = self.cam.project(deformed_marker_xyz(self.sensor_t, mk))          # [B,m,2]
        flow = pix - self.pix_rest                                             # [B,m,2]
        img = render_dots(pix, self.px, self.px, self.sigma, **self.render_kw)  # [B,1,H,W]
        return img, flow

    def _render_chunked(self, field, chunk=32):
        """no_grad render of a (possibly large) batch, chunked to bound memory."""
        imgs, flows = [], []
        with torch.no_grad():
            for i in range(0, field.shape[0], chunk):
                im, fl = self._field_to_image_flow(field[i:i + chunk])
                imgs.append(im); flows.append(fl)
        return torch.cat(imgs), torch.cat(flows)

    # -- env API -----------------------------------------------------------

    def reset(self, split="test", batch=None, idx=None):
        """Sample a context (sphere indentor + target tactile imprint).

        Returns the TARGET sensor image (the goal to servo toward). The matching
        policy input is cached in `self.feat`; the true action that produced the
        target is `self.true_action` (oracle reference)."""
        if idx is None:
            pool = self.pool_te if split == "test" else self.pool_tr
            batch = batch or len(pool)
            sel = self.rng.choice(pool, size=min(batch, len(pool)), replace=False)
            idx = torch.tensor(np.sort(sel), device=DEV)
        self.idx = idx
        self.ctx = context_tensors(self.S, idx)
        self.target_img, self.target_flow = self._render_chunked(self.ctx["ystar"])
        self.feat = (context_features(self.ctx) - self.feat_m) / self.feat_s
        self.true_action = self.ctx["params"][:, 4:6]
        return self.target_img

    def differentiable_step(self, action):
        """action [B,2] (matching the current ctx) -> (obs image, reward [B], info).

        Fully differentiable in `action`: obs = sensor(FNO(action, ctx)),
        reward = -mean square error to the target (image or flow space)."""
        field = fno_field(self.S, action, self.ctx)            # [B,3,H,W] raw disp
        img, flow = self._field_to_image_flow(field)
        if self.reward_mode == "image":
            reward = -((img - self.target_img) ** 2).mean(dim=(1, 2, 3))
        else:
            reward = -((flow - self.target_flow) ** 2).mean(dim=(1, 2))
        return img, reward, {"field": field, "flow": flow}

    @torch.no_grad()
    def step(self, action):
        """Gym-style detached step. Adds camera noise to the returned observation
        (realistic sensor view); the reward is still computed on the clean image."""
        a = action if torch.is_tensor(action) else torch.tensor(
            np.asarray(action, np.float32), device=DEV)
        if a.ndim == 1:
            a = a[None]
        img, reward, info = self.differentiable_step(a)
        obs = img
        if self.noise_read > 0:
            obs = add_camera_noise(img, photons=self.photons, read_noise=self.noise_read)
        return obs, reward, True, False, {"flow": info["flow"]}

    def random_action(self, batch):
        """Uniform action in the physical box [-a_scale, a_scale]."""
        u = torch.tensor(self.rng.uniform(-1, 1, size=(batch, 2)).astype(np.float32), device=DEV)
        return u * self.a_scale


# ---------------------------------------------------------------------------
# optional gymnasium adapter (obs/action spaces); falls back to the plain class
# ---------------------------------------------------------------------------

def make_gym_env(S, **kw):
    """Return a gymnasium.Env wrapping TactileEnv if gymnasium is importable, else None."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception:
        return None

    class GymTactileEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.env = TactileEnv(S, **kw)
            a = float(self.env.a_scale[0])
            self.action_space = spaces.Box(-a, a, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(
                0.0, 1.0, shape=(1, self.env.px, self.env.px), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs = self.env.reset(split="test", batch=1)
            return obs[0].cpu().numpy(), {"feat": self.env.feat[0].cpu().numpy()}

        def step(self, action):
            obs, reward, term, trunc, info = self.env.step(action)
            return obs[0].cpu().numpy(), float(reward[0]), True, False, info

    return GymTactileEnv()


# ---------------------------------------------------------------------------
# policy training THROUGH the env (proves the env is the integration point)
# ---------------------------------------------------------------------------

def train_policy_in_env(env, *, steps=300, bs=32, lr=1e-2, seed=0):
    """Train a Phase-4 PolicyMLP by backprop through env.differentiable_step.

    Each step: reset to a fresh train batch -> action=policy(feat) -> reward ->
    loss=-reward.mean() -> Adam. Returns the trained policy and the reward curve."""
    torch.manual_seed(seed)
    policy = PolicyMLP(env.feat_dim, 2, env.a_scale).to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve = []
    for step in range(steps):
        env.reset(split="train", batch=bs)
        action = policy(env.feat)
        _, reward, _ = env.differentiable_step(action)
        loss = -reward.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 25 == 0 or step == steps - 1:
            curve.append((step, float(reward.mean().detach())))
    return policy, curve


@torch.no_grad()
def mean_reward(env, action_fn, *, split="test", batch=None, chunk=64):
    """Mean reward of an action policy over a reset context, chunked for memory.
    action_fn(env) -> [B,2] action, sized to the current (chunked) ctx."""
    env.reset(split=split, batch=batch)
    n = env.idx.shape[0]
    full = dict(ctx=env.ctx, feat=env.feat, img=env.target_img,
                flow=env.target_flow, true=env.true_action)
    rs = []
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        env.ctx = {k: v[sl] for k, v in full["ctx"].items()}
        env.feat = full["feat"][sl]
        env.target_img, env.target_flow = full["img"][sl], full["flow"][sl]
        env.true_action = full["true"][sl]
        rs.append(action_and_reward(env, action_fn))
    env.ctx, env.feat = full["ctx"], full["feat"]
    env.target_img, env.target_flow = full["img"], full["flow"]
    env.true_action = full["true"]
    return torch.cat(rs)


def action_and_reward(env, action_fn):
    action = action_fn(env)
    return env.differentiable_step(action)[1]


def gradcheck_action(env, *, batch=4, eps=None):
    """Finite-difference gradcheck: confirm reward grad flows to the action.

    The frozen FNO is float32, so we check agreement at achievable precision via
    central differences rather than a float64 torch.autograd.gradcheck."""
    env.reset(split="test", batch=batch)
    n = env.idx.shape[0]
    a0 = env.true_action.clone().detach()
    eps = eps if eps is not None else float(env.a_scale[0]) * 1e-3

    a = a0.clone().requires_grad_(True)
    _, reward, _ = env.differentiable_step(a)
    g_auto = torch.autograd.grad(reward.sum(), a)[0]                 # [n,2]

    g_num = torch.zeros_like(a0)
    for i in range(n):
        for j in range(2):
            ap = a0.clone(); ap[i, j] += eps
            am = a0.clone(); am[i, j] -= eps
            with torch.no_grad():
                rp = env.differentiable_step(ap)[1].sum()
                rm = env.differentiable_step(am)[1].sum()
            g_num[i, j] = (rp - rm) / (2 * eps)
    rel = float((g_auto - g_num).norm() / (g_num.norm() + 1e-12))
    return {"grad_norm_autograd": float(g_auto.norm()),
            "grad_norm_numeric": float(g_num.norm()),
            "rel_error": rel, "flows_to_action": bool(g_auto.norm() > 0),
            "passed": bool(rel < 0.05 and g_auto.norm() > 0)}


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

def rollout_preview(env, policy, out_path, k=4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"plot_error: {e}"
    env.reset(split="test", batch=k)
    pr = env.pix_rest[0].cpu().numpy()
    with torch.no_grad():
        a_pol = policy(env.feat)
        a_rnd = env.random_action(env.idx.shape[0])
        img_pol, _, info_pol = env.differentiable_step(a_pol)
        img_rnd, _, info_rnd = env.differentiable_step(a_rnd)
    tgt = env.target_img.cpu().numpy(); tflow = env.target_flow.cpu().numpy()
    ip = img_pol.cpu().numpy(); fp = info_pol["flow"].cpu().numpy()
    ir = img_rnd.cpu().numpy(); fr = info_rnd["flow"].cpu().numpy()
    im_kw = dict(cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    cols = [("target (goal)", tgt, tflow), ("random action", ir, fr), ("policy action", ip, fp)]
    fig, axes = plt.subplots(k, 3, figsize=(9, 3 * k), squeeze=False)
    for r in range(k):
        for c, (title, imgs, flows) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(imgs[r, 0], **im_kw)
            fl = flows[r]
            ax.quiver(pr[:, 0], pr[:, 1], fl[:, 0], -fl[:, 1], color="red",
                      scale_units="xy", angles="xy", scale=0.5, width=0.005)
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xlim(0, env.px); ax.set_ylim(env.px, 0)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Phase 6a tactile env: target vs random vs Phase-4 policy", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return str(out_path)


def run_demo(S, args):
    env = TactileEnv(S, sensor_side=args.sensor_side, px=args.px,
                     reward_mode=args.reward_mode, noise_read=args.noise_read)
    print(f"env: sphere-only  train={len(env.pool_tr)} test={len(env.pool_te)}  "
          f"feat_dim={env.feat_dim}  px={env.px}  markers={env.m}  reward={env.reward_mode}")

    gym_env = make_gym_env(S, sensor_side=args.sensor_side, px=args.px,
                           reward_mode=args.reward_mode)
    print(f"gymnasium adapter: {'available' if gym_env is not None else 'not installed (plain class)'}")

    gc = gradcheck_action(env, batch=args.gradcheck_batch)
    print(f"gradcheck: ||g_auto||={gc['grad_norm_autograd']:.3e}  "
          f"||g_num||={gc['grad_norm_numeric']:.3e}  rel_err={gc['rel_error']:.2e}  "
          f"passed={gc['passed']}")

    t0 = time.perf_counter()
    policy, curve = train_policy_in_env(env, steps=args.steps, bs=args.bs, lr=args.policy_lr)
    train_s = time.perf_counter() - t0

    r_rand = mean_reward(env, lambda e: e.random_action(e.feat.shape[0]), split="test")
    r_true = mean_reward(env, lambda e: e.true_action, split="test")
    r_pol = mean_reward(env, lambda e: policy(e.feat), split="test")
    rand_m, pol_m, true_m = float(r_rand.mean()), float(r_pol.mean()), float(r_true.mean())
    # fraction of the random->oracle reward gap the policy closes
    closed = (pol_m - rand_m) / (true_m - rand_m + 1e-12)
    improves = pol_m > rand_m
    print(f"\nmean reward (test, higher=better):")
    print(f"  random action : {rand_m:.4e}")
    print(f"  Phase-4 policy: {pol_m:.4e}")
    print(f"  true action   : {true_m:.4e}  (oracle reference)")
    print(f"policy closes {closed*100:.0f}% of random->oracle gap   "
          f"improves_over_random={improves}")

    phase_dir = RUNS / "phase6"; ensure(phase_dir)
    preview = rollout_preview(env, policy, phase_dir / "env_demo.png", k=args.preview_k)
    out = {"gt": os.path.basename(args.data), "gt_path": args.data,
           "data": args.data, "device": str(DEV), "geometry": "sphere-only",
           "n_train": len(env.pool_tr), "n_test": len(env.pool_te),
           "px": env.px, "markers": env.m, "feat_dim": env.feat_dim,
           "reward_mode": env.reward_mode, "noise_read": args.noise_read,
           "gymnasium": gym_env is not None,
           "fno": {"train_s": round(S.fno_train_s, 1), "params": S.fno_params},
           "gradcheck": gc, "policy_train_s": round(train_s, 1),
           "reward_curve": curve,
           "reward": {"random": rand_m, "policy": pol_m, "true_oracle": true_m,
                      "gap_closed_frac": closed, "policy_improves_over_random": improves},
           "preview": preview}
    json.dump(out, open(phase_dir / "env_demo.json", "w"), indent=2, default=float)
    print(f"\nsaved {phase_dir/'env_demo.json'}")
    print(f"saved {preview}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sensor-side", type=int, default=11)
    ap.add_argument("--px", type=int, default=64)
    ap.add_argument("--reward-mode", default="image", choices=["image", "flow"])
    ap.add_argument("--noise-read", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--policy-lr", type=float, default=1e-2)
    ap.add_argument("--gradcheck-batch", type=int, default=4)
    ap.add_argument("--preview-k", type=int, default=4)
    args = ap.parse_args()

    S = Setup(args)
    if args.demo:
        run_demo(S, args)
    else:
        print("nothing to do: pass --demo")


if __name__ == "__main__":
    main()
