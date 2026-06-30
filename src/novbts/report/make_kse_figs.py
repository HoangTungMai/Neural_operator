"""Regenerate the KSE2026 paper figures from the canonical run JSONs.

The paper figures must always reflect the *current* benchmark/policy JSONs (the
ground truth switched PhysX -> IPC/GIPC, and the figures were previously hand-made
and went stale). This module is the single reproducible source for them.

Figures written to ``docs/kse2026/figs/``:
  - fidelity_speed.png      (RQ1/RQ3: rel-L2 vs throughput)   <- runs/phase3_fem/benchmark.json
  - policy_servo_curve.png  (RQ4: control convergence)        <- runs/phase4/policy_servo.json

The sensor panel (sensor_gt_vs_fno.png) needs the differentiable renderer and is
produced by the sensor pipeline, not here.

Run:  python -m novbts.report.make_kse_figs
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from novbts import paths

FIGS = paths.DOCS / "kse2026" / "figs"
BENCH = paths.RUNS / "phase3_fem" / "benchmark.json"
POLICY = paths.RUNS / "phase4" / "policy_servo.json"


def fig_fidelity_speed() -> None:
    """RQ1xRQ3: relative-L2 (lower = better) vs inference throughput (log)."""
    b = json.load(open(BENCH))
    rl2 = lambda m: b["RQ1"][m]["relative_l2"]["overall"]
    fps = b["RQ3"]["throughput_fps"]
    # (name, fps, rel-L2, colour, label offset in points, ha)
    pts = [
        ("MLP", fps["mlp"], rl2("mlp"), "tab:blue", (-10, 6), "right"),
        ("FNO", fps["fno"], rl2("fno"), "tab:orange", (10, -14), "left"),
        ("FNO+slip(a)", fps["fno_mt_a"], rl2("fno_mt_a"), "tab:green", (-10, 8), "right"),
    ]
    solver_fps = fps.get("gt_solver", fps["physx_fem_shear_solver"])
    solver_k3_fps = fps.get("gt_solver_k3_averaged")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, x, y, c, off, ha in pts:
        ax.scatter([x], [y], s=130, color=c, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off,
                    fontsize=11, ha=ha)

    ax.set_xscale("log")
    ax.set_xlabel("throughput (frames/s, log)")
    ax.set_ylabel("relative L2 (lower = more accurate)")
    ax.set_title("Fidelity vs. speed (IPC/GIPC ground truth)")
    ax.set_ylim(0.0, max(p[2] for p in pts) * 1.18)
    xmin = (solver_k3_fps or solver_fps) * 0.55
    ax.set_xlim(xmin, fps["mlp"] * 3.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.axvline(solver_fps, ls="--", color="0.45", lw=1.4, zorder=1)
    ax.text(solver_fps * 1.35, 0.30,
            f"single IPC solve\n{solver_fps:.3f} fps",
            rotation=90, va="center", ha="left", color="0.35", fontsize=8.5)
    if solver_k3_fps is not None:
        ax.axvline(solver_k3_fps, ls=":", color="0.55", lw=1.4, zorder=1)
        ax.text(solver_k3_fps * 1.35, 0.11,
                f"K=3 target\n{solver_k3_fps:.3f} fps",
                rotation=90, va="center", ha="left", color="0.45", fontsize=8.5)
    fig.tight_layout()
    out = FIGS / "fidelity_speed.png"
    run_out = paths.RUNS / "phase3_fem" / "fidelity_speed.png"
    paths.ensure(run_out.parent)
    fig.savefig(out, dpi=150)
    fig.savefig(run_out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} and {run_out}  (FNO {rl2('fno'):.3f} @ {fps['fno']:.0f} fps, "
          f"MLP {rl2('mlp'):.3f}, solver {solver_fps:.3f} fps)")


def fig_policy_servo() -> None:
    """RQ4: validation loss vs cumulative forward queries (both log)."""
    p = json.load(open(POLICY))
    ag = p["autograd"]["curve_mean"]
    es = p["es"]["curve_mean"]
    floor = p["references"]["per_instance_floor"]
    baseline = p["references"]["mean_action_baseline"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([q for q, _ in ag], [v for _, v in ag], "-o", ms=3,
            color="tab:blue", label="autograd (policy)")
    ax.plot([q for q, _ in es], [v for _, v in es], "-o", ms=3,
            color="tab:orange", label="es (policy)")
    ax.axhline(floor, ls="--", color="tab:green", lw=1.6,
               label="per-instance oracle floor")
    ax.axhline(baseline, ls=":", color="0.5", lw=1.6,
               label="mean-action baseline")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative FNO-env forward queries (log)")
    ax.set_ylabel("validation loss (log)")
    ax.set_title("Phase 4: differentiable (autograd) vs gradient-free (ES) policy")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = FIGS / "policy_servo_curve.png"
    run_out = paths.RUNS / "phase4" / "policy_servo_curve.png"
    paths.ensure(run_out.parent)
    fig.savefig(out, dpi=150)
    fig.savefig(run_out, dpi=150)
    plt.close(fig)
    print(f"wrote {out} and {run_out}  (autograd final {ag[-1][1]:.2e}, es final {es[-1][1]:.2e}, "
          f"floor {floor:.2e})")


def main() -> None:
    paths.ensure(FIGS)
    fig_fidelity_speed()
    fig_policy_servo()


if __name__ == "__main__":
    main()
