"""FOSSIL 2.0: 2-D CEGIS counterexample visualisation.

Captures the verifier's 20 random perturbations + exact returned candidate
(last tensor row), separately from the consolidator's gradient-ascent trace.

Public entry point:
    plot_cex_iteration(domains=..., net=..., dynamics=..., cex=...,
                       trajectory=..., iteration=..., output_dir=...)

Supports 2D fossil.domains.Rectangle and fossil.domains.Sphere, with a
closed-loop dynamics instance for controlled systems. Does not mutate the net.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
# Only choose Agg if pyplot has not already been loaded by the application.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as RectanglePatch, Circle
import numpy as np
import torch
import torch.nn.functional as F


LABELS = {"init": "Initial", "unsafe": "Unsafe", "lie": "Lie"}
POINT_COLORS = {"init": "#087f43", "unsafe": "#c62828", "lie": "#d17a00"}


def _to_numpy_2d(value, *, n=2):
    """Return an (N, n) float ndarray, including for empty lists."""
    if value is None:
        return np.empty((0, n), dtype=float)
    if isinstance(value, (list, tuple)) and not value:
        return np.empty((0, n), dtype=float)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, n), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != n:
            raise ValueError(f"Expected a {n}-D point, got shape {arr.shape}")
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n:
        raise ValueError(f"Counterexamples must have shape (N,{n}), got {arr.shape}")
    return arr


def _geometry(geometry):
    """Decode FOSSIL Rectangle/Sphere via their public attributes."""
    if hasattr(geometry, "lower_bounds") and hasattr(geometry, "upper_bounds"):
        lo = np.asarray(geometry.lower_bounds, dtype=float)
        hi = np.asarray(geometry.upper_bounds, dtype=float)
        if lo.shape != (2,) or hi.shape != (2,):
            raise ValueError("Only 2D Rectangle domains can be plotted")
        return "rectangle", lo, hi
    if hasattr(geometry, "centre") and hasattr(geometry, "radius"):
        center = np.asarray(geometry.centre, dtype=float)
        r = float(geometry.radius)
        if center.shape != (2,) or r <= 0:
            raise ValueError("Expected a 2D Sphere with positive radius")
        return "sphere", center, r
    raise TypeError(
        f"Unsupported domain type: {type(geometry).__name__}. "
        "Expected fossil.domains.Rectangle or fossil.domains.Sphere."
    )


def _bounds(domain):
    kind, a, b = _geometry(domain)
    if kind == "rectangle":
        return (a[0], b[0]), (a[1], b[1])
    return (a[0] - b, a[0] + b), (a[1] - b, a[1] + b)


def _inside(domain, xy):
    kind, a, b = _geometry(domain)
    if kind == "rectangle":
        return np.all((xy >= a) & (xy <= b), axis=1)
    return np.sum((xy - a) ** 2, axis=1) <= b**2


def _draw_region(ax, region, *, facecolor, name, fill_alpha=0.16, lw=1.5):
    if region is None:
        return
    kind, a, b = _geometry(region)
    if kind == "rectangle":
        patch = RectanglePatch(
            tuple(a), b[0] - a[0], b[1] - a[1],
            facecolor=facecolor, edgecolor=facecolor,
            alpha=fill_alpha, linewidth=lw, zorder=1,
        )
        ax.add_patch(patch)
        ax.add_patch(RectanglePatch(tuple(a), b[0] - a[0], b[1] - a[1],
                                    facecolor="none", edgecolor=facecolor,
                                    linewidth=lw, zorder=5))
    else:
        ax.add_patch(Circle(tuple(a), b, facecolor=facecolor,
                            edgecolor=facecolor, alpha=fill_alpha, zorder=1))
        ax.add_patch(Circle(tuple(a), b, facecolor="none",
                            edgecolor=facecolor, linewidth=lw, zorder=5))


def _module_spec(net):
    params = list(net.parameters())
    if params:
        return params[0].device, params[0].dtype
    return torch.device("cpu"), torch.float32


def _model_value_and_grad(net, xt):
    """Match LearnerCT semantics: certificate B = NN * configured factor."""
    nn_value = net(xt).reshape(-1)
    factor = net.compute_factors(xt)[0] if hasattr(net, "compute_factors") else 1
    B = nn_value * factor
    gradB = torch.autograd.grad(B.sum(), xt, create_graph=False)[0]
    return B, gradB


def _fields(net, dynamics, xx, yy, *, batch_size=4096, normalize_barrier=False,
            eps_norm=1e-8):
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    B_out, lie_out, dx_out, dy_out = [], [], [], []
    device, dtype = _module_spec(net)
    with torch.enable_grad():
        for lo in range(0, len(grid_points), batch_size):
            segment = grid_points[lo:lo + batch_size]
            xy = torch.as_tensor(segment, dtype=dtype, device=device).clone().detach()
            xy.requires_grad_(True)
            B, gB = _model_value_and_grad(net, xy)
            dyn = dynamics(xy)
            if isinstance(dyn, (list, tuple)):
                dyn = torch.stack(dyn, dim=1)
            if not torch.is_tensor(dyn) or dyn.shape != xy.shape:
                raise ValueError(f"Dynamics must return (N,2) Tensor, got {type(dyn)} / {getattr(dyn, 'shape', '?')}")
            lie = (gB * dyn).sum(dim=1)
            if normalize_barrier:
                gradB_norm = torch.linalg.vector_norm(gB, dim=1).clamp_min(eps_norm)
                B = B / gradB_norm
                lie = F.cosine_similarity(gB, dyn, dim=1, eps=eps_norm)
            B_out.append(B.detach().cpu().numpy())
            lie_out.append(lie.detach().cpu().numpy())
            v = dyn.detach().cpu().numpy()
            dx_out.append(v[:, 0]); dy_out.append(v[:, 1])
    shape = xx.shape
    return tuple(np.concatenate(chunks).reshape(shape) for chunks in
                 (B_out, lie_out, dx_out, dy_out))


def _add_vector_field(ax, xx, yy, vx, vy, *, n_arrows=17):
    stride = max(1, int(np.ceil(max(xx.shape) / n_arrows)))
    xxq, yyq = xx[::stride, ::stride], yy[::stride, ::stride]
    ux, uy = vx[::stride, ::stride], vy[::stride, ::stride]
    length = np.hypot(ux, uy)
    keep = length > 1e-10
    # Normalize only arrow DISPLAY so large polynomial dynamics do not hide the domain.
    ux = np.where(keep, ux / np.maximum(length, 1e-10), 0.0)
    uy = np.where(keep, uy / np.maximum(length, 1e-10), 0.0)
    ax.quiver(xxq, yyq, ux, uy, color="#53616c", pivot="middle",
              angles="xy", scale_units="inches", scale=8,
              alpha=0.48, width=0.002, zorder=0)


def _contour_if_present(ax, xx, yy, Z, levels, colors, styles, widths, *, zorder=4):
    lo = np.nanmin(Z); hi = np.nanmax(Z)
    active = [i for i, level in enumerate(levels) if lo <= level <= hi]
    if not active:
        return
    ax.contour(xx, yy, Z, levels=[levels[i] for i in active],
               colors=[colors[i] for i in active],
               linestyles=[styles[i] for i in active],
               linewidths=[widths[i] for i in active], zorder=zorder)


def _scatter_stage(ax, label, batch, traj):
    """FOSSIL verifier batch order: [noise0..noise19, original_point]."""
    color = POINT_COLORS.get(label, "#7937ac")
    if len(batch):
        noisy, seed = batch[:-1], batch[-1:]
        if len(noisy):
            ax.scatter(noisy[:, 0], noisy[:, 1], s=18, marker="o", c=color,
                       alpha=0.55, edgecolors="none", zorder=9,
                       label=f"{label}: verifier noise ({len(noisy)})")
        ax.scatter(seed[:, 0], seed[:, 1], s=110, marker="*", c=color,
                   edgecolors="black", linewidth=0.7, zorder=12,
                   label=f"{label}: verifier")
    if label == "lie" and len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color="#7137aa", lw=1.5,
                marker="x", markersize=4.2, zorder=10,
                label=f"Consolidator ({len(traj)} pts)")
        # Consolidator starts at the verifier seed, so no extra 'new seed'.
        if len(traj) > 1:
            ax.scatter(traj[-1, 0], traj[-1, 1], s=48,
                       marker="D", c="#7137aa", edgecolors="white",
                       linewidths=0.7, zorder=11)


def _draw_verifier_zoom(ax, label, batch):
    """Inset needed because default FOSSIL perturbation std is only 0.0005."""
    if len(batch) < 2:
        return
    color = POINT_COLORS.get(label, "#7937ac")
    seed = batch[-1]
    noise = batch[:-1]
    radius = max(0.002, float(np.max(np.abs(noise - seed))) * 1.45)
    # Put inset in the lower-right, away from the legend in upper-right.
    detail = ax.inset_axes([0.67, 0.045, 0.30, 0.30], zorder=18)
    detail.scatter(noise[:, 0], noise[:, 1], s=14, c=color, alpha=0.75)
    detail.scatter([seed[0]], [seed[1]], s=105, c=color, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=20)
    detail.set_xlim(seed[0] - radius, seed[0] + radius)
    detail.set_ylim(seed[1] - radius, seed[1] + radius)
    detail.set_aspect("equal", adjustable="box")
    detail.set_title(f"{label}: verifier zoom (20 noise)", fontsize=7)
    detail.tick_params(labelsize=6)
    detail.grid(alpha=0.2)
    detail.set_facecolor("#f9fafc")
    for spine in detail.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.2)


def _plot_one(axs, xx, yy, B, lie, vx, vy, domains, *, group,
              groups, trajectory, epsilon, normed_B, extent, fig):
    (xmin, xmax), (ymin, ymax) = extent
    for k, ax in enumerate(axs):
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(alpha=0.08)
        _draw_region(ax, domains.get("lie"), facecolor="#92a0b2", name="Domain",
                     fill_alpha=0.025)
        _draw_region(ax, domains.get("init"), facecolor="#008743", name="Initial")
        _draw_region(ax, domains.get("unsafe"), facecolor="#d92929", name="Unsafe")
        _add_vector_field(ax, xx, yy, vx, vy)

    _contour_if_present(
        axs[0], xx, yy, B, [-epsilon, 0.0, epsilon],
        ["#5a6268", "#0759a8", "#5a6268"],
        ["dashed", "solid", "dashed"], [1.25, 2.3, 1.25]
    )
    axs[0].set_title(
        ("Normed Barrier $B/\\|\\nabla B\\|$" if normed_B else "Barrier $B(x)$")
    )
    
    if np.nanmax(lie) > np.nanmin(lie):
        lim = float(np.nanpercentile(np.abs(lie), 95))
        lim = max(lim, 1e-8)
        im = axs[1].contourf(
            xx, yy, np.clip(lie, -lim, lim),
            levels=np.linspace(-lim, lim, 19), cmap="RdBu_r", alpha=0.27, zorder=0
        )
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label=r"$\dot B=\nabla B\cdot f$")
        distinct = sorted(set(float(v) for v in np.percentile(lie, [10, 25, 50, 75, 90])))
        # At least three levels, not all equal.
        levels = [v for v in distinct if v < 0 or v > 0]
        if levels:
            _contour_if_present(axs[1], xx, yy, lie, levels,
                                ["#555555"]*len(levels), ["solid"]*len(levels),
                                [0.65]*len(levels), zorder=3)
    _contour_if_present(axs[1], xx, yy, lie, [0.0], ["#efb000"], ["solid"], [2.25])
    # Draw B=0 on Lie map for context.
    _contour_if_present(axs[1], xx, yy, B, [0.0], ["#0759a8"], ["dashed"], [1.25])
    axs[1].set_title(
        ("Normed Lie derivative $\\dot B(x)$ | yellow: normed $\\dot B=0$" 
         if normed_B else 
         "Lie derivative $\\dot B(x)$ | yellow: $\\dot B=0$")
    )

    for ax in axs:
        for label in group:
            _scatter_stage(ax, label, groups[label], trajectory)
        # compact key legend with actual scattered groups and regions.
        handles, names = ax.get_legend_handles_labels()
        handles.extend([
            Line2D([], [], color="#087f43", lw=4, alpha=0.4),
            Line2D([], [], color="#c62828", lw=4, alpha=0.4),
            Line2D([], [], color="#0759a8", lw=2),
        ])
        names.extend(["Initial set", "Unsafe set", "B=0"])
        ax.legend(handles, names, loc="upper right", fontsize=7, framealpha=0.84)
    # An inset on each focused figure shows all 20 tiny verifier perturbations.
    if len(group) == 1:
        name = group[0]
        _draw_verifier_zoom(axs[0], name, groups[name])


def plot_cex_iteration(*, domains, net, dynamics, cex, trajectory, iteration,
                       output_dir="results/cex_plots", epsilon=0.05,
                       grid_size=151, normalize_barrier=False,
                       plot_combined=True, batch_size=4096):
    """Save snapshots for each verifier condition and optional combined plot.

    Args:
        domains: CegisConfig.DOMAINS, mapping 'init'/'unsafe'/'lie' to 2D sets.
        net: current FOSSIL Barrier learner (not a translated SMT expression).
        dynamics: current closed-loop model, e.g. SingleCegis.f.
        cex: state[CegisStateKeys.cex] *before* process_cex appends trajectory.
        trajectory: state[CegisStateKeys.trajectory] from Consolidator.
        iteration: zero-based CEGIS iteration (not epoch).
        output_dir: directory for generated PNGs.
        epsilon: B=+-epsilon levels (normalised B when normalize_barrier=True).
        grid_size: number of grid samples on each axis.
        normalize_barrier: display B/||grad B|| instead of raw B on left panel.

    Returns:
        list[pathlib.Path]: paths of PNGs written.

    Notes:
        Candidate points include approximate delta-sat centres (dReal); noise and
        consolidator trace are training points, *not* certified violations.
        Consolidator is gradient ascent on Lie derivative, NOT physical f-trajectory.
        Only 2D Rectangle/Sphere geometry is currently supported.
    """
    if cex is None or not isinstance(cex, dict):
        return []
    if int(grid_size) < 8:
        raise ValueError("grid_size must be >= 8")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if "lie" not in domains:
        raise KeyError("domains must contain XD/domain key 'lie'")
    groups = {name: _to_numpy_2d(cex.get(name)) for name in LABELS}
    active = [name for name in LABELS if len(groups[name])]
    if not active:
        return []
    traj = _to_numpy_2d(trajectory)
    # Trajectory belongs only to Lie; ignore stale/mismatched values otherwise.
    if not len(groups["lie"]):
        traj = np.empty((0, 2), dtype=float)
    (xmin, xmax), (ymin, ymax) = _bounds(domains["lie"])
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    B, lie, vx, vy = _fields(net, dynamics, xx, yy, batch_size=batch_size,
                             normalize_barrier=normalize_barrier)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    selections = [(name,) for name in active]
    if plot_combined and len(active) > 1:
        selections.append(tuple(active))
    for group in selections:
        fig, axs = plt.subplots(1, 2, figsize=(15.5, 6.1), constrained_layout=True)
        _plot_one(axs, xx, yy, B, lie, vx, vy, domains, group=group,
                  groups=groups, trajectory=traj, epsilon=epsilon,
                  normed_B=normalize_barrier,
                  extent=((xmin, xmax), (ymin, ymax)), fig=fig)
        title = "+".join(group)
        fig.suptitle(
            f"CEGIS iteration {iteration:04d} | {title} | "
            f"verifier: " + ", ".join(f"{name} {len(groups[name])}" for name in group)
            + (f" | consolidator: {len(traj)}" if "lie" in group else ""),
            fontsize=13
        )
        path = out_dir / f"cegis_{iteration:04d}_{title.replace('+', '_')}.png"
        fig.savefig(path, dpi=175, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved
