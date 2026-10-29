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

    domain_type = type(domain).__name__

    if domain_type in (
        "Union",
        "Intersection",
        "SetMinus",
    ):

        (x1a, x1b), (y1a, y1b) = _bounds(domain.S1)

        if domain_type == "SetMinus":
            return (x1a, x1b), (y1a, y1b)

        (x2a, x2b), (y2a, y2b) = _bounds(domain.S2)

        if domain_type == "Union":
            return (
                (min(x1a, x2a), max(x1b, x2b)),
                (min(y1a, y2a), max(y1b, y2b)),
            )

        return (
            (max(x1a, x2a), min(x1b, x2b)),
            (max(y1a, y2a), min(y1b, y2b)),
        )

    kind, a, b = _geometry(domain)

    if kind == "rectangle":
        return (a[0], b[0]), (a[1], b[1])

    return (
        (a[0] - b, a[0] + b),
        (a[1] - b, a[1] + b),
    )


def _inside(domain, xy):
    """
    Check whether points belong to a domain.

    Supports:
        Rectangle
        Sphere
        Union
        SetMinus
        Intersection

    Args:
        domain: FOSSIL domain object
        xy: ndarray, shape (N, 2)

    Returns:
        mask: ndarray, shape (N,), dtype bool
    """

    domain_type = type(domain).__name__

    # Union: S1 ∪ S2
    if domain_type == "Union":
        return (
            _inside(domain.S1, xy)
            | _inside(domain.S2, xy)
        )

    # SetMinus: S1 \ S2
    if domain_type == "SetMinus":
        return (
            _inside(domain.S1, xy)
            & ~_inside(domain.S2, xy)
        )

    # Intersection: S1 ∩ S2
    if domain_type == "Intersection":
        return (
            _inside(domain.S1, xy)
            & _inside(domain.S2, xy)
        )

    # Original Rectangle / Sphere
    kind, a, b = _geometry(domain)

    if kind == "rectangle":
        return np.all(
            (xy >= a) & (xy <= b),
            axis=1
        )

    return (
        np.sum((xy - a) ** 2, axis=1)
        <= b ** 2
    )

def _draw_region(
    ax,
    region,
    *,
    facecolor,
    name,
    fill_alpha=0.16,
    lw=1.5,
    mask_resolution=601,
):
    """
    Draw FOSSIL domains.

    Primitive sets:
        Rectangle, Sphere

    Composite sets:
        Union, SetMinus, Intersection
    """

    if region is None:
        return

    domain_type = type(region).__name__

    # =====================================================
    # 1. Composite domain
    # =====================================================
    if domain_type in (
        "Union",
        "SetMinus",
        "Intersection",
    ):

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Generate a dense grid for region visualization
        xs = np.linspace(
            xmin, xmax, mask_resolution
        )
        ys = np.linspace(
            ymin, ymax, mask_resolution
        )

        xx, yy = np.meshgrid(
            xs, ys, indexing="xy"
        )

        points = np.column_stack([
            xx.ravel(),
            yy.ravel()
        ])

        # Membership mask
        mask = _inside(
            region,
            points
        ).reshape(xx.shape)

        if not np.any(mask):
            return

        if np.all(mask):
            ax.add_patch(
                RectanglePatch(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor=facecolor,
                    edgecolor=facecolor,
                    alpha=fill_alpha,
                    linewidth=lw,
                    zorder=1,
                )
            )
            return

        mask_float = mask.astype(float)

        # Fill only the unsafe/safe region
        ax.contourf(
            xx,
            yy,
            mask_float,
            levels=[0.5, 1.5],
            colors=[facecolor],
            alpha=fill_alpha,
            zorder=1,
        )

        # Draw composite region boundary
        ax.contour(
            xx,
            yy,
            mask_float,
            levels=[0.5],
            colors=[facecolor],
            linewidths=[lw],
            zorder=5,
        )

        return

    # =====================================================
    # 2. Rectangle
    # =====================================================
    kind, a, b = _geometry(region)

    if kind == "rectangle":

        ax.add_patch(
            RectanglePatch(
                tuple(a),
                b[0] - a[0],
                b[1] - a[1],
                facecolor=facecolor,
                edgecolor=facecolor,
                alpha=fill_alpha,
                linewidth=lw,
                zorder=1,
            )
        )

        ax.add_patch(
            RectanglePatch(
                tuple(a),
                b[0] - a[0],
                b[1] - a[1],
                facecolor="none",
                edgecolor=facecolor,
                linewidth=lw,
                zorder=5,
            )
        )

    # =====================================================
    # 3. Sphere
    # =====================================================
    else:

        ax.add_patch(
            Circle(
                tuple(a),
                b,
                facecolor=facecolor,
                edgecolor=facecolor,
                alpha=fill_alpha,
                zorder=1,
            )
        )

        ax.add_patch(
            Circle(
                tuple(a),
                b,
                facecolor="none",
                edgecolor=facecolor,
                linewidth=lw,
                zorder=5,
            )
        )

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
                B = B / gB.norm(dim=1).clamp_min(eps_norm)
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
    """Only draw requested levels that occur in the sampled field range."""
    finite_values = np.asarray(Z)[np.isfinite(Z)]
    if finite_values.size == 0:
        return
    lo, hi = float(finite_values.min()), float(finite_values.max())
    if lo == hi:  # No well-defined isoline on a constant-valued field.
        return
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
        # The first Consolidator point is the original verifier cex.
        if len(traj) > 1:
            ax.scatter(traj[-1, 0], traj[-1, 1], s=48,
                       marker="D", c="#7137aa", edgecolors="white",
                       linewidths=0.7, zorder=11)


def _barrier_grid(net, xx, yy, *, batch_size=4096, normalize_barrier=False,
                  eps_norm=1e-8):
    """Evaluate just the plotted Barrier on a local grid (no dynamics needed).

    The contour levels refer to the SAME quantity used on the main Barrier panel:
    B or B / max(||grad B||, eps_norm). Network parameters are not modified.
    """
    points = np.column_stack((xx.ravel(), yy.ravel()))
    device, dtype = _module_spec(net)
    parts = []
    for lo in range(0, len(points), batch_size):
        xy = torch.as_tensor(points[lo:lo + batch_size],
                             dtype=dtype, device=device).clone().detach()
        if normalize_barrier:
            with torch.enable_grad():
                xy.requires_grad_(True)
                B, gradB = _model_value_and_grad(net, xy)
                B = B / gradB.norm(dim=1).clamp_min(eps_norm)
        else:
            with torch.no_grad():
                nn_value = net(xy).reshape(-1)
                factor = (net.compute_factors(xy)[0]
                          if hasattr(net, "compute_factors") else 1)
                B = nn_value * factor
        parts.append(B.detach().cpu().numpy())
    return np.concatenate(parts).reshape(xx.shape)


def _draw_barrier_contours(ax, xx, yy, B, epsilon, *, zorder=4,
                           negative_color="#5a6268", zero_color="#0759a8",
                           positive_color="#5a6268", thin=1.0, thick=2.3):
    """Consistent B contour encoding in main panel and noise-level inset.

    B=-eps: dashed/thin; B=0: solid/thick; B=+eps: solid/thin.
    """
    _contour_if_present(
        ax, xx, yy, B, [-epsilon, 0.0, epsilon],
        [negative_color, zero_color, positive_color],
        ["dashed", "solid", "solid"],
        [thin, thick, thin], zorder=zorder,
    )


def _draw_zoom(
    ax, label, batch, traj, net, epsilon, normed_B,
    *,
    batch_size=4096,
    zoom_grid_size=121,
    zoom_epsilon=0.001,
):
    """
    Draw two independent insets:
      1. Verifier-noise zoom: range determined ONLY by the 20 noise points.
      2. Consolidator zoom: range determined by the Consolidator trajectory.

    Both insets display B=0 and B=+-zoom_epsilon when those levels
    exist inside the corresponding plotting window.

    The global epsilon and local zoom_epsilon are independent.
    """
    if zoom_epsilon <= 0:
        raise ValueError("zoom_epsilon must be positive")

    color = POINT_COLORS.get(label, "#7937ac")
    zoom_eps = float(zoom_epsilon)

    # ---------------------------------------------------------
    # Helper: draw Barrier contours on a locally sampled grid.
    # ---------------------------------------------------------
    def draw_local_barrier(detail, xlo, xhi, ylo, yhi):
        xs = np.linspace(xlo, xhi, zoom_grid_size)
        ys = np.linspace(ylo, yhi, zoom_grid_size)
        xx_zoom, yy_zoom = np.meshgrid(xs, ys, indexing="xy")

        B_zoom = _barrier_grid(
            net,
            xx_zoom,
            yy_zoom,
            batch_size=batch_size,
            normalize_barrier=normed_B,
        )

        # Draw B=-zoom_epsilon, B=0, B=+zoom_epsilon.
        # Levels outside the local function range are omitted.
        _draw_barrier_contours(
            detail,
            xx_zoom,
            yy_zoom,
            B_zoom,
            zoom_eps,
            zorder=6,
            thin=0.80,
            thick=1.8,
        )

        detail.set_xlim(xlo, xhi)
        detail.set_ylim(ylo, yhi)
        detail.set_aspect("equal", adjustable="box")
        detail.tick_params(labelsize=5.5)
        detail.grid(alpha=0.18, linewidth=0.5)
        detail.set_facecolor("#f9fafc")

        return detail

    # ---------------------------------------------------------
    # 1. Verifier-noise zoom.
    #    Its range depends ONLY on the 20 noise points.
    # ---------------------------------------------------------
    if len(batch) >= 2:
        seed = batch[-1]
        noise = batch[:-1]

        radius = max(
            0.002,
            float(np.max(np.abs(noise - seed))) * 1.45,
        )

        xlo = seed[0] - radius
        xhi = seed[0] + radius
        ylo = seed[1] - radius
        yhi = seed[1] + radius

        noise_ax = ax.inset_axes(
            [0.60, 0.01, 0.42, 0.28],
            zorder=18,
        )
        noise_ax.patch.set_facecolor("#f9fafc")
        noise_ax.patch.set_alpha(0.75)

        draw_local_barrier(
            noise_ax, xlo, xhi, ylo, yhi
        )

        noise_ax.scatter(
            noise[:, 0],
            noise[:, 1],
            s=13,
            c=color,
            alpha=0.75,
            edgecolors="none",
            zorder=9,
        )

        noise_ax.scatter(
            [seed[0]],
            [seed[1]],
            s=85,
            c=color,
            marker="*",
            edgecolors="black",
            linewidths=0.6,
            zorder=12,
        )

        noise_ax.set_title(
            rf"Verifier ({len(noise)+1} pts, $\epsilon={zoom_eps:g}$)",
            fontsize=6.5,
            pad=2,
        )

        for spine in noise_ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.0)

    # ---------------------------------------------------------
    # 2. Consolidator zoom.
    #    Its range depends on the trajectory, not the noise.
    # ---------------------------------------------------------
    if label == "lie" and len(traj):
        traj = np.asarray(traj, dtype=float)

        valid_mask = np.isfinite(traj).all(axis=1)
        valid_traj = traj[valid_mask]

        if len(valid_traj):
            # Include the original Verifier point in the window
            # whenever it is available.
            if len(batch):
                seed = batch[-1]
                focus_points = np.vstack([valid_traj, seed])
            else:
                seed = None
                focus_points = valid_traj

            pmin = focus_points.min(axis=0)
            pmax = focus_points.max(axis=0)

            center = 0.5 * (pmin + pmax)
            span = float(np.max(pmax - pmin))

            # Add padding around the trajectory. The minimum window
            # size prevents a degenerate zoom if the trajectory barely moves.
            radius = max(0.002, 0.65 * span)

            xlo = center[0] - radius
            xhi = center[0] + radius
            ylo = center[1] - radius
            yhi = center[1] + radius

            consol_ax = ax.inset_axes(
                [0.01, 0.01, 0.42, 0.28],
                zorder=18,
            )
            consol_ax.patch.set_facecolor("#f9fafc")
            consol_ax.patch.set_alpha(0.75)

            draw_local_barrier(
                consol_ax, xlo, xhi, ylo, yhi
            )

            # Draw the Consolidator gradient-ascent trajectory.
            # NaN entries break the line at invalid trajectory points.
            consol_ax.plot(
                traj[:, 0],
                traj[:, 1],
                color="#7137aa",
                linewidth=1.1,
                marker="x",
                markersize=3.5,
                zorder=10,
            )

            # Mark the original Verifier counterexample.
            if seed is not None:
                consol_ax.scatter(
                    [seed[0]],
                    [seed[1]],
                    s=75,
                    c=color,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=12,
                )

            # Mark the final valid Consolidator point.
            last_point = valid_traj[-1]

            consol_ax.scatter(
                [last_point[0]],
                [last_point[1]],
                s=28,
                marker="D",
                c="#7137aa",
                edgecolors="white",
                linewidths=0.5,
                zorder=13,
            )

            consol_ax.set_title(
                rf"Consolidator ({len(valid_traj)} pts, $\epsilon={zoom_eps:g}$)",
                fontsize=6.5,
                pad=2,
            )

            for spine in consol_ax.spines.values():
                spine.set_color("#7137aa")
                spine.set_linewidth(1.0)


def _plot_one(axs, xx, yy, B, lie, vx, vy, domains, *, group,
              groups, trajectory, epsilon, normed_B, extent, fig, net,
              batch_size, zoom_grid_size, zoom_epsilon):
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

    _draw_barrier_contours(axs[0], xx, yy, B, epsilon)
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
        fig.colorbar(
            im, ax=axs[1], fraction=0.046, pad=0.04,
            label=(r"$\cos(\nabla B,f)$" if normed_B
                   else r"$\dot B=\nabla B\cdot f$"),
        )
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
        _draw_zoom(
            axs[0], name, groups[name], trajectory, net, epsilon, normed_B,
            batch_size=batch_size, zoom_grid_size=zoom_grid_size,
            zoom_epsilon=zoom_epsilon,
        )


def plot_cex_iteration(*, domains, net, dynamics, cex, trajectory, iteration,
                       output_dir="results/cex_plots", epsilon=0.001,
                       grid_size=151, normalize_barrier=False,
                       plot_combined=True, batch_size=4096,
                       zoom_grid_size=301, zoom_epsilon=0.001):
    """Save snapshots for each verifier condition and optional combined plot.

    Args:
        domains: CegisConfig.DOMAINS, mapping 'init'/'unsafe'/'lie' to 2D sets.
        net: current FOSSIL Barrier learner (not a translated SMT expression).
        dynamics: current closed-loop model, e.g. SingleCegis.f.
        cex: state[CegisStateKeys.cex] *before* process_cex appends trajectory.
        trajectory: state[CegisStateKeys.trajectory] from Consolidator.
        iteration: zero-based CEGIS iteration (not epoch).
        output_dir: directory for generated PNGs.
        epsilon: shared global/inset B=+-epsilon levels (default 0.05).
            Levels outside the inset's B range are not drawn.
        grid_size: number of grid samples on each axis.
        normalize_barrier: display B/||grad B|| instead of raw B on left panel;
            also switches the displayed Lie field to cosine similarity.
        zoom_grid_size: independently evaluate this many local grid points per axis.
        zoom_epsilon: deprecated compatibility keyword. Must be None or equal
            to epsilon; the global and inset contours ALWAYS use the same
            exact levels (-epsilon, 0, +epsilon).

    Returns:
        list[pathlib.Path]: paths of PNGs written.

    Notes:
        Candidate points include approximate delta-sat centres (dReal); noise and
        consolidator trace are training points, *not* certified violations.
        Consolidator is gradient ascent on its configured Lie objective (raw or
        normalized); it is NOT the physical trajectory of the system.
        Only 2D Rectangle/Sphere geometry is currently supported.
    """
    if cex is None or not isinstance(cex, dict):
        return []
    if int(grid_size) < 8:
        raise ValueError("grid_size must be >= 8")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if int(zoom_grid_size) < 8:
        raise ValueError("zoom_grid_size must be >= 8")
    if zoom_epsilon is not None and not np.isclose(
        float(zoom_epsilon), float(epsilon), rtol=0, atol=1e-12
    ):
        raise ValueError(
            "zoom_epsilon must be None or equal to epsilon; "
            "the inset and main plot use identical contour levels"
        )
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
                  extent=((xmin, xmax), (ymin, ymax)), fig=fig,
                  net=net, batch_size=batch_size,
                  zoom_grid_size=int(zoom_grid_size), zoom_epsilon=zoom_epsilon)
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