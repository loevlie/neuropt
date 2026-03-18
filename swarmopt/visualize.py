"""Visualization helpers for SwarmTuner results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def _pick_2d_dims(df, param_names):
    """Pick two dimensions to plot. For >2D, choose the two with most variance."""
    if len(param_names) <= 2:
        return param_names[0], param_names[1] if len(param_names) == 2 else param_names[0]
    variances = {name: df[name].var() for name in param_names}
    sorted_names = sorted(variances, key=variances.get, reverse=True)
    return sorted_names[0], sorted_names[1]


def _pick_3d_dims(df, param_names):
    """Pick three dimensions to plot. For >3D, choose the three with most variance."""
    if len(param_names) <= 3:
        return param_names[:3]
    variances = {name: df[name].var() for name in param_names}
    sorted_names = sorted(variances, key=variances.get, reverse=True)
    return sorted_names[0], sorted_names[1], sorted_names[2]


def plot(results_df, search_space, save_path="pso_results.png"):
    """Static 3-panel figure: all evaluations, trajectories, convergence.

    Uses 3D scatter for 3+ dimensions, 2D otherwise.

    Args:
        results_df: DataFrame from ResultStore.to_dataframe()
        search_space: dict of {name: Dimension} from the tuner
        save_path: where to save the figure

    Returns:
        matplotlib Figure
    """
    param_names = list(search_space.keys())
    df = results_df.copy()
    scores = df["score"].values
    iters = df["iteration"].values
    particles = df["particle"].values

    # Convert params to internal space for plotting
    internal = {}
    for name, dim in search_space.items():
        internal[name] = np.array([dim.to_internal(v) for v in df[name].values])

    n_iters = int(iters.max())
    n_p = int(particles.max()) + 1
    best_idx = scores.argmin()
    use_3d = len(param_names) >= 3

    if use_3d:
        dim_names = _pick_3d_dims(df, param_names)
        x_vals = internal[dim_names[0]]
        y_vals = internal[dim_names[1]]
        z_vals = internal[dim_names[2]]

        fig = plt.figure(figsize=(20, 6))

        # Panel 1: all evals colored by score (3D)
        ax = fig.add_subplot(1, 3, 1, projection="3d")
        sc = ax.scatter(x_vals, y_vals, z_vals, c=scores, cmap="viridis_r",
                        s=50, edgecolors="k", linewidths=0.3, alpha=0.85)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], z_vals[best_idx],
                   c="red", s=200, marker="*", edgecolors="k", linewidths=1,
                   zorder=5, label=f"Best: {scores[best_idx]:.4f}")
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])
        ax.set_title("All Evaluations")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(sc, ax=ax, label="score", shrink=0.6)

        # Panel 2: particle trajectories (3D)
        ax = fig.add_subplot(1, 3, 2, projection="3d")
        cmap_traj = cm.get_cmap("tab10", n_p)
        for p in range(n_p):
            mask = particles == p
            tx, ty, tz = x_vals[mask], y_vals[mask], z_vals[mask]
            color = cmap_traj(p)
            ax.plot(tx, ty, tz, "-o", color=color, markersize=3,
                    alpha=0.7, linewidth=1.2, label=f"p{p}")
            ax.scatter(tx[0], ty[0], tz[0], color=color, marker="s", s=50,
                       edgecolors="k", zorder=5)
            ax.scatter(tx[-1], ty[-1], tz[-1], color=color, marker="D", s=50,
                       edgecolors="k", zorder=5)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], z_vals[best_idx],
                   c="red", s=200, marker="*", edgecolors="k", linewidths=1,
                   zorder=6)
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])
        ax.set_title("Particle Trajectories")
        ax.legend(loc="upper right", fontsize=6, ncol=2)

        # Panel 3: convergence (always 2D)
        ax = fig.add_subplot(1, 3, 3)
    else:
        if len(param_names) >= 2:
            x_name, y_name = _pick_2d_dims(df, param_names)
        else:
            x_name = param_names[0]
            y_name = None

        x_vals = internal[x_name]
        y_vals = internal[y_name] if y_name else np.zeros_like(x_vals)
        x_label = f"{x_name} (internal)" if y_name else x_name
        y_label = f"{y_name} (internal)" if y_name else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: all evals colored by score
        ax = axes[0]
        sc = ax.scatter(x_vals, y_vals, c=scores, cmap="viridis_r", s=50,
                        edgecolors="k", linewidths=0.3, alpha=0.85)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], c="red", s=200,
                   marker="*", edgecolors="k", linewidths=1, zorder=5,
                   label=f"Best: score={scores[best_idx]:.4f}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("All Evaluations (color = score)")
        ax.legend(loc="upper right")
        fig.colorbar(sc, ax=ax, label="score")

        # Panel 2: particle trajectories
        ax = axes[1]
        cmap_traj = cm.get_cmap("tab10", n_p)
        for p in range(n_p):
            mask = particles == p
            traj_x, traj_y = x_vals[mask], y_vals[mask]
            color = cmap_traj(p)
            ax.plot(traj_x, traj_y, "-o", color=color, markersize=4,
                    alpha=0.7, linewidth=1.2, label=f"p{p}")
            ax.scatter(traj_x[0], traj_y[0], color=color, marker="s", s=60,
                       edgecolors="k", zorder=5)
            ax.scatter(traj_x[-1], traj_y[-1], color=color, marker="D", s=60,
                       edgecolors="k", zorder=5)
        ax.scatter(x_vals[best_idx], y_vals[best_idx], c="red", s=200,
                   marker="*", edgecolors="k", linewidths=1, zorder=6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Particle Trajectories")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        # Panel 3: convergence
        ax = axes[2]

    best_so_far = np.minimum.accumulate(scores)
    eval_nums = np.arange(1, len(scores) + 1)
    ax.plot(eval_nums, best_so_far, "b-", linewidth=2, label="Best so far")
    ax.scatter(eval_nums, scores, c="gray", s=15, alpha=0.5, label="Each eval")
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("score")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for it in range(1, n_iters + 1):
        ax.axvline(x=it * n_p, color="gray", linestyle="--", alpha=0.3,
                   linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    return fig


def animate(results_df, search_space, save_path="pso_trajectories.gif", fps=2):
    """Animated GIF of particle trajectories over iterations.

    Uses 3D with a slowly rotating camera for 3+ dimensions.

    Args:
        results_df: DataFrame from ResultStore.to_dataframe()
        search_space: dict of {name: Dimension} from the tuner
        save_path: where to save the GIF
        fps: frames per second
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    param_names = list(search_space.keys())
    df = results_df.copy()
    particles = df["particle"].values
    iters = df["iteration"].values
    scores = df["score"].values

    internal = {}
    for name, dim in search_space.items():
        internal[name] = np.array([dim.to_internal(v) for v in df[name].values])

    n_iters = int(iters.max())
    n_p = int(particles.max()) + 1
    use_3d = len(param_names) >= 3

    cmap_traj = cm.get_cmap("tab10", n_p)

    if use_3d:
        dim_names = _pick_3d_dims(df, param_names)
        x_vals = internal[dim_names[0]]
        y_vals = internal[dim_names[1]]
        z_vals = internal[dim_names[2]]

        # Build per-particle trajectories
        trajs = {p: [] for p in range(n_p)}
        for i in range(len(df)):
            trajs[int(particles[i])].append(
                (x_vals[i], y_vals[i], z_vals[i], scores[i]))

        pad = 0.2
        x_min, x_max = x_vals.min() - pad, x_vals.max() + pad
        y_min, y_max = y_vals.min() - pad, y_vals.max() + pad
        z_min, z_max = z_vals.min() - pad, z_vals.max() + pad

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

        lines = []
        dots = []
        for p in range(n_p):
            color = cmap_traj(p)
            line, = ax.plot([], [], [], "-", color=color, alpha=0.4,
                            linewidth=1.5)
            dot, = ax.plot([], [], [], "o", color=color, markersize=9,
                           markeredgecolor="k", markeredgewidth=0.8)
            lines.append(line)
            dots.append(dot)

        best_star, = ax.plot([], [], [], "*", color="red", markersize=18,
                             markeredgecolor="k", markeredgewidth=1)
        title = ax.set_title("")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel(dim_names[0])
        ax.set_ylabel(dim_names[1])
        ax.set_zlabel(dim_names[2])

        # Starting camera angle
        base_elev = 25
        base_azim = 30

        def update(frame):
            best_cost = np.inf
            bx, by, bz = 0, 0, 0

            for p in range(n_p):
                xs = [t[0] for t in trajs[p][:frame + 1]]
                ys = [t[1] for t in trajs[p][:frame + 1]]
                zs = [t[2] for t in trajs[p][:frame + 1]]
                lines[p].set_data_3d(xs, ys, zs)
                dots[p].set_data_3d([xs[-1]], [ys[-1]], [zs[-1]])

                for t in trajs[p][:frame + 1]:
                    if t[3] < best_cost:
                        best_cost = t[3]
                        bx, by, bz = t[0], t[1], t[2]

            best_star.set_data_3d([bx], [by], [bz])
            # Slowly rotate camera
            ax.view_init(elev=base_elev, azim=base_azim + frame * 8)
            title.set_text(
                f"Iteration {frame}  |  best score = {best_cost:.4f}")
            return lines + dots + [best_star, title]

        anim = FuncAnimation(fig, update, frames=n_iters + 1,
                             interval=int(1000 / fps), blit=False)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"Trajectory GIF saved to {save_path}")

    else:
        # 2D animation (original path)
        if len(param_names) >= 2:
            x_name, y_name = _pick_2d_dims(df, param_names)
        else:
            x_name = param_names[0]
            y_name = None

        x_vals = internal[x_name]
        y_vals = internal[y_name] if y_name else np.zeros_like(x_vals)

        trajs = {p: [] for p in range(n_p)}
        for i in range(len(df)):
            trajs[int(particles[i])].append(
                (x_vals[i], y_vals[i], scores[i]))

        pad = 0.2
        x_min, x_max = x_vals.min() - pad, x_vals.max() + pad
        y_min, y_max = y_vals.min() - pad, y_vals.max() + pad

        fig, ax = plt.subplots(figsize=(7, 6))

        lines = []
        dots = []
        for p in range(n_p):
            color = cmap_traj(p)
            line, = ax.plot([], [], "-", color=color, alpha=0.4, linewidth=1.5)
            dot, = ax.plot([], [], "o", color=color, markersize=10,
                           markeredgecolor="k", markeredgewidth=0.8)
            lines.append(line)
            dots.append(dot)

        best_star, = ax.plot([], [], "*", color="red", markersize=20,
                             markeredgecolor="k", markeredgewidth=1)
        title = ax.set_title("")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(f"{x_name} (internal)")
        ax.set_ylabel(f"{y_name} (internal)" if y_name else "")
        ax.grid(True, alpha=0.3)

        def update(frame):
            best_cost = np.inf
            best_x, best_y = 0, 0

            for p in range(n_p):
                xs = [t[0] for t in trajs[p][:frame + 1]]
                ys = [t[1] for t in trajs[p][:frame + 1]]
                lines[p].set_data(xs, ys)
                dots[p].set_data([xs[-1]], [ys[-1]])

                for t in trajs[p][:frame + 1]:
                    if t[2] < best_cost:
                        best_cost = t[2]
                        best_x, best_y = t[0], t[1]

            best_star.set_data([best_x], [best_y])
            title.set_text(
                f"Iteration {frame}  |  best score = {best_cost:.4f}")
            return lines + dots + [best_star, title]

        anim = FuncAnimation(fig, update, frames=n_iters + 1,
                             interval=int(1000 / fps), blit=True)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"Trajectory GIF saved to {save_path}")
