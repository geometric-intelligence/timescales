import numpy as np
import matplotlib.pyplot as plt


def get_timescale_groups(
    model,
    group_method="discrete",  # or "binned"
    n_groups=None,
):
    """
    Extract timescale groups and create group assignments.

    Args:
        model: The neural network model
        n_groups: Number of groups to create (for continuous distributions)
        group_method: "discrete" (use unique values) or "binned" (bin continuous values)
    """
    W_rec = model.rnn_step.W_rec.weight.detach().cpu().numpy()
    timescales = model.rnn_step.timescales.cpu().numpy()

    if group_method == "discrete":
        # Original behavior: use unique timescale values
        unique_timescales = np.unique(timescales)
        n_groups_actual = len(unique_timescales)

        if n_groups_actual > 20:  # Probably continuous, warn user
            print(
                f"Warning: Found {n_groups_actual} unique timescales. Consider using group_method='binned'"
            )

        print(f"Found {n_groups_actual} discrete timescale groups:")
        for i, ts in enumerate(unique_timescales):
            count = np.sum(timescales == ts)
            print(
                f"  Group {i}: τ={ts:.4f}, {count} neurons ({count/len(timescales)*100:.1f}%)"
            )

        # Create group assignment for each neuron
        group_assignment = np.zeros(len(timescales), dtype=int)
        for i, ts in enumerate(unique_timescales):
            group_assignment[timescales == ts] = i

    elif group_method == "binned":
        # Bin continuous timescales into groups
        if n_groups is None:
            n_groups = 4  # Default to 4 groups

        # Create bins based on timescale range
        min_ts, max_ts = timescales.min(), timescales.max()

        # Option 1: Linear binning
        # bin_edges = np.linspace(min_ts, max_ts, n_groups + 1)

        # Option 2: Log-space binning (better for power-law distributions)
        if min_ts <= 0:
            # Add small offset if min is zero or negative
            min_ts_safe = max(min_ts, 1e-6)
            bin_edges = np.logspace(
                np.log10(min_ts_safe), np.log10(max_ts), n_groups + 1
            )
        else:
            bin_edges = np.logspace(np.log10(min_ts), np.log10(max_ts), n_groups + 1)

        # Assign each neuron to a bin
        group_assignment = np.digitize(timescales, bin_edges) - 1
        group_assignment = np.clip(
            group_assignment, 0, n_groups - 1
        )  # Handle edge cases

        # Compute representative timescales for each group (group means)
        unique_timescales = np.zeros(n_groups)
        for i in range(n_groups):
            mask = group_assignment == i
            if mask.sum() > 0:
                unique_timescales[i] = np.mean(timescales[mask])
            else:
                # Empty group, use bin center
                unique_timescales[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

        print(f"Binned {len(timescales)} neurons into {n_groups} groups:")
        print(f"Timescale range: [{min_ts:.4f}, {max_ts:.4f}]")
        for i in range(n_groups):
            mask = group_assignment == i
            count = mask.sum()
            ts_range = f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]"
            mean_ts = unique_timescales[i]
            print(
                f"  Group {i}: τ_mean={mean_ts:.4f}, range={ts_range}, {count} neurons ({count/len(timescales)*100:.1f}%)"
            )

    else:
        raise ValueError(
            f"Unknown group_method: {group_method}. Use 'discrete' or 'binned'"
        )

    return W_rec, timescales, unique_timescales, group_assignment


def plot_group_connectivity_magnitude_stats(
    W_rec,
    group_assignment,
    unique_timescales,
    figsize=(10, 6),
    save_fig=False,
    save_fig_name=None,
):
    """
    Plot magnitude-based connectivity statistics by group.
    Shows absolute values of self-connections, outgoing, and incoming connections.
    """
    n_groups = len(unique_timescales)

    # Compute group connectivity matrix
    group_connectivity = np.zeros((n_groups, n_groups))
    group_abs_connectivity = np.zeros((n_groups, n_groups))

    for pre_group in range(n_groups):
        for post_group in range(n_groups):
            pre_mask = group_assignment == pre_group
            post_mask = group_assignment == post_group

            if pre_mask.sum() > 0 and post_mask.sum() > 0:
                submatrix = W_rec[np.ix_(post_mask, pre_mask)]
                group_connectivity[post_group, pre_group] = np.mean(submatrix)
                group_abs_connectivity[post_group, pre_group] = np.mean(
                    np.abs(submatrix)
                )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Prepare data
    x_pos = np.arange(n_groups)
    width = 0.25

    # 1. Self-connections (diagonal) - use absolute values
    self_conn_mag = np.diag(group_abs_connectivity)

    # 2. Average outgoing connections (row means excluding diagonal) - use absolute values
    outgoing_mag = []
    for i in range(n_groups):
        mask = np.ones(n_groups, dtype=bool)
        mask[i] = False  # Exclude diagonal
        outgoing_mag.append(np.mean(group_abs_connectivity[i, mask]))

    # 3. Average incoming connections (column means excluding diagonal) - use absolute values
    incoming_mag = []
    for j in range(n_groups):
        mask = np.ones(n_groups, dtype=bool)
        mask[j] = False  # Exclude diagonal
        incoming_mag.append(np.mean(group_abs_connectivity[mask, j]))

    # Plot bars
    ax.bar(
        x_pos - width,
        self_conn_mag,
        width,
        label="Self-connections",
        alpha=0.8,
        color="purple",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x_pos,
        outgoing_mag,
        width,
        label="Avg outgoing",
        alpha=0.8,
        color="orange",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x_pos + width,
        incoming_mag,
        width,
        label="Avg incoming",
        alpha=0.8,
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )

    # # Add value labels on bars
    # def add_value_labels(bars, values):
    #     for bar, value in zip(bars, values):
    #         height = bar.get_height()
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2.0,
    #             height + height * 0.01,
    #             f"{value:.1e}",
    #             ha="center",
    #             va="bottom",
    #             fontsize=8,
    #             rotation=0,
    #         )

    # add_value_labels(bars1, self_conn_mag)
    # add_value_labels(bars2, outgoing_mag)
    # add_value_labels(bars3, incoming_mag)

    # Formatting
    ax.set_xlabel("Timescale Group", fontsize=12)
    ax.set_ylabel("Mean |Connection Strength|", fontsize=12)
    ax.set_title("Connection Magnitude Patterns by Group", fontsize=14, pad=20)
    ax.set_xticks(x_pos)

    # Create detailed tick labels
    tick_labels = []
    for i, ts in enumerate(unique_timescales):
        neuron_count = np.sum(group_assignment == i)
        tick_labels.append(f"Group {i}\nτ={ts:.3f}\n({neuron_count}n)")

    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Print detailed statistics
    print("Magnitude-based connectivity statistics:")
    print("\nSelf-connections |magnitude|:")
    for i, (ts, mag) in enumerate(zip(unique_timescales, self_conn_mag, strict=False)):
        print(f"  Group {i} (τ={ts:.4f}): {mag:.6f}")

    print("\nOutgoing connections |magnitude|:")
    for i, (ts, mag) in enumerate(zip(unique_timescales, outgoing_mag, strict=False)):
        print(f"  Group {i} (τ={ts:.4f}): {mag:.6f}")

    print("\nIncoming connections |magnitude|:")
    for i, (ts, mag) in enumerate(zip(unique_timescales, incoming_mag, strict=False)):
        print(f"  Group {i} (τ={ts:.4f}): {mag:.6f}")

    # Compute ratios for additional insights
    print("\nConnection strength ratios:")
    for i, ts in enumerate(unique_timescales):
        self_vs_out = (
            self_conn_mag[i] / outgoing_mag[i] if outgoing_mag[i] > 0 else float("inf")
        )
        self_vs_in = (
            self_conn_mag[i] / incoming_mag[i] if incoming_mag[i] > 0 else float("inf")
        )
        print(
            f"  Group {i} (τ={ts:.4f}): Self/Out={self_vs_out:.2f}, Self/In={self_vs_in:.2f}"
        )
    if save_fig:
        plt.savefig(save_fig_name, dpi=300, bbox_inches="tight", format="pdf")

    return fig, {
        "self_connections": self_conn_mag,
        "outgoing": outgoing_mag,
        "incoming": incoming_mag,
        "timescales": unique_timescales,
    }


def plot_group_bipartite_with_asymmetry(
    W_rec,
    group_assignment,
    unique_timescales,
    timescales,
    figsize=(12, 10),
    save_fig=False,
    save_fig_name=None,
):
    """
    Plot bipartite graph with line thickness = mean absolute weight (magnitude only).
    """
    n_groups = len(unique_timescales)

    # Compute statistics for each group pair
    group_stats = {}

    for pre_group in range(n_groups):
        for post_group in range(n_groups):
            pre_mask = group_assignment == pre_group
            post_mask = group_assignment == post_group

            if pre_mask.sum() > 0 and post_mask.sum() > 0:
                submatrix = W_rec[np.ix_(post_mask, pre_mask)]
                weights = submatrix.flatten()

                # Compute magnitude statistics
                mean_abs_weight = np.mean(np.abs(weights))
                n_total = len(weights)

                group_stats[(post_group, pre_group)] = {
                    "mean_abs_weight": mean_abs_weight,
                    "n_total": n_total,
                    "weights": weights,
                }

    # Get max mean absolute weight for normalization
    max_abs_weight = max([stats["mean_abs_weight"] for stats in group_stats.values()])
    min_abs_weight = min([stats["mean_abs_weight"] for stats in group_stats.values()])

    print(f"Connection magnitude range: [{min_abs_weight:.1e}, {max_abs_weight:.1e}]")

    # Create figure with space for external legend
    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]}
    )

    # Layout: presynaptic on left, postsynaptic on right
    pre_y = np.linspace(0.1, 0.9, n_groups)
    post_y = np.linspace(0.1, 0.9, n_groups)

    # Presynaptic groups (left side) - all black
    for i, y in enumerate(pre_y):
        neuron_count = np.sum(group_assignment == i)
        size = 300 + 500 * (neuron_count / len(timescales))
        ax.scatter(
            0.2, y, s=size, c="black", alpha=0.8, edgecolors="black", linewidth=2
        )
        ax.text(
            0.02,
            y,
            f"G{i+1}\n({neuron_count}n)",
            ha="right",
            va="center",
            fontsize=10,
            weight="bold",
        )

    # Postsynaptic groups (right side) - all black
    for i, y in enumerate(post_y):
        neuron_count = np.sum(group_assignment == i)
        size = 300 + 500 * (neuron_count / len(timescales))
        ax.scatter(
            0.8, y, s=size, c="black", alpha=0.8, edgecolors="black", linewidth=2
        )
        ax.text(
            0.98,
            y,
            f"G{i+1}\n({neuron_count}n)",
            ha="left",
            va="center",
            fontsize=10,
            weight="bold",
        )

    # Draw connections
    connection_info = []

    for pre_i in range(n_groups):
        for post_i in range(n_groups):
            key = (post_i, pre_i)
            if key in group_stats:
                stats = group_stats[key]

                # Line thickness proportional to mean absolute weight
                thickness = 1 + 8 * (stats["mean_abs_weight"] / max_abs_weight)

                # Alpha proportional to connection strength
                alpha = 0.4 + 0.6 * (stats["mean_abs_weight"] / max_abs_weight)

                # Draw the connection (purple lines)
                ax.plot(
                    [0.2, 0.8],
                    [pre_y[pre_i], post_y[post_i]],
                    color="purple",
                    alpha=alpha,
                    linewidth=thickness,
                    solid_capstyle="round",
                )

                # Store info for summary
                connection_info.append(
                    {
                        "pre_group": pre_i,
                        "post_group": post_i,
                        "pre_ts": unique_timescales[pre_i],
                        "post_ts": unique_timescales[post_i],
                        "mean_abs_weight": stats["mean_abs_weight"],
                        "n_total": stats["n_total"],
                    }
                )

    # Set up the main plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Group-to-Group Connectivity\n" + "Thickness ∝ Mean |Weight|",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks([0.2, 0.8])
    ax.set_xticklabels(["Presynaptic\nGroups", "Postsynaptic\nGroups"], fontsize=12)
    ax.set_yticks([])

    # Create legend in separate subplot
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")  # Remove axes

    # Add thickness legend in the legend subplot
    thicknesses = [1, 3, 6, 9]  # Example thicknesses
    legend_y_start = 0.7

    # Title for legend
    legend_ax.text(0.1, 0.85, "Line Thickness", fontsize=12, weight="bold")
    legend_ax.text(0.1, 0.8, "Mean |Weight|", fontsize=10)

    for i, thick in enumerate(thicknesses):
        y_pos = legend_y_start - i * 0.1

        # Draw example line
        legend_ax.plot(
            [0.1, 0.4],
            [y_pos, y_pos],
            color="purple",
            linewidth=thick,
            alpha=0.8,
        )

        # Convert thickness back to weight for label
        weight = (thick - 1) / 8 * max_abs_weight
        legend_ax.text(
            0.5,
            y_pos,
            f"{weight:.1e}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    # Print summary statistics
    print("\nConnection Summary (by magnitude):")
    for info in sorted(
        connection_info, key=lambda x: x["mean_abs_weight"], reverse=True
    ):
        print(
            f"G{info['pre_group']+1} → G{info['post_group']+1}: "
            f"|W|={info['mean_abs_weight']:.1e} ({info['n_total']} connections)"
        )
    if save_fig:
        plt.savefig(save_fig_name, dpi=300, bbox_inches="tight", format="pdf")

    return fig, group_stats, connection_info


def plot_group_connectivity_distributions(
    W_rec,
    group_assignment,
    unique_timescales,
    figsize=(15, 12),
    bins=50,
    common_ylim=False,
    paper_ready=False,
    save_fig=False,
    save_fig_name="connectivity.png",
):
    """
    Plot histograms of raw connectivity weight distributions for all group-to-group connections.

    Args:
        W_rec: Full connectivity matrix
        group_assignment: Array assigning each neuron to a timescale group
        unique_timescales: Array of unique timescale values
        figsize: Figure size
        bins: Number of histogram bins
        common_ylim: If True, use common y-axis limits across all plots
        paper_ready: If True, create cleaner version suitable for publication
    """
    n_groups = len(unique_timescales)

    # Create 4x4 subplot grid
    fig, axes = plt.subplots(n_groups, n_groups, figsize=figsize)

    # If only one group, ensure axes is 2D
    if n_groups == 1:
        axes = np.array([[axes]])
    elif n_groups == 2:
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)

    # Collect all weights for global statistics and find max variance
    all_weights = []
    group_weights = {}
    group_stds = {}

    for pre_group in range(n_groups):
        for post_group in range(n_groups):
            # Find neurons in each group
            pre_mask = group_assignment == pre_group
            post_mask = group_assignment == post_group

            # Extract all individual connections for this group pair
            if pre_mask.sum() > 0 and post_mask.sum() > 0:
                submatrix = W_rec[np.ix_(post_mask, pre_mask)]  # post x pre
                weights = submatrix.flatten()
                group_weights[(post_group, pre_group)] = weights
                group_stds[(post_group, pre_group)] = weights.std()
                all_weights.extend(weights)

    # Find the group pair with highest standard deviation
    max_std = max(group_stds.values()) if group_stds else 1.0

    # Set common x-limits: ±3 standard deviations of the most variable group
    xlim_range = 3 * max_std
    common_xlim = (-xlim_range, xlim_range)

    # First pass: compute all histograms to find max density if using common y-limits
    if common_ylim:
        max_density = 0
        for pre_group in range(n_groups):
            for post_group in range(n_groups):
                if (post_group, pre_group) in group_weights:
                    weights = group_weights[(post_group, pre_group)]
                    counts, _ = np.histogram(
                        weights, bins=bins, range=common_xlim, density=True
                    )
                    max_density = max(max_density, counts.max())

        common_ylim_range = (0, max_density * 1.05)  # Add 5% padding
        if not paper_ready:
            print(f"Common y-limits: [0, {max_density:.2f}]")

    # Convert to numpy for global statistics
    all_weights = np.array(all_weights)
    global_mean = all_weights.mean()
    global_std = all_weights.std()
    global_min, global_max = all_weights.min(), all_weights.max()

    if not paper_ready:
        print("Global weight statistics:")
        print(f"  Mean: {global_mean:.6f}")
        print(f"  Std: {global_std:.6f}")
        print(f"  Range: [{global_min:.6f}, {global_max:.6f}]")
        print(f"  Max group std: {max_std:.6f}")
        print(f"  Common x-limits: [{common_xlim[0]:.6f}, {common_xlim[1]:.6f}]")
        print(f"  Total connections: {len(all_weights):,}")

    # Plot histograms
    for post_group in range(n_groups):
        for pre_group in range(n_groups):
            ax = axes[post_group, pre_group]

            if (post_group, pre_group) in group_weights:
                weights = group_weights[(post_group, pre_group)]

                # Plot histogram with common x-limits
                edge_width = 0.3 if paper_ready else 0.5
                counts, bins_edges, patches = ax.hist(
                    weights,
                    bins=bins,
                    alpha=0.8 if paper_ready else 0.7,
                    density=True,
                    edgecolor="black",  # if not paper_ready else "none",
                    linewidth=edge_width,
                    range=common_xlim,
                )

                # Color bars by sign
                for i, patch in enumerate(patches):
                    bin_center = (bins_edges[i] + bins_edges[i + 1]) / 2
                    if bin_center > 0:
                        patch.set_facecolor("red")
                        patch.set_alpha(0.8 if paper_ready else 0.7)
                    elif bin_center < 0:
                        patch.set_facecolor("blue")
                        patch.set_alpha(0.8 if paper_ready else 0.7)
                    else:
                        patch.set_facecolor("gray")
                        patch.set_alpha(0.8 if paper_ready else 0.7)

                # Add statistics
                mean_weight = weights.mean()
                std_weight = weights.std()
                n_connections = len(weights)

                # Add vertical lines (simplified for paper)
                if paper_ready:
                    # ax.axvline(mean_weight, color="black", linestyle="--", linewidth=1.5, alpha=0.9)
                    ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.7)
                else:
                    # ax.axvline(mean_weight, color="black", linestyle="--", linewidth=2, alpha=0.8)
                    ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.7)

                # Set common x-limits
                ax.set_xlim(common_xlim)

                # Set common y-limits if requested
                if common_ylim:
                    ax.set_ylim(common_ylim_range)

                # Title with group info (only for non-paper version)
                if not paper_ready:
                    pre_ts = unique_timescales[pre_group]
                    post_ts = unique_timescales[post_group]
                    title = f"Pre: τ={pre_ts:.3f} → Post: τ={post_ts:.3f}"
                    ax.set_title(title, fontsize=10)

                # Add text box with statistics (only for non-paper version)
                if not paper_ready:
                    stats_text = (
                        f"n={n_connections:,}\nμ={mean_weight:.1e}\nσ={std_weight:.1e}"
                    )
                    ax.text(
                        0.02,
                        0.98,
                        stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                # Set axis labels only for edge subplots
                label_fontsize = 10 if paper_ready else 9
                if post_group == n_groups - 1:
                    ax.set_xlabel("Weight", fontsize=label_fontsize)
                if pre_group == 0:
                    ax.set_ylabel("Density", fontsize=label_fontsize)

                # Cleaner tick labels for paper
                if paper_ready:
                    ax.tick_params(labelsize=8)
                    # Reduce number of ticks and make them smaller
                    ax.locator_params(axis="x", nbins=3)
                    ax.locator_params(axis="y", nbins=3)

            else:
                # No connections for this group pair
                if paper_ready:
                    ax.text(
                        0.5,
                        0.5,
                        "—",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=16,
                        color="gray",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No\nconnections",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )

                ax.set_xlim(common_xlim)
                if common_ylim:
                    ax.set_ylim(common_ylim_range)
                if post_group == n_groups - 1:
                    ax.set_xlabel("Weight", fontsize=label_fontsize)
                if pre_group == 0:
                    ax.set_ylabel("Density", fontsize=label_fontsize)

    # Add overall title (only for non-paper version)
    if not paper_ready:
        ylim_text = " (Common Y-limits)" if common_ylim else " (Individual Y-limits)"
        fig.suptitle(
            "Connection Weight Distributions by Group Pairs\n"
            + "Red=Positive weights, Blue=Negative weights, Dashed line=Group mean\n"
            + f"All plots: x ∈ ±3σ of most variable group (σ_max = {max_std:.1e}){ylim_text}",
            fontsize=14,
            y=0.98,
        )

    if paper_ready:
        # Eliminate all spacing between subplots
        plt.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.0, hspace=0.0
        )
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

    if save_fig:
        plt.savefig(save_fig_name, dpi=300, bbox_inches="tight", format="pdf")

    return fig, group_weights
