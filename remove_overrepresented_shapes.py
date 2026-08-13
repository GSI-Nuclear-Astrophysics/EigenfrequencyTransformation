import numpy as np
import matplotlib.pyplot as plt

# Build normalized mean shape for one cluster

def build_cluster_mean_shape(
    cluster_id,
    raw_peak_results,
    target_len=200
):

    parts = []

    for r in raw_peak_results:

        if r["cluster_id"] != cluster_id:
            continue

        y = np.maximum(
            np.asarray(r["y"]),
            0
        )

        if len(y) < 5:
            continue


        norm = np.linalg.norm(y)

        if norm == 0:
            continue

        y = y / norm

        y = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(y)),
            y
        )

        parts.append(y)


    if len(parts) == 0:
        return None

    mean_shape = np.mean(
        parts,
        axis=0
    )

    # erneut L2-normalisieren

    mean_shape /= (
        np.linalg.norm(mean_shape)
        + 1e-12
    )

    return mean_shape

# Find groups of clusters with similar shapes

def group_similar_shapes(
    cluster_ids,
    sim_matrix,
    similarity_threshold=0.9
):

    cluster_ids = list(cluster_ids)

    n = len(cluster_ids)

    visited = set()

    shape_groups = []

    for i in range(n):

        cid = cluster_ids[i]

        if cid in visited:
            continue


        group = [cid]

        visited.add(cid)

        for j in range(n):

            if i == j:
                continue

            other = cluster_ids[j]

            if other in visited:
                continue


            similarity = sim_matrix[i, j]


            if similarity >= similarity_threshold:

                group.append(other)

                visited.add(other)


        shape_groups.append(group)


    return shape_groups

# Build averaged shape of several clusters

def build_group_mean_shape(
    group,
    shapes
):

    valid_shapes = [
        shapes[cid]
        for cid in group
        if cid in shapes
    ]

    if len(valid_shapes) == 0:
        return None


    mean_shape = np.mean(
        valid_shapes,
        axis=0
    )



    mean_shape /= (
        np.linalg.norm(mean_shape)
        + 1e-12
    )


    return mean_shape

# optional: Plot 

def plot_cluster_shapes(
    raw_peak_results,
    cluster_positions,
    sim_matrix,
    cluster_ids,
    similarity_threshold=0.9,
    figsize=(12, 8)
):

    shapes = {}

    counts = {}


    for cid in cluster_ids:

        shape = build_cluster_mean_shape(
            cid,
            raw_peak_results,
            target_len=200
        )


        if shape is None:
            continue


        shapes[cid] = shape

        counts[cid] = sum(
            1
            for r in raw_peak_results
            if r["cluster_id"] == cid
        )


    valid_cluster_ids = [
        cid
        for cid in cluster_ids
        if cid in shapes
    ]


    shape_groups = group_similar_shapes(
        valid_cluster_ids,
        sim_matrix,
        similarity_threshold=similarity_threshold
    )


    print("=" * 70)
    print("SHAPE GROUPS")
    print("=" * 70)

    for i, group in enumerate(shape_groups):

        total_peaks = sum(
            counts[cid]
            for cid in group
        )

        print(
            f"Shape {i+1}: "
            f"Clusters = {group} | "
            f"N clusters = {len(group)} | "
            f"N peaks = {total_peaks}"
        )


    print()


    fig, ax = plt.subplots(
        figsize=figsize
    )


    x = np.linspace(
        0,
        1,
        200
    )

    cmap = plt.cm.tab20


    for group_index, group in enumerate(shape_groups):

        group_shape = build_group_mean_shape(
            group,
            shapes
        )


        if group_shape is None:
            continue


        color = cmap(
            group_index % 20
        )


        positions = [
            cluster_positions[cid] / 1e6
            for cid in group
            if cid in cluster_positions
        ]


        total_peaks = sum(
            counts[cid]
            for cid in group
        )


        cluster_text = ", ".join(
            str(cid)
            for cid in group
        )


        label = (
            f"Shape {group_index+1}: "
            #f"C[{cluster_text}] "
            f"(N={total_peaks})"
        )

        # mean shape

        ax.plot(
            x,
            group_shape,
            color=color,
            linewidth=2.5,
            alpha=0.9,
            label=label
        )

        # every peak shown in the background 

        for cid in group:

            ax.plot(
                x,
                shapes[cid],
                color=color,
                alpha=0.12,
                linewidth=0.8
            )



    ax.set_xlabel(
        "Normalized peak position"
    )

    ax.set_ylabel(
        "L2-normalized amplitude"
    )

    ax.set_title(
        "Mean normalized peak shapes grouped by similarity"
    )


    ax.grid(
        alpha=0.2
    )


    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )


    plt.tight_layout()

    plt.show()

    return shape_groups

import numpy as np


def remove_overrepresented_shape_clusters(
    clusters,
    shape_groups,
    expected_repeats=5,
    max_factor=1.8
):

    print("=" * 70)
    print("REMOVE OVERREPRESENTED SHAPE CLUSTERS")
    print("=" * 70)

    max_allowed = expected_repeats * max_factor

    print(
        f"Expected repeats : {expected_repeats}"
    )

    print(
        f"Maximum allowed  : {max_allowed:.1f}"
    )

    print()


    removed_cluster_ids = set()


    for i, shape_group in enumerate(shape_groups):

        print(
            f"Shape {i+1}: "
            f"{len(shape_group)} clusters"
        )

        print(
            f"Clusters: {shape_group}"
        )


        if len(shape_group) > max_allowed:

            print(
                "  -> REMOVE overrepresented shape"
            )

            removed_cluster_ids.update(
                shape_group
            )

        else:

            print(
                "  -> KEEP"
            )

        print()

    all_cluster_ids = set(
        range(len(clusters))
    )

    remaining_cluster_ids = (
        all_cluster_ids
        - removed_cluster_ids
    )


    remaining_cluster_ids = sorted(
        remaining_cluster_ids
    )


    # new clusters with original cluster ids

    clusters_new = [
        clusters[cid]
        for cid in remaining_cluster_ids
    ]


    print("=" * 70)

    print(
        f"Original clusters: "
        f"{len(clusters)}"
    )

    print(
        f"Removed clusters: "
        f"{len(removed_cluster_ids)}"
    )

    print(
        f"Remaining clusters: "
        f"{len(remaining_cluster_ids)}"
    )

    print()

    print(
        "Removed original IDs:"
    )

    print(
        sorted(removed_cluster_ids)
    )

    print()

    print(
        "Remaining original IDs:"
    )

    print(
        remaining_cluster_ids
    )

    print("=" * 70)


    return (
        clusters_new,
        sorted(removed_cluster_ids),
        remaining_cluster_ids
    )
