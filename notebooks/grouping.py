# Grouping by frequency 
def group_clusters_strict_periodic(
    sim_matrix,
    cluster_ids,
    cluster_positions,
    cluster_amplitudes,
    cluster_activity,
    freq_min,
    freq_max,
    min_repeats,
    max_repeats,
    sim_threshold=0.6,
    harmonic_tolerance=2e4,
    max_spacing_error=2e4,
    max_harmonic_deviation=0.02   # possible deviation from integer harmonic
):

    candidate_groups = []


    for i, base in enumerate(cluster_ids):

        base_pos = cluster_positions[base]

        possible_frev = []


        for j, other in enumerate(cluster_ids):

            if other == base:
                continue


            if sim_matrix[i,j] < sim_threshold:
                continue


            delta = abs(
                cluster_positions[other]
                -
                base_pos
            )


            if freq_min <= delta <= freq_max:

                possible_frev.append(delta)



        if len(possible_frev) == 0:
            continue



        for frev0 in possible_frev:


            group = []
            harmonics = []


            for h in range(max_repeats+1):

                target = (
                    base_pos
                    +
                    h*frev0
                )


                best = None
                best_score = -np.inf


                for cid in cluster_ids:

                    pos = cluster_positions[cid]


                    error = abs(
                        pos-target
                    )


                    if error > harmonic_tolerance:
                        continue



                    idx1 = cluster_ids.index(base)
                    idx2 = cluster_ids.index(cid)


                    shape = sim_matrix[
                        idx1,
                        idx2
                    ]


                    activity = cluster_activity.get(
                        cid,
                        0
                    )


                    local_score = (

                        0.55*shape

                        +

                        0.25*activity

                        +

                        0.20*np.exp(
                            -0.5*
                            (error/30000)**2
                        )

                    )


                    if local_score > best_score:

                        best_score = local_score
                        best = cid



                if best is not None:

                    if best not in group:

                        group.append(best)
                        harmonics.append(h)



            if len(group) < min_repeats:
                continue



            if len(np.unique(harmonics)) != len(harmonics):
                continue



            # linear fit

            x = np.asarray(
                harmonics
            )


            y = np.asarray([
                cluster_positions[c]
                for c in group
            ])


            fit = np.polyfit(
                x,
                y,
                1
            )


            f_rev = fit[0]
            f0 = fit[1]


            fitted = (
                f0
                +
                x*f_rev
            )


            residuals = (
                y-fitted
            )


            fit_error = np.mean(
                np.abs(residuals)
            )



            if fit_error > max_spacing_error:
                continue



            # check harmonics 

            harmonic_values = (
                (y) / f_rev
            )


            harmonic_deviation = np.abs(
                harmonic_values
                -
                np.round(harmonic_values)
            )


            if np.any(
                harmonic_deviation > max_harmonic_deviation
            ):
                continue



            candidate_groups.append({

                "clusters":group,

                "harmonics":np.asarray(
                    harmonics
                ),

                "harmonic_spacing":f_rev,

                "f0_fit":f0,

                "fit_error":fit_error,

                "harmonic_values":harmonic_values,

                "harmonic_deviation":harmonic_deviation

            })



    # remove double candidates 

    unique = []

    seen = set()


    for g in candidate_groups:

        key = tuple(
            sorted(
                g["clusters"]
            )
        )


        if key in seen:
            continue


        seen.add(key)
        unique.append(g)



    unique = sorted(
        unique,
        key=lambda g: (
            -len(g["clusters"]),
            np.mean(g["harmonic_deviation"]),
            g["fit_error"]
        )
    )



    final = []

    occupied = set()


    for g in unique:

        overlap = len(
            set(g["clusters"])
            &
            occupied
        )


        if overlap > 0:

            if overlap / len(g["clusters"]) > 0.8:
                continue



        final.append(g)

        occupied.update(
            g["clusters"]
        )
    print("\nNumber of candidate groups before overlap filtering:")
    print(len(unique))

    for i,g in enumerate(unique[:20]):

        print(
            i,
            g["clusters"],
            np.round(
                g["harmonic_values"],
                3
            ),
            "f_rev=",
            g["harmonic_spacing"]/1e6
        )

    return final

# conditions combined 
def score_periodic_groups(
    groups,
    cluster_amplitudes,
    cluster_activity,
    cluster_ids,
    sim_matrix
):

    scored=[]


    max_activity = (
        max(cluster_activity.values())
        +1e-12
    )


    for g in groups:

        clusters = g["clusters"]


        fit_error = g["fit_error"]


        freq_score = np.exp(
            -0.5*(fit_error/15000)**2
        )



        sim_values=[]


        for i in range(len(clusters)):

            for j in range(i+1,len(clusters)):

                idx1=cluster_ids.index(clusters[i])
                idx2=cluster_ids.index(clusters[j])

                sim_values.append(
                    sim_matrix[idx1,idx2]
                )


        shape_score=np.mean(sim_values)



        amps=np.array([
            cluster_amplitudes[c]
            for c in clusters
        ])


        median_amp=np.median(amps)

        amp_dev=np.mean(
            np.abs(
                amps-median_amp
            )
            /
            (median_amp+1e-12)
        )


        amp_score=np.exp(-amp_dev)



        activity_score=np.mean([
            cluster_activity[c]/max_activity
            for c in clusters
        ])



        size_score=min(
            len(clusters)/5,
            1.0
        )
        order_score = g.get(
            "order_score",
            0
        )


        total=(

            0.30*freq_score
            +
            0.15*shape_score
            +
            0.10*amp_score
            +
            0.05*activity_score
            +
            0.05*size_score
            +
            0.35*order_score

        )

        g["order_score"]=order_score
        g["score"]=total
        g["freq_score"]=freq_score
        g["shape_score"]=shape_score
        g["amp_score"]=amp_score
        g["activity_score"]=activity_score
        g["size_score"]=size_score


        scored.append(g)



    return sorted(
        scored,
        key=lambda x:x["score"],
        reverse=True
    )

import copy
import itertools
import numpy as np


def select_final_groups(
    groups,
    min_clusters=2
):
    """
    Selects the combination of groups resulting in the maximum number
    of final groups.

    Rules
    -----
    1. Clusters must not be used more than once.

    2. In the event of a conflict, a cluster may be removed
       from a group.

    3. After removal, a group must contain at least `min_clusters`
       clusters.

    4. Primary objective:
           maximum number of groups

    5. Secondary objective:
           maximum number of clusters

    6. Tertiary objective:
           minimal fit error
    """

    groups = copy.deepcopy(groups)

    if len(groups) == 0:
        return []

    best_solution = None
    best_key = None


    for r in range(1, len(groups) + 1):

        for combination in itertools.combinations(
            groups,
            r
        ):


            combination = sorted(
                combination,
                key=lambda g: len(g["clusters"]),
                reverse=True
            )


            used_clusters = set()

            current_groups = []

            valid = True


            for group in combination:

                clusters = list(
                    group["clusters"]
                )

                harmonics = list(
                    group["harmonics"]
                )


                new_clusters = []
                new_harmonics = []

                for cid, h in zip(
                    clusters,
                    harmonics
                ):

                    if cid in used_clusters:
                        continue

                    new_clusters.append(cid)
                    new_harmonics.append(h)



                if len(new_clusters) < min_clusters:

                    valid = False
                    break


                new_group = copy.deepcopy(group)

                new_group["clusters"] = new_clusters

                new_group["harmonics"] = np.asarray(
                    new_harmonics
                )


                current_groups.append(
                    new_group
                )

                used_clusters.update(
                    new_clusters
                )


            if not valid:
                continue


            number_of_groups = len(
                current_groups
            )


            number_of_clusters = sum(
                len(g["clusters"])
                for g in current_groups
            )


            total_fit_error = sum(
                g.get("fit_error", 0)
                for g in current_groups
            )

            key = (
                number_of_groups,
                number_of_clusters,
                -total_fit_error
            )


            if (
                best_key is None
                or key > best_key
            ):

                best_key = key

                best_solution = current_groups

    if best_solution is None:

        return []
    
    print("=" * 70)
    print("FINAL GROUP SELECTION")
    print("=" * 70)

    print(
        f"Final number of groups: "
        f"{len(best_solution)}"
    )

    print(
        f"Total clusters: "
        f"{sum(len(g['clusters']) for g in best_solution)}"
    )


    for i, g in enumerate(best_solution):

        print(
            f"\nGroup {i+1}"
        )

        print(
            "clusters:",
            g["clusters"]
        )

        print(
            "harmonics:",
            g["harmonics"]
        )

        print(
            f"f_rev = "
            f"{g['harmonic_spacing']/1e6:.6f} MHz"
        )

        print(
            f"fit error = "
            f"{g['fit_error']/1e3:.2f} kHz"
        )


    return best_solution

