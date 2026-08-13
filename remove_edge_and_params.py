import numpy as np
def remove_edge_artifacts(
    frequency,
    amplitude,
    mean_window=501,
    gradient_window=201,
    gradient_threshold_factor=0.1,
    amplitude_threshold_factor=5,
    min_region=200,
    symmetric_edges=True,
    debug=True
):

    frequency = np.asarray(frequency)
    amplitude = np.asarray(amplitude)


    if mean_window % 2 == 0:
        mean_window += 1


    half = mean_window // 2


    # derivative

    grad = np.zeros_like(amplitude)


    for i in range(
        half,
        len(amplitude)-half
    ):

        left_mean = np.mean(
            amplitude[i-half:i]
        )

        right_mean = np.mean(
            amplitude[i:i+half]
        )

        grad[i] = right_mean - left_mean



    # mean value of the derivative

    kernel = np.ones(
        gradient_window
    ) / gradient_window


    grad_mean = np.convolve(
        grad,
        kernel,
        mode="same"
    )


    abs_grad = np.abs(grad_mean)


    gradient_threshold = (
        gradient_threshold_factor
        *
        np.max(abs_grad)
    )


    background = np.median(
        amplitude[
            len(amplitude)//4:
            3*len(amplitude)//4
        ]
    )


    amplitude_threshold = (
        amplitude_threshold_factor
        *
        background
    )


    if debug:

        print("max gradient:",
              np.max(abs_grad))

        print("gradient threshold:",
              gradient_threshold)

        print("background:",
              background)

        print("amplitude threshold:",
              amplitude_threshold)



    left = 0


    for i in range(
        0,
        len(amplitude)-min_region
    ):

        grad_region = np.mean(
            abs_grad[i:i+min_region]
        )

        amp_region = np.mean(
            amplitude[i:i+min_region]
        )


        if (
            grad_region < gradient_threshold
            and
            amp_region < amplitude_threshold
        ):

            left = i
            break



    right = len(amplitude)-1


    for i in range(
        len(amplitude)-1,
        min_region,
        -1
    ):

        grad_region = np.mean(
            abs_grad[i-min_region:i]
        )

        amp_region = np.mean(
            amplitude[i-min_region:i]
        )


        if (
            grad_region < gradient_threshold
            and
            amp_region < amplitude_threshold
        ):

            right = i
            break



    # local peak safe, so peaks on the edge don't get cut off

    def has_local_peak(
        region,
        drop_fraction=0.5
    ):

        if len(region) < 50:
            return False


        peak_idx = np.argmax(region)

        peak_val = region[peak_idx]


        if peak_val <= 0:
            return False


        left_side = region[:peak_idx]

        right_side = region[peak_idx:]


        if (
            len(left_side) < 10
            or
            len(right_side) < 10
        ):
            return False


        left_min = np.min(
            left_side
        )

        right_min = np.min(
            right_side
        )


        return (
            left_min < drop_fraction * peak_val
            and
            right_min < drop_fraction * peak_val
        )



    edge_length = min_region * 5

    # right edge

    right_region = amplitude[
        max(0, right-edge_length):
        right
    ]


    if has_local_peak(right_region):

        if debug:
            print(
                "Right side contains local peak -> keep peak"
            )

        peak_idx = np.argmax(
            right_region
        )

        right = (
            max(0, right-edge_length)
            +
            peak_idx
        )



    # left edge

    left_region = amplitude[
        left:
        min(
            len(amplitude),
            left+edge_length
        )
    ]


    if has_local_peak(left_region):

        if debug:
            print(
                "Left side contains local peak -> keep peak"
            )

        peak_idx = np.argmax(
            left_region
        )

        left = (
            left
            +
            peak_idx
        )

    # symmetric cut

    if symmetric_edges:

        left_cut = left

        right_cut = (
            len(amplitude)
            -
            right
            -
            1
        )


        symmetric_cut = min(
            left_cut,
            right_cut
        )


        if debug:

            print("="*60)
            print("Symmetric edge correction")
            print("="*60)

            print(
                f"Detected cuts: "
                f"left={left_cut}, "
                f"right={right_cut}"
            )

            print(
                f"Applied symmetric cut: "
                f"{symmetric_cut}"
            )


        left = symmetric_cut

        right = (
            len(amplitude)
            -
            symmetric_cut
            -
            1
        )


    if debug:

        print("="*60)
        print("Edge artifact removal")
        print("="*60)

        print(
            f"Left cut : {left}"
        )

        print(
            f"Right cut: {len(amplitude)-right-1}"
        )

        print(
            f"Remaining points: {right-left+1}"
        )


    return (
        frequency[left:right+1],
        amplitude[left:right+1]
    )

def estimate_spectrum_parameters(
    frequency,
    expected_frev=2.00e6,  # values for ESR- change if needed
    frev_tolerance=0.10e6,  # values for ESR
    safety_margin=2
):

    f_min = np.min(frequency)
    f_max = np.max(frequency)

    bandwidth = f_max - f_min

    # number of possible repetitions 

    min_repeats = max(
        3,
        int(np.floor(bandwidth / (expected_frev + frev_tolerance)))
    )
    min_repeats = min_repeats-1
    max_repeats = int(
        np.ceil(
            bandwidth /
            (expected_frev - frev_tolerance)
        )
    ) + safety_margin

    # expected harmoincs

    harmonic_min = int(
        np.floor(
            (f_min /
            (expected_frev + frev_tolerance))-5
        )
    )

    harmonic_max = int(
        np.ceil(
            (f_max /
            (expected_frev - frev_tolerance))+5
        )
    )


    print("="*60)
    print("Automatic spectrum parameters")
    print("="*60)

    print(f"Spectrum range : {f_min/1e6:.3f} - {f_max/1e6:.3f} MHz")
    print(f"Bandwidth      : {bandwidth/1e6:.3f} MHz")
    print()
    print(f"Expected f_rev : {expected_frev/1e6:.3f} MHz")
    print(f"Tolerance      : ±{frev_tolerance/1e6:.3f} MHz")
    print()
    print(f"Harmonics      : {harmonic_min} ... {harmonic_max}")
    print(f"Repeats        : {min_repeats} ... {max_repeats}")

    return {
        "freq_min": expected_frev-frev_tolerance,
        "freq_max": expected_frev+frev_tolerance,
        "harmonic_min": harmonic_min,
        "harmonic_max": harmonic_max,
        "min_repeats": min_repeats,
        "max_repeats": max_repeats
    }

