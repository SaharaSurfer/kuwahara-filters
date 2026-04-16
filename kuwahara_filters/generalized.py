import cv2
import numpy as np


def generate_sector_weights(radius, N=8, sigma=None):
    """
    Generate a set of fuzzy sector kernels for the Generalized Kuwahara filter.

    Parameters
    ----------
    radius : int
        The spatial extent of the kernel, defining a square window of size
        (2*radius + 1).
    N : int, optional
        The number of sectors to divide the neighborhood into. A higher N
        provides more directional sensitivity. Default is 8.
    sigma : float, optional
        The standard deviation of the Gaussian spatial window. If not
        provided, it defaults to radius / 2.0.

    Returns
    -------
    list of ndarray
        A list of N normalized kernels, each of shape (2\*radius+1, 2*radius+1),
        representing a specific directional subwindow.
    """
    if sigma is None:
        sigma = radius / 2.0

    # Coordinate grid centered at zero
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]

    # Base radial Gaussian window
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Polar angles for each pixel in the window
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)

    # Smoothing factor to create "fuzzy" sector boundaries
    sigma_v = sigma / 4.0

    weights = []
    for i in range(N):
        # Define bounds for the current angular sector
        theta_start = (i - 0.5) * (2 * np.pi / N)
        theta_end = (i + 0.5) * (2 * np.pi / N)

        u = np.zeros_like(theta, dtype=np.float32)

        mask = (theta >= theta_start) & (theta <= theta_end)
        u[mask] = 1.0

        # Handle wrap-around for sectors crossing the 0/2pi boundary
        if theta_end > 2 * np.pi:
            u[theta <= (theta_end - 2 * np.pi)] = 1.0
        if theta_start < 0:
            u[theta >= (2 * np.pi + theta_start)] = 1.0

        # Blur the angular mask to create a smooth transition between sectors
        v = cv2.GaussianBlur(
            u, (0, 0), sigmaX=sigma_v, borderType=cv2.BORDER_REPLICATE
        )

        # Combine radial Gaussian with the angular mask and normalize
        w = g * v
        w = w / np.sum(w)
        weights.append(w)

    return weights


def generalized_kuwahara(image, radius=5, N=8, q=8.0):
    """
    Apply the Generalized Kuwahara filter for edge-preserving smoothing.

    Unlike the classic Kuwahara filter which selects a single subwindow, this
    algorithm calculates a weighted average of all subwindows, where weights
    are inversely proportional to the local variance.

    Parameters
    ----------
    image : ndarray
        The input grayscale (H, W) or color (H, W, C) image, expected as a
        uint8 array.
    radius : int, optional
        The filter radius that determines the (2*radius + 1) window size.
        Default is 5.
    N : int, optional
        The number of directional sectors used, with higher values offering
        better preservation of corners. Default is 8.
    q : float, optional
        A tuning parameter controlling the "sharpness" of the weighting.
        Higher values cause the filter to approximate the classic "hard-min"
        Kuwahara selection, while lower values result in a Gaussian-like
        blending of sectors. Default is 8.0.

    Returns
    -------
    ndarray
        The filtered image of the same shape and type (uint8) as the input,
        exhibiting smoothed textures and enhanced edges.

    Notes
    -----
    The Generalized Kuwahara filter addresses the instability and blocky
    artifacts of the classic implementation. Instead of a hard minimum
    selection, it uses a "soft-min" approach where the output is a weighted
    sum of the means ($\mu_i$) of $N$ sectors.

    This approach allows for a smoother transition between regions while
    significantly enhancing edges and corners through the use of overlapping
    fuzzy angular kernels.

    References
    ----------
    .. [1] G. Papari, N. Petkov, and P. Campisi, "Artistic Edge and Corner
       Enhancing Smoothing," IEEE Transactions on Image Processing, vol. 16,
       no. 10, pp. 2449-2462, 2007.
    """
    if radius == 0:
        return image.copy()

    image_float = image.astype(np.float32) / 255.0
    if image_float.ndim == 2:
        image_float = image_float[..., np.newaxis]

    image_float_sq = image_float**2

    H, W, C = image_float.shape
    numerator = np.zeros((H, W, C), dtype=np.float32)
    denominator = np.zeros((H, W, 1), dtype=np.float32)

    sector_weights = generate_sector_weights(radius, N=N)
    for sector_weight in sector_weights:
        # Calculate E[X] for the subwindow
        mean = cv2.filter2D(
            image_float, -1, sector_weight, borderType=cv2.BORDER_REPLICATE
        ).reshape(H, W, C)

        # Calculate E[X^2] for the subwindow
        mean_sq = cv2.filter2D(
            image_float_sq, -1, sector_weight, borderType=cv2.BORDER_REPLICATE
        ).reshape(H, W, C)

        # Calculate variance as Var(X) = E[X^2] - (E[X])^2
        var_per_channel = np.clip(mean_sq - mean**2, 0.0, None)
        total_var = np.sum(var_per_channel, axis=-1, keepdims=True)
        total_std = np.sqrt(total_var) + 1e-4

        # Accumulate the weighted mean and the total weight for normalization
        weight = np.power(total_std, -q)
        numerator += mean * weight
        denominator += weight

    result = np.clip(numerator / denominator * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        result = result.squeeze(axis=-1)

    return result
