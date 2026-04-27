import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter, map_coordinates

from kuwahara_filters.generalized import generate_sector_weights


def anisotropic_kuwahara(
    image, radius=5, N=8, q=8.0, alpha=1.0, sigma_g=1.0, sigma_s=2.0
):
    """
    Apply the Anisotropic Kuwahara filter for structure-aware image abstraction.

    Unlike the Generalized Kuwahara filter, this method adapts the shape,
    scale and orientation of the local filtering window to the image structure.

    Parameters
    ----------
    image : ndarray
        Input image of shape (H, W) or (H, W, C), expected as a uint8 array.
    radius : int, optional
        Radius of the base isotropic kernel used to build the sector weights.
        The actual support of the filter is larger because the ellipse can be
        stretched up to a factor of `(alpha + 1) / alpha`. Default is 5.
    N : int, optional
        Number of angular sectors. Default is 8.
    q : float, optional
        Sharpness parameter for the soft variance weighting. Higher values
        give harder selection of the most homogeneous sector. Default is 8.0.
    alpha : float, optional
        Tuning parameter that controls the maximum eccentricity of the
        anisotropic kernel. `alpha = 1` yields a maximum axis ratio of 4:1.
        As `alpha -> inf` the kernel becomes circular. Default is 1.0.
    sigma_g : float, optional
        Standard deviation of the Gaussian used to compute image gradients.
        Default is 1.0.
    sigma_s : float, optional
        Standard deviation of the Gaussian used to smooth the structure tensor
        components. Default is 2.0.

    Returns
    -------
    ndarray
        The filtered image of the same shape and type (uint8) as the input.

    Notes
    -----
    Because every pixel has a different ellipse orientation and eccentricity,
    the spatial convolution cannot be used so the filtering is performed
    explicitly.

    References
    ----------
    .. [1] J. E. Kyprianidis, H. Kang, and J. Döllner, "Image and Video
       Abstraction by Anisotropic Kuwahara Filtering," Computer Graphics Forum,
       vol. 28, no. 7, pp. 1955-1963, 2009.
    """
    image_float = cp.asarray(image, dtype=cp.float32)
    if image_float.ndim == 2:
        image_float = image_float[..., cp.newaxis]

    # Calculate partial derivatives
    Ix = gaussian_filter(
        image_float, sigma=sigma_g, order=(0, 1), mode="nearest", axes=(0, 1)
    )
    Iy = gaussian_filter(
        image_float, sigma=sigma_g, order=(1, 0), mode="nearest", axes=(0, 1)
    )

    # Calculate structure tensor
    E = gaussian_filter(cp.sum(Ix * Ix, axis=-1), sigma=sigma_s, mode="nearest")
    F = gaussian_filter(cp.sum(Ix * Iy, axis=-1), sigma=sigma_s, mode="nearest")
    G = gaussian_filter(cp.sum(Iy * Iy, axis=-1), sigma=sigma_s, mode="nearest")

    # Calculate eigenvalues of structure tensor
    discriminant_root = cp.sqrt(cp.clip((E - G) ** 2 + 4 * F**2, 0, None))
    lambda1 = (E + G + discriminant_root) / 2.0  # across edges
    lambda2 = (E + G - discriminant_root) / 2.0  # along edges

    # Calculate anisotropy
    A = cp.clip((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-4), 0, 1)

    # Components of stretching matrix
    scale_u = alpha / (alpha + A)
    scale_v = (alpha + A) / alpha

    # Calculate local orientation (angle of vector lying along edges)
    phi = cp.arctan2(-F, lambda1 - E)

    # Precalculate sin and cos for rotation matrix
    sin_phi = cp.sin(phi)
    cos_phi = cp.cos(phi)

    # The ellipse can stretch by up to (alpha+1)/alpha, so the effective
    # radius is larger than the base kernel radius.
    effective_radius = int(np.ceil(radius * (alpha + 1.0) / alpha))
    effective_radius_sq = effective_radius**2

    # Replicate-pad the image so that shifts never read out of bounds
    padded_img = cp.pad(
        image_float,
        pad_width=(
            (effective_radius, effective_radius),
            (effective_radius, effective_radius),
            (0, 0),
        ),
        mode="edge",
    )
    padded_img_sq = padded_img**2

    H, W, C = image_float.shape
    mean = cp.zeros((N, H, W, C), dtype=cp.float32)
    mean_sq = cp.zeros((N, H, W, C), dtype=cp.float32)
    w_sum = cp.zeros((N, H, W, 1), dtype=cp.float32)

    K_weights = cp.asarray(generate_sector_weights(radius, N=N))

    # Loop over offsets (dy, dx) forming square that contains ellipse
    for dy in range(-effective_radius, effective_radius + 1):
        for dx in range(-effective_radius, effective_radius + 1):
            # Skip offsets that lie outside circle that contains ellipse
            if dx * dx + dy * dy > effective_radius_sq:
                continue

            # (u, v) are the coordinates inside the base circular kernel.
            u = scale_u * (dx * cos_phi + dy * sin_phi)
            v = scale_v * (-dx * sin_phi + dy * cos_phi)

            map_x = u + radius
            map_y = v + radius

            coords = cp.stack([map_y, map_x], axis=0)

            shifted_img = padded_img[
                effective_radius + dy : effective_radius + dy + H,
                effective_radius + dx : effective_radius + dx + W,
            ]
            shifted_img_sq = padded_img_sq[
                effective_radius + dy : effective_radius + dy + H,
                effective_radius + dx : effective_radius + dx + W,
            ]

            # Accumulate each sector's weighted statistics.
            for i in range(N):
                W_img = map_coordinates(
                    K_weights[i], coords, order=1, mode="constant", cval=0.0
                )[..., cp.newaxis]

                mean[i] += W_img * shifted_img
                mean_sq[i] += W_img * shifted_img_sq
                w_sum[i] += W_img

    mean = mean / (w_sum + 1e-4)
    mean_sq = mean_sq / (w_sum + 1e-4)

    # Calculate variance as Var(X) = E[X^2] - (E[X])^2
    var_per_channel = cp.clip(mean_sq - mean**2, 0, None)
    total_std = cp.sqrt(cp.sum(var_per_channel, axis=-1, keepdims=True))

    # Accumulate the weighted mean and the total weight for normalization
    weight = 1.0 / (1.0 + cp.power(total_std, q))
    numerator = cp.sum(mean * weight, axis=0)
    denominator = cp.sum(weight, axis=0)

    result = cp.clip(numerator / denominator, 0, 255).astype(cp.uint8)
    if image.ndim == 2:
        result = result.squeeze(axis=-1)

    return cp.asnumpy(result)
