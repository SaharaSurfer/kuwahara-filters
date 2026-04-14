import cv2
import numpy as np


def classic_kuwahara(image, radius: int = 3, return_decision_map: bool = False):
    """
    Apply the classic Kuwahara filter to an image for edge-preserving smoothing.

    Parameters
    ----------
    image : ndarray
        The input grayscale (H, W) or color (H, W, C) image, expected as a
        uint8 array.
    radius : int, optional
        The filter radius that determines the (2*radius + 1) window size. For
        instance, a radius of 3 results in a 7x7 window split into four
        overlapping 4x4 subwindows. Default is 3.
    return_decision_map : bool, optional
        If True, also returns a map indicating which of the 4 subwindows
        was chosen for each pixel. Default is False.

    Returns
    -------
    result : ndarray
        The filtered image of the same shape and type (uint8) as the input.
    decision_map : ndarray, optional
        An integer array of shape (H, W) where values (0-3) represent
        the subwindow with the minimum variance. Only returned if
        `return_decision_map` is True.

    Notes
    -----
    The Kuwahara filter works on a window divided into 4 overlapping
    subwindows. In each subwindow, the mean and variance are computed.
    The output value (located at the center of the window) is set to
    the mean of the subwindow with the smallest variance.

    Example for a 5x5 pixel window (radius=2):

    ```
    ( a  a  ab   b  b)
    ( a  a  ab   b  b)
    (ac ac abcd bd bd)
    ( c  c  cd   d  d)
    ( c  c  cd   d  d)
    ```

    References
    ----------
    .. [1] M. Kuwahara, K. Hachimura, S. Eiho, and M. Kinoshita, "Processing of
       RI-Angiocardiographic Images," in Digital Processing of Biomedical
       Images, K. Preston Jr. et al., Eds. New York: Plenum Press, 1976,
       pp. 187-202.
    .. [2] M. Kuwahara, K. Hachimura, and M. Kinoshita, "Nonlinear Spatial
       Filtering of the RI-Angiocardiographic Images," Automation Research
       Laboratory, Kyoto University and Shiga Medical School, 1976.
    .. [3] W. Burger and M. J. Burge, Digital Image Processing: An Algorithmic
       Introduction, 3rd ed. Springer Cham, 2022.
    """
    if radius == 0:
        return image.copy()

    image_float = image.astype(np.float32) / 255.0
    if image_float.ndim == 2:
        image_float = image_float[..., np.newaxis]

    image_float_sq = image_float**2

    H, W, C = image_float.shape
    means = np.zeros((4, H, W, C), dtype=np.float32)
    variances = np.zeros((4, H, W), dtype=np.float32)

    subwindow_size = (radius + 1, radius + 1)

    # 'anchor' parameter of boxFilter is used to calculate the mean
    # of a shifted window without manual cropping or padding.
    subwindow_anchors = [
        (radius, radius),  # Top-left subwindow (a)
        (0, radius),  # Top-right subwindow (b)
        (radius, 0),  # Bottom-left subwindow (c)
        (0, 0),  # Bottom-right subwindow (d)
    ]

    for i, anchor in enumerate(subwindow_anchors):
        # Calculate E[X] for the subwindow
        mean = cv2.boxFilter(
            src=image_float,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(H, W, C)

        # Calculate E[X^2] for the subwindow
        mean_sq = cv2.boxFilter(
            src=image_float_sq,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(H, W, C)

        # Calculate variance as Var(X) = E[X^2] - (E[X])^2
        var_per_channel = np.clip(mean_sq - mean**2, 0.0, None)
        total_var = np.sum(var_per_channel, axis=-1)

        means[i] = mean
        variances[i] = total_var

    decision_map = np.argmin(variances, axis=0)

    yy, xx = np.indices((H, W))
    result = means[decision_map, yy, xx]

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        result = result.squeeze(axis=-1)

    if return_decision_map:
        return result, decision_map

    return result


def classic_kuwahara_flawed(image, radius: int = 3):
    """
    Apply a per-channel version of the classic Kuwahara filter.

    Parameters
    ----------
    image : ndarray
        The input grayscale (H, W) or color (H, W, C) image, expected as a
        uint8 array.
    radius : int, optional
        The filter radius that determines the (2*radius + 1) window size. For
        instance, a radius of 3 results in a 7x7 window split into four
        overlapping 4x4 subwindows. Default is 3.

    Returns
    -------
    result : ndarray
        The filtered image of the same shape and type (uint8) as the input,
        processed using per-channel variance minimization.

    Notes
    -----
    In this specific "flawed" version, the variance minimization is performed
    independently for each color channel. It often introduces color artifacts
    as the red, green, and blue components of a single pixel may originate from
    different subwindows.

    References
    ----------
    .. [1] W. Burger and M. J. Burge, Digital Image Processing: An Algorithmic
       Introduction, 3rd ed. Springer Cham, 2022.
    """
    if radius == 0:
        return image.copy()

    image_float = image.astype(np.float32) / 255.0
    if image_float.ndim == 2:
        image_float = image_float[..., np.newaxis]

    image_float_sq = image_float**2

    H, W, C = image_float.shape
    means = np.zeros((4, H, W, C), dtype=np.float32)
    variances = np.zeros((4, H, W, C), dtype=np.float32)

    subwindow_size = (radius + 1, radius + 1)

    # 'anchor' parameter of boxFilter is used to calculate the mean
    # of a shifted window without manual cropping or padding.
    subwindow_anchors = [
        (radius, radius),  # Top-left subwindow (a)
        (0, radius),  # Top-right subwindow (b)
        (radius, 0),  # Bottom-left subwindow (c)
        (0, 0),  # Bottom-right subwindow (d)
    ]

    for i, anchor in enumerate(subwindow_anchors):
        # Calculate E[X] for the subwindow
        mean = cv2.boxFilter(
            src=image_float,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(H, W, C)

        # Calculate E[X^2] for the subwindow
        mean_sq = cv2.boxFilter(
            src=image_float_sq,
            ddepth=-1,
            ksize=subwindow_size,
            anchor=anchor,
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(H, W, C)

        # Calculate variance as Var(X) = E[X^2] - (E[X])^2
        var_per_channel = np.clip(mean_sq - mean**2, 0.0, None)

        means[i] = mean
        variances[i] = var_per_channel

    # Find the best subwindow index for each pixel AND each channel
    min_var_idx = np.argmin(variances, axis=0)

    yy, xx = np.indices((H, W))
    result = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        # min_var_idx[:,:,c] can differ for R, G, and B
        result[:, :, c] = means[min_var_idx[:, :, c], yy, xx, c]

    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        result = result.squeeze(axis=-1)

    return result
