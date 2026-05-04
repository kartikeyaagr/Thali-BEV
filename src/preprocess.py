import cv2
import numpy as np


def mean_shift(image: np.ndarray, sp: int = 20, sr: int = 40) -> np.ndarray:
    """
    Smooth intra-region texture while preserving compartment edges.

    sp (spatial radius): pixels within sp are candidates for merging.
    sr (color radius):   pixels with colour distance < sr get merged.

    Higher sr → more aggressive colour merging (good for noisy food textures).
    Higher sp → larger spatial neighbourhoods (can bleed across thin compartment walls).
    """
    return cv2.pyrMeanShiftFiltering(image, sp=sp, sr=sr)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Enhance local contrast in the L channel (LAB space).
    Useful when the thali is unevenly lit (one side brighter than the other).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = eq.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess(
    image: np.ndarray,
    sp: int = 20,
    sr: int = 40,
    use_clahe: bool = False,
) -> np.ndarray:
    """
    Full pre-processing chain before SAM.
    CLAHE first (improves edges in dark regions), then mean shift (smooths regions).
    """
    if use_clahe:
        image = clahe(image)
    return mean_shift(image, sp=sp, sr=sr)
