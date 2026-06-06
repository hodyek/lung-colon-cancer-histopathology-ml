"""
features.py
Handcrafted feature extraction for histopathology images.
Three feature families are computed: colour, texture (GLCM and LBP), and
shape or edge structure (HOG). Each feature carries a readable name so the
explainability stage can report which family drives each class.
"""

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from scipy.stats import skew

# Feature settings kept in one place so notebooks and the report agree.
HOG_PPC = (56, 56)        # pixels per cell, gives a moderate HOG length
HOG_CPB = (2, 2)          # cells per block
HOG_ORI = 9               # gradient orientation bins
LBP_P, LBP_R = 8, 1       # neighbours and radius for Local Binary Patterns
GLCM_LEVELS = 32          # grey levels used to keep the GLCM small and fast
GLCM_DIST = [1, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity",
              "energy", "correlation", "ASM"]


def _color_features(img_u8):
    # Colour moments (mean, standard deviation, skewness) in RGB, HSV and LAB,
    # plus a small HSV histogram. Colour matters in H and E histology because
    # haematoxylin stains nuclei blue and eosin stains cytoplasm pink.
    rgb = img_u8.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb)
    lab = rgb2lab(rgb)
    feats, names = [], []
    for tag, arr in [("RGB", rgb), ("HSV", hsv), ("LAB", lab)]:
        for ch in range(3):
            v = arr[:, :, ch].ravel()
            feats += [float(v.mean()), float(v.std()), float(skew(v))]
            names += [f"color_{tag}_c{ch}_mean",
                      f"color_{tag}_c{ch}_std",
                      f"color_{tag}_c{ch}_skew"]
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=16, range=(0, 1), density=True)
        feats += hist.tolist()
        names += [f"color_HSVhist_c{ch}_b{i}" for i in range(16)]
    return feats, names


def _glcm_features(gray_q):
    # Grey Level Co-occurrence Matrix texture descriptors (Haralick features).
    # Values are averaged over the four directions at each distance.
    glcm = graycomatrix(gray_q, distances=GLCM_DIST, angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS, symmetric=True, normed=True)
    feats, names = [], []
    for p in GLCM_PROPS:
        vals = graycoprops(glcm, p)
        for di, d in enumerate(GLCM_DIST):
            feats.append(float(vals[di].mean()))
            names.append(f"glcm_{p}_d{d}")
    return feats, names


def _lbp_features(gray_u8):
    # Local Binary Pattern histogram, a compact description of micro texture.
    lbp = local_binary_pattern(gray_u8, LBP_P, LBP_R, method="uniform")
    n_bins = LBP_P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    names = [f"lbp_b{i}" for i in range(n_bins)]
    return hist.tolist(), names


def _hog_features(gray_u8):
    # Histogram of Oriented Gradients, capturing edge and structure layout.
    feats = hog(gray_u8, orientations=HOG_ORI, pixels_per_cell=HOG_PPC,
                cells_per_block=HOG_CPB, block_norm="L2-Hys", feature_vector=True)
    names = [f"hog_{i}" for i in range(len(feats))]
    return feats.tolist(), names


def extract_all_features(img_rgb_u8, apply_macenko=True):
    # Run the full feature pipeline on one RGB image and return a single vector
    # together with the matching feature names.
    from src.dataset import macenko_normalize
    img = macenko_normalize(img_rgb_u8) if apply_macenko else img_rgb_u8
    gray = (rgb2gray(img.astype(np.float32) / 255.0) * 255).astype(np.uint8)
    gray_q = (gray // (256 // GLCM_LEVELS)).astype(np.uint8)
    f1, n1 = _color_features(img)
    f2, n2 = _glcm_features(gray_q)
    f3, n3 = _lbp_features(gray)
    f4, n4 = _hog_features(gray)
    vec = np.array(f1 + f2 + f3 + f4, dtype=np.float32)
    names = n1 + n2 + n3 + n4
    return vec, names


def feature_family(name):
    # Map a feature name to its family for grouped importance analysis.
    if name.startswith("color"):
        return "color"
    if name.startswith("glcm"):
        return "glcm"
    if name.startswith("lbp"):
        return "lbp"
    if name.startswith("hog"):
        return "hog"
    return "other"
