"""
dataset.py
Image listing, stratified train/val/test splitting, image loading,
and Macenko stain normalisation for the LC25000 machine learning pipeline.
"""

import os
import glob
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# Fixed class order used everywhere in the project
CLASSES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def list_images(data_dir):
    # Walk the dataset folder and collect every image with a known class label.
    # The class label is taken from the name of the folder the image sits in.
    paths = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        paths += glob.glob(os.path.join(data_dir, "**", ext), recursive=True)
    rows = []
    for p in paths:
        label = os.path.basename(os.path.dirname(p))
        if label in CLASS_TO_IDX:
            rows.append({"path": p, "label": label, "y": CLASS_TO_IDX[label]})
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    return df


def make_splits(df, seed=42, train_frac=0.70, val_frac=0.15):
    # Stratified split into 70 percent train, 15 percent validation, 15 percent test.
    # The same seed reproduces the same split, so it can match the deep learning pipeline.
    temp_frac = 1.0 - train_frac
    train_df, temp_df = train_test_split(
        df, test_size=temp_frac, stratify=df["y"], random_state=seed
    )
    rel_test = (1.0 - train_frac - val_frac) / temp_frac
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["y"], random_state=seed
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[val_df.index, "split"] = "val"
    out.loc[test_df.index, "split"] = "test"
    return out


def load_image(path, size=224):
    # Read an image, convert from BGR to RGB, and resize to a fixed square size.
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def macenko_normalize(img, Io=240, alpha=1, beta=0.15):
    # Custom Macenko stain normalisation using only numpy and OpenCV.
    # staintools and spams fail to install on the Colab Python build, so this
    # standalone version is used for the whole project. It maps the stain colours
    # of any H and E image onto a fixed reference, reducing colour variation
    # between slides. If the maths fails on an unusual image, the original is returned.
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    h, w, c = img.shape
    flat = img.reshape((-1, 3)).astype(float)
    try:
        OD = -np.log((flat + 1.0) / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        if ODhat.shape[0] < 10:
            return img
        cov = np.cov(ODhat.T)
        _, eigvecs = np.linalg.eigh(cov)
        Vtop = eigvecs[:, 1:3]
        That = ODhat.dot(Vtop)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        vMin = Vtop.dot(np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = Vtop.dot(np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T
        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        maxC[maxC == 0] = 1e-6
        C2 = C * (maxCRef / maxC)[:, np.newaxis]
        Inorm = Io * np.exp(-HERef.dot(C2))
        Inorm = np.clip(Inorm, 0, 255)
        Inorm = Inorm.T.reshape(h, w, 3).astype(np.uint8)
        return Inorm
    except Exception:
        return img
