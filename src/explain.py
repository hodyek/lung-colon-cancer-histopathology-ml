"""
explain.py
Explainability helpers for the classical models. Feature importance is grouped
by family so the analysis stays readable, and permutation importance gives a
model-agnostic check. SHAP is run directly in the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def aggregate_importance_by_family(importances, feature_names, family_fn):
    # Sum importance values within each feature family and return shares that
    # add up to one. This answers which family of features the model relies on.
    fam = {}
    for imp, name in zip(importances, feature_names):
        f = family_fn(name)
        fam[f] = fam.get(f, 0.0) + float(imp)
    total = sum(fam.values()) or 1.0
    return {k: v / total for k, v in sorted(fam.items(), key=lambda x: -x[1])}


def plot_family_importance(fam_dict, title, save_path=None):
    plt.figure(figsize=(7, 4))
    plt.bar(list(fam_dict.keys()), list(fam_dict.values()), color="steelblue")
    plt.ylabel("Share of total importance")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_top_features(importances, feature_names, title, top=20, save_path=None):
    order = np.argsort(importances)[::-1][:top]
    names = [feature_names[i] for i in order]
    vals = [importances[i] for i in order]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(names))[::-1], vals, color="indianred")
    plt.yticks(range(len(names))[::-1], names, fontsize=8)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def permutation_importance_df(model, X, y, feature_names, n_repeats=5, seed=42):
    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, n_jobs=-1)
    df = pd.DataFrame({"feature": feature_names,
                       "importance": r.importances_mean,
                       "std": r.importances_std})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
