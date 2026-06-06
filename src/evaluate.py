"""
evaluate.py
Metrics and plots shared by every model notebook: accuracy, macro AUC-ROC,
macro F1, sensitivity, specificity, confusion matrix, ROC curves, learning
curves, validation curves, and the train versus test overfitting gap.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve, validation_curve


def get_scores(model, X):
    # Return class scores for AUC and ROC. Use probabilities when available,
    # otherwise fall back to the decision function (used by the SVM).
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def specificity_per_class(cm):
    # Specificity for each class, computed from the confusion matrix as
    # true negatives over true negatives plus false positives.
    spec = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(spec)


def evaluate_model(y_true, y_pred, scores, class_names):
    # Compute the full metric set in one call and return a dictionary.
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    sens = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    spec = float(specificity_per_class(cm).mean())
    yb = label_binarize(y_true, classes=list(range(len(class_names))))
    try:
        auc_macro = roc_auc_score(yb, scores, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = float("nan")
    return {"accuracy": float(acc), "auc": float(auc_macro), "f1": float(f1),
            "sensitivity": float(sens), "specificity": float(spec), "cm": cm}


def print_metrics(name, m):
    print(f"Model: {name}")
    print(f"  Accuracy    : {m['accuracy']:.4f}")
    print(f"  AUC-ROC     : {m['auc']:.4f}")
    print(f"  F1 (macro)  : {m['f1']:.4f}")
    print(f"  Sensitivity : {m['sensitivity']:.4f}")
    print(f"  Specificity : {m['specificity']:.4f}")


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true, scores, class_names, title, save_path=None):
    yb = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(yb[:, i], scores[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_curve(estimator, X, y, title, save_path=None, cv=3):
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy")
    plt.figure(figsize=(7, 5))
    plt.plot(sizes, train_scores.mean(axis=1), "o-", label="Training accuracy")
    plt.plot(sizes, val_scores.mean(axis=1), "o-", label="Cross-validation accuracy")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_validation_curve(estimator, X, y, param_name, param_range, title,
                          save_path=None, cv=3, logx=False):
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring="accuracy", n_jobs=-1)
    plt.figure(figsize=(7, 5))
    plt.plot(param_range, train_scores.mean(axis=1), "o-", label="Training accuracy")
    plt.plot(param_range, val_scores.mean(axis=1), "o-", label="Cross-validation accuracy")
    if logx:
        plt.xscale("log")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def overfitting_gap(model, X_train, y_train, X_test, y_test):
    tr = accuracy_score(y_train, model.predict(X_train))
    te = accuracy_score(y_test, model.predict(X_test))
    return tr, te, tr - te
