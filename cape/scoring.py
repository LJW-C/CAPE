"""Empirical prototypes, Prototype-Energy, known-only calibration, and metrics."""

import math

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


METRICS = ("known_acc", "closed_known_acc", "unknown_auroc", "unknown_f1", "h_score",
           "KAR_clean_avg", "KAR_deg_avg", "URR_clean_avg", "URR_deg_avg", "H4_avg")


def empirical_prototypes(features, labels, known_classes):
    centers = []
    for cls in known_classes:
        group = features[np.asarray(labels) == cls]
        if len(group) == 0:
            raise ValueError(f"No fitting sample for known class {cls}")
        mean = group.mean(axis=0)
        centers.append(mean / max(np.linalg.norm(mean), 1e-8))
    return np.stack(centers).astype(np.float32)


def prototype_energy(features, prototypes, temperature=0.2):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Energy temperature must be positive")
    similarities = features @ prototypes.T
    predicted = similarities.argmax(axis=1)
    logits = similarities / float(temperature)
    maximum = logits.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1).clip(min=1e-12))
    return -float(temperature) * logsumexp, predicted


def calibrate_known(scores, target_fpr=0.05):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite known-validation scores")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0,1)")
    rejected = math.floor(len(scores) * target_fpr)
    threshold = float(np.sort(scores)[len(scores) - rejected - 1])
    actual = int((scores > threshold).sum())
    if actual != rejected:
        raise ValueError("Score ties prevent the requested empirical rejection count")
    return threshold, {"samples": len(scores), "target_fpr": target_fpr,
                       "rejected": actual, "actual_fpr": actual / len(scores)}


def condition_metrics(labels, scores, predicted_local, threshold, known_classes, unknown_classes):
    labels = np.asarray(labels)
    known, unknown = np.isin(labels, known_classes), np.isin(labels, unknown_classes)
    if not np.all(known | unknown) or not known.any() or not unknown.any():
        raise ValueError("Test labels must contain the specified known/unknown classes")
    reject = np.asarray(scores) > threshold
    correct = np.asarray(known_classes)[predicted_local] == labels
    accuracy = float((correct[known] & ~reject[known]).mean())
    recall = float(reject[unknown].mean())
    return {"known_acc": accuracy, "closed_known_acc": float(correct[known].mean()),
            "unknown_recall": recall, "unknown_auroc": float(roc_auc_score(unknown.astype(int), scores)),
            "unknown_f1": float(f1_score(unknown.astype(int), reject.astype(int), zero_division=0)),
            "h_score": 2 * accuracy * recall / max(accuracy + recall, 1e-8),
            "KAR": float((~reject[known]).mean()), "URR": recall}


def four_group(clean, degraded):
    rates = [clean["KAR"], degraded["KAR"], clean["URR"], degraded["URR"]]
    return {"KAR_clean": rates[0], "KAR_deg": rates[1], "URR_clean": rates[2],
            "URR_deg": rates[3], "H4": 4.0 / sum(1.0 / (r + 1e-8) for r in rates),
            "known_acc_deg": degraded["known_acc"],
            "closed_known_acc_deg": degraded["closed_known_acc"]}


def summarize_conditions(clean, conditions):
    if not conditions:
        raise ValueError("At least one perturbation is required")
    row = {key: clean[key] for key in ("known_acc", "closed_known_acc", "unknown_auroc",
                                       "unknown_f1", "h_score")}
    row.update({key + "_avg": float(np.mean([r[key] for r in conditions]))
                for key in ("KAR_clean", "KAR_deg", "URR_clean", "URR_deg", "H4")})
    return row
