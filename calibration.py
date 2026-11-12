"""Train-only covariance interventions; RCA/whitening are established controls."""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def statistics(x, labels):
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    means = np.asarray([x[labels == c].mean(0) for c in classes])
    center = means.mean(0)
    within = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    total = np.zeros_like(within)
    usable = 0
    counts = []
    for c, mean in zip(classes, means):
        xx = x[labels == c]
        counts.append(len(xx))
        vv = xx - center
        total += vv.T @ vv / len(xx) / len(classes)
        if len(xx) > 1:
            vv = xx - mean
            within += vv.T @ vv / (len(xx) - 1)
            usable += 1
    if usable < 2:
        raise ValueError('At least two classes need independent images')
    within /= usable
    return center, within, total, {'classes': len(classes), 'covariance_classes': usable,
        'images': len(x), 'min_class_images': min(counts), 'max_class_images': max(counts)}


def whitening(covariance, ridge_fraction=0.1):
    values, vectors = np.linalg.eigh(covariance)
    scale = np.trace(covariance) / len(values)
    if scale <= 0:
        raise ValueError('Degenerate covariance')
    ridge = ridge_fraction * scale
    factors = np.maximum(values + ridge, ridge) ** -0.5
    # A shared scalar is immaterial after final L2 normalization.
    transform = (vectors * factors) @ vectors.T
    transform /= np.linalg.norm(transform, 'fro') / np.sqrt(len(values))
    return transform, {'minimum_eigenvalue': float(values.min()), 'maximum_eigenvalue': float(values.max()),
        'ridge': float(ridge), 'regularized_condition_number': float((values.max() + ridge) / ridge)}


def fit_interventions(x, labels, split=512):
    center, within, total, info = statistics(x, labels)
    _, permuted, _, _ = statistics(x, np.random.default_rng(20261029).permutation(labels))
    block = within.copy()
    block[:split, split:] = 0
    block[split:, :split] = 0
    matrices = {'CENTER': np.eye(x.shape[1])}
    details = {}
    for name, cov in [('TOTAL', total), ('BLOCK', block), ('PERM', permuted), ('JOINT', within)]:
        matrices[name], details[name] = whitening(cov)
    info.update(cross_block_within_fraction=float(np.linalg.norm(within[:split, split:]) * np.sqrt(2) / np.linalg.norm(within)),
        covariance_trace=float(np.trace(within)), center_norm=float(np.linalg.norm(center)),
        train_only=True, ridge_fraction=0.1, exponent=-0.5,
        candidate='JOINT', controls=['RAW', 'D2', 'CENTER', 'TOTAL', 'BLOCK', 'PERM'],
        novelty_claim=False, limitations='Class-centered covariance is a nuisance proxy, not verified morphology or acquisition ground truth.')
    return center, matrices, {'data': info, 'transforms': details}


def transform_features(x, center, matrix):
    return normalize((np.asarray(x, dtype=np.float64) - center) @ matrix).astype(np.float32)


def fixed_global_features(x, center, matrix):
    x = np.asarray(x, dtype=np.float64)
    global_part = normalize((x-center) @ matrix[:, :512])
    local_part = normalize(x[:, 512:])
    return np.concatenate([np.sqrt(.75)*global_part, .5*local_part], axis=1).astype(np.float32)
