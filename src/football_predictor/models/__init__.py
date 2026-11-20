"""Reproducible, training-only preprocessing and fixed outcome ordering."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classifier(name, params):
    if name == 'KNN':
        estimator = KNeighborsClassifier(**params)
    elif name == 'Random Forest':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, **params)
    else:
        raise ValueError(f'Unknown classifier: {name}')
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()), ('model', estimator)])


def aligned_probabilities(model, x):
    probabilities = np.zeros((len(x), 3))
    probabilities[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return probabilities
