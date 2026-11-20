"""Date-blocked chronological evaluation; no randomized football splits."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss


def evaluate_probabilities(model, y_true, probabilities):
    """Multiclass Brier is the mean sum of three squared errors."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if (p.shape != (len(y), 3) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError('Probabilities must be finite normalized three-class rows.')
    predicted = p.argmax(axis=1)
    metrics = dict(model=model, accuracy=float(accuracy_score(y, predicted)),
                   macro_f1=float(f1_score(y, predicted, labels=[0, 1, 2], average='macro', zero_division=0)),
                   log_loss=float(log_loss(y, p, labels=[0, 1, 2])),
                   brier=float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1))),
                   n_test=len(y))
    return metrics, confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist()


def chronological_split(frame):
    """Return earliest 60%, next 20%, last 20% of unique dates."""
    dates = np.sort(pd.to_datetime(frame['date']).unique())
    if len(dates) < 5:
        raise ValueError('At least five distinct match dates are required.')
    first = max(1, int(len(dates) * .6))
    second = max(first + 1, int(len(dates) * .8))
    return tuple(frame.loc[frame.date.isin(block)].sort_values('date').copy()
                 for block in (dates[:first], dates[first:second], dates[second:]))
