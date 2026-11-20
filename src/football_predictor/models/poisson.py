"""Independent goal Poisson with adaptive, tail-controlled score grids."""
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from scipy.special import gammaln


class PoissonModel:
    """Optimize log scale plus home intercept on training goal likelihood only."""
    columns = ['poisson_home', 'poisson_away']

    def fit(self, x, goals):
        base = np.asarray(x[self.columns], dtype=float)
        goals = np.asarray(goals, dtype=float)
        if base.shape != goals.shape or not np.isfinite(base).all() or (base <= 0).any():
            raise ValueError('Positive two-column historical rates and aligned goal pairs required.')
        if not np.isfinite(goals).all() or (goals < 0).any():
            raise ValueError('Training goals must be finite and nonnegative.')

        def objective(params):
            log_rates = np.log(base) + params[0] + np.array([params[1], 0.])
            return float(np.mean(np.sum(np.exp(log_rates) - goals * log_rates + gammaln(goals + 1), axis=1)))

        initial = np.zeros(2)
        trace = [objective(initial)]
        result = minimize(objective, initial, method='L-BFGS-B', bounds=[(-2., 2.), (-1.5, 1.5)],
                          callback=lambda params: trace.append(objective(params)))
        if not np.isfinite(result.fun):
            raise RuntimeError('Poisson optimization produced a nonfinite likelihood.')
        self.params_ = result.x
        self.optimization_ = dict(success=bool(result.success), message=str(result.message),
                                  iterations=int(result.nit), initial_nll=trace[0], final_nll=float(result.fun),
                                  trace=trace, n_train=len(goals),
                                  log_scale=float(result.x[0]), home_intercept=float(result.x[1]),
                                  method='L-BFGS-B', objective='mean independent Poisson goal negative log likelihood')
        return self

    def expected_goals(self, x):
        return np.asarray(x[self.columns], dtype=float) * np.exp(self.params_[0] + np.array([self.params_[1], 0.]))

    def predict_proba(self, x):
        return np.asarray([score_distribution(*rates)[1] for rates in self.expected_goals(x)])



def historical_rates(matches, window=10, smoothing=5.):
    """Venue-specific rolling attack/defence, smoothed toward earlier league rates.

    Fixed cold-start priors are 1.5 home and 1.2 away goals. All observations
    on a date are consumed only after that date's covariates are generated.
    Missing goals allow a future fixture to be appended without updating history.
    """
    history = defaultdict(lambda: deque(maxlen=window))
    totals = np.zeros(2)
    count = 0
    output = pd.DataFrame(index=matches.index, columns=['poisson_home', 'poisson_away'], dtype=float)
    for _, group in matches.sort_values('date').groupby('date', sort=True):
        league = (totals + smoothing * np.array([1.5, 1.2])) / (count + smoothing)
        for row in group.itertuples():
            home = np.asarray(history[(row.home_team, 'home')], dtype=float).reshape(-1, 2)
            away = np.asarray(history[(row.away_team, 'away')], dtype=float).reshape(-1, 2)
            home_mean = (home.sum(axis=0) + smoothing * league) / (len(home) + smoothing)
            away_mean = (away.sum(axis=0) + smoothing * league[::-1]) / (len(away) + smoothing)
            rates = [home_mean[0] * away_mean[1] / league[0],
                     away_mean[0] * home_mean[1] / league[1]]
            output.loc[row.Index] = np.maximum(rates, .05)
        for row in group.itertuples():
            if pd.notna(row.home_goals) and pd.notna(row.away_goals):
                history[(row.home_team, 'home')].append([row.home_goals, row.away_goals])
                history[(row.away_team, 'away')].append([row.away_goals, row.home_goals])
                totals += [row.home_goals, row.away_goals]
                count += 1
    return output



def score_distribution(home_rate, away_rate, tolerance=1e-9):
    rates = np.asarray([home_rate, away_rate], dtype=float)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Goal rates must be finite and nonnegative.')
    maximum = max(8, int(np.ceil(rates.max())))
    while 1 - float(np.prod(poisson.cdf(maximum, rates))) >= tolerance:
        maximum += max(1, maximum // 4)
    goals = np.arange(maximum + 1)
    matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    tail = max(0., 1. - float(matrix.sum()))
    matrix /= matrix.sum()
    outcomes = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
    return matrix, outcomes, tail
