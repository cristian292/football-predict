"""Regularized, time-weighted football team log-strength goal model.

Defence parameters are *conceding* strengths: larger means more goals allowed.
Unknown teams have zero (league-average) attack and defence parameters.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .poisson import score_distribution as independent_score_distribution


class DixonColesModel:
    """Time-weighted team-strength Dixon-Coles goal likelihood.

    ``fit`` consumes date/home_team/away_team/home_goals/away_goals; prediction
    consumes only home_team/away_team. Decay is measured from the latest training
    timestamp in UTC, never the current clock or a prediction fixture's date.
    ``half_life_days=None`` gives equal weights. The minimized objective is the
    weighted *mean* joint goal NLL plus ``ridge / 2 * sum(team_strength ** 2)``.

    Parameters are bounded for finite numerical optimization: intercept [-6, 4],
    home advantage [-3, 3], team log strengths [-4, 4], rho [-0.99, 0.99]. Stored
    attack and conceding-defence strengths are then centered to zero mean,
    adjusting the intercept to preserve known-team rates (including ridge=0).
    ``params_`` order is intercept, home advantage, attacks, defences, optional
    rho, with both team blocks ordered by ``teams_``. Unseen teams have zero
    strengths, not an arbitrary observed team's coefficients.

    Corrected fits use SLSQP with analytic objective/constraint derivatives and
    tau >= 1e-6 constraints on distinct training pairings. Independent fits use
    L-BFGS-B. A finite unconverged result is retained with a RuntimeWarning and
    ``optimization_['success']=False``; nonfinite results raise RuntimeError.
    """

    def __init__(self, half_life_days=365.0, ridge=0.1, use_correction=True, maxiter=300):
        from numbers import Integral, Real

        if half_life_days is not None and (
                isinstance(half_life_days, (bool, np.bool_)) or
                not isinstance(half_life_days, Real) or
                not np.isfinite(half_life_days) or half_life_days <= 0):
            raise ValueError('half_life_days must be positive and finite, or None.')
        if (isinstance(ridge, (bool, np.bool_)) or not isinstance(ridge, Real) or
                not np.isfinite(ridge) or ridge < 0):
            raise ValueError('ridge must be finite and nonnegative.')
        if (isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, Integral) or
                maxiter <= 0):
            raise ValueError('maxiter must be a positive integer.')
        if not isinstance(use_correction, (bool, np.bool_)):
            raise ValueError('use_correction must be a boolean.')
        self.half_life_days = half_life_days
        self.ridge = ridge
        self.use_correction = use_correction
        self.maxiter = maxiter

    def fit(self, frame):
        required = {'date', 'home_team', 'away_team', 'home_goals', 'away_goals'}
        if (not isinstance(frame, pd.DataFrame) or frame.empty or
                not frame.columns.is_unique or not required.issubset(frame.columns)):
            raise ValueError('A nonempty training DataFrame with unique required columns is required.')
        self._validate_teams(frame)
        goals = frame[['home_goals', 'away_goals']].to_numpy(dtype=float)
        if (not np.isfinite(goals).all() or (goals < 0).any() or
                (goals != np.floor(goals)).any()):
            raise ValueError('Training goals must be finite nonnegative integers.')
        dates = pd.to_datetime(frame['date'], utc=True, format='mixed')
        if dates.isna().any():
            raise ValueError('Training dates must not be missing.')
        self.training_cutoff_ = dates.max()
        self.teams_ = sorted(set(frame['home_team']) | set(frame['away_team']))
        self.team_index_ = {team: i for i, team in enumerate(self.teams_)}
        n = len(self.teams_)
        home = frame['home_team'].map(self.team_index_).to_numpy()
        away = frame['away_team'].map(self.team_index_).to_numpy()
        age = (self.training_cutoff_ - dates).dt.total_seconds().to_numpy() / 86400.0
        # Overflow of age/half-life means zero historical weight, not an
        # invalid fit. At least the latest-date fixtures always retain weight 1.
        with np.errstate(over='ignore', under='ignore'):
            weights = (np.ones(len(frame)) if self.half_life_days is None else
                       np.exp2(-age / self.half_life_days))
        self.sample_weight_ = weights.copy()
        weights /= weights.sum()
        self._training = (home, away, goals, weights)
        self._constraint_pairs = np.unique(np.column_stack((home, away)), axis=0).T
        initial = np.zeros(2 + 2 * n + int(self.use_correction))
        mean = np.maximum(np.average(goals, weights=weights, axis=0), 0.1)
        initial[0] = np.log(mean[1])
        initial[1] = np.log(mean[0]) - initial[0]
        bounds = [(-6., 4.), (-3., 3.)] + [(-4., 4.)] * (2 * n)
        if self.use_correction:
            bounds.append((-0.99, 0.99))
        initial = np.clip(initial, *np.asarray(bounds).T)
        initial_nll = self._objective(initial)[0]
        method = 'SLSQP' if self.use_correction else 'L-BFGS-B'
        constraints = ({'type': 'ineq', 'fun': self._tau_constraints,
                        'jac': self._tau_constraint_jacobian},) if self.use_correction else ()
        result = minimize(self._objective, initial, jac=True, method=method,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': self.maxiter, 'ftol': 1e-11})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError('Dixon-Coles optimization produced nonfinite parameters or likelihood.')
        self.params_ = result.x.copy()
        # Fix both additive gauges without changing any known-team intensity.
        for block in (slice(2, 2 + n), slice(2 + n, 2 + 2 * n)):
            shift = float(self.params_[block].mean())
            self.params_[block] -= shift
            self.params_[0] += shift
        self.rho_ = float(result.x[-1]) if self.use_correction else 0.0
        final_nll, final_gradient = self._objective(self.params_)
        if not np.isfinite(final_nll) or not np.isfinite(final_gradient).all():
            raise RuntimeError('Dixon-Coles optimization produced nonfinite likelihood or gradient.')
        feasible = not self.use_correction or bool((self._tau_constraints(self.params_) >= -1e-8).all())
        success = bool(result.success) and feasible
        self.optimization_ = dict(success=success, message=str(result.message),
                                  status='converged' if success else 'not_converged',
                                  tau_feasible=feasible,
                                  iterations=int(result.nit), initial_nll=float(initial_nll),
                                  final_nll=float(final_nll), rho=float(self.rho_),
                                  training_cutoff=self.training_cutoff_.isoformat(),
                                  n_train=int(len(frame)), method=method)
        if not success:
            warnings.warn('Dixon-Coles optimization did not converge: ' + str(result.message),
                          RuntimeWarning, stacklevel=2)
        return self

    @staticmethod
    def _validate_teams(frame):
        for column in ('home_team', 'away_team'):
            if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError('Team identifiers must be nonempty strings.')
        if (frame['home_team'] == frame['away_team']).any():
            raise ValueError('Home and away teams must differ.')

    def _tau_constraints(self, params):
        """Enforce positivity for each distinct training pairing (not each row)."""
        home, away = self._constraint_pairs
        n = len(self.teams_)
        attack, defence = params[2:2 + n], params[2 + n:2 + 2 * n]
        lam = np.exp(params[0] + params[1] + attack[home] + defence[away])
        mu = np.exp(params[0] + attack[away] + defence[home])
        rho = params[-1]
        return np.concatenate((1 + rho * lam, 1 + rho * mu,
                               1 - rho * lam * mu, [1 - rho])) - 1e-6

    def _tau_constraint_jacobian(self, params):
        home, away = self._constraint_pairs
        n = len(self.teams_)
        count = len(home)
        attack, defence = params[2:2 + n], params[2 + n:2 + 2 * n]
        lam = np.exp(params[0] + params[1] + attack[home] + defence[away])
        mu = np.exp(params[0] + attack[away] + defence[home])
        home_log_jac = np.zeros((count, len(params)))
        away_log_jac = np.zeros_like(home_log_jac)
        rows = np.arange(count)
        home_log_jac[:, :2] = 1.0
        home_log_jac[rows, 2 + home] = 1.0
        home_log_jac[rows, 2 + n + away] = 1.0
        away_log_jac[:, 0] = 1.0
        away_log_jac[rows, 2 + away] = 1.0
        away_log_jac[rows, 2 + n + home] = 1.0
        rho = params[-1]
        jac = np.vstack((rho * lam[:, None] * home_log_jac,
                         rho * mu[:, None] * away_log_jac,
                         -rho * (lam * mu)[:, None] * (home_log_jac + away_log_jac),
                         np.zeros((1, len(params)))))
        jac[:, -1] = np.concatenate((lam, mu, -lam * mu, [-1.0]))
        return jac

    def _objective(self, params):
        """Penalized weighted negative log likelihood and analytic gradient."""
        home, away, goals, weights = self._training
        n = len(self.teams_)
        attack, defence = params[2:2 + n], params[2 + n:2 + 2 * n]
        log_home = params[0] + params[1] + attack[home] + defence[away]
        log_away = params[0] + attack[away] + defence[home]
        logs = np.column_stack((log_home, log_away))
        rates = np.exp(logs)
        loss = np.sum(weights[:, None] * (rates - goals * logs + gammaln(goals + 1)))
        residual = weights[:, None] * (rates - goals)
        rho_gradient = 0.0
        if self.use_correction:
            rho = params[-1]
            lam, mu = rates.T
            product = lam * mu
            taus = np.column_stack((1 - product * rho, 1 + lam * rho,
                                    1 + mu * rho, np.full(len(home), 1 - rho)))
            d_home = np.column_stack((-product * rho, lam * rho,
                                     np.zeros(len(home)), np.zeros(len(home))))
            d_away = np.column_stack((-product * rho, np.zeros(len(home)),
                                     mu * rho, np.zeros(len(home))))
            d_rho = np.column_stack((-product, lam, mu, -np.ones(len(home))))
            # A C1 log extension keeps rejected line-search candidates finite.
            # Every negative tau (not only the observed cell) is penalized.
            floor = 1e-6
            violation = np.minimum(taus - floor, 0.0)
            loss += 1e4 * np.sum(weights[:, None] * violation ** 2)
            tau_gradient = 2e4 * weights[:, None] * violation
            low = (goals[:, 0] <= 1) & (goals[:, 1] <= 1)
            rows = np.flatnonzero(low)
            columns = (2 * goals[low, 0] + goals[low, 1]).astype(int)
            observed = taus[rows, columns]
            safe = np.maximum(observed, floor)
            delta = np.minimum(observed - floor, 0.0) / floor
            loss -= np.sum(weights[low] * (np.log(safe) + delta - 0.5 * delta ** 2))
            tau_gradient[rows, columns] -= weights[low] * (1 - delta) / safe
            residual[:, 0] += np.sum(tau_gradient * d_home, axis=1)
            residual[:, 1] += np.sum(tau_gradient * d_away, axis=1)
            rho_gradient = np.sum(tau_gradient * d_rho)
        gradient = np.zeros_like(params)
        gradient[0] = residual.sum()
        gradient[1] = residual[:, 0].sum()
        np.add.at(gradient[2:2 + n], home, residual[:, 0])
        np.add.at(gradient[2:2 + n], away, residual[:, 1])
        np.add.at(gradient[2 + n:2 + 2 * n], away, residual[:, 0])
        np.add.at(gradient[2 + n:2 + 2 * n], home, residual[:, 1])
        loss += 0.5 * self.ridge * np.dot(params[2:2 + 2 * n], params[2:2 + 2 * n])
        gradient[2:2 + 2 * n] += self.ridge * params[2:2 + 2 * n]
        if self.use_correction:
            gradient[-1] = rho_gradient
        return float(loss), gradient

    def expected_goals(self, frame):
        if not hasattr(self, 'params_'):
            raise RuntimeError('Call fit before predicting expected goals.')
        if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or
                not {'home_team', 'away_team'}.issubset(frame.columns)):
            raise ValueError('Fixture DataFrame must have unique home_team and away_team columns.')
        self._validate_teams(frame)
        n = len(self.teams_)
        home = frame['home_team'].map(self.team_index_).fillna(n).to_numpy(dtype=int)
        away = frame['away_team'].map(self.team_index_).fillna(n).to_numpy(dtype=int)
        attack = np.append(self.params_[2:2 + n], 0.0)
        defence = np.append(self.params_[2 + n:2 + 2 * n], 0.0)
        return np.exp(np.column_stack((
            self.params_[0] + self.params_[1] + attack[home] + defence[away],
            self.params_[0] + attack[away] + defence[home])))

    def predict_proba(self, frame):
        """Return home win, draw, away win probabilities, in that order."""
        return np.asarray([self.score_distribution(*rates)[1]
                           for rates in self.expected_goals(frame)]).reshape(-1, 3)

    def score_distribution(self, home_rate, away_rate):
        """Return a normalized home-row/away-column grid, H/D/A, omitted tail.

        For new rate pairs, rho is projected into the pair's tau-positive
        interval with a 1e-12 margin. This changes *rho*, not individual cells,
        so the four-cell mass-preserving Dixon-Coles correction is retained.
        ``last_effective_rho_`` records the actual correction for this call;
        ``rho_`` remains the fitted training value. Tail is the unnormalized
        omitted Poisson mass (the correction has zero net mass inside the grid).
        """
        matrix, outcomes, tail = independent_score_distribution(home_rate, away_rate)
        rho = self.rho_ if self.use_correction else 0.0
        largest, smallest = max(home_rate, away_rate), min(home_rate, away_rate)
        # A reciprocal outside float64 range is an unrestrictive bound.
        with np.errstate(over='ignore'):
            lower = -(1 - 1e-12) / largest if largest else -np.inf
        upper = ((1 - 1e-12) / largest / smallest
                 if largest > 1 and smallest > 1 / largest else 1 - 1e-12)
        rho = float(np.clip(rho, lower, upper))
        self.last_effective_rho_ = rho
        if rho == 0.0:
            return matrix, outcomes, tail
        matrix[:2, :2] *= [[1 - home_rate * away_rate * rho, 1 + home_rate * rho],
                          [1 + away_rate * rho, 1 - rho]]
        matrix /= matrix.sum()
        outcomes = np.array([np.tril(matrix, -1).sum(), np.trace(matrix),
                             np.triu(matrix, 1).sum()])
        return matrix, outcomes, tail
