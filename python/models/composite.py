from commons import cm_vote_thresholds as vote_thresh, cm_zero_thresholds as zero_thresh, \
    cm_vote_thresh_opt as opt_vote_thresh, cm_zero_thresh_opt as opt_zero_thresh
import numpy as np
from scipy.stats import threshold
from commons import cv_n_folds as n_folds
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def fit_composite_model(setup, models):
    return CompositeModel(models=models, supporting_model=LinearRegression(fit_intercept=False)) \
        .select_predictors(zero_thresh, vote_thresh, setup.x_tune, setup.y_tune) \
        .fit(X=setup.x_train, y=setup.y_train)


def fit_composite_model_opt(setup, models):
    return CompositeModel(models=models, supporting_model=LinearRegression(fit_intercept=False)) \
        .select_predictors([opt_zero_thresh], [opt_vote_thresh]) \
        .fit(X=setup.x_train, y=setup.y_train)


class CompositeModel:
    full_coef_ = None
    n_sub_models = None
    n_total_predictors = None
    supporting_model = None

    fraction_votes_ = None
    selected_predictors_ = None
    coef_ = None
    params_ = None

    def __init__(self, models, supporting_model):
        self.full_coef = np.array(
            [np.array(model.coef_) for name, model in models.items() if not isinstance(model, CompositeModel)])
        (self.n_sub_models, self.n_total_predictors) = self.full_coef.shape
        self.supporting_model = supporting_model

    # Retain all predictors which have non-zero coefficients in a vote_thresholds fraction of sub-models
    def select_predictors(self, zero_thresholds=[0.01], vote_thresholds=[0.5], X=None, y=None):
        if len(zero_thresholds) > 1 or len(vote_thresholds) > 1:
            best_mse = 999999
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

            for zero_thresh in zero_thresholds:
                for vote_thresh in vote_thresholds:

                    votes = np.count_nonzero(threshold(np.absolute(self.full_coef), zero_thresh), axis=0)
                    current_fraction_votes = votes / float(self.n_sub_models)
                    current_predictors = np.where(current_fraction_votes >= vote_thresh)[0]
                    X_sel = X[:, current_predictors]

                    errors = []
                    for training, holdout in kf.split(X_sel):
                        self.supporting_model = self.supporting_model.fit(X=X_sel[training, :], y=y[training])
                        errors.append(mean_squared_error(y[holdout], self.supporting_model.predict(X_sel[holdout, :])))
                    mse = np.mean(errors)

                    # print("Zero thresh = %.2f,\t Vote thresh = %.2f,\t MSE = %.2f" % (zero_thresh, vote_thresh, mse))
                    if mse < best_mse:
                        best_mse = mse
                        zero_threshold = zero_thresh
                        vote_threshold = vote_thresh
                        fraction_votes = current_fraction_votes
                        selected_predictors = current_predictors
        else:
            zero_threshold = zero_thresholds[0]
            vote_threshold = vote_thresholds[0]
            votes = np.count_nonzero(threshold(np.absolute(self.full_coef), zero_threshold), axis=0)
            fraction_votes = votes / float(self.n_sub_models)
            selected_predictors = np.where(fraction_votes >= vote_threshold)[0]

        self.fraction_votes_ = fraction_votes
        self.selected_predictors_ = selected_predictors
        self.params_ = {"Zero threshold": zero_threshold, "Vote threshold": vote_threshold}
        return self

    def get_fraction_votes(self):
        return self.fraction_votes_

    def set_vote_threshold(self, threshold_new):
        self.params_["Vote threshold"] = threshold_new
        self.selected_predictors_ = np.where(self.fraction_votes_ >= threshold_new)[0]

    def fit(self, X, y):
        if len(self.selected_predictors_) == 0:
            coef = [0] * self.n_total_predictors
        else:
            selected_coef = self.supporting_model.fit(X=X[:, self.selected_predictors_], y=y).coef_
            selected_indices = self.selected_predictors_.tolist()
            coef = [selected_coef[selected_indices.index(i)] if i in self.selected_predictors_ else 0.0 for i in
                    range(self.n_total_predictors)]
        self.coef_ = np.array(coef)
        return self

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1)
