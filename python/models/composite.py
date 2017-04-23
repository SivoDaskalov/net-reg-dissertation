from commons import cm_magnitude_std_thresholds as mstd_thresh, cm_vote_thresholds as vote_thresh, \
    cm_zero_thresholds as zero_thresh
import numpy as np
from scipy.stats import threshold
from commons import cv_n_folds as n_folds
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def fit_composite_vote_model(setup, models):
    return CompositeModel(models=models, supporting_model=LinearRegression(fit_intercept=False)) \
        .select_predictors_by_vote(zero_thresh, vote_thresh, setup.x_tune, setup.y_tune) \
        .fit(X=setup.x_train, y=setup.y_train)


def fit_composite_magnitude_model(setup, models):
    return CompositeModel(models=models, supporting_model=LinearRegression(fit_intercept=False)) \
        .select_predictors_by_magnitude(zero_thresh, mstd_thresh, setup.x_tune, setup.y_tune) \
        .fit(X=setup.x_train, y=setup.y_train)


class CompositeModel:
    full_coef_ = None
    n_sub_models = None
    n_total_predictors = None
    supporting_model = None

    selected_predictors = None
    coef_ = None
    params_ = None

    def __init__(self, models, supporting_model):
        self.full_coef = np.array(
            [np.array(model.coef_) for name, model in models.items() if not isinstance(model, CompositeModel)])
        (self.n_sub_models, self.n_total_predictors) = self.full_coef.shape
        self.supporting_model = supporting_model

    # Retain all predictors which have non-zero coefficients in a vote_thresholds fraction of sub-models
    def select_predictors_by_vote(self, zero_thresholds=[0.01], vote_thresholds=[0.5], X=None, y=None):
        if len(zero_thresholds) > 1 or len(vote_thresholds) > 1:
            best_mse = 999999
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

            for zero_thresh in zero_thresholds:
                for vote_thresh in vote_thresholds:

                    votes = np.count_nonzero(threshold(np.absolute(self.full_coef), zero_thresh), axis=0)
                    fraction_votes = votes / float(self.n_sub_models)
                    current_predictors = np.where(fraction_votes > vote_thresh)[0]
                    X_sel = X[:, current_predictors]

                    errors = []
                    for training, holdout in kf.split(X_sel):
                        model = self.supporting_model.fit(X=X_sel[training, :], y=y[training])
                        errors.append(mean_squared_error(y[holdout], self.supporting_model.predict(X_sel[holdout, :])))
                    mse = np.mean(errors)

                    # print("Zero thresh = %.2f,\t Vote thresh = %.2f,\t MSE = %.2f" % (zero_thresh, vote_thresh, mse))
                    if mse < best_mse:
                        best_mse = mse
                        zero_threshold = zero_thresh
                        vote_threshold = vote_thresh
                        selected_predictors = current_predictors
        else:
            zero_threshold = zero_thresholds[0]
            vote_threshold = vote_thresholds[0]
            votes = np.count_nonzero(threshold(np.absolute(self.full_coef), zero_threshold), axis=0)
            fraction_votes = votes / float(self.n_sub_models)
            selected_predictors = np.where(fraction_votes > vote_threshold)[0]

        self.selected_predictors = selected_predictors
        self.params_ = {"Zero threshold": zero_threshold, "Vote threshold": vote_threshold}
        return self

    # Retain all predictors whose coefficients' L1 norm is > mean() + magnitude_std_thresholds * std()
    def select_predictors_by_magnitude(self, zero_thresholds=[0.01], magnitude_std_thresholds=[0.0], X=None, y=None):
        if len(zero_thresholds) > 1 or len(magnitude_std_thresholds) > 1:
            best_mse = 999999
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

            for zero_thresh in zero_thresholds:
                for magn_std_thresh in magnitude_std_thresholds:

                    coef_magnitudes = np.sum(threshold(np.absolute(self.full_coef), zero_thresh), axis=0)
                    mean = np.mean(coef_magnitudes)
                    std = np.mean(coef_magnitudes)
                    current_predictors = np.where(coef_magnitudes > mean + std * magn_std_thresh)[0]
                    X_sel = X[:, current_predictors]

                    errors = []
                    for training, holdout in kf.split(X_sel):
                        model = self.supporting_model.fit(X=X_sel[training, :], y=y[training])
                        errors.append(mean_squared_error(y[holdout], self.supporting_model.predict(X_sel[holdout, :])))
                    mse = np.mean(errors)

                    # print("Zero thresh = %.2f,\t Magnitude std thresh = %.2f,\t MSE = %.2f" % (
                    #     zero_thresh, magn_std_thresh, mse))
                    if mse < best_mse:
                        best_mse = mse
                        zero_threshold = zero_thresh
                        magnitude_std_threshold = magn_std_thresh
                        selected_predictors = current_predictors
        else:
            zero_threshold = zero_thresholds[0]
            magnitude_std_threshold = magnitude_std_thresholds[0]
            coef_magnitudes = np.sum(threshold(np.absolute(self.full_coef), zero_threshold), axis=0)
            mean = np.mean(coef_magnitudes)
            std = np.mean(coef_magnitudes)
            selected_predictors = np.where(coef_magnitudes > mean + std * magnitude_std_threshold)[0]

        self.selected_predictors = selected_predictors
        mst = magnitude_std_threshold
        thresh_label = "" if mst == 0.0 else "+%d*std" % mst if mst > 0 else "%d * std" % mst
        self.params_ = {"Zero threshold": zero_threshold, "Coef. thresh": "mean%s" % (thresh_label)}
        return self

    def fit(self, X, y):
        selected_coef = self.supporting_model.fit(X=X[:, self.selected_predictors], y=y).coef_
        selected_indices = self.selected_predictors.tolist()
        coef = [selected_coef[selected_indices.index(i)] if i in self.selected_predictors else 0.0 for i in
                range(self.n_total_predictors)]
        self.coef_ = np.array(coef)
        return self

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1)
