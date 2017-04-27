from commons import cv_n_folds as n_folds, glm_n_alphas as n_alphas, glm_l1_ratios as l1_ratios, glm_max_iter as iter, \
    glm_alpha_opt as opt_alpha, glm_l1_ratio_opt as opt_l1_ratio
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from models import Model


def fit_lasso(setup):
    # Tuning
    model = LassoCV(n_alphas=n_alphas, cv=n_folds, n_jobs=-1, max_iter=iter, random_state=1, fit_intercept=False)
    model.fit(X=setup.x_tune, y=setup.y_tune)

    # Training
    model = Lasso(alpha=model.alpha_, random_state=1, fit_intercept=False, max_iter=iter)
    model.fit(setup.x_train, y=setup.y_train)

    return Model(model.coef_, params={"alpha": model.alpha}, from_matlab=False)


def fit_enet(setup):
    # Tuning
    model = ElasticNetCV(n_alphas=n_alphas, l1_ratio=l1_ratios, cv=n_folds, n_jobs=-1, max_iter=iter, random_state=1,
                         fit_intercept=False)
    model.fit(X=setup.x_tune, y=setup.y_tune)

    # Training
    model = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_, random_state=1, fit_intercept=False, max_iter=iter)
    model.fit(setup.x_train, y=setup.y_train)

    return Model(model.coef_, params={"alpha": model.alpha, "l1_ratio": model.l1_ratio}, from_matlab=False)


def fit_lasso_opt(setup):
    model = Lasso(alpha=opt_alpha, random_state=1, fit_intercept=False, max_iter=iter)
    model.fit(setup.x_train, y=setup.y_train)
    return Model(model.coef_, params={"alpha": opt_alpha}, from_matlab=False)


def fit_enet_opt(setup):
    model = ElasticNet(alpha=opt_alpha, l1_ratio=opt_l1_ratio, random_state=1, fit_intercept=False, max_iter=iter)
    model.fit(setup.x_train, y=setup.y_train)
    return Model(model.coef_, params={"alpha": opt_alpha, "l1_ratio": opt_l1_ratio}, from_matlab=False)
