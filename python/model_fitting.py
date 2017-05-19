from models.glm import fit_lasso, fit_enet, fit_enet_opt, fit_lasso_opt
from models.grace import fit_grace, fit_agrace, fit_agrace_opt, fit_grace_opt
from models.gblasso import fit_gblasso, fit_gblasso_opt
from models.linf import fit_linf, fit_alinf, fit_alinf_opt, fit_linf_opt
from models.tlp import fit_ttlp, fit_ltlp, fit_ttlp_opt, fit_ltlp_opt
from models.composite import fit_composite_model, fit_composite_model_opt
from commons import Setup, timestamp
import model_utils as modut
import matlab.engine
import os.path
import pickle
import time
import math

full_method_list = ["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf", "ttlp", "ltlp", "composite"]
real_data_method_list = ["lasso", "enet", "grace", "gblasso", "linf", "composite"]


def fit_or_load(setup, method_name, load_dump, fitting_func, args, base_dump_url):
    dump_url = base_dump_url + method_name
    if load_dump and os.path.exists(dump_url) and method_name != "composite":
        print("%sLoaded %s model for %s" % (timestamp(), method_name, setup.label))
        with open(dump_url, 'rbU') as f:
            fit = pickle.load(f)
    else:
        print("%sFitting %s model for %s" % (timestamp(), method_name, setup.label))
        t_ = time.clock()
        fit = fitting_func(setup, *args)
        print(
            "%sFitting %s model for %s took %.0f seconds\n" % (
                timestamp(), method_name, setup.label, time.clock() - t_))
        if method_name != "composite":
            with open(dump_url, 'wb') as f:
                pickle.dump(fit, f)
    return fit


def fit_models(setup, engine, methods=full_method_list, load_dump=True, base_dump_url=None):
    if base_dump_url is None:
        base_dump_url = "dumps/%s_n%d_p%d/%s_n%d_p%d_" % (setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1],
                                                          setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1])
        dir_url = "dumps/%s_n%d_p%d" % (setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1])
        if not os.path.exists(dir_url):
            os.makedirs(dir_url)

    models = {}
    t_ = time.clock()

    if "agrace" in methods and "enet" not in methods:
        methods.append("enet")
    if "alinf" in methods and "linf" not in methods:
        methods.append("linf")
    if ("ttlp" in methods or "ltlp" in methods) and "lasso" not in methods:
        methods.append("lasso")

    if "lasso" in methods:
        method = "lasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_lasso, [], base_dump_url)

    if "enet" in methods:
        method = "enet"
        models[method] = fit_or_load(setup, method, load_dump, fit_enet, [], base_dump_url)

    if "grace" in methods:
        method = "grace"
        models[method] = fit_or_load(setup, method, load_dump, fit_grace, [engine], base_dump_url)

    if "agrace" in methods:
        method = "agrace"
        models[method] = fit_or_load(setup, method, load_dump, fit_agrace, [engine, models["enet"]], base_dump_url)

    if "gblasso" in methods:
        method = "gblasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_gblasso, [], base_dump_url)

    if "linf" in methods:
        method = "linf"
        models[method] = fit_or_load(setup, method, load_dump, fit_linf, [engine], base_dump_url)

    if "alinf" in methods:
        method = "alinf"
        models[method] = fit_or_load(setup, method, load_dump, fit_alinf, [engine, models["linf"]], base_dump_url)

    if "ttlp" in methods:
        method = "ttlp"
        models[method] = fit_or_load(setup, method, load_dump, fit_ttlp, [engine, models["lasso"]], base_dump_url)

    if "ltlp" in methods:
        method = "ltlp"
        models[method] = fit_or_load(setup, method, load_dump, fit_ltlp, [engine, models["lasso"]], base_dump_url)

    if "composite" in methods:
        method = "composite"
        models[method] = fit_or_load(setup, method, load_dump, fit_composite_model, [models], base_dump_url)

    print("%sFitting models for %s took %.0f seconds\n" % (timestamp(), setup.label, time.clock() - t_))
    return models


def batch_fit_models(setups, methods=full_method_list, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    models = [(setup, fit_models(setup=setup, engine=engine, methods=methods, load_dump=load_dump)) for setup in setups]
    similarities = modut.batch_evaluate_similarities(models)
    return models, similarities


def fit_models_opt_params(setup, engine, methods=real_data_method_list, load_dump=True, base_dump_url=None):
    if base_dump_url is None:
        base_dump_url = "dumps/%s_n%d_p%d/%s_n%d_p%d_" % (setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1],
                                                          setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1])
    models = {}
    t_ = time.clock()

    if "agrace" in methods and "enet" not in methods:
        methods.append("enet")
    if "alinf" in methods and "linf" not in methods:
        methods.append("linf")
    if ("ttlp" in methods or "ltlp" in methods) and "lasso" not in methods:
        methods.append("lasso")

    if "lasso" in methods:
        method = "lasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_lasso_opt, [], base_dump_url)

    if "enet" in methods:
        method = "enet"
        models[method] = fit_or_load(setup, method, load_dump, fit_enet_opt, [], base_dump_url)

    if "grace" in methods:
        method = "grace"
        models[method] = fit_or_load(setup, method, load_dump, fit_grace_opt, [engine], base_dump_url)

    if "agrace" in methods:
        method = "agrace"
        models[method] = fit_or_load(setup, method, load_dump, fit_agrace_opt, [engine, models["enet"]], base_dump_url)

    if "gblasso" in methods:
        method = "gblasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_gblasso_opt, [], base_dump_url)

    if "linf" in methods:
        method = "linf"
        models[method] = fit_or_load(setup, method, load_dump, fit_linf_opt, [engine], base_dump_url)

    if "alinf" in methods:
        method = "alinf"
        models[method] = fit_or_load(setup, method, load_dump, fit_alinf_opt, [engine, models["linf"]], base_dump_url)

    if "ttlp" in methods:
        method = "ttlp"
        models[method] = fit_or_load(setup, method, load_dump, fit_ttlp_opt, [engine, models["lasso"]], base_dump_url)

    if "ltlp" in methods:
        method = "ltlp"
        models[method] = fit_or_load(setup, method, load_dump, fit_ltlp_opt, [engine, models["lasso"]], base_dump_url)

    if "composite" in methods:
        method = "composite"
        models[method] = fit_or_load(setup, method, load_dump, fit_composite_model_opt, [models], base_dump_url)

    print("%sFitting models for %s took %.0f seconds\n" % (timestamp(), setup.label, time.clock() - t_))
    return models


def batch_fit_real_data(datasets, methods=full_method_list, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")

    fits_dir = "fits"
    if not os.path.exists(fits_dir):
        os.makedirs(fits_dir)

    fits = []
    for dataset in datasets:
        dataset_dir = "%s/%s" % (fits_dir, dataset.label)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        netwk = dataset.network
        deg = dataset.degrees

        x_full = dataset.methylation.as_matrix()
        test_fraction = 0.25

        train_test_cutoff = x_full.shape[0] - int(math.ceil(x_full.shape[0] * test_fraction))
        x_tr = x_full[:train_test_cutoff]
        x_ts = x_full[train_test_cutoff:]

        dataset_fits = []
        for gene in dataset.expression.columns:
            gene_dir = "%s/%s" % (dataset_dir, gene)
            if not os.path.exists(gene_dir):
                os.makedirs(gene_dir)
            base_dump_url = "%s/" % gene_dir

            y_full = dataset.expression.loc[:, gene].values
            y_tr = y_full[:train_test_cutoff]
            y_ts = y_full[train_test_cutoff:]

            setup = Setup(label="%s_%s" % (gene, dataset.label), network=netwk, degrees=deg, true_coefficients=None,
                          x_tune=x_tr, y_tune=y_tr, x_train=x_tr, y_train=y_tr, x_test=x_ts, y_test=y_ts)
            dataset_fits.append(
                (setup, fit_models_opt_params(setup=setup, engine=engine, methods=methods, load_dump=load_dump,
                                              base_dump_url=base_dump_url)))

        with open("dumps/%s_models" % dataset.label, 'wb') as f:
            pickle.dump(dataset_fits, f)

        with open("dumps/%s_models" % dataset.label, 'rbU') as f:
            dataset_fits = pickle.load(f)
        modut.assemble_mappings(dataset, dataset_fits, methods)
        fits += dataset_fits
    return fits
