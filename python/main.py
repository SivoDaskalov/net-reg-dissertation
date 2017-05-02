from commons import real_data_methods, dump, load
import orchestrated_tuning as orctun
import data_gen_utils as gen
import import_utils as imp
import model_fitting as fitting
import model_metrics as metrics
import plotting_utils as plut
import pandas as pd
import time

time.clock()


def tune_method_parameters_with_generated_dataset():
    n_trans_factors = 50
    n_regulated_genes_per_trans_factor = 10
    p = n_trans_factors * (n_regulated_genes_per_trans_factor + 1)
    pd.set_option('display.width', 200)

    setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
                                       n_trans_factors=n_trans_factors, load_dump=True,
                                       n_tune_obs=200, n_train_obs=100, n_test_obs=100)

    fits = fitting.batch_fit_models(setups, load_dump=True)
    results = metrics.batch_evaluate_models(fits)
    print(results)
    # plut.plot_results(results)

    # model_dump_url = "dumps/cache/orctun_models_p%d" % p
    # orctun_fits = orctun.batch_do_orchestrated_tuning(setups, load_dump=True)
    # dump(orctun_fits, model_dump_url)
    # orctun_fits = load(model_dump_url)
    # orctun_results = metrics.batch_evaluate_models(orctun_fits, "results/orctun_results_%s_p%d.csv" % (
    #     orctun.optimization_method, p))
    # print(orctun_results)
    # plut.plot_results(orctun_results)


def fit_optimal_parameter_models_on_real_data(methods=real_data_methods):
    datasets = imp.batch_import_datasets()
    fits = fitting.batch_fit_tumor_data(datasets, methods=methods)
    results = metrics.batch_evaluate_models(fits, filename="results/tumor_data.csv")


tune_method_parameters_with_generated_dataset()
# fit_optimal_parameter_models_on_real_data(methods=["lasso", "enet", "grace", "agrace", "composite"])
print("Total time elapsed: %.0f seconds" % time.clock())
