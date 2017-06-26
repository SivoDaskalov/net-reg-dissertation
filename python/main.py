from commons import real_data_methods, dump, load
from orchestrated_tuning import batch_do_orchestrated_tuning
from orchestrated_tuning.utilities import load_custom_start_points
from data_gen_utils import batch_generate_setups
from import_utils import batch_import_datasets
from model_fitting import batch_fit_models, batch_fit_real_data
from model_metrics import batch_evaluate_models, summarize_results
from model_utils import export_errors, batch_evaluate_similarities
from plotting_utils import plot_summary_comparison, plot_parameter_tuning, plot_mapping_summary, plot_mapping_errors, \
    plot_mapping_fraction_votes, plot_convergence_statistics
from models.composite import fit_composite_model_opt
import pandas as pd
import time

time.clock()

n_trans_factors = 50
n_regulated_genes_per_trans_factor = 10
p = n_trans_factors * (n_regulated_genes_per_trans_factor + 1)
relevant_trans_factor_groups = [1, 2, 3, 4, 5]
setups = batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor, load_dump=True,
                               n_trans_factors=n_trans_factors, train_on_tuning_dataset=True, n_tune_obs=300,
                               n_test_obs=100, n_relevant_trans_factor_groups=relevant_trans_factor_groups)


def cv_mse_tune_generated_data(methods, load_dump=False):
    pd.set_option('display.width', 200)
    fits = batch_fit_models(setups, methods=methods, load_dump=load_dump)
    similarities = batch_evaluate_similarities(fits, summary_url="results/similarities.csv",
                                               figure_url="figures/cv_mse_similarities.png",
                                               title='Method similarities, CV of MSE tuning')
    results = batch_evaluate_models(fits, "results/p%d" % p)
    summary = summarize_results(results, "results/p%d_summary" % p)
    print(summary)


def orchestrated_tune_generated_data(load_dump=False, opt_method="coef_correlation"):
    pd.set_option('display.width', 200)
    model_dump_url = "dumps/cache/orctun_models_%s_p%d" % (opt_method, p)
    orctun_fits = batch_do_orchestrated_tuning(setups, load_dump=load_dump, opt_method=opt_method)
    dump(orctun_fits, model_dump_url)
    orctun_fits = load(model_dump_url)

    recalculate_composite_models = False
    if recalculate_composite_models:
        for setup, fits in orctun_fits:
            fits["composite"] = fit_composite_model_opt(setup, fits)

    delete_composite_models = True
    if delete_composite_models:
        for setup, fits in orctun_fits:
            del fits["composite"]

    similarities = batch_evaluate_similarities(orctun_fits, summary_url="results/orctun_similarities.csv",
                                               figure_url="figures/orchestrated_similarities.png",
                                               title='Method similarities, Orchestrated tuning')
    orctun_results = batch_evaluate_models(orctun_fits, "results/orctun_results_%s_p%d" % (opt_method, p))
    summary = summarize_results(orctun_results, "results/orctun_results_%s_p%d_summary" % (opt_method, p))
    print(summary)


def fit_optimal_parameter_models_on_real_data(methods=real_data_methods, labels=["body", "prom"], load_dump=False):
    datasets = batch_import_datasets(labels=labels)
    fits = batch_fit_real_data(datasets, methods=methods, load_dump=load_dump)
    results = batch_evaluate_models(fits, filename="results/real_data")
    export_errors(results)


def summarize_mapping_results():
    plot_mapping_summary()
    plot_mapping_errors()
    plot_mapping_fraction_votes()


cv_mse_tune_generated_data(
    methods=["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf", "ttlp", "ltlp", "composite"],
    load_dump=True)
load_custom_start_points("results/p550.csv")
orchestrated_tune_generated_data(opt_method="coef_correlation", load_dump=True)
plot_convergence_statistics()

plot_summary_comparison(summary_urls={"CV of MSE tuning": "results/p550_summary.csv"},
                        figure_url="figures/cv_mse_tuning.png",
                        suptitle="Model metrics by regression method, CV-MSE tuning")
plot_summary_comparison(color="#FF7F0E",
    summary_urls={"Orchestrated tuning": "results/orctun_results_coef_correlation_p550_summary.csv"},
    figure_url="figures/orchestrated_tuning.png",
    suptitle="Model metrics by regression method, Orchestrated tuning")
plot_summary_comparison(summary_urls={
    "CV-MSE tuning": "results/p550_summary.csv",
    "Orchestrated tuning": "results/orctun_results_coef_correlation_p550_summary.csv"},
    figure_url="figures/tuning_method_comparison.png",
    suptitle="Model metrics comparison by tuning method")
plot_parameter_tuning(results_file_urls=["results/p550.csv", "results/orctun_results_coef_correlation_p550.csv"])

fit_optimal_parameter_models_on_real_data(methods=["lasso", "enet", "grace", "linf", "composite"], load_dump=True)
summarize_mapping_results()
print("Total time elapsed: %.0f seconds" % time.clock())
