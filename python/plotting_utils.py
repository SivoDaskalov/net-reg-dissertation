from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import math
from orchestrated_tuning.utilities import converge_iterations
from commons import full_method_list as method_order, glm_l1_ratios, grace_lambda1_values, grace_lambda2_values, \
    gblasso_gamma_values, gblasso_lambda_values, linf_c_values, alinf_e_values, cm_vote_thresholds, mapping_files, \
    error_files, fraction_votes_files


def simple_bar_plot(title, methods, values):
    plt.figure()
    x = range(len(methods))
    plt.bar(x, values, align='center')
    plt.xticks(x, methods, rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("figures/gitignore/%s.png" % title)
    plt.clf()
    plt.close()


def plot_results(results, columns=["mse", "predictors", "correlation", "sens", "spec", "prec"]):
    setups = results.setup.unique()
    for setup in setups:
        subframe = results[results["setup"] == setup]
        methods = subframe["model"].values.tolist()
        for column in columns:
            values = subframe[column].values.tolist()
            simple_bar_plot("%s %s" % (setup, column), methods, values)


def plot_similarities_heatmap(similarities, methods, url="figures/similarities.png", title='Method similarities',
                              cmap=plt.cm.Blues):
    plt.figure()
    plt.title(title, y=1.15)
    plt.imshow(similarities, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(methods))
    plt.xticks(tick_marks, methods, rotation=45)
    plt.gca().xaxis.tick_top()
    plt.yticks(tick_marks, methods)

    thresh = (similarities.max() + similarities.min()) / 2.
    for i, j in itertools.product(range(similarities.shape[0]), range(similarities.shape[1])):
        plt.text(j, i, "%.2f" % similarities[i, j],
                 horizontalalignment="center", color="white" if similarities[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(url)


def plot_summary_comparison(summary_urls, figure_url="figures/tuning_method_comparison.png",
                            suptitle="Properties by tuning method"):
    summaries = {}
    for label, url in sorted(summary_urls.iteritems(), reverse=True):
        summaries[label] = pd.read_csv(url, index_col=0)

    methods = set(method_order)
    for label, summary in summaries.iteritems():
        methods = methods.intersection(summary.index.values)
    methods = [method for method in method_order if method in methods]

    n_tuning_methods = len(summary_urls.keys())
    ind = np.arange(len(methods))
    width = 0.8 / n_tuning_methods

    subplots_info = [("MSE", "mse",), ("Sensitivity", "sens"), ("Specificity", "spec"), ("Precision", "prec")]
    figure, subplots = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(4):
        subplot = subplots[int(i / 2), int(i % 2)]
        title = "%s" % subplots_info[i][0]
        subplot.set_title(title)
        subplot.set_xticks(np.subtract(ind + width / 2, 0.4 if n_tuning_methods == 1 else 0.0))
        subplot.set_xticklabels(methods, rotation=45 if n_tuning_methods == 1 else 0)

        j = 0
        for label, summary in sorted(summaries.iteritems()):
            sub_summary = summary.loc[methods]
            subplot.bar(ind + j * width, sub_summary["%s mean" % subplots_info[i][1]], width,
                        yerr=sub_summary["%s std" % subplots_info[i][1]], label=label)
            j += 1
        subplot.set_ylim((0.0, subplot.get_ylim()[1]))
        if i != 0:
            subplot.set_ylim((0.0, 1.0))

    h, l = subplot.get_legend_handles_labels()
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if n_tuning_methods > 1:
        plt.figlegend(h, l, bbox_to_anchor=[0.5, 0.02], loc='center', ncol=len(summaries.keys()))
        plt.subplots_adjust(bottom=0.07)
    plt.savefig(figure_url)


def plot_parameter_tuning(results_file_urls=["results/p550.csv"]):
    import seaborn as sns
    parsed_params = {}

    for results_file_url in results_file_urls:
        results = pd.read_csv(results_file_url, index_col=0)
        for idx, row in results.iterrows():
            model = row['model']
            if model == "composite" and results_file_url == "results/orctun_results_coef_correlation_p550.csv":
                continue
            cur_params = [key_value.strip().split('=') for key_value in row["params"].split(',')]

            if not model in parsed_params:
                parsed_params[model] = {}
                for param_name, param_value in cur_params:
                    parsed_params[model][param_name] = []

            for param_name, param_value in cur_params:
                parsed_params[model][param_name].append(float(param_value))

    for model, params in parsed_params.iteritems():
        if len(params) == 1:
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()
            if model in xticks.keys():
                ax.set_xticks(xticks[model])
                bins = [xticks[model][0]] + [(j + i) / 2 for i, j in zip(xticks[model][:-1], xticks[model][1:])] + [
                    xticks[model][-1]]
            else:
                bins = 15
            if model in xlim.keys():
                ax.set_xlim(xlim[model])
            sns.distplot(params.values()[0], bins=bins, kde=False, ax=ax)
            ax.set_xscale(xscale[model])
            ax.set_xlabel(xlabel[model])
            ax.set_ylabel("Number of times selected")

        elif len(params) == 2:
            count = pd.DataFrame(params).groupby([params.keys()[0], params.keys()[1]]).size()
            data = pd.DataFrame({'count': count}).reset_index()

            g = sns.JointGrid(x=data.columns[0], y=data.columns[1], data=data)
            ax = g.ax_joint
            ax.set_xlim(xlim[model])
            ax.set_ylim(ylim[model])
            ax.set_xticks(xticks[model])
            ax.set_xlabel(xlabel[model])
            ax.set_yticks(yticks[model])
            ax.set_ylabel(ylabel[model])

            if xscale[model] == "log":
                ax.set_xscale(xscale[model])
                g.ax_marg_x.set_xscale(xscale[model])
                xbins = np.logspace(math.log10(xticks[model][0]), math.log10(xticks[model][-1]),
                                    2 * len(xticks[model]))
                xbins = [xticks[model][0]] + [np.logspace(math.log10(i), math.log10(j), 3)[1] for i, j in
                                              zip(xticks[model][:-1], xticks[model][1:])] + [
                            xticks[model][-1]]
            else:
                xbins = [xticks[model][0]] + [(j + i) / 2 for i, j in zip(xticks[model][:-1], xticks[model][1:])] + [
                    xticks[model][-1]]

            if yscale[model] == "log":
                ax.set_yscale(yscale[model])
                g.ax_marg_y.set_yscale(yscale[model])
                ybins = np.logspace(math.log10(yticks[model][0]), math.log10(yticks[model][-1]),
                                    2 * len(yticks[model]) - 1)
                ybins = [yticks[model][0]] + [np.logspace(math.log10(i), math.log10(j), 3)[1] for i, j in
                                              zip(yticks[model][:-1], yticks[model][1:])] + [
                            yticks[model][-1]]
            else:
                ybins = [yticks[model][0]] + [(j + i) / 2 for i, j in zip(yticks[model][:-1], yticks[model][1:])] + [
                    yticks[model][-1]]

            xhist_values = [val for sublist in
                            [[data.iloc[idx, 0]] * int(data.iloc[idx, 2]) for idx in range(data.shape[0])]
                            for val in sublist]
            yhist_values = [val for sublist in
                            [[data.iloc[idx, 1]] * int(data.iloc[idx, 2]) for idx in range(data.shape[0])]
                            for val in sublist]
            g.ax_marg_x.hist(xhist_values, color="b", alpha=.6, bins=xbins)
            g.ax_marg_y.hist(yhist_values, color="b", alpha=.6, orientation="horizontal", bins=ybins)

            plt.sca(ax)
            plt.scatter(x=data.iloc[:, 0], y=data.iloc[:, 1], s=data.iloc[:, 2] * 100, c=data.iloc[:, 2], cmap="Blues",
                        edgecolors="Blue")

        else:
            continue
            # No tuning plots for the TLP methods

        plt.suptitle("Parameter tuning for the %s method" % model)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig("figures/tuning/%s.png" % model)


xlabel = {
    "lasso": "Alpha",
    "enet": "Alpha",
    "grace": "Lambda 1",
    "agrace": "Lambda 1",
    "gblasso": "Gamma",
    "linf": "C",
    "alinf": "E",
    "composite": "Fraction of votes"
}

ylabel = {
    "enet": "L1 ratio",
    "grace": "Lambda 2",
    "agrace": "Lambda 2",
    "gblasso": "Lambda"
}

xticks = {
    # "lasso": [x * 0.02 for x in range(11)],
    "enet": [x * 0.1 for x in range(11)],
    "grace": grace_lambda1_values,
    "agrace": grace_lambda1_values,
    "gblasso": gblasso_gamma_values,
    "linf": linf_c_values,
    "alinf": alinf_e_values,
    "composite": cm_vote_thresholds
}

yticks = {
    "enet": glm_l1_ratios,
    "grace": grace_lambda2_values,
    "agrace": grace_lambda2_values,
    "gblasso": gblasso_lambda_values
}

xlim = {
    # "lasso": (0.0, 0.2),
    "enet": (0.0, 1.0),
    "grace": (min(grace_lambda1_values), max(grace_lambda1_values)),
    "agrace": (min(grace_lambda1_values), max(grace_lambda1_values)),
    "gblasso": (min(gblasso_gamma_values), max(gblasso_gamma_values)),
    "linf": (min(linf_c_values), max(linf_c_values)),
    "alinf": (min(alinf_e_values), max(alinf_e_values)),
    "composite": (min(cm_vote_thresholds), max(cm_vote_thresholds))
}

ylim = {
    "enet": (0.0, 1.0),
    "grace": (min(grace_lambda2_values), max(grace_lambda2_values)),
    "agrace": (min(grace_lambda2_values), max(grace_lambda2_values)),
    "gblasso": (min(gblasso_lambda_values), max(gblasso_lambda_values))
}

xscale = {
    "lasso": "linear",
    "enet": "linear",
    "grace": "log",
    "agrace": "log",
    "gblasso": "linear",
    "linf": "linear",
    "alinf": "linear",
    "composite": "linear"
}

yscale = {
    "enet": "linear",
    "grace": "log",
    "agrace": "log",
    "gblasso": "log"
}


def plot_distribution_hist(ax, data, title, xlabel, ylabel, bins=10, yscale="linear", normed=True,
                           customize_x_axis=False):
    if customize_x_axis:
        lim = (0, max(data))
        ticks = range(lim[1] + 1)
        bins = np.subtract(ticks, 0.5).tolist() + [ticks[-1] + 0.5]
        ticks = [x for x in ticks if x % 2 == 0]
        ax.set_xlim(lim)
        ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.hist(data, bins=bins, normed=normed)


def plot_mapping_summary(mapping_files=mapping_files):
    summary = pd.DataFrame(columns=["Dataset", "Method", "Resolved", "Unresolved", "Influencing", "Non-influencing"])
    for label, url in mapping_files.iteritems():
        mapping = pd.read_csv(url, index_col=0)
        method, dataset = label.split("_")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        dependencies = np.count_nonzero(mapping, axis=1)
        plot_distribution_hist(ax=axes[0], data=dependencies[dependencies < 100], xlabel="Number of covariates",
                               ylabel="Fraction of data",
                               title="Distribution of gene dependencies", customize_x_axis=True)
        n_resolved = np.count_nonzero(dependencies)
        n_unresolved = len(dependencies) - n_resolved

        influences = np.count_nonzero(mapping, axis=0)
        plot_distribution_hist(ax=axes[1], data=influences, bins=30, title="Distribution of influenced genes",
                               xlabel="Number of influenced genes", ylabel="Fraction of data")
        n_influencers = np.count_nonzero(influences)
        n_non_influencers = len(influences) - n_influencers

        plt.suptitle("Gene interactions for %s method on %s dataset" % (method, dataset))
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig("figures/mappings/distributions/%s.png" % label)

        summary.loc[label] = [dataset, method, n_resolved, n_unresolved, n_influencers, n_non_influencers]
    summary = summary.sort_values(["Dataset", "Method"])
    summary.to_csv("results/mapping_summary.csv", sep=',', index=False)


def plot_mapping_errors(error_files=error_files):
    summary = pd.DataFrame(columns=["Dataset", "Method", "MSE", "STD"])
    for dataset, url in error_files.iteritems():
        errors = pd.read_csv(url, index_col=0)
        mses = np.mean(errors, axis=0)
        stds = np.std(errors, axis=0)
        for method in errors.columns:
            summary.loc[len(summary.index)] = [dataset, method, mses[method], stds[method]]
    summary = summary.sort_values(["Dataset", "Method"])
    summary.to_csv("results/mapping_errors.csv", sep=',', index=False)

    plt.figure()
    ind = range(len(summary.index))
    width = 0.8

    ax = plt.gca()
    ax.bar(ind, summary["MSE"], width, yerr=summary["STD"])
    ax.set_xticks(ind)
    ax.set_xticklabels([summary.iloc[i, 0] + " " + summary.iloc[i, 1] for i in range(len(summary.index))], rotation=45)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MSE")
    plt.title("Prediction errors by method, real data")
    plt.tight_layout()
    plt.savefig("figures/mappings/errors.png")


def plot_mapping_fraction_votes(fraction_votes_files=fraction_votes_files):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    i = 0
    for dataset, url in fraction_votes_files.iteritems():
        fracvotes = pd.read_csv(url, index_col=0).as_matrix().flatten()
        plot_distribution_hist(ax=axes[i], data=fracvotes, title="%s dataset" % dataset,
                               yscale="log", xlabel="Fraction of votes", ylabel="Number of cases", bins=10,
                               normed=False)
        i += 1
    plt.suptitle("Gene importance distribution of method votes")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig("figures/mappings/vote_distribution.png")


def plot_convergence_statistics():
    plt.figure()
    ax = plt.gca()
    plot_distribution_hist(ax, converge_iterations, "Orchestrated tuning iterations required to converge",
                           "Iterations required", "Fraction of cases")
    plt.tight_layout()
    plt.savefig("figures/tuning/convergence_statistics.png")
