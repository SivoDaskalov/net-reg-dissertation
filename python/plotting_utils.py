from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from commons import full_method_list as method_order


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


def plot_similarities_heatmap(similarities, methods, title='Method similarities', cmap=plt.cm.Blues):
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
    plt.savefig("figures/similarities.png")


def plot_summary_comparison(summary_urls):
    summaries = {}
    for label, url in summary_urls.iteritems():
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
        subplot.set_xticks(ind + width / 2)
        subplot.set_xticklabels(methods)

        j = 0
        for label, summary in summaries.iteritems():
            subplot.bar(ind + j * width, summary["%s mean" % subplots_info[i][1]], width,
                        yerr=summary["%s std" % subplots_info[i][1]], label=label)
            j += 1
        subplot.set_ylim((0.0, subplot.get_ylim()[1]))
        if i != 0:
            subplot.set_ylim((0.0, 1.0))

    h, l = subplot.get_legend_handles_labels()
    plt.suptitle("Properties by tuning method")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.figlegend(h, l, bbox_to_anchor=[0.5, 0.02], loc='center', ncol=len(summaries.keys()))
    plt.savefig("figures/tuning_method_comparison.png")
