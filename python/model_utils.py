from __future__ import division
from import_utils import tumor_data_files
from sklearn.metrics.pairwise import cosine_similarity
from plotting_utils import plot_similarities_heatmap
from commons import full_method_list as export_column_order
import pandas as pd
import numpy as np
import os.path


def assemble_mappings(dataset, fits, methods):
    coef = {}
    votes = pd.DataFrame(data=None, columns=dataset.expression.columns, index=dataset.expression.columns)
    for method in methods:
        coef[method] = pd.DataFrame(data=None, columns=dataset.expression.columns, index=dataset.expression.columns)

    for fit in fits:
        gene = fit[0].label.split("_")[0]
        for method, model in fit[1].iteritems():
            coef[method][gene] = model.coef_
            if method == "composite":
                votes[gene] = model.fraction_votes_

    mappings_dir = "mappings/%s" % dataset.label
    if not os.path.exists(mappings_dir):
        os.makedirs(mappings_dir)
    for method, coef_matrix in coef.iteritems():
        coef_matrix.transpose().to_csv(os.path.join(mappings_dir, "%s.csv" % method), sep=',')
    votes.transpose().to_csv(os.path.join(mappings_dir, "composite_fraction_votes.csv"), sep=',')
    return coef


def export_errors(results):
    labels = tumor_data_files.keys()
    errors = {}

    for label in labels:
        genes = [gene.split('_', 1)[0] for gene in results["setup"].unique() if label in gene]
        methods = [method for method in export_column_order if method in results["model"].unique()]
        errors[label] = pd.DataFrame(data=None, columns=methods, index=genes)

    for index, row in results.iterrows():
        gene, label = row["setup"].split("_")
        method = row["model"]
        errors[label][method][gene] = row["mse"]

    for label, frame in errors.iteritems():
        mappings_dir = "mappings/%s" % label
        if not os.path.exists(mappings_dir):
            os.makedirs(mappings_dir)
        frame.to_csv(os.path.join(mappings_dir, "errors.csv"), sep=',')


def applier(a, b, op):
    return map(lambda ro: map(op, ro[0], ro[1]), zip(a, b))


def batch_evaluate_similarities(models, summary_url="results/similarities.csv", figure_url="figures/similarities.png",
                                title='Method similarities'):
    methods = [method for method in export_column_order if method in models[0][1].keys()]
    similarities = np.zeros((len(methods), len(methods), len(models)))
    for i in range(len(models)):
        cur_models = models[i][1]
        for row_idx in range(len(methods)):
            for col_idx in range(len(methods)):
                m1 = cur_models[methods[row_idx]].coef_
                m2 = cur_models[methods[col_idx]].coef_
                similarity = cosine_similarity(m1.reshape(1, -1), m2.reshape(1, -1))
                similarities[row_idx][col_idx][i] = round(similarity, 4)

    sim_mean = np.mean(similarities, 2)
    sim_std = np.std(similarities, 2)
    sim_merged = applier(sim_mean, sim_std, lambda mean, std: "%.3f (%.3f)" % (mean, std))

    summary_similarities = pd.DataFrame(data=sim_merged, index=methods, columns=methods)
    summary_similarities.to_csv(summary_url, sep=',')
    plot_similarities_heatmap(sim_mean, methods, url=figure_url, title=title)
    return summary_similarities
