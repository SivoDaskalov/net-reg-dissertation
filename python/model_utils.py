from import_utils import tumor_data_files
import pandas as pd
import os.path

export_column_order = ["lasso", "enet", "grace", "gblasso", "linf", "ttlp", "composite"]


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
