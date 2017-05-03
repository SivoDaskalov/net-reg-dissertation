import pandas as pd
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


import pickle


def export_errors(results):
    tmp = "dumps/tmp"
    with open(tmp, 'rb') as f:
        results = pickle.load(f)
    return None
