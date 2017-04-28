import pandas as pd


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

    for method, coef_matrix in coef.iteritems():
        coef_matrix.to_csv("mappings/%s_%s.csv" % (dataset.label, method), sep=',')
    votes.to_csv("mappings/%s_composite_fraction_votes.csv" % dataset.label, sep=',')
    return coef
