import pandas as pd


def assemble_mappings(dataset, fits, methods):
    coef = {}
    for method in methods:
        coef[method] = pd.DataFrame(data=None, columns=dataset.expression.columns, index=dataset.expression.columns)

    for fit in fits:
        gene = fit[0].label.split("_")[0]
        for method, model in fit[1].iteritems():
            coef[method][gene] = model.coef_

    for method, coef_matrix in coef.iteritems():
        coef_matrix.to_csv("mappings/%s.csv" % method, sep=',')
    return coef
