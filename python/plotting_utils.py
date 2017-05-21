import matplotlib.pyplot as plt
import numpy as np
import itertools


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
    plt.title(title)
    plt.imshow(similarities, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(methods))
    plt.xticks(tick_marks, methods, rotation=45)
    plt.gca().xaxis.tick_top()
    plt.yticks(tick_marks, methods)

    thresh = (similarities.max() + similarities.min()) / 2.
    for i, j in itertools.product(range(similarities.shape[0]), range(similarities.shape[1])):
        plt.text(j, i, "%.3f" % similarities[i, j],
                 horizontalalignment="center", color="white" if similarities[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.subplots_adjust(top=0.93)
    plt.savefig("figures/similarities.png")
