from pathlib import Path
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from prettytable import PrettyTable

"""Plot ROC curve for all methods"""


def parse_methods_scores(files_):
    methods_list = [Path(file).stem.split('-score')[0] for file in files_]
    scores_list = [np.load(file) for file in files_]
    methods_ = np.array(methods_list)
    scores_ = dict(zip(methods_, scores_list))
    return methods_, scores_


def sample_colours_from_colourmap(n_colours, colour_map):
    cm = plt.get_cmap(colour_map)
    return [cm(1. * i / n_colours)[:3] for i in range(n_colours)]


def calculate_result(label_, scores_):
    fpr_, tpr_, _ = metrics.roc_curve(label_, scores_)
    roc_auc_ = metrics.auc(fpr_, tpr_)
    fpr_ = np.flipud(fpr_)
    tpr_ = np.flipud(tpr_)  # select largest tpr at same fpr
    return fpr_, tpr_, roc_auc_


def add_table_row(tpr_fpr_table_, method_, x_labels_, fpr_, tpr_):
    tpr_fpr_row = [method_]
    for fpr_iter in np.arange(len(x_labels_)):
        _, min_index = min(list(zip(abs(fpr_ - x_labels_[fpr_iter]), range(len(fpr_)))))
        tpr_fpr_row.append('%.4f' % tpr_[min_index])

    tpr_fpr_table_.add_row(tpr_fpr_row)


def plot_curve(x_labels_):
    plt.xlim([10 ** -6, 0.1])
    plt.ylim([0.01, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels_)
    plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on IJB-C')
    plt.legend(loc="lower right")
    # plt.tight_layout()
    plt.show()


def main(files_):

    # Load necessary data
    label = np.load('meta/ijbc_labels.npy')
    methods, scores = parse_methods_scores(files_)

    # Plot and table settings
    colours = dict(zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
    # x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
    x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    tpr_fpr_table = PrettyTable(['Methods'] + list(map(str, x_labels)))
    fig = plt.figure()

    # Calculate results and fit into plot and table
    for method in methods:

        print(f"Analyzing {method} ... ")
        fpr, tpr, roc_auc = calculate_result(label, scores[method])
        plt.plot(fpr, tpr, color=colours[method], lw=1,
                 label=('[%s (AUC = %0.4f %%)]' % (method, roc_auc * 100)))
        add_table_row(tpr_fpr_table, method, x_labels, fpr, tpr)

    print("Done analyzing. Rendering plot and tpr-fpr table ... ")

    # Plot and save curve
    plot_curve(x_labels)
    fig.savefig('ROC-on-ijbc.png')

    # Show table
    print(tpr_fpr_table)


if __name__ == '__main__':
    scores = [n for n in Path('scores').iterdir()]
    main(scores)
