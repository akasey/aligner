import pandas as pd
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from collections import defaultdict

def jpt():
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 2],
                           height_ratios=[4, 1]
                           )

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    plt.show()

def all_kmers(str, K):
    all = []
    for i in range(len(str)-K+1):
        all.append(str[i:i+K])
    return all

memo = {}
def get_activations_neuron_kmer(df, neuron, kmer):
    key = kmer + '-' + str(neuron)
    if key not in memo:
        memo[key] = df[df.Kmer==kmer][neuron].as_matrix()[0]
    return memo[key]


def main():
    plt.rcParams['axes.linewidth'] = 0.1

    data_dir = '/Users/akash/PycharmProjects/aligner/sample_classification_run'
    values_df = pd.read_csv(data_dir + '/Y_3.csv')
    columns = ['5', '10', '12', '15', '19', '24', '30', '50']

    def make_series(kmers, common_kmers, neuron, invert=False):
        x_series, x_common_series = [], []
        y_series, y_common_series = [], []
        for idx, kmer in enumerate(kmers):
            if kmer in common_kmers:
                x_common_series.append(idx)
                y_common_series.append(get_activations_neuron_kmer(values_df, neuron, kmer))
            else:
                x_series.append(idx)
                y_series.append(get_activations_neuron_kmer(values_df, neuron, kmer))
        if invert:
            y_series = [-1*y for y in y_series]
            y_common_series = [-1*y for y in y_common_series]
        return (x_series, y_series), (x_common_series, y_common_series)

    def plot_for(neuron, upright_kmers, upsidedown_kmers, common_kmers, fig, y_label_enable=False):
        (x_series, y_series), (x_common_series1, y_common_series1) = make_series(upright_kmers, common_kmers, neuron)
        a = fig.bar(x_series, y_series, width=0.8, align="center", color="#448FA3")
        (x_series, y_series), (x_common_series2, y_common_series2) = make_series(upsidedown_kmers, common_kmers, neuron, invert=False)
        x_series = [x+10 for x in x_series]
        x_common_series2 = [x+10 for x in x_common_series2]
        b = fig.bar(x_series, y_series, width=0.8, align="center", color="#65A344")
        c = fig.bar(x_common_series1+x_common_series2, y_common_series1+y_common_series2, width=0.9, align="center", color="#A5A5A5")
        fig.axhline(0, color="black", linewidth=0.5)
        if y_label_enable:
            plt.ylabel("# activations", fontsize=8)

        plt.xticks([])
        plt.yticks([])
        return a, b, c

    # TAGAAATTT--TTCTTG
    # TAGAAATTTAATTCT--

    # AA-AA-AAAAATTTTTT
    # AAC-AT-AAAATTTTTT

    kmers1 = all_kmers("TAGAAATTTTTCTTG", 7)
    kmers2 = all_kmers("AAAAAAAAATTTTTT", 7)
    common_kmers = set(kmers1).intersection(set(kmers2))

    fig = plt.figure(figsize=(5,3), edgecolor="gray")
    for idx, neuron in enumerate(columns):
        ax = plt.subplot(3, 3, idx+1)
        a, b, c = plot_for(neuron, kmers1, kmers2, common_kmers, ax, idx%3==0)

    fig.legend((a,b,c), ("Kmer-X", "Kmer-Y", "Common kmer"), "lower right")
    plt.savefig(data_dir+"/original_kmers.png")
    plt.show()

    kmers1 = all_kmers("TAGAAATTTAATTCT", 7)
    kmers2 = all_kmers("AACATAAAATTTTTT", 7)
    common_kmers = set(kmers1).intersection(set(kmers2))
    fig = plt.figure(figsize=(5, 3), edgecolor="gray")
    for idx, neuron in enumerate(columns):
        ax = plt.subplot(3, 3, idx+1)
        a, b, c = plot_for(neuron, kmers1, kmers2, common_kmers, ax, idx%3==0)

    fig.legend((a,b), ("Kmer-X'", "Kmer-Y'"), "lower right")
    plt.savefig(data_dir + "/mutated_kmers.png")
    plt.show()


if __name__ == "__main__":
    # jpt()
    main()