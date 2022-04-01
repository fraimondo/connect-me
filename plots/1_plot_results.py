import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import models_labels, features_labels, metrics_labels

ys = {
    'doc.enrollment': 'Enrollment',
    'doc.discharge': 'Discharge'
}
models = ['gssvm', 'rf']

sns.set_context('paper', rc={'font.size': 10, 'axes.labelsize': 10,
                             'lines.linewidth': 1,
                             'xtick.labelsize': 8, 'ytick.labelsize': 10})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
sns.set_color_codes()
current_palette = sns.color_palette()

# Configuration
kind = ['unimodal', 'merged', 'multimodal']
results_folder = 'kfold'

kind_titles = {
    'unimodal': 'Unimodal models on all samples',
    'merged': 'Unimodal models on same samples',
    'multimodal': 'Multimodal models on same samples'
}

kind_prefix = {
    'unimodal': '1_unimodal',
    'merged': '2_unimodalmerged',
    'multimodal': '3_multimodal',
}

data_path = Path(__file__).parent.parent \
    / 'models' / 'results' / results_folder
out_path = Path(__file__).parent.parent / 'plots' / 'figs' / results_folder
out_path.mkdir(exist_ok=True, parents=True)


hue_order = ['Enrollment', 'Discharge']

for t_kind in kind:
    all_df = []
    for t_model in models:
        for t_y, t_y_label in ys.items():
            g_pattern = f'{kind_prefix[t_kind]}_{t_model}_{t_y}*.csv'
            fnames = data_path.glob(g_pattern)
            for fname in fnames:
                t_df = pd.read_csv(
                    data_path / fname, sep=';')
                t_df['target'] = t_y_label
                all_df.append(t_df)

    all_df = pd.concat(all_df)
    if all_df['repeat'].unique().shape[0] > 1:
        all_df = all_df.groupby(
            ['model', 'target', 'features', 'repeat']).mean().reset_index()
    order = []
    t_features = all_df['features'].unique()
    for k, v in features_labels.items():
        if k in t_features:
            order.append(v)
    all_df.replace(dict(features=features_labels), inplace=True)

    metrics = ['test_roc_auc', 'test_precision', 'test_recall']
    for t_metric in metrics:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(models), sharey=False, sharex=True,
            figsize=(12, 6))
        for i_col, t_model in enumerate(models):
            t_df = all_df[all_df['model'] == t_model]
            t_ax = axes[i_col]

            sns.swarmplot(
                x='features', y=t_metric, order=order, hue_order=hue_order,
                data=t_df, hue='target', dodge=True,
                ax=t_ax, alpha=.5, size=2)

            sns.boxplot(
                x='features', y=t_metric, order=order, hue_order=hue_order,
                data=t_df, hue='target', dodge=True,
                ax=t_ax,
                whis=[2.5, 97.5],
                palette=['w', 'w'],
                color='w', zorder=1,
                showfliers=False,
            )
            [t_ax.axhline(x, color='k', linestyle='--', alpha=0.3)
             for x in np.arange(0, 1.01, 0.1)]
            t_ax.set_ylabel('')
            t_ax.set_title(models_labels[t_model])
            t_ax.set_ylim(0, 1)
            t_ax.set_yticks(np.arange(0, 1.01, 0.1))
            if i_col == 0:
                t_ax.set_ylabel(metrics_labels[t_metric])
            t_ax.set_xlabel('')
            handles, labels = t_ax.get_legend_handles_labels()
            t_ax.legend(handles[2:], labels[2:])

        plt.suptitle(kind_titles[t_kind])
        fig.tight_layout()

        fname = f'results_{t_metric}_{t_kind}.pdf'
        fig.savefig(out_path / fname)
        plt.close(fig)
