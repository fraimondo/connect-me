import pandas as pd
import numpy as np
import itertools
import string
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import features_labels, metrics_labels, compute_ci, models_labels


sns.set_context('paper', rc={'font.size': 10, 'axes.labelsize': 10,
                             'lines.linewidth': 1,
                             'xtick.labelsize': 7, 'ytick.labelsize': 10})
sns.set_style('white',
              {'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.edgecolor': '.8'})

mpl.rcParams.update({'font.weight': 'ultralight'})
sns.set_color_codes()
current_palette = sns.color_palette()

# Configuration
results_folder = 'kfold'

ys = {
    'doc.enrollment': 'Enrollment',
    'doc.discharge': 'Discharge'
}
models = ['gssvm', 'rf']


data_path = Path(__file__).parent.parent \
    / 'models' / 'results' / results_folder
out_path = Path(__file__).parent.parent / 'plots' / 'figs' / results_folder
out_path.mkdir(exist_ok=True, parents=True)

# Output of this script
# Figure 1: Unimodal, all samples, ROC AUC, enrollment and discharge,
# RF and SVM
# Table 1: Features | AUC | Precision | Recall (enrolmment and discharge)


hue_order = ['Enrollment', 'Discharge']

kind_prefixes = ['6_unimodalmergednotwlst', '7_multimodalnotwlst']
# kind_prefixes = ['6_unimodalmergednotwlst']
t_metric = 'test_roc_auc'
title = 'Unimodal and Multimodal {model_name}s (same samples)'
out_fname = '4_comparison_roc_auc_{t_model}.pdf'

models_dfs = []
# Merge dataframes
for kind_prefix in kind_prefixes:
    for t_model in models:
        for t_y, t_y_label in ys.items():
            g_pattern = f'{kind_prefix}_{t_model}_{t_y}*.csv'
            fnames = data_path.glob(g_pattern)
            for fname in fnames:
                t_df = pd.read_csv(
                    data_path / fname, sep=';')
                t_df['target'] = t_y_label
                models_dfs.append(t_df)


all_df = pd.concat(models_dfs)
order = []
t_features = all_df['features'].unique()
for k, v in features_labels.items():
    if k in t_features:
        order.append(v)
all_df.replace(dict(features=features_labels), inplace=True)
all_df.rename(columns=metrics_labels, inplace=True)  # type: ignore
all_df.rename(
    columns={'model': 'Model', 'target': 'Target',
             'features': 'Features'}, inplace=True)  # type: ignore
metrics = ['ROC AUC', 'Precision', 'Recall']

mean_df = all_df.groupby(
    ['Model', 'Target', 'Features', 'repeat'])[metrics].mean().reset_index()
features_char_map = {
    k: string.ascii_uppercase[i] for i, k in enumerate(order)
}


new_names_mean = {k: f'{k}\n({features_char_map[k]})' for k in order}
mean_df.replace(dict(Features=new_names_mean), inplace=True)

for t_model in models:
    t_mean_df = mean_df[mean_df['Model'] == t_model]

    axes_titles = ['Unimodal', 'Multimodal']
    features_splits = [
        list(new_names_mean.values())[:7],
        list(new_names_mean.values())[7:]]

    t_fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=False, sharex=False, figsize=(10, 4))
    t_metric = 'ROC AUC'
    for t_ax, t_ax_title, t_features_split in zip(
            axes, axes_titles, features_splits):  # type: ignore

        t_split_df = t_mean_df[t_mean_df['Features'].isin(t_features_split)]
        sns.swarmplot(
            x='Features', y=t_metric, order=t_features_split,
            hue_order=hue_order,
            data=t_split_df, hue='Target', dodge=True,
            ax=t_ax, alpha=.5, size=2)

        sns.boxplot(
            x='Features', y=t_metric, order=t_features_split,
            hue_order=hue_order,
            data=t_split_df, hue='Target', dodge=True,
            ax=t_ax,
            whis=(2.5, 97.5),  # type: ignore
            palette=['w', 'w'],
            color='w', zorder=1,
            showfliers=False,
        )
        [t_ax.axhline(x, color='k', linestyle='--', alpha=0.3)
            for x in np.arange(0, 1.01, 0.1)]
        t_ax.set_title(t_ax_title)
        t_ax.set_ylim(0.4, 1)
        t_ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        t_ax.set_ylabel(t_metric)
        t_ax.set_xlabel('')
        handles, labels = t_ax.get_legend_handles_labels()
        t_ax.legend(handles[2:], labels[2:])

    model_name = models_labels[t_model]
    t_fig.suptitle(eval(f"f'{title}'"))
    t_fig.tight_layout()
    t_out_fname = eval(f"f'{out_fname}'")
    t_fig.savefig(out_path / t_out_fname)
    t_fig.savefig(out_path / t_out_fname.replace('.pdf', '.png'), dpi=300)
    plt.close(t_fig)


title = 'Pairwise model comparison ({model_name}) for {t_target}'
out_fname = '3_pairwise_roc_auc_{t_model}_{t_target}.pdf'

for t_model in models:
    for t_target in hue_order:

        t_df = all_df[
            (all_df['Model'] == t_model) & (all_df['Target'] == t_target)]

        t_order = order
        t_order_r = list(reversed(t_order))
        n_features = len(t_order)

        t_fig, axes = plt.subplots(
            nrows=n_features-1, ncols=n_features-1, sharex=True,
            figsize=(10, 5))

        for row, col in itertools.product(range(n_features-1), repeat=2):
            t_ax = axes[row, col]  # type: ignore
            f1 = t_order_r[row]
            f2 = t_order[col]
            if (row + col) >= (n_features - 1):
                t_ax.axis('off')
            else:
                print(f'{f1} ({features_char_map[f1]}) vs '
                      f'{f2} ({features_char_map[f2]})')
                t_df1 = t_df[
                    t_df['Features'] == f1].set_index(['repeat', 'fold'])
                t_df2 = t_df[
                    t_df['Features'] == f2].set_index(['repeat', 'fold'])
                t_df1 = t_df1[metrics]
                t_df2 = t_df2[metrics]
                diff = (t_df1 - t_df2).reset_index().groupby('repeat').mean()
                # sns.swarmplot(
                #     x=t_metric, data=diff, color='k', ax=t_ax, alpha=.5,
                #     size=0.5)  # type: ignore
                sns.boxplot(
                    x=t_metric, data=diff, ax=t_ax,
                    whis=(2.5, 97.5),  # type: ignore
                    color='w', zorder=1, showfliers=False,)
                t_ax.axvline(0, color='k', linestyle='--')
                t_ax.set_xlabel('')
                t_ax.set_ylabel('')

                t_ax.set_xlim([-0.5, 0.5])
                if row == 0:
                    t_ax.set_xticklabels(['-0.5', '0', '0.5'])
                t_ax.set_yticks([])
            if row == 0:
                t_ax.set_title(features_char_map[f2])
            if col == 0:
                t_ax.set_ylabel(
                    f'{features_char_map[f1]} - ', rotation=0, va='center',
                    ha='right')

        model_name = models_labels[t_model]
        t_fig.suptitle(eval(f"f'{title}'"))
        t_fig.subplots_adjust(
            left=0.038, right=0.99, top=0.88, bottom=0.052,
            wspace=0.03, hspace=0.17)
        # t_fig.tight_layout()
        t_out_fname = eval(f"f'{out_fname}'")
        t_fig.savefig(out_path / t_out_fname)
        t_fig.savefig(out_path / t_out_fname.replace('.pdf', '.png'), dpi=300)
        plt.close(t_fig)


# Now build the table as in with all samples
table_data = all_df.groupby(
    ['Model', 'Target', 'Features', 'repeat'])[metrics].mean().reset_index()
new_labels = {x: x.replace('\n', ' ') for x in order}
new_order = [x.replace('\n', ' ') for x in order]
table_data.replace(
    dict(Features=new_labels, Model=models_labels), inplace=True)
metrics = ['ROC AUC', 'Precision', 'Recall']

# Compute CI
table_data = table_data.groupby(
    ['Model', 'Target', 'Features'])[metrics].agg(compute_ci)

# Reshape
table_data = table_data.unstack('Target')
# Reorder columns
table_data = table_data.swaplevel(axis=1).reindex(hue_order, axis=1, level=0)
# Reorder rows
table_data = table_data.reindex(new_order, level='Features')

# Format the values
table_data = table_data.applymap(
    lambda x: f'{x[0]:.2f} [{x[1]:.2f} {x[2]:.2f}]')
table_data.columns.names = ['Target', 'Metric']
print(table_data)
table_data.to_csv(out_path / '4_nowlst.csv', sep=';')
