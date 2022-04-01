import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from utils import features_labels, metrics_labels, compute_ci, models_labels


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

kind_prefix = '1_unimodal'
models = ['rf', 'gssvm']
t_metric = 'test_roc_auc'
title = 'Unimodal models (maximum available samples)'
out_fname = '1_unimodal_all_samples_roc_auc.pdf'
t_ax_titles = ['Random Forest', 'Support Vector Machine']

t_fig, axes = plt.subplots(
    nrows=1, ncols=2, sharey=False, sharex=True, figsize=(10, 4))

models_dfs = []
# Merge dataframes
for t_model, t_ax, t_ax_title in zip(
        models, axes, t_ax_titles):  # type: ignore
    t_all_df = []
    for t_y, t_y_label in ys.items():
        g_pattern = f'{kind_prefix}_{t_model}_{t_y}*.csv'
        fnames = data_path.glob(g_pattern)
        for fname in fnames:
            t_df = pd.read_csv(
                data_path / fname, sep=';')
            t_df['target'] = t_y_label
            t_all_df.append(t_df)

    all_df = pd.concat(t_all_df)
    models_dfs.append(all_df)
    if all_df['repeat'].unique().shape[0] > 1:
        all_df = all_df.groupby(
            ['model', 'target', 'features', 'repeat']).mean().reset_index()
    order = []
    t_features = all_df['features'].unique()
    for k, v in features_labels.items():
        if k in t_features:
            order.append(v)
    all_df.replace(dict(features=features_labels), inplace=True)
    t_df = all_df[all_df['model'] == t_model]

    sns.swarmplot(
        x='features', y=t_metric, order=order, hue_order=hue_order,
        data=t_df, hue='target', dodge=True,
        ax=t_ax, alpha=.5, size=2)

    sns.boxplot(
        x='features', y=t_metric, order=order, hue_order=hue_order,
        data=t_df, hue='target', dodge=True,
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
    t_ax.set_ylabel(metrics_labels[t_metric])
    t_ax.set_xlabel('')
    handles, labels = t_ax.get_legend_handles_labels()
    t_ax.legend(handles[2:], labels[2:])
t_fig.suptitle(title)
t_fig.tight_layout()
t_fig.savefig(out_path / out_fname)
t_fig.savefig(out_path / out_fname.replace('.pdf', '.png'), dpi=300)

plt.close(t_fig)

# Now build the table
all_data = pd.concat(models_dfs)
new_labels = {k: v.replace('\n', ' ') for k, v in features_labels.items()}
new_order = [x.replace('\n', ' ') for x in order]
all_data.replace(
    dict(features=new_labels, model=models_labels), inplace=True)
all_data.rename(columns=metrics_labels, inplace=True)  # type: ignore
all_data.rename(
    columns={'model': 'Model', 'target': 'Target',
             'features': 'Features'}, inplace=True)  # type: ignore
metrics = ['ROC AUC', 'Precision', 'Recall']

# Compute mean across folds (one value per repeat)
all_data = all_data.groupby(
    ['Model', 'Target', 'Features', 'repeat'])[metrics].mean().reset_index()

# Compute CI
all_data = all_data.groupby(
    ['Model', 'Target', 'Features'])[metrics].agg(compute_ci)

# Reshape
all_data = all_data.unstack('Target')
# Reorder columns
all_data = all_data.swaplevel(axis=1).reindex(hue_order, axis=1, level=0)
# Reorder rows
all_data = all_data.reindex(new_order, level='Features')

# Format the values
all_data = all_data.applymap(lambda x: f'{x[0]:.2f} [{x[1]:.2f} {x[2]:.2f}]')
all_data.columns.names = ['Target', 'Metric']
print(all_data)

all_data.to_csv(out_path / '1_unimodal_all_samples.csv', sep=';')

