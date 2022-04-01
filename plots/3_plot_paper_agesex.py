import pandas as pd
import matplotlib as mpl
import seaborn as sns

from pathlib import Path
from utils import (agesex_features_labels, metrics_labels, compute_ci,
                   models_labels)


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

models_dfs = []
# Merge dataframes
for t_model in models:
    for t_y, t_y_label in ys.items():
        g_pattern = f'5_agesex_{t_model}_{t_y}*.csv'
        fnames = data_path.glob(g_pattern)
        for fname in fnames:
            t_df = pd.read_csv(
                data_path / fname, sep=';')
            t_df['target'] = t_y_label
            models_dfs.append(t_df)


all_df = pd.concat(models_dfs)
order = []
t_features = all_df['features'].unique()
for k, v in agesex_features_labels.items():
    if k in t_features and v not in order:
        order.append(v)

all_df['Age Sex'] = ['Shuffled' if x.endswith('shuffled') else 'Real'
                     for x in all_df['features']]

all_df.replace(dict(features=agesex_features_labels), inplace=True)
all_df.rename(columns=metrics_labels, inplace=True)  # type: ignore
all_df.rename(
    columns={'model': 'Model', 'target': 'Target',
             'features': 'Features'}, inplace=True)  # type: ignore
metrics = ['ROC AUC', 'Precision', 'Recall']
hue_order = ['Enrollment', 'Discharge']

# Now build the table as in with all samples
table_data = all_df.groupby(
    ['Model', 'Target', 'Features', 'Age Sex', 'repeat'])[
        metrics].mean().reset_index()


new_labels = {x: x.replace('\n', ' ') for x in order}
new_order = [x.replace('\n', ' ') for x in order]
table_data.replace(
    dict(Features=new_labels, Model=models_labels), inplace=True)
metrics = ['ROC AUC', 'Precision', 'Recall']

# Compute CI
table_data = table_data.groupby(
    ['Model', 'Target', 'Features', 'Age Sex'])[metrics].agg(compute_ci)

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
table_data = table_data.unstack('Age Sex')
print(table_data)
table_data.to_csv(out_path / '3_agesex.csv', sep=';')
