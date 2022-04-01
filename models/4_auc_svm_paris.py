from sklearn.metrics import roc_auc_score

from utils import get_data
# Check the AUCs of the SVM models
df = get_data(eeg_model=True)

targets = ['doc.enrollment', 'doc.discharge']

for target in targets:
    for model in ['pmcs.rest', 'pmc.stim']:
        t_df = df[[target, model]].dropna()
        print(f'Target {target} | Model {model}')
        y_true = t_df[target] == 'nonUWS'
        y_pred = t_df[model]
        print(roc_auc_score(y_true.values, y_pred.values))
