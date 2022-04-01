import pandas as pd
from pathlib import Path
import julearn
from sklearn.model_selection import RepeatedStratifiedKFold, \
    StratifiedShuffleSplit, StratifiedKFold

TARGETS = ['doc.enrollment', 'doc.discharge']
FMRI_FEATURES = [
    'dmn.dmn', 'fpn.fpn', 'sn.sn', 'an.an', 'smn.smn', 'vn.vn', 'dmn.fpn',
    'dmn.sn', 'dmn.an', 'dmn.smn', 'dmn.vn', 'fpn.sn', 'fpn.an', 'fpn.smn',
    'fpn.vn', 'sn.an', 'sn.smn', 'sn.vn', 'an.smn', 'an.vn', 'smn.vn'
]

EEG_VISUAL_FEATURES = ['eeg.visual.synek']
EEG_ABCD = ['abcd']
EEG_MODEL = ['pmcs.rest', 'pmc.stim']
AGESEX = ['age', 'sex']

_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}


def get_data(fmri=False, eeg_visual=False, eeg_abcd=False, eeg_model=False,
             eeg_features=False, agesex=False,
             eeg_electrodes='all', eeg_kind='all'):
    t_path = Path(__file__).parent.parent / 'data' / 'dataset.csv'
    df = pd.read_csv(t_path, sep=';', decimal=',')
    to_keep = TARGETS.copy() + ['Id']
    if fmri is True:
        to_keep += FMRI_FEATURES
    if eeg_visual is True:
        to_keep += EEG_VISUAL_FEATURES
    if eeg_abcd is True:
        to_keep += EEG_ABCD
    if eeg_model is True:
        to_keep += EEG_MODEL
    if agesex is True:
        to_keep += AGESEX

    t_df = df[to_keep]
    # Drop NANS
    t_df = t_df.dropna()

    if fmri is True:
        t_df[FMRI_FEATURES] = t_df[FMRI_FEATURES].astype(float)

    if eeg_abcd:
        t_df['abcd'] = t_df['abcd'].apply(lambda x: _map[x])

    if eeg_model is True:
        t_df[EEG_MODEL] = t_df[EEG_MODEL].astype(float)

    if agesex is True:
        t_df['sex'] = (t_df['sex'] == 'Female').astype(int)

    t_df = t_df.rename(columns={'Id': 'Subject'}).set_index('Subject')
    if eeg_features is True:
        df_eeg = get_eeg_features(electrodes=eeg_electrodes, kind=eeg_kind)
        eeg_markers = [x for x in df_eeg.columns if 'nice/marker' in x]
        t_df = t_df.join(df_eeg[eeg_markers + ['Electrodes']]).dropna()
    return t_df


def _red_to_str(reduction):
    out = ''
    if 'gfp/trim_mean80' in reduction:
        out = 'std_trim_mean80'
    elif 'gfp/std' in reduction:
        out = 'std_std'
    elif '/trim_mean80' in reduction:
        out = 'mean_trim_mean80'
    elif 'std' in reduction:
        out = 'mean_std'
    else:
        raise ValueError('WTF')
    return out


def _reduction_to_columns(df, t_kind):
    markers = [x for x in df.columns if x.startswith('nice')]
    df['Electrodes'] = df['Reduction'].apply(lambda x: 19 if '19' in x else 25)
    df['Reduction'] = df['Reduction'].apply(_red_to_str)
    df = df.set_index(
        ['Subject', 'Electrodes', 'Reduction'])[markers].unstack('Reduction')
    df.columns = [f'{t_kind}_{"_".join(x)}' for x in df.columns]
    return df.reset_index()


def get_eeg_features(electrodes='all', kind='all'):
    t_path = Path(__file__).parent.parent / 'data' / 'eeg'

    all_data = []
    if kind == 'all':
        kind = ['stim', 'resting']
    else:
        kind = [kind]
    for t_kind in kind:
        tk_path = t_path / t_kind
        dfs = []
        if electrodes in ['19', 'all', 19]:
            dfs.append(
                pd.read_csv(
                    tk_path / 'copenhagen_19' / 'all_scalars.csv', sep=';'))
        if electrodes in ['25', 'all', 25]:
            dfs.append(
                pd.read_csv(
                    tk_path / 'copenhagen_25' / 'all_scalars.csv', sep=';'))

        df = pd.concat([_reduction_to_columns(x, t_kind) for x in dfs])
        all_data.append(df)

    if len(all_data) == 2:
        df0 = all_data[0].set_index(['Subject', 'Electrodes'])
        df1 = all_data[1].set_index(['Subject', 'Electrodes'])
        df = df0.join(df1).reset_index().set_index(['Subject'])
    else:
        df = all_data[0].set_index(['Subject'])

    meta_df = get_data()
    df = meta_df.join(df).dropna()
    return df


def test_input(
        df, X, y, title, model, cv, confounds=None,
        deconfound_vars=None):
    if cv == 'kfold':
        cv = RepeatedStratifiedKFold(
            n_splits=5, n_repeats=100, random_state=42)
    elif cv == 'mc':
        cv = StratifiedShuffleSplit(
            n_splits=100, test_size=0.3, random_state=42)
    else:
        raise ValueError('Unknown CV scheme ["kfold" or "mc"]')

    extra_params = {}
    if model == 'svm':
        extra_params['model'] = 'svm'
        extra_params['preprocess_X'] = 'zscore'
        extra_params['model_params'] = {'svm__kernel': 'linear'}
    elif model == 'rf':
        extra_params['model'] = 'rf'
        extra_params['model_params'] = {
            'rf__n_estimators': 500,
        }
    elif model == 'gssvm':
        extra_params['model'] = 'svm'
        extra_params['preprocess_X'] = 'zscore'
        extra_params['model_params'] = {
            'svm__kernel': 'linear',
            'svm__C': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
            'cv': StratifiedKFold(n_splits=5, random_state=77, shuffle=True)
        }
    if confounds is not None and deconfound_vars is not None:
        extra_params['model_params'][  # type: ignore
            'remove_confound__apply_to'] = [
                deconfound_vars +
                [f'{x}__:type:__confound' for x in confounds]]
    cv_results = julearn.api.run_cross_validation(
        X=X, y=y, data=df, problem_type='binary_classification',
        pos_labels='nonUWS', scoring=['roc_auc', 'precision', 'recall'],
        cv=cv, confounds=confounds, **extra_params
    )
    print('=============================')
    print(title)
    print(cv_results.mean())  # type: ignore
    print('=============================')
    return cv_results
