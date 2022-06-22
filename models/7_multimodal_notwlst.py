from pathlib import Path
import julearn
from julearn.utils import logger
import pandas as pd
from utils import (get_data, FMRI_FEATURES, EEG_ABCD, EEG_MODEL,
                   EEG_VISUAL_FEATURES, test_input)
from argparse import ArgumentParser
julearn.utils.configure_logging(level="INFO")

parser = ArgumentParser(description='Compute unimodal models')


parser.add_argument('--features', metavar='features', nargs='+', type=int,
                    help='Features set to compute [1-7].',
                    required=True)

parser.add_argument('--model', metavar='model', nargs=1, type=str,
                    help='model to use "gssvm" or "rf".',
                    required=True)

parser.add_argument('--target', metavar='target', nargs=1, type=str,
                    help='target to use "doc.enrollment" or "doc.discharge".',
                    required=True)

parser.add_argument('--cv', metavar='cv', nargs=1, type=str,
                    help='CV to use "mc" or "kfold".',
                    required=True)

args = parser.parse_args()
feature_set = args.features
cv = args.cv[0]
model = args.model[0]
y = args.target[0]

logger.info(f'Features set: {feature_set}')
logger.info(f'Model: {model}')
logger.info(f'Target: {y}')
logger.info(f'CV: {cv}')
df_all = get_data(fmri=True, eeg_visual=True, eeg_abcd=True,
                  eeg_model=True, eeg_features=True, notwlst=True)

out_path = Path(__file__).parent / 'results' / cv
out_path.mkdir(parents=True, exist_ok=True)

all_dfs = []
# Define EEG features sets
EEG_RESTING_FEATURES = [
    x for x in df_all.columns if x.startswith('resting_nice')]
EEG_STIM_FEATURES = [
    x for x in df_all.columns if x.startswith('stim_nice')]

if 1 in feature_set:
    X = FMRI_FEATURES + EEG_RESTING_FEATURES
    title = 'FMRI + EEG resting features (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,
        confounds=confounds,
        deconfound_vars=EEG_RESTING_FEATURES)
    result_df['features'] = 'fmri_eegresting'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 2 in feature_set:
    X = FMRI_FEATURES + EEG_STIM_FEATURES
    title = 'FMRI + EEG stim features (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,
        confounds=confounds,
        deconfound_vars=EEG_STIM_FEATURES)
    result_df['features'] = 'fmri_eegstim'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 3 in feature_set:
    X = FMRI_FEATURES + EEG_STIM_FEATURES + EEG_RESTING_FEATURES
    title = 'FMRI + EEG all features (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,
        confounds=confounds,
        deconfound_vars=EEG_STIM_FEATURES + EEG_RESTING_FEATURES)
    result_df['features'] = 'fmri_eegall'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 4 in feature_set:
    X = FMRI_FEATURES + EEG_ABCD + EEG_MODEL + EEG_VISUAL_FEATURES
    title = 'FMRI + EEG all but features (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv)
    result_df['features'] = 'fmri_eegmeta'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 5 in feature_set:
    X = EEG_ABCD + EEG_MODEL + EEG_VISUAL_FEATURES
    title = 'EEG all but features (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,)
    result_df['features'] = 'eegmeta'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 6 in feature_set:
    X = EEG_ABCD + EEG_MODEL + EEG_VISUAL_FEATURES + \
        EEG_STIM_FEATURES + EEG_RESTING_FEATURES
    title = 'EEG all (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,
        confounds=confounds,
        deconfound_vars=EEG_STIM_FEATURES + EEG_RESTING_FEATURES)
    result_df['features'] = 'eegall'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

if 7 in feature_set:
    X = EEG_ABCD + EEG_MODEL + EEG_VISUAL_FEATURES + \
        EEG_STIM_FEATURES + \
        EEG_RESTING_FEATURES + FMRI_FEATURES
    title = 'fMRI + EEG all (merged)'
    confounds = ['Electrodes']
    result_df = test_input(
        df_all, X, y, title=title, model=model, cv=cv,
        confounds=confounds,
        deconfound_vars=EEG_STIM_FEATURES + EEG_RESTING_FEATURES)
    result_df['features'] = 'all'  # type: ignore
    result_df['model'] = model  # type: ignore
    all_dfs.append(result_df)  # type: ignore

all_dfs = pd.concat(all_dfs)
if len(feature_set) > 1:
    suffix = '_'.join(feature_set)
else:
    suffix = f'{feature_set[0]}'
all_dfs.to_csv(
    out_path / f'7_multimodalnotwlst_{model}_{y}_{suffix}.csv', sep=';')
