import julearn
from julearn.utils import logger
import pandas as pd
from utils import (get_data, FMRI_FEATURES, EEG_ABCD, EEG_MODEL,
                   EEG_VISUAL_FEATURES, test_input)
from argparse import ArgumentParser
julearn.utils.configure_logging(level="INFO")

# Configuration
ys = ['doc.enrollment', 'doc.discharge']
# y = ['doc.discharge']
models = ['gssvm', 'rf']

parser = ArgumentParser(description='Compute unimodal models')

parser.add_argument('--features', metavar='features', nargs='+', type=int,
                    help='Features set to compute [1-7].',
                    required=True)

feature_set = parser.parse_args().features

logger.info(f'Features set: {feature_set}')

# Now do the same thing, but only with the subjects that have all data
df_all = get_data(fmri=True, eeg_visual=True, eeg_abcd=True,
                  eeg_model=True, eeg_features=True)

for y in ys:
    for model in models:
        all_dfs = []
        # Define EEG features sets
        EEG_RESTING_FEATURES = [
            x for x in df_all.columns if x.startswith('resting_nice')]
        EEG_STIM_FEATURES = [
            x for x in df_all.columns if x.startswith('stim_nice')]

        if 1 in feature_set:
            X = FMRI_FEATURES
            title = 'FMRI (merged)'
            result_df = test_input(df_all, X, y, title=title, model=model)
            result_df['features'] = 'fmri_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 2 in feature_set:
            X = EEG_VISUAL_FEATURES
            title = 'EEG visual (merged)'
            result_df = test_input(df_all, X, y, title=title, model=model)
            result_df['features'] = 'eeg_visual_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 3 in feature_set:
            X = EEG_ABCD
            title = 'EEG abcd (merged)'
            result_df = test_input(df_all, X, y, title=title, model=model)
            result_df['features'] = 'eeg_abcd_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 4 in feature_set:
            X = EEG_MODEL
            title = 'EEG model (merged)'
            result_df = test_input(df_all, X, y, title=title, model=model)
            result_df['features'] = 'eeg_model_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 5 in feature_set:
            X = EEG_RESTING_FEATURES + EEG_STIM_FEATURES
            title = 'EEG features (all, merged)'
            confounds = ['Electrodes']
            result_df = test_input(
                df_all, X, y, title=title, model=model, confounds=confounds)

            result_df['features'] = 'eeg_features_both_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 6 in feature_set:
            X = EEG_STIM_FEATURES
            title = 'EEG features (stim, merged)'
            confounds = ['Electrodes']
            result_df = test_input(
                df_all, X, y, title=title, model=model, confounds=confounds)

            result_df['features'] = 'eeg_features_stim_merged'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 7 in feature_set:
            X = EEG_RESTING_FEATURES
            title = 'EEG features (resting, merged)'
            confounds = ['Electrodes']
            result_df = test_input(
                df_all, X, y, title=title, model=model, confounds=confounds)

            result_df[  # type: ignore
                'features'] = 'eeg_features_resting_merged'
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)  # type: ignore

        all_dfs = pd.concat(all_dfs)
        if len(feature_set) > 1:
            suffix = '_'.join(feature_set)
        else:
            suffix = f'{feature_set[0]}'
        all_dfs.to_csv(
            f'results/2_unimodalmerged_{model}_{y}_{suffix}.csv', sep=';')
