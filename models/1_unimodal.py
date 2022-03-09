import julearn
from julearn.utils import logger
import pandas as pd
from utils import (get_data, FMRI_FEATURES, EEG_ABCD, EEG_MODEL,
                   EEG_VISUAL_FEATURES, test_input)

from argparse import ArgumentParser
julearn.utils.configure_logging(level="INFO")

# Configuration
ys = ['doc.enrollment', 'doc.discharge']
# ys = ['doc.discharge']
models = ['gssvm', 'rf']
# models = ['rf']

parser = ArgumentParser(description='Compute unimodal models')

parser.add_argument('--features', metavar='features', nargs='+', type=int,
                    help='Features set to compute [1-7].',
                    required=True)

feature_set = parser.parse_args().features

logger.info(f'Features set: {feature_set}')

for y in ys:
    for model in models:

        # Compute univariate models, using maximum available data
        all_dfs = []
        if 1 in feature_set:
            df = get_data(fmri=True)
            X = FMRI_FEATURES
            title = 'FMRI (not merged)'
            result_df = test_input(df, X, y, title=title, model=model)
            result_df['features'] = 'fmri_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 2 in feature_set:
            df = get_data(eeg_visual=True)
            X = EEG_VISUAL_FEATURES
            title = 'EEG VISUAL (not merged)'
            result_df = test_input(df, X, y, title=title, model=model)
            result_df['features'] = 'eeg_visual_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 3 in feature_set:
            df = get_data(eeg_abcd=True)
            X = EEG_ABCD
            title = 'EEG ABCD (not merged)'
            result_df = test_input(df, X, y, title=title, model=model)
            result_df['features'] = 'eeg_abcd_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 4 in feature_set:
            X = EEG_MODEL
            df = get_data(eeg_model=True)
            title = 'EEG MODEL (not merged)'
            result_df = test_input(df, X, y, title=title, model=model)
            result_df['features'] = 'eeg_model_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 5 in feature_set:
            df = get_data(eeg_features=True, eeg_kind='all')
            X = [x for x in df.columns if 'nice' in x]
            confounds = ['Electrodes']
            title = 'EEG Features (all, not merged)'
            result_df = test_input(
                df, X, y, title=title, confounds=confounds, model=model)

            result_df['features'] = 'eeg_features_both_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 6 in feature_set:
            df = get_data(eeg_features=True, eeg_kind='stim')
            X = [x for x in df.columns if 'nice' in x]
            confounds = ['Electrodes']
            title = 'EEG Features (stim, not merged)'
            result_df = test_input(
                df, X, y, title=title, confounds=confounds, model=model)

            result_df['features'] = 'eeg_features_stim_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        if 7 in feature_set:
            df = get_data(eeg_features=True, eeg_kind='resting')
            X = [x for x in df.columns if 'nice' in x]
            confounds = ['Electrodes']
            title = 'EEG Features (resting, not merged)'
            result_df = test_input(
                df, X, y, title=title, confounds=confounds, model=model)

            result_df['features'] = 'eeg_features_resting_full'  # type: ignore
            result_df['model'] = model  # type: ignore
            all_dfs.append(result_df)

        all_dfs = pd.concat(all_dfs)
        suffix = '_'.join(feature_set)
        all_dfs.to_csv(f'results/1_unimodal_{model}_{y}_{suffix}.csv', sep=';')