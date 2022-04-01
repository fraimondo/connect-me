import numpy as np
from scipy import stats


models_labels = {
    'rf': 'Random Forest',
    'gssvm': 'Support Vector Machine'
}

features_labels = {
    'eeg_visual_full': 'Visual\n(EEG)',
    'eeg_abcd_full': 'ABCD\n(EEG)',
    'eeg_features_both_full': 'Markers\n(EEG)',
    'eeg_features_resting_full': 'Markers R\n(EEG)',
    'eeg_features_stim_full': 'Markers S\n(EEG)',
    'eeg_model_full': 'SVM Model\n(EEG)',
    'fmri_full': 'FC\n(fMRI)',

    'eeg_visual_merged': 'Visual\n(EEG)',
    'eeg_abcd_merged': 'ABCD\n(EEG)',
    'eeg_features_both_merged': 'Markers\n(EEG)',
    'eeg_features_resting_merged': 'Markers R\n(EEG)',
    'eeg_features_stim_merged': 'Markers S\n(EEG)',
    'eeg_model_merged': 'SVM Model\n(EEG)',
    'fmri_merged': 'FC\n(fMRI)',

    'fmri_eegall': 'FC (fMRI)\nMarkers (EEG)',
    'fmri_eegresting': 'FC (fMRI)\nMarkers R\n(EEG)',
    'fmri_eegstim': 'FC (fMRI)\nMarkers S\n(EEG)',
    'fmri_eegmeta': 'FC (fMRI)\nABCD (EEG)\nSVM (EEG)\nVisual (EEG)',
    'eegmeta': 'ABCD (EEG)\nSVM (EEG)\nVisual (EEG)',
    'eegall': 'ABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)',
    'all': 'FC (fMRI)\nABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)',
}

agesex_features_labels = {
    'agesex_eegvisual': 'Visual\n(EEG)',
    'agesex_eegvisual_shuffled': 'Visual\n(EEG)',
    'agesex_eegabcd': 'ABCD\n(EEG)',
    'agesex_eegabcd_shuffled': 'ABCD\n(EEG)',
    'agesex_eegmodel': 'SVM Model\n(EEG)',
    'agesex_eegmodel_shuffled': 'SVM Model\n(EEG)',
    'agesex_fmrieegall': 'FC (fMRI)\nMarkers (EEG)',
    'agesex_fmrieegall_shuffled': 'FC (fMRI)\nMarkers (EEG)',
    'agesex_eegall': 'ABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)',
    'agesex_eegall_shuffled':
        'ABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)',
    'agesex_all':
        'FC (fMRI)\nABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)',
    'agesex_all_shuffled':
        'FC (fMRI)\nABCD (EEG)\nSVM (EEG)\nVisual (EEG)\nMarker (EEG)'
}


metrics_labels = {
    'test_roc_auc': 'ROC AUC',
    'test_precision': 'Precision',
    'test_recall': 'Recall',
}


def compute_ci(data, ci=95, use_percentile=True):
    x = np.mean(data)

    if use_percentile:
        ci_lower, ci_upper = np.percentile(
            data, [(100 - ci) / 2, ci + ((100 - ci) / 2)])
    else:
        n_samples = len(data)
        ci = ci / 100
        if n_samples <= 30:
            ci_lower, ci_upper = stats.t.interval(
                ci, len(data)-1, loc=x, scale=stats.sem(data))
        else:
            ci_lower, ci_upper = stats.norm.interval(
                alpha=ci, loc=np.mean(data), scale=stats.sem(data))
    return x, ci_lower, ci_upper
